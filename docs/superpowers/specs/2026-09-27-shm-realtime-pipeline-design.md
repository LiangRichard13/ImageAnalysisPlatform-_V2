# 共享内存实时图像链路改造设计（采集端 / 分析端 / 孪生端）

- 日期：2026-09-27
- 状态：待评审
- 关联代码：`anomaly_detection_tab.py`、`utils/anomaly_detection_client.py`、`utils/rule_based_wrinkle.py`、`utils/dinomaly_engine.py`

## 1. 背景与问题

### 1.1 现状链路（全文件式）

```
相机端(IKapExpert 封闭软件, 存 PNG 到监控目录 G:\TestImg\...)
  → 写盘 + PNG 编码
anomaly_detection_tab.py: BatchProcessingThread 每 5s scandir 轮询
  → 解码读图 → shutil.copy2 原图到 download/anomaly_api/input/{id}/
  → 引擎推理 → 编码写 output/{id}.png / _heatmap.png / .json
孪生数字界面(外部 Qt 程序) 轮询读 input/ + output/ → 再解码展示
```

### 1.2 慢的根源

1. 一张 31901×1000 线阵大图（GRAY8 ≈ 32MB）在链路上被**编解码 3 次以上**（相机端存图编码 → 分析端解码 → 结果 PNG 编码 → 孪生端解码）；
2. 批处理线程 5 秒轮询间隔 + 文件落盘延迟；
3. 硬盘位于实时链路正中，受 Defender/缓存穿透影响。

### 1.3 约束条件（已确认）

| 项 | 值 |
|---|---|
| 部署形态 | 三端同机（Windows 10 Pro，双路 Xeon + RTX 4090） |
| 相机端 | IKapExpert 封闭软件，**不可改代码**；配套 IKapLibrary SDK（C/C++/C#/Python 示例齐全） |
| 孪生端 | Qt/原生桌面，可直读 `OpenFileMapping`，**无需 Web 桥接** |
| 分析端 | 本 repo（PyQt5 + rule_based / dinomaly 双引擎） |
| 出图频率 | ≈1 fps（秒级出图） |
| 图像规格 | 31901×1000，灰度 8bit（协议字段化，允许未来 16bit/彩色） |

## 2. 目标与非目标

### 目标

1. 帧就绪 → 孪生端可读 **≤ 100ms**（现状 ≥ 5s + 编码耗时）；
2. 分析链路（帧就绪 → 结果写入结果 ring）≤ 推理耗时 + 50ms；
3. 硬盘移出实时链路：存图/存档全部异步后台化，归档目录结构与现状**完全一致**（孪生端过渡期可双路读取）；
4. 回退能力：`.env` 一键切回现有文件链路（`IPC_MODE=file`），行为零改动。

### 非目标

- 不修改相机端封闭软件本体；
- 不做跨机传输（同机假设成立，跨机留待 V2 评估 ZeroMQ/DDS）;
- 不做历史回放/录播功能（归档仍按现有目录结构落盘）。

## 3. 总体架构与进程拓扑

```
[Phase 3 过渡] IKapExpert 存图 ──> shm_file_bridge.py(解码入ring) ─┐
[Phase 4 终态] IKap采集进程(SDK帧回调直写) ────────────────────────┤
                                                                   ▼
              ┌──────────── Local\IAP_SHM_FRAME_V1 (8槽环形) ─────────────┐
              │  latest-wins：环满覆盖最旧，生产者永不阻塞                  │
              └───────┬──────────────────┬──────────────────┬────────────┘
                      ▼                  ▼                  ▼
              分析端(本repo)         孪生端(只读)        归档worker(异步)
              event唤醒→memcpy出     QImage 直 wrap      后台编码 PNG 落盘
              → 引擎推理(ndarray)    零拷贝显示          目录结构不变
                      │
                      ▼  Local\IAP_SHM_RESULT_V1 (结果ring, 8槽)
              ┌──────────────────────────────────────────────┐
              └───────┬──────────────────┬───────────────────┘
                      ▼                  ▼
                 孪生端展示         归档worker(同一进程)
```

设计要点：**原始帧 ring 与结果 ring 解耦**。dinomaly 秒级推理拖慢的只是结果链路；孪生端始终实时看到最新原始帧，不受推理速度影响。

## 4. 共享内存协议规范（V1）

### 4.1 对象与命名

三个命名对象，页文件 backed（`PAGE_READWRITE`，不碰磁盘），`Local\` 命名空间，版本号进对象名。命名空间前缀可由 `.env SHM_NAMESPACE` 覆盖（默认 `Local\IAP`），便于多套环境并行。

| 对象名 | 大小 | 用途 |
|---|---|---|
| `IAP_SHM_CTRL_V1` | 4KB | 控制块：全局配置与状态 |
| `IAP_SHM_FRAME_V1` | 8 × 36MB = 288MB | 原始帧环形缓冲 |
| `IAP_SHM_RESULT_V1` | 8 × 36MB = 288MB | 检测结果环形缓冲 |

内存预算合计 ≈ 576MB（预算上限 700MB）。

所有偏移为 slot/块内字节偏移；多字节整数一律**小端**（x86/x64 原生序）；时间戳统一 `timestamp_ns`（Unix 纳秒），心跳为毫秒。

### 4.2 CTRL 块布局（4KB）

| 偏移 | 大小 | 字段 | 说明 |
|---|---|---|---|
| 0 | 4 | `magic` | `IAPS`（0x49 0x41 0x50 0x53） |
| 4 | 4 | `version` | 协议版本，本版 = 1；不匹配必须拒绝 attach 并报错 |
| 8 | 4 | `frame_width` | u32，如 31901 |
| 12 | 4 | `frame_height` | u32，如 1000 |
| 16 | 4 | `pixel_format` | u32 枚举：0=GRAY8，1=GRAY16，2=RGB8（预留） |
| 20 | 4 | `frame_slots` | u32 = 8 |
| 24 | 4 | `frame_slot_bytes` | u32 = 37748736（36MB） |
| 28 | 4 | `result_slots` | u32 = 8 |
| 32 | 4 | `result_slot_bytes` | u32 = 37748736（36MB） |
| 36 | 4 | `reserved` | 0 |
| 40 | 8 | `frame_write_seq` | u64，最新**已完成**原始帧的全局序号（从 1 递增，0=无帧） |
| 48 | 8 | `result_write_seq` | u64，最新已完成结果的序号 |
| 56 | 8 | `producer_heartbeat_ms` | u64，生产者每次写帧前更新 |
| 64 | 8 | `analyzer_heartbeat_ms` | u64，分析端每轮循环更新 |
| 72 | 8 | `archiver_heartbeat_ms` | u64，归档 worker 更新 |
| 80 | 8 | `twin_heartbeat_ms` | u64，孪生端可选更新（不强制） |
| 88 | 8 | `dropped_by_overwrite` | u64，生产者覆盖未读槽的计数（调试用） |
| 96 | 8 | `dropped_by_lag` | u64，消费者落后一整圈而丢帧的计数 |
| 104 | ~3992 | `reserved` | 全 0，未来扩展（V2 字段只能追加在此区域） |

### 4.3 FRAME ring slot 布局（每槽 36MB）

| 偏移 | 大小 | 字段 | 说明 |
|---|---|---|---|
| 0 | 8 | `seq` | u64，完成序号（= 写入时的 `frame_write_seq`）；**0 = 空/无效**，≥1 且 ≤ `frame_write_seq` 为已完成 |
| 8 | 8 | `timestamp_ns` | u64，采集/入队时刻 |
| 16 | 8 | `frame_id` | u64，生产者自定义帧号（如 IKap 帧计数） |
| 24 | 4 | `crc32` | u32，数据区前 `data_bytes` 字节的 CRC32（消费者可抽查校验） |
| 28 | 4 | `data_bytes` | u32，实际数据长度 = w×h×每像素字节 |
| 32 | 4 | `status` | u32：0=empty，1=writing，2=ready（**仅调试用途**，完成判据以 `seq` 为准） |
| 36 | 28 | `reserved` | 0 |
| 64 | 32MB-64 | `data` | 像素数据，行对齐 = width（无 padding） |

### 4.4 RESULT ring slot 布局（每槽 36MB）

| 偏移 | 大小 | 字段 | 说明 |
|---|---|---|---|
| 0 | 8 | `seq` | u64，结果完成序号 |
| 8 | 8 | `frame_seq` | u64，对应的原始帧 `seq`（溯源用） |
| 16 | 8 | `timestamp_ns` | u64，结果完成时刻 |
| 24 | 4 | `json_bytes` | u32，JSON 区长度（≤ 8192） |
| 28 | 4 | `pred_bytes` | u32，预测标注图长度 = w×h×1 |
| 32 | 4 | `heat_bytes` | u32，热力图长度 = (w/4)×(h/4)×1 |
| 36 | 4 | `crc32` | u32，三个数据区拼接的 CRC32 |
| 40 | 24 | `reserved` | 0 |
| 64 | 8192 | `json` | 检测结果 JSON（与现行 output/{id}.json 字段一致） |
| 64+8192 | pred_bytes | `pred` | 预测标注图 GRAY8（白底黑标线，语义同现行 {id}.png 的灰度版） |
| 其后 | heat_bytes | `heat` | 热力图 GRAY8，1/4 分辨率（显示用途足够；需原始分数图时在 V2 追加字段） |

### 4.5 同步原语与语义

**正确性靠序号，事件只做唤醒。** 理由：manual-reset 事件被多消费者共享时 ResetEvent 时机存在竞态；auto-reset 事件会被一个消费者"偷走"唤醒。因此事件仅为性能优化（避免纯轮询），配 50ms 兜底轮询保证不漏帧。

事件对象（均为 **manual-reset**，命名随命名空间前缀）：

| 事件名 | 置位者 | 用途 |
|---|---|---|
| `IAP_EVT_FRAME` | 帧生产者 | 有新原始帧 |
| `IAP_EVT_RESULT` | 分析端 | 有新结果 |

**生产者写帧流程**（顺序不可交换，x86-64 TSO 下 store 按序可见；C/C++ 端相关字段须 `volatile`/atomic，C# 端用 `Volatile.Write`，Python ctypes 端天然顺序写）：

1. `producer_heartbeat_ms` 更新；
2. 定位 slot = `(frame_write_seq + 1) % frame_slots`，置 `slot.status = 1`；
3. 写 `slot.data`、`data_bytes`、`timestamp_ns`、`frame_id`，算 `crc32`；
4. `slot.seq = frame_write_seq + 1`（**此写为完成标记**），`slot.status = 2`；
5. `frame_write_seq += 1`；
6. `SetEvent(IAP_EVT_FRAME)`。

**消费者读取流程**（latest-wins）：

1. 等待事件（`WaitForSingleObject`，超时 50ms）或 50ms 兜底轮询；
2. `s = frame_write_seq`；若 `s == 0` 或 `s == 上次已见 seq` → 返回；
3. slot = `s % frame_slots`；预读 `slot.seq`，若 ≠ `s` → 数据已被覆盖，`dropped_by_lag++`，回到第 2 步追最新；
4. 读数据（孪生端直接 wrap 零拷贝；分析端 memcpy 出，32MB ≈ 10ms）；
5. 复读 `slot.seq`，若 ≠ `s` → 读取期间被覆盖，丢弃本次，回到第 2 步；
6. 若 `s == frame_write_seq`（已追平）→ `ResetEvent(IAP_EVT_FRAME)`（最后一个追平者重置；竞态无害，兜底轮询兜底）。

### 4.6 消费语义：三端统一 latest-wins

- **孪生端**：只看最新帧，显示完成即释放（零拷贝 wrap 的生命周期必须在本帧渲染结束前结束，跨越下一帧写入前必须解除引用；若实现方无法保证，退化为 memcpy 出）；
- **分析端**：每次推理取当时最新帧，跳过的帧自然丢弃（推理速度 < 生产速度时的设计意图）；
- **归档 worker**：尽力而为，队列容量 16，满则丢最旧归档项并告警计数（归档不影响实时链路）。

生产者**永不阻塞**：环满直接覆盖最旧槽，`dropped_by_overwrite++`（1 fps × 8 槽 = 8 秒缓冲，正常消费远追得上）。

### 4.7 生命周期与版本兼容

- **create-or-open**：首个进程 `CreateFileMappingW` 创建，其余 `OpenFileMappingW` attach；attach 时校验 `magic`/`version`/`frame_slot_bytes` 一致，不一致直接报错退出（禁止静默错读）；
- **句柄与回收**：Windows 在所有句柄关闭后自动回收页文件 backed mapping；进程崩溃无残留文件；
- **生产者接管**：新生产者（含桥接进程重启）接管后，`frame_write_seq` **继续递增、不回零**，消费者以 seq 单调性判断新旧；
- **对端死亡判定**：heartbeat 超时阈值 10s（可配 `.env SHM_HEARTBEAT_TIMEOUT_MS`）→ 消费端 UI 显示"源离线"，mapping 保留等待接管。

## 5. 各端改造清单

### 5.1 分析端（本 repo）

| # | 改动 | 内容 |
|---|---|---|
| A1 | 新增 `utils/ipc/shm_ring.py` | 纯 ctypes 封装（零第三方依赖）：`CreateFileMappingW/OpenFileMappingW/MapViewOfFile/CreateEventW/SetEvent/WaitForSingleObject` + CTRL/slot 结构体读写 + 上述生产/消费流程实现。提供 `FrameRingProducer` / `FrameRingConsumer` / `ResultRingProducer` / `ResultRingConsumer` 四个类 |
| A2 | 新增 `SharedMemoryIngestThread(QThread)` | 取代 `BatchProcessingThread` 的轮询：事件唤醒 → latest-wins 读帧 → memcpy 出 → 调引擎 → 结果写 RESULT ring → 投递归档队列 |
| A3 | `utils/rule_based_wrinkle.py` 拆 `process_array(gray)` | `process_array(gray: np.ndarray) -> (pred_u8, heat_u8, metrics_dict)`；现有 `process_to_dir` 改为薄壳（文件模式继续可用，行为不变） |
| A4 | `utils/dinomaly_engine.py` 补直通入口 | 已有 `process_pil_image(image) -> Dict`；新增 ndarray/PIL 进 → `(pred_u8, heat_u8, json_dict)` 出的包装；`process_to_dir` 保持不变 |
| A5 | `utils/anomaly_detection_client.py` 新增 `process_from_array()` | 内存进出入口；原路径式方法不动 |
| A6 | 新增归档 worker（分析端进程内后台线程） | 从队列取 `(frame_seq, frame ndarray, result ndarray×2, json)` → 按**现行目录结构**落盘：`input/{id}/{frame_id}.png + request.json`、`output/{id}.png / {id}_heatmap.png / {id}.json`。编码在线程池执行，不阻塞实时链路 |
| A7 | `anomaly_detection_tab.py` 接入 | shm 模式下显示"实时链路：在线/离线 + 最新 seq + 丢帧计数"；文件轮询面板置灰提示"由实时链路接管"；引擎选择逻辑复用 |
| A8 | `.env` 配置项 | `IPC_MODE=shm|file`（默认 `file` 回退）、`SHM_NAMESPACE=Local\IAP`、`SHM_FRAME_SLOTS=8`、`SHM_RESULT_SLOTS=8`、`SHM_HEARTBEAT_TIMEOUT_MS=10000` |

### 5.2 文件桥接（Phase 3，我们实现）

`tools/shm_file_bridge.py`（独立进程）：

1. 监听目录 = `.env ONLINE_PROCESSING_AD_DIR`（与现批处理同源）；`ReadDirectoryChangesW`（pywin32）事件驱动 + 200ms 兜底扫描；
2. 新文件完整后（size 稳定判定：两次采样同 size）解码（PIL/cv2 → GRAY8）→ 写入 FRAME ring；
3. **已入 ring 的文件移动到 `<监控目录>/.ingested/` 子目录**（防重复入 ring；如 IKapExpert 对目录内容有依赖导致不可移动，降级为 journal 文件记录已处理文件名——实现时二选一，默认移动）；
4. 桥接进程兼 FRAME ring 的默认创建者（create-or-open 语义，见 4.7）。

链路中仅剩"封闭软件存图编码 + 桥接解码"一段旧路径，其余全部走内存。

### 5.3 采集端交付物（Phase 4，相机端团队实施）

1. 协议文档 `docs/ipc/SHM_PROTOCOL.md`：本文档第 4 节的独立完整版（布局图、字段表、时序图、错误码、心跳/超时）；
2. **C# 与 Python 双参考生产者实现**（Python 版对齐 IKapLibrary SDK 的 `GrabContinuous` 帧回调模式：回调内 memcpy 进 slot；C# 版用 `MemoryMappedFile` + `EventWaitHandle`）；
3. 验收工具 `tools/shm_inspect.py`：dump CTRL 状态/心跳/丢帧计数/单帧导出 PNG，供三端联调自检。

### 5.4 孪生端接入说明

- 事件等待：`QWinEventNotifier`（Qt）包 `OpenEventW` 句柄，或独立线程 `WaitForSingleObject`；
- 图像直读：GRAY8 数据按 width/height 构造 `QImage(data, w, h, QImage::Format_Grayscale8)`，注意零拷贝 wrap 的生命周期约束（4.6 节）；
- 结果读取：同一套 latest-wins 流程读 RESULT ring；
- 过渡期兼容：归档目录（现行 input/output 结构）持续存在，孪生端可先接归档再切实时。

## 6. 实施阶段

| 阶段 | 内容 | 交付物 | 验收标准 |
|---|---|---|---|
| Phase 1 | 协议 + 参考实现 | `utils/ipc/shm_ring.py`、协议文档、单元测试、双进程基准脚本 | 单测全绿；双进程 1fps×32MB 连续 30min，零数据损坏（CRC 抽查全过）、端到端延迟 < 50ms |
| Phase 2 | 分析端接入 | A2–A8 全部改动 | 模拟生产者打帧：结果 ring 时延 ≤ 推理耗时+50ms；`IPC_MODE=file` 回归通过；归档目录结构 diff 与现状一致 |
| Phase 3 | 文件桥接 | `tools/shm_file_bridge.py` | IKapExpert 实际落图 → 孪生端（用 shm_inspect 模拟）读到帧 ≤ 2s（含桥接解码）；重复入 ring = 0 |
| Phase 4 | 采集进程替换 | 协议+参考实现交相机端团队 | SDK 回调 → ring 端到端 ≤ 100ms；心跳/接管语义验证 |

## 7. 错误处理与回退

| 场景 | 行为 |
|---|---|
| 生产者心跳超时（>10s） | 孪生/分析端 UI 显示"源离线"，ring 保留；新生产者接管后续写，seq 不回零 |
| 消费者崩溃 | 生产者无感知（latest-wins 语义天然免疫），无需恢复协议 |
| attach 时 magic/version/slot 尺寸不匹配 | 拒绝 attach，日志输出明确的版本与尺寸对比，退出 |
| ring 创建失败（内存不足） | 启动失败并明确提示，不静默降级 |
| 归档队列满（容量 16） | 丢最旧归档项 + 告警计数，实时链路不受影响 |
| 任何实时链路异常 | `.env IPC_MODE=file` 切回现有文件链路（代码零改动） |

## 8. 测试与验收

1. **单元测试**（Phase 1）：slot 读写、seq 完成语义、落后一圈覆盖检测、CRC 校验、事件唤醒 + 兜底轮询、create-or-open 并发启动；
2. **集成压测**（Phase 1）：双进程 1fps×32MB×30min，CRC 全量抽查，断言 `dropped_by_lag` 语义正确；
3. **回归测试**（Phase 2）：`IPC_MODE=file` 模式下现有批处理/单张处理/检查点行为全部不变；
4. **验收指标**：帧就绪→孪生可读 ≤ 100ms；分析链路 ≤ 推理耗时 + 50ms；内存预算 ≤ 700MB；归档不阻塞实时链路（归档线程暂停注入测试）。

## 9. 风险与对策

| 风险 | 对策 |
|---|---|
| Defender/杀软干预 | 页文件 backed mapping 不经磁盘，影响面小；如压测发现异常，将三端进程加入排除列表 |
| Python GIL 阻塞事件等待 | `WaitForSingleObject` 经 ctypes 调用会释放 GIL；解码/编码在线程池执行 |
| IKapExpert 无法自动连续落图（手动触发模式） | Phase 3 桥接依然成立（有图才入 ring），仅实时性受限于上游 |
| IKap SDK 回调只给副本不给指针 | 协议允许生产侧一次 memcpy（32MB ≈ 10ms），"零拷贝"以 SDK 实际能力为准，语义不受影响 |
| 孪生端实现偏差 | 交付 `shm_inspect.py` 自检工具 + CRC 抽查 + 参考代码，联调时逐端验证 |
| 31901 宽 QImage wrap 对齐问题 | GRAY8 无行 padding（行对齐=width），QImage 构造时确认 bytesPerLine 参数，参考代码中显式传入 |

## 10. 已决策记录

| # | 决策 | 理由 |
|---|---|---|
| 1 | 方案 A：命名共享内存环形缓冲 + 命名事件（否决 ZeroMQ/iceoryx2/mmapped 文件） | 同机部署、32MB 大帧零拷贝刚需、零外部依赖、全栈原生可访问；ZeroMQ 有拷贝，iceoryx2 依赖重且 Windows 成熟度未验证，mmapped 文件没真正脱离磁盘 |
| 2 | latest-wins 而非逐帧队列 | 推理（秒级）慢于生产（1fps），逐帧队列必然无限积压；三端解耦后各取所需 |
| 3 | 分析端 memcpy 出 ring 再推理 | 推理秒级，不能长期持有 slot 引用阻塞生产者覆盖 |
| 4 | 热力图 1/4 分辨率 GRAY8 入 ring | 显示用途足够，控制结果槽体积；原始分数图需求出现时在 V2 追加字段 |
| 5 | 归档目录结构与现状完全一致 | 孪生端零成本过渡，可双路并行验证 |
| 6 | 事件仅做唤醒、seq 保证正确性 | manual-reset 事件的 ResetEvent 竞态跨语言最难讲清；正确性路径越简单越好 |
| 7 | 文件桥接先行（Phase 3），SDK 采集进程殿后（Phase 4） | 相机端封闭不可改，先用桥接拿到 80% 收益，Phase 4 消除最后一段编码 |
| 8 | `slot.status` 仅作调试，完成判据 = `seq` | 单字段判据最不易错；status 供人工诊断 |
