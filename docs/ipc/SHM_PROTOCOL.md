# IAP 共享内存实时图像链路协议（V1）

- 版本：1（变更须同步三份端侧指南与本文件，版本号进对象名）
- 权威来源：`docs/superpowers/specs/2026-09-27-shm-realtime-pipeline-design.md` §4
- 端侧指南：[相机端](相机端接入指南.md) · [分析端](分析端实施方案.md) · [孪生端](孪生端接入指南.md)

## 0. 总则

1. 三端同机 Windows，进程一律 **64 位**；
2. 命名对象统一 `Local\` 前缀（或不带前缀，默认即会话内命名空间），**禁用 `Global\`**（需特权）；
3. 多字节整数一律**小端**；`timestamp_ns` 为 Unix 纳秒，心跳为毫秒；
4. 所有数据区 CRC32 校验，消费者可抽查；
5. 页文件 backed（`PAGE_READWRITE`），不经磁盘，进程全部退出后由 OS 自动回收。

## 1. 对象清单与常量

| 名称 | 类型 | 大小 | 说明 |
|---|---|---|---|
| `Local\IAP_SHM_CTRL_V1` | mapping | 4096B | 控制块 |
| `Local\IAP_SHM_FRAME_V1` | mapping | 8 × 36MB = 288MB | 原始帧环形缓冲（单写多读） |
| `Local\IAP_SHM_RESULT_V1` | mapping | 8 × 64MB = 512MB | 检测结果环形缓冲（单写多读） |
| `Local\IAP_EVT_FRAME` | manual-reset event | — | 有新原始帧（唤醒提示，不保证正确性） |
| `Local\IAP_EVT_RESULT` | manual-reset event | — | 有新结果 |
| `Local\IAP_PRODUCER_MUTEX` | mutex | — | FRAME ring 单生产者占用（同时只允许一个进程写 FRAME） |

内存预算合计 ≈ 800MB（上限 1GB）。心跳超时阈值 10s；事件等待超时/兜底轮询 50ms。

角色约定：**FRAME 生产者**（相机端/桥接进程，持 `IAP_PRODUCER_MUTEX`）、**分析端**（FRAME 消费者 + RESULT 生产者）、**孪生端**（FRAME 与 RESULT 双只读消费者）。

## 2. CTRL 块（4096B）

| 偏移 | 大小 | 字段 | 说明 |
|---|---|---|---|
| 0 | 4 | `magic` | `IAPS` |
| 4 | 4 | `version` | = 1；不匹配必须拒绝 attach 并报错 |
| 8 | 4 | `frame_width` | u32（当前 31901） |
| 12 | 4 | `frame_height` | u32（当前 1000） |
| 16 | 4 | `pixel_format` | 0=GRAY8（V1 交付），1=GRAY16，2=RGB8（预留） |
| 20 | 4 | `frame_slots` | = 8 |
| 24 | 4 | `frame_slot_bytes` | = 37748736（36MB） |
| 28 | 4 | `result_slots` | = 8 |
| 32 | 4 | `result_slot_bytes` | = 67108864（64MB） |
| 36 | 4 | `reserved` | 0 |
| 40 | 8 | `frame_write_seq` | u64，最新已完成原始帧序号；从 1 递增，0=无帧；**单调不回零** |
| 48 | 8 | `result_write_seq` | u64，最新已完成结果序号 |
| 56 | 8 | `producer_heartbeat_ms` | 生产者每轮写帧前更新 |
| 64 | 8 | `analyzer_heartbeat_ms` | 分析端每轮循环更新 |
| 72 | 8 | `archiver_heartbeat_ms` | 归档 worker 更新 |
| 80 | 8 | `twin_heartbeat_ms` | 孪生端可选更新 |
| 88 | 8 | `dropped_by_overwrite` | u64，生产者覆盖未读槽计数（调试） |
| 96 | 8 | `dropped_by_lag` | u64，消费者落后整圈丢帧计数 |
| 104 | ~3992 | `reserved` | 全 0；V2 字段只能追加于此 |

启用 GRAY16/RGB8 时，创建者按 `w×h×每像素字节 + 64B` 向上取整到整 MB 重算 `frame_slot_bytes`，消费者以 CTRL 实际值校验。

## 3. FRAME slot（每槽 36MB）

| 偏移 | 大小 | 字段 | 说明 |
|---|---|---|---|
| 0 | 8 | `seq` | u64，完成序号；**0=空/无效**；≥1 且 ≤ `frame_write_seq` 为已完成。**完成判据以本字段为准** |
| 8 | 8 | `timestamp_ns` | 采集/入队时刻 |
| 16 | 8 | `frame_id` | 生产者帧号 |
| 24 | 4 | `crc32` | 数据区前 `data_bytes` 字节的 CRC32 |
| 28 | 4 | `data_bytes` | = w×h×每像素字节 |
| 32 | 4 | `status` | 0=empty / 1=writing / 2=ready（**仅调试**） |
| 36 | 28 | `reserved` | 0 |
| 64 | 其余 | `data` | 像素数据，行对齐 = width（无 padding） |

槽定位：`slot_index = seq % frame_slots`（seq 从 1 起）。

## 4. RESULT slot（每槽 64MB）

| 偏移 | 大小 | 字段 | 说明 |
|---|---|---|---|
| 0 | 8 | `seq` | u64，结果完成序号（判据同上） |
| 8 | 8 | `frame_seq` | u64，对应原始帧 seq（溯源） |
| 16 | 8 | `timestamp_ns` | 结果完成时刻 |
| 24 | 4 | `json_bytes` | ≤ 8192 |
| 28 | 4 | `pred_bytes` | = w×h×1 |
| 32 | 4 | `heat_bytes` | = w×h×1（全分辨率） |
| 36 | 4 | `crc32` | 三数据区拼接的 CRC32 |
| 40 | 24 | `reserved` | 0 |
| 64 | 8192 | `json` | 检测结果 JSON（字段与现行 output/{id}.json 一致） |
| 64+8192 | pred_bytes | `pred` | 预测标注图 GRAY8 全分辨率 |
| 其后 | heat_bytes | `heat` | 异常强度图 GRAY8 全分辨率（未上色；显示端套 JET 伪彩，等效 `cv2.applyColorMap(gray, cv2.COLORMAP_JET)`） |

## 5. 生产者时序（FRAME；RESULT 同构，对象名替换）

顺序不可交换。x86-64 TSO 下 store 按序可见；C/C++ 相关字段须 `volatile`/atomic，C# 用 `Volatile.Write`，Python ctypes 顺序写即可。

1. 更新 `producer_heartbeat_ms`；
2. `next = frame_write_seq + 1`；`slot = next % frame_slots`；`slot.status = 1`；
3. 写 `slot.data`、`data_bytes`、`timestamp_ns`、`frame_id`，计算 `crc32`；
4. `slot.seq = next`（**完成标记**）；`slot.status = 2`；
5. `frame_write_seq = next`；
6. `SetEvent(IAP_EVT_FRAME)`。

生产者**永不阻塞**：环满覆盖最旧槽，`dropped_by_overwrite++`。

## 6. 消费者时序（latest-wins）

1. 等待事件（超时 50ms）或 50ms 兜底轮询；
2. `s = frame_write_seq`；`s == 0` 或 `s == 上次已见` → 返回；
3. `slot = s % frame_slots`；预读 `slot.seq` ≠ `s` → 已被覆盖，`dropped_by_lag++`，回步骤 2 追最新；
4. 读数据（孪生端零拷贝 wrap / 分析端 memcpy 出）；
5. 复读 `slot.seq` ≠ `s` → 读取期间被覆盖，丢弃本次，回步骤 2；
6. `s == frame_write_seq`（追平）→ `ResetEvent`（最后追平者重置；竞态无害，兜底轮询兜底）。

生命周期：create-or-open（首建者 `CreateFileMappingW`，其余 `OpenFileMappingW`）；attach 时校验 `magic`/`version`/slot 尺寸，不符即报错退出；生产者接管后续写 seq **不回零**；心跳超时 10s 判对端离线，mapping 保留等接管。
