# Dinomaly 全流程耗时基准（100 张长条图抽样）

- 日期：2026-10-16
- 脚本：`tools/bench_dinomaly_time.py`
- 数据集：`E:\TestImg\2026-09-22(14.02)`（129 张 jpg，抽样 100 张，`seed=42`）
- 图像规格：31901×1000 灰度（L）长条图

## 测试环境

| 项目 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 4090（`cuda:0`） |
| torch | 2.11.0+cu128 |
| Python | conda env `analysis_system` |
| 处理模式 | `fixed_crop`（全图 25 等分，25 子图/张） |
| 判定配置 | 默认（`ANOMALY_CROP_EXCLUDE_JUDGE=2`，detect/display=0） |

代码路径完全复用产线实现（`core/anomaly/dinomaly/dinomaly_service.py` 的
`_crop_fixed` → `process_pil_image`×25 → `_merge_crop_results` → `_save_merged_results`），
含归一化、JET 伪彩、PNG 落盘全流程。预热 2 张不计入统计。

## 结果（98 张计入统计）

### 分阶段平均耗时（单张 31901×1000 长条图）

| 阶段 | 平均耗时 | 占比 | 内容 |
|---|---|---|---|
| load | 79.3 ms | 3.8% | JPEG 解码 + 转 RGB |
| crop | 158.6 ms | 7.6% | 中心裁剪 + 25 等分切片 |
| **infer** | **1069.3 ms** | **50.9%** | 25 次前向 + 异常图计算；单次前向 ≈ 42.8 ms |
| merge | 85.0 ms | 4.0% | 子图异常图回缩放合并 + 总判定 |
| save | 706.9 ms | 33.7% | min-max 归一化 + JET 伪彩 + 2 张 PNG 编码写盘 + JSON |
| **total** | **2099.1 ms** | 100% | — |

### 耗时分布

| 阶段 | min | p95 | max |
|---|---|---|---|
| infer | 0.956 s | 1.150 s | 1.176 s |
| total | 1.969 s | 2.197 s | 2.256 s |

- 端到端平均 **≈ 2.1 s/张**，约 **28.6 张/分钟**
- 纯计时墙钟 207.4 s
- total 极差 0.29 s（±7%），波动主要来自 infer 的 GPU 调度抖动

## 结论

1. **耗时与图像内容复杂度基本无关**：98 张内容各异的图耗时高度一致（±7%），
   计算量由输入尺寸唯一决定（每子图固定 resize 到 448×448 → 392×392）。
2. **save 占 1/3**：两张 2500 万像素 PNG 的编码 + 写盘是大头，是首要优化点
   （调低 `IMWRITE_PNG_COMPRESSION` / 改 JPEG / 落盘异步化）。
3. 单次子图前向 ≈ 42.8 ms；`ANOMALY_CROP_EXCLUDE_DETECT` 每排除 1 份/侧
   省约 2×42.8 ms。

## shm 实时链路模式对比（同一批 100 张，seed=42）

复现 `shm_ingest._process_frame` 每帧实际工作：
`process_from_array`（引擎内存直通）→ UI 预览 QImage 转换；归档单独计时（真实系统为后台线程）。

| 阶段 | file 模式 | shm 模式 | 说明 |
|---|---|---|---|
| load / decode | 79.3 ms | 44.7 ms | shm 真实帧内存直达，decode 仅参考值 |
| crop | 158.6 ms | ┐ | |
| infer（25 前向） | 1069.3 ms | ├ 1878.0 ms | engine 合计（含 `np.repeat` 灰度转 RGB 与归一化） |
| merge | 85.0 ms | ┘ | |
| save（PNG 落盘） | 706.9 ms | → archive **790.2 ms** | 异步线程执行，不占帧周期 |
| UI 预览转换 | — | 139.1 ms | 2×灰度 QImage + JET 上色（32M 像素） |
| **realtime 帧周期** | — | **2017.1 ms** | engine + ui；min 1.887 / p95 2.210 / max 2.240 |

### engine 内部拆分（单张探针）

| 步骤 | 耗时 | 备注 |
|---|---|---|
| `np.repeat` 灰度→RGB | ≈ 440 ms | 首调用值；纯 numpy 逐元素复制 96 MB，未走 SIMD |
| `_crop_fixed` | ≈ 190-225 ms | 全宽 25 等分（1277×1000/片，与 file 的中心 1000×1000 略不同） |
| 25× 前向 | ≈ 1070-1200 ms | 与 file 模式 infer 一致 |
| `_merge_crop_results`（含归一化） | ≈ 275 ms | file 模式归一化计在 save；engine 移入 merge |

### 结论

1. **shm 实时帧周期 ≈ 2.0 s，与 file 模式全程（2.1 s）几乎持平**：省掉的
   PNG 落盘（~707 ms）被 `np.repeat`（~440 ms）+ 归一化并入 merge 的增量
   （~190 ms）+ UI 上色拷贝（139 ms）抵消。
2. **2.0 s > 1 fps 生产周期（1 s）**：协议为 latest-wins 语义，消费跟不上时
   跳帧（`dropped_by_lag` 增长），等效检测率约 0.5 fps——每两帧检一帧。
3. `np.repeat` 劣化探针中随循环恶化（441→2659→4657 ms），但基准 98 张全程
   engine 稳定（min 1.744 / max 2.109 s），批量场景下未见失控。

### 可优化点（按收益排序，均未实施）

1. **灰度→RGB 改 `cv2.cvtColor`**（`dinomaly_engine.py:375`）：SIMD 实现，
   预计 440 ms → 20-40 ms，一行改动、数值等价。
2. **子图批推理**：25 片合成 batch=25 一次前向，GPU 利用率提升，
   infer 有望从 ~1070 ms 显著下降（需验证 batch 维度数值一致性）。
3. **UI 预览降采样后上色**：预览只需屏幕分辨率，32M 像素 JET+三次全尺寸
   tobytes 拷贝（139 ms）可先缩放再上色（需产品确认缩放交互不受影响）。

以上 1+2+3 全部落地后，realtime 有望压进 1 s 预算内，跟上 1 fps 生产节奏。

## 复现

```bash
conda activate analysis_system
# file 模式（文件轮询服务链路）
python tools/bench_dinomaly_time.py --image_dir "E:\TestImg\2026-09-22(14.02)" --num 100 --seed 42
# shm 模式（实时链路帧周期）
python tools/bench_dinomaly_time.py --image_dir "E:\TestImg\2026-09-22(14.02)" --num 100 --seed 42 --mode shm
```
