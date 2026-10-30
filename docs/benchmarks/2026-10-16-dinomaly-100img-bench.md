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

## 优化落地结果（2026-10-16，同机同图三轮均值）

| 阶段 | 优化前基线 | ①②③后 | ④后 | 优化内容 |
|---|---|---|---|---|
| engine | 1878.0 ms | 1294.9 ms | 938.4 ms | ① cvtColor −293ms（A/B）② 合批 −493ms ④ transform 线程池 −357ms |
| ui | 139.1 ms | 105.2 ms | ~112 ms | ③ 显示 QImage 上抛前降采样 |
| **realtime** | **2017.1 ms** | **≈1400.2 ms** | **≈1050.6 ms** | 累计 **−48%** |

①②③三轮：1412.5 / 1379.4 / 1408.7 ms；④三轮：1061.3 / 1044.0 / 1046.5 ms
（98 张计入统计，seed=42）。

### ④ transform 线程池并行（bitwise 等价）

- Step 0 实测拆分：transform 296-335ms / stack+H2D 8 / 前向(B=25) 215-228 / 后处理 2-3
- 8-worker 常驻 ThreadPoolExecutor：transform 378→64ms（5.9×；PIL Resize 在 C 层、
  ToTensor/Normalize 在 ATen CPU kernel，主体 GIL 外执行）
- verify_batch_equiv 20 张各项与池化前逐项相同（bitwise 等价坐实）；105 测试全绿

### 数值等价性（tools/verify_batch_equiv.py，20 张真实权重 GPU）

- **操作等价达标**：500 crops 判定级别 0 变化、overall 判定逐图一致、
  max|Δoverall_score| = 1.9e-3
- prompt_idx 翻转 3/500：近平局 prompt（top2 相似度差距 ~4e-6）在 B=1 与
  B=25 注意力 GEMM 归约顺序差异下随机翻转——FP 非结合性，严格逐位等价
  原理上不可达；翻转仅使该 crop 分数 ±4e-3 抖动
- VRAM 峰值 2.74 GiB（batch=25 一次前向），确认默认 `MAX_INFERENCE_BATCH=25`

### 未达 1s 的剩余差距与下一杠杆

realtime ~1051ms 距 1fps 预算（1000ms）尚差 ~51ms。候选（均未实施）：
`_crop_fixed` 的 PIL 往返消除（全图 `np.array(PIL)` 96MB 回拷，估 −30-50ms，
bitwise）；Step B CPU/GPU 流水线（transform 64ms 与前向 228ms 重叠，−64ms，
引入 13+12 chunk 形状噪声需复核等价性）。两者都做 ≈ 940-980ms。

## 复现

## 规则化引擎（rule_based）优化（2026-10-19，实验→定位→优化→复测循环）

基线（30 张 shm）realtime 699.7ms（engine 585 + ui 114）。定位：preprocess 345ms
（medianBlur 100 / CLAHE 46 / adaptiveThreshold 100 / not 13 / CLOSE 64 / OPEN 69，
cv2 全核并行下均为最优算子）+ 归一化 cast 186ms。

| 轮次 | 改动 | bitwise | engine | realtime |
|---|---|---|---|---|
| 基线 | — | — | 585.4 | 699.7 |
| R1 | 32MP float min-max 归一化 → 256 项 LUT 查表（同公式逐值等价，186→~5ms） | ✓ | 436.9* | 562.5* |
| R2 | adaptiveThreshold 直接 THRESH_BINARY_INV（255-x == not(x)，省一趟全幅） | ✓ | ↑合并 | ↑合并 |
| R3 | emit 处 pred/heat 同对象只降采样一次（两引擎共享，ui −33ms） | ✓ | — | 516.4 |

\* R1-R3 合并复测。终验（100 张×3）见下。锚点测试 3 条钉死与旧公式/旧链逐位一致。

### 终验数字（100 张 ×3，seed=42）

realtime：465.9 / 509.0 / 502.9 ms → **均值 ≈493ms**（engine ≈412 + ui ≈81），
较基线 699.7ms **−30%**；1fps 预算裕量约一倍。

### 循环终止判定

剩余 engine ~438ms 中 preprocess 占 ~76%（332ms）——cv2 内置最优算子且全核并行，
语义不变前提下无进一步空间。软件层下一候选为 OpenCL/UMat 迁移 preprocess
（预计 −200ms 级），但 CLAHE 插值等存在 bitwise 风险，**未实施**，需权衡后决策。
收益曲线（700→562→516）呈明显递减，bitwise 红线内判定"无明显收益"。

```bash
conda activate analysis_system
# file 模式（文件轮询服务链路）
python tools/bench_dinomaly_time.py --image_dir "E:\TestImg\2026-09-22(14.02)" --num 100 --seed 42
# shm 模式（实时链路帧周期，dinomaly 引擎）
python tools/bench_dinomaly_time.py --image_dir "E:\TestImg\2026-09-22(14.02)" --num 100 --seed 42 --mode shm
# shm 模式（规则化引擎）
python tools/bench_dinomaly_time.py --image_dir "E:\TestImg\2026-09-22(14.02)" --num 100 --seed 42 --mode shm --engine rule_based
```

## 优化落地结果 ①③（2026-10-30，同机同图三轮均值）

| 阶段 | 优化前 | ①③后 | 变化 |
|---|---|---|---|
| engine | 938.4 ms | **540.2 ms** | −398.2 ms |
| ui | ≈112 ms | 95.3 ms | −17 ms |
| **realtime** | **1050.6 ms** | **635.5 ms** | **−415.1 ms（−39.5%）** |

三轮：engine 540.9 / 533.6 / 546.2；realtime 637.6 / 631.3 / 637.6 ms
（98 张计入统计，seed=42，min 0.574 / p95 0.670 / max 0.777）。

### ① `_crop_fixed` 兼容 ndarray 直入（消除 PIL 全图往返）

- 原路径：`Image.fromarray(rgb)` 全图 96MB 拷贝 + `_crop_fixed` 内 `np.array(image)` 再一次
  96MB 回拷；A/B 实测 353.8 ms → 142.3 ms，**−211.5 ms**（提案估 −30-50 ms 严重低估）。
- 实现：`np.asanyarray` 零拷贝接收 ndarray；逐片仍 `Image.fromarray` 进原 transform 栈
  （PIL BICUBIC 不动），`test_fixed_crop_ndarray_matches_pil_bitwise` 钉死逐片像素逐位一致。
- 回归哨兵：`test_process_array_very_long_skips_full_pil` 白盒计数 `Image.fromarray`，
  出现全宽调用即失败（只允许 25 次逐片调用）。

### ③ merge 归一化 GPU 单趟（`_normalize_canvas_uint8`）

- 原路径：`_min_max_norm` CPU 六趟全幅扫描 195.0 ms → GPU 单趟上传/归一化/下传 24.9 ms，
  **−170.2 ms**；IEEE float32 逐元素 + uint8 截断语义与 CPU 公式一致，`np.array_equal` 逐位相同
  （`tests/test_dinomaly_gpu_norm.py` 3 条，含 flat 画布 NaN→0 守卫与无 torch 回退）。
- resize/merge 组装仍在 CPU：cv2 与 torch interpolate 不等价，不越 bitwise 红线。

### 检测精度等价性验证（tools/verify_engine_opt_equiv.py）

真实权重 + 20 张真实图 × 两套排除配置（生产 2/2/2 含边缘置零、基准 0/2/0），
A 路按优化前代码顺序重放（全图 PIL + CPU 归一化），B 路走现产线 `process_array`：

| 比对项 | 结果 |
|---|---|
| 25 片裁剪像素 / mode / 位置区间（①） | 逐位一致 |
| merge 前画布（float32，含置零） | 逐位一致，无 NaN/Inf |
| `anomaly_map_uint8`（③ 输出） | max\|Δ\|=0，差异像素 0 |
| overall_score | bit 级相等 |
| overall / 逐 crop 判定级别、分数、位置 | 全部一致 |

40 次比对（20 图 × 2 配置）零差异 → **①③ 为纯拷贝/算子迁移，未触碰数值路径**。
全链中唯一非逐位等价环节仍是既有合批推理（B=25 的 GEMM 归约顺序，见上节
`verify_batch_equiv`：500 crops 判定 0 变化、近平局 prompt 翻转 3/500、分数抖动 ±4e-3）。

### 优化后 engine 内部拆分（单张探针，31901×1000）

| 步骤 | 耗时 | 性质 |
|---|---|---|
| cvtColor 灰度→RGB | 25.5 ms | 内存带宽 |
| `_crop_fixed` | 112.4 ms | 逐片 25×96MB/25 拷贝（PIL 入 transform 必需） |
| 25× 前向（B=25 单 chunk） | 298.8 ms | GPU 计算 + transform 线程池 64 ms |
| `_merge_crop_results` | 154.1 ms | 其中归一化 25.9 ms，其余为 25 片 resize + 画布拼装 |

### ② CPU/GPU 流水线重叠：**不实施**（前提不成立）

`MAX_INFERENCE_BATCH=25` 使 very long 为**单 chunk**，不存在"上一 chunk 的 CPU 侧"
可重叠；拆 13+12 两 chunk 实测前向回归 +34~74 ms，大于可重叠上限 64 ms（transform），净收益为负。

### 结论

realtime 635 ms 已较 1 fps 预算（1000 ms）留出约 36% 裕量，丢帧（`dropped_by_lag`）解除。
剩余 engine 540 ms 中约 300 ms 为 GPU 前向、约 150 ms 为 96MB 级内存拷贝（带宽下界），
Python 层可再掘空间已降至 ~50-80 ms 量级——继续投入性价比低，判定收敛。

file 模式链路（`core/anomaly/dinomaly/dinomaly_service.py` 自有裁剪副本）不在本次改动内，
20 张回归冒烟与基线一致（infer 1037.9 / total 2261.0 ms）。

```bash
# 本次终验命令（三轮）
python tools/bench_dinomaly_time.py --image_dir "E:\TestImg6-09-22(14.02)" --num 100 --seed 42 --mode shm
```
