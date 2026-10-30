#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dinomaly 引擎优化等价性验证（真实权重 + 真实图纸）。

针对 bench 2026-10-16「下一杠杆」①（`_crop_fixed` ndarray 直入）③（merge
归一化 GPU 单趟）两项提速，逐张真实图比对 **优化前路径（A 路）** 与
**现产线路径（B 路）**，确认没有以检测精度换速度：

- A 路：全图 `Image.fromarray` → PIL 入 `_crop_fixed`（原双 96MB 拷贝）→
  merge 用 CPU 归一化 `_min_max_norm`（原公式）→ 旧产线结果
- B 路：`process_array`（rgb ndarray 直入 + GPU 归一化）→ 现产线结果

比对项（全部按"逐位"验收，出现任何差异即视为回归）：
  1. 裁剪像素、mode、位置区间（①）
  2. merge 前画布（float32，含展示排除置零）逐位（③ 的输入等价）
  3. 归一化后 `anomaly_map_uint8` 逐位（③ 的输出等价）
  4. overall_score（bit 级）、overall_anomaly_level、逐 crop 判定与分数
  5. 画布 NaN/Inf 检查（GPU 归一化对 NaN 无定义，须先确认不存在）

两套排除配置各跑一遍：生产 `.env`（detect/judge/display=2/2/2，含边缘置零）
与基准配置（0/2/0，无置零）。

用法：
    python tools/verify_engine_opt_equiv.py --image_dir "E:\\TestImg\\2026-09-22(14.02)" --num 20
退出码 0 = 全部逐位一致。
"""
import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.dinomaly_engine import _min_max_norm, get_dinomaly_engine  # noqa: E402

CONFIGS = [
    ("production(detect=2,judge=2,display=2)", 2, 2, 2),
    ("bench(detect=0,judge=2,display=0)", 0, 2, 0),
]


def _cpu_norm(canvas: np.ndarray) -> np.ndarray:
    """优化前归一化（原产线公式，逐字符复刻）。"""
    return (_min_max_norm(canvas) * 255).astype(np.uint8)


def _legacy_path(engine, gray: np.ndarray, detect_n: int, judge_n: int, display_n: int):
    """A 路：按优化前 `process_array` 的代码顺序重放（全图 PIL + CPU 归一化）。"""
    rgb = cv2.cvtColor(np.ascontiguousarray(gray), cv2.COLOR_GRAY2RGB)
    image = Image.fromarray(rgb, "RGB")
    crops_full = engine._crop_fixed(image)                 # ① 旧入参：PIL 全图
    crops = crops_full
    if detect_n > 0 and len(crops) > 2 * detect_n:
        crops = crops[detect_n:-detect_n]                  # 检测排除：切片时完成
    results = engine.process_pil_images([c for c, _ in crops])
    crops_results = list(zip(results, [p for _, p in crops]))
    merged = engine._merge_crop_results(crops_results, gray.shape[::-1],
                                        detect_n=detect_n, judge_n=judge_n,
                                        display_n=display_n, include_heatmap=False)
    return crops_full, crops_results, merged


def _compare(tag, failures, path, crops_full, crops_b, canvas_a, canvas_b,
             merged_a, pred_b, metrics_b) -> None:
    def check(cond: bool, msg: str) -> None:
        if not cond:
            failures.append(f"[{tag}] {path.name}: {msg}")

    # 1. 裁剪几何与像素（①）
    check(len(crops_full) == len(crops_b), "裁剪数量不一致 "
          f"{len(crops_full)} vs {len(crops_b)}")
    check([p for _, p in crops_full] == [p for _, p in crops_b], "裁剪位置区间不一致")
    for (img_a, _), (img_b, _) in zip(crops_full, crops_b):
        check(img_a.mode == img_b.mode, f"裁剪 mode 不一致 {img_a.mode} vs {img_b.mode}")
        if not np.array_equal(np.asarray(img_a), np.asarray(img_b)):
            check(False, "裁剪像素不一致")

    # 2. merge 前画布（含展示排除置零）
    check(canvas_a.shape == canvas_b.shape, f"merge 画布形状不一致 {canvas_a.shape} vs {canvas_b.shape}")
    if canvas_a.shape == canvas_b.shape:
        check(np.array_equal(canvas_a, canvas_b),
              f"merge 画布不一致 max|Δ|={np.abs(canvas_a - canvas_b).max()}")
    check(np.isfinite(canvas_a).all(), "画布含 NaN/Inf（GPU 归一化前提不成立）")

    # 3. 归一化 uint8 输出（③）
    diff = np.abs(merged_a["anomaly_map_uint8"].astype(np.int16) - pred_b.astype(np.int16))
    check(diff.max() == 0, f"anomaly_map_uint8 不逐位一致 max|Δ|={diff.max()}"
                           f" 差异像素={int((diff > 0).sum())}")

    # 4. 判定与分数
    check(merged_a["overall_score"].hex() == float(metrics_b["sample_score"]).hex(),
          f"overall_score 不一致 {merged_a['overall_score']!r} vs {metrics_b['sample_score']!r}")
    check(merged_a["overall_anomaly_level"] == metrics_b["anomaly_level"],
          f"overall_anomaly_level 不一致 {merged_a['overall_anomaly_level']} vs "
          f"{metrics_b['anomaly_level']}")
    check(merged_a["crop_results"] == metrics_b["crop_results"], "逐 crop 判定/分数/位置不一致")
    check(merged_a["num_crops"] == metrics_b["num_crops"], "num_crops 不一致")
    check(merged_a["excluded_edge_crops"] == metrics_b["excluded_edge_crops"],
          "excluded_edge_crops 不一致")


def main() -> int:
    parser = argparse.ArgumentParser(description="引擎优化等价性验证（① 裁剪直入 / ③ GPU 归一化）")
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--num", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    files = sorted(p for p in Path(args.image_dir).iterdir()
                   if p.is_file() and p.suffix.lower() in exts)
    if not files:
        print(f"[FAIL] 目录无图片: {args.image_dir}")
        return 2
    picked = random.Random(args.seed).sample(files, min(args.num, len(files)))
    print(f"images: {len(picked)} / {len(files)}  seed={args.seed}")

    engine = get_dinomaly_engine()
    print(f"device: {engine.device}  threshold: {engine.threshold}")

    failures: list = []
    for tag, detect_n, judge_n, display_n in CONFIGS:
        print(f"\n===== 排除配置 {tag} =====")
        for path in picked:
            gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                failures.append(f"[{tag}] {path.name}: 解码失败")
                continue
            original = engine._normalize_canvas_uint8
            cap_a, cap_b = [], []

            def cpu_capture(canvas):
                cap_a.append(canvas.copy())
                return _cpu_norm(canvas)

            def gpu_capture(canvas):
                cap_b.append(canvas.copy())
                return original(canvas)

            engine._normalize_canvas_uint8 = cpu_capture
            try:
                crops_full, _cr, merged_a = _legacy_path(engine, gray, detect_n, judge_n, display_n)
            finally:
                engine._normalize_canvas_uint8 = original

            rgb = cv2.cvtColor(np.ascontiguousarray(gray), cv2.COLOR_GRAY2RGB)
            crops_b = engine._crop_fixed(rgb)               # ① 新入参：ndarray

            engine._normalize_canvas_uint8 = gpu_capture
            try:
                pred_b, _heat_b, metrics_b = engine.process_array(
                    gray, image_type="very long",
                    exclude_edge_crops=judge_n,
                    exclude_detect_crops=detect_n,
                    exclude_display_crops=display_n)
            finally:
                engine._normalize_canvas_uint8 = original

            _compare(tag, failures, path, crops_full, crops_b, cap_a[0], cap_b[0],
                     merged_a, pred_b, metrics_b)
            print(f"  {path.name}: score={metrics_b['sample_score']:.6f} "
                  f"level={metrics_b['anomaly_level']} crops={metrics_b['num_crops']} "
                  f"ui8max|Δ|={int(np.abs(merged_a['anomaly_map_uint8'].astype(np.int16) - pred_b.astype(np.int16)).max())}")

    print("\n===== 结论 =====")
    if failures:
        print(f"[FAIL] {len(failures)} 项不一致：")
        for f in failures:
            print("  -", f)
        return 1
    print("[PASS] 全部逐位一致：裁剪像素 / merge 画布 / uint8 输出 / "
          "overall_score(bit) / 判定级别 / 逐 crop 结果")
    return 0


if __name__ == "__main__":
    sys.exit(main())
