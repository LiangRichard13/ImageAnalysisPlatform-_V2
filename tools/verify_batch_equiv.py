#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dinomaly 合批推理数值一致性验证（真实权重 + GPU）。

逐片单批（A 路）vs 合批（B 路）逐样本比对。批形状不同 → 注意力 GEMM
归约顺序不同 → 特征噪声 ~1e-6，近平局 prompt（top2 相似度差距同量级）
会随机翻转——浮点非结合性所致，严格逐位等价原理上不可达。故本工具验收
**操作等价**：
  - 硬门槛：逐 crop 与 overall 的 anomaly_level 判定完全一致；
    overall_score 差 ≤ LEVEL_TOL（远离阈值才可能达标的宽容差）
  - 观测项：prompt_idx 翻转数、Δscore/Δmap 分布、合并 uint8 LSB、
    VRAM 峰值（定 MAX_INFERENCE_BATCH 默认值）

用法：
    python tools/verify_batch_equiv.py --image_dir "E:\\TestImg\\..." --num 20
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.dinomaly_engine import get_dinomaly_engine  # noqa: E402

LEVEL_TOL = 5e-3  # overall_score 容差：翻转只在近平局 prompt 上产生 ~2e-3 跳变


def main() -> None:
    parser = argparse.ArgumentParser(description="合批推理数值一致性验证")
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--num", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    import random
    files = sorted(p for p in Path(args.image_dir).iterdir()
                   if p.is_file() and p.suffix.lower() in exts)
    picked = random.Random(args.seed).sample(files, min(args.num, len(files)))

    engine = get_dinomaly_engine()

    # forward hook 记录每次 Prompt 前向的 prompt_idx（B, top_k）
    prompt_log = []

    def _hook(module, inputs, output):
        prompt_log.append(output["prompt_idx"].detach().cpu().clone())

    engine.model.model.prompt.register_forward_hook(_hook)

    # torch.cuda.max_memory_allocated
    import torch
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    max_dscore = 0.0
    max_dmap = 0.0
    flips = 0
    level_changes = 0
    overall_level_changes = 0
    max_doverall = 0.0
    compared = 0
    max_lsb = 0
    batch_sizes = []

    level = engine._get_anomaly_level  # 判定函数（阈值同产线）

    for path in picked:
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        rgb = cv2.cvtColor(np.ascontiguousarray(gray), cv2.COLOR_GRAY2RGB)
        from PIL import Image
        image = Image.fromarray(rgb, "RGB")
        crops = engine._crop_fixed(image)
        crop_images = [c for c, _ in crops]

        prompt_log.clear()
        results_a = [engine.process_pil_image(c) for c in crop_images]  # 逐片 B=1
        idx_a = torch.cat(prompt_log, dim=0)                            # (25, 1)

        prompt_log.clear()
        results_b = engine.process_pil_images(crop_images)              # 合批
        idx_b = prompt_log[0]                                           # (25, 1)
        batch_sizes.append(prompt_log[0].shape[0])

        flips += int((idx_a != idx_b).sum().item())
        compared += len(crop_images)
        for ra, rb in zip(results_a, results_b):
            max_dscore = max(max_dscore, abs(ra["sample_score"] - rb["sample_score"]))
            dma = np.max(np.abs(ra["anomaly_map"] - rb["anomaly_map"]))
            max_dmap = max(max_dmap, float(dma))
            if level(ra["sample_score"]) != level(rb["sample_score"]):
                level_changes += 1

        # 全链合并比对：A 路手工合并 vs process_array（内部走合批）
        merged_a = engine._merge_crop_results(
            list(zip(results_a, [p for _, p in crops])), image.size,
            detect_n=0, judge_n=2, display_n=0, include_heatmap=False)
        pred_b, _, metrics_b = engine.process_array(gray, image_type="very long")
        max_lsb = max(max_lsb, int(np.max(np.abs(
            merged_a["anomaly_map_uint8"].astype(int) - pred_b.astype(int)))))
        doverall = abs(merged_a["overall_score"] - metrics_b["sample_score"])
        max_doverall = max(max_doverall, doverall)
        if merged_a["overall_anomaly_level"] != metrics_b["anomaly_level"]:
            overall_level_changes += 1
        print(f"{path.name}: flip={flips} Δoverall={doverall:.2e} "
              f"level(A)={merged_a['overall_anomaly_level']} "
              f"level(B)={metrics_b['anomaly_level']}")

    vram_gb = (torch.cuda.max_memory_allocated() / 2**30
               if torch.cuda.is_available() else 0.0)
    print("\n===== verify_batch_equiv =====")
    print(f"images: {len(picked)}  crops compared: {compared}")
    print(f"crop anomaly_level 变化   = {level_changes}  (硬门槛，要求 0)")
    print(f"overall anomaly_level 变化 = {overall_level_changes}  (硬门槛，要求 0)")
    print(f"max |Δoverall_score| = {max_doverall:.3e}  (tol {LEVEL_TOL})")
    print(f"[观测] prompt_idx flips = {flips}（近平局随机翻转，FP 非结合性）")
    print(f"[观测] max |Δcrop_score| = {max_dscore:.3e}")
    print(f"[观测] max |Δanomaly_map| = {max_dmap:.3e}")
    print(f"[观测] merged uint8 max Δ = {max_lsb} LSB")
    print(f"[观测] batch sizes = {sorted(set(batch_sizes))}")
    print(f"VRAM peak = {vram_gb:.2f} GiB")

    ok = (level_changes == 0 and overall_level_changes == 0
          and max_doverall <= LEVEL_TOL)
    print("RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
