#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dinomaly 处理耗时基准：随机抽取 N 张长条图，按产线流程跑全程，分阶段计时。

--mode file（默认）：文件轮询服务链路（dinomaly_service._crop_fixed →
    process_pil_image×25 → _merge_crop_results → _save_merged_results，
    含归一化 + JET 伪彩 + PNG 落盘）。
--mode shm：共享内存实时链路每帧做的事（shm_ingest._process_frame 复现）：
    cv2 解码（参考值，真实帧内存直达无此步）→ process_from_array（引擎内存
    直通：切片+推理+合并+归一化，不落盘）→ UI 预览 QImage 转换；归档
    （ArchiveWorker.archive：原图+三件套落盘）单独计时（真实系统为异步线程）。

用法：
    python tools/bench_dinomaly_time.py --image_dir "E:\\TestImg\\..." --num 100
    python tools/bench_dinomaly_time.py --image_dir "E:\\TestImg\\..." --num 100 --mode shm
"""

import argparse
import random
import shutil
import sys
import time
from pathlib import Path

import cv2
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SERVICE_DIR = PROJECT_ROOT / "core" / "anomaly" / "dinomaly"
sys.path.append(str(SERVICE_DIR))
sys.path.append(str(SERVICE_DIR.parent.parent / "algo_packages"))


def run_file_mode(args, picked: list) -> dict:
    from dinomaly_service import (
        DEFAULT_ANOMALY_MODEL_PATH, AnomalyProcessorService, SharedModelManager)

    model_manager = SharedModelManager(model_path=args.model_path
                                       or DEFAULT_ANOMALY_MODEL_PATH, device=args.device)
    tmp_out = Path(args.image_dir).parent / "_bench_dinomaly_out"
    shutil.rmtree(tmp_out, ignore_errors=True)
    tmp_out.mkdir(parents=True, exist_ok=True)
    service = AnomalyProcessorService(
        model_manager=model_manager, input_dir=str(Path(args.image_dir)),
        output_dir=str(tmp_out))

    def run_one(path: Path, pid: str) -> dict:
        t = {}
        t0 = time.perf_counter()
        image = Image.open(path).convert("RGB")
        t["load"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        crops = service._crop_fixed(image)
        t["crop"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        crops_results = [(model_manager.process_pil_image(img), pos) for img, pos in crops]
        t["infer"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        merged = service._merge_crop_results(crops_results, image.size,
                                             detect_n=0, judge_n=2, display_n=0)
        t["merge"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        service._save_merged_results(pid, merged, "fixed_crop")
        t["save"] = time.perf_counter() - t0
        t["total"] = sum(t.values())
        return t

    keys = ["load", "crop", "infer", "merge", "save", "total"]
    cleanup = tmp_out
    return {"run_one": run_one, "keys": keys, "cleanup": cleanup,
            "warmup_pid": lambda i: f"warmup_{i}", "picked_pid": lambda p: p.stem}


def run_shm_mode(args, picked: list) -> dict:
    from utils.anomaly_detection_client import (
        ENGINE_DINOMALY, ENGINE_RULE_BASED, AnomalyDetectionClient)
    from utils.archive_worker import ArchiveItem, ArchiveWorker
    from utils.dinomaly_engine import DEFAULT_WEIGHT_PATH
    from utils.shm_ingest import (
        _PREVIEW_MAX_EDGE, _RESULT_MAX_EDGE, _downscale,
        _to_qimage_gray, _to_qimage_jet)

    engine_name = ENGINE_RULE_BASED if args.engine == "rule_based" else ENGINE_DINOMALY
    client = AnomalyDetectionClient()
    tmp_archive = Path(args.image_dir).parent / "_bench_dinomaly_archive_out"
    shutil.rmtree(tmp_archive, ignore_errors=True)
    tmp_archive.mkdir(parents=True, exist_ok=True)
    archiver = ArchiveWorker(tmp_archive)  # 不 start()，同步调用 archive() 计时

    def run_one(path: Path, pid: str) -> dict:
        t = {}
        t0 = time.perf_counter()
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        t["decode"] = time.perf_counter() - t0  # 参考值：真实 shm 帧内存直达，无此开销

        t0 = time.perf_counter()
        pred, heat, metrics = client.process_from_array(gray, engine_name, frame_id=0)
        t["engine"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        # 与 shm_ingest._process_frame 的 emit 镜像：先降采样到显示端上限再转 QImage
        # （pred/heat 同对象时只降一次，与 shm_ingest 同款）
        pred_small = _downscale(pred, _RESULT_MAX_EDGE)
        heat_small = pred_small if heat is pred else _downscale(heat, _RESULT_MAX_EDGE)
        _to_qimage_gray(_downscale(gray, _PREVIEW_MAX_EDGE))
        _to_qimage_gray(pred_small)
        _to_qimage_jet(heat_small)
        t["ui"] = time.perf_counter() - t0

        t["realtime"] = t["engine"] + t["ui"]  # 帧周期（未含 ring 写入 memcpy，memcpy 级）

        t0 = time.perf_counter()
        archiver.archive(ArchiveItem(
            frame_seq=0, frame_id=0, process_id=metrics["process_id"],
            image_type="very long", gray=gray, pred_u8=pred, heat_u8=heat,
            json_dict=metrics))
        t["archive"] = time.perf_counter() - t0  # 真实系统为后台异步线程
        t["sum"] = t["decode"] + t["realtime"] + t["archive"]
        return t

    keys = ["decode", "engine", "ui", "realtime", "archive", "sum"]
    return {"run_one": run_one, "keys": keys, "cleanup": tmp_archive,
            "warmup_pid": lambda i: i, "picked_pid": lambda p: p.stem}


def main() -> None:
    parser = argparse.ArgumentParser(description="Dinomaly 全流程耗时基准（file/shm 双模式）")
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--num", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--mode", choices=["file", "shm"], default="file")
    parser.add_argument("--engine", choices=["dinomaly", "rule_based"], default="dinomaly",
                        help="shm 模式推理引擎（file 模式固定走服务器版 dinomaly 链路）")
    args = parser.parse_args()

    if args.mode == "file" and args.engine != "dinomaly":
        raise SystemExit("--engine 仅作用于 --mode shm（file 模式固定走服务器版 dinomaly 链路）")

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    files = sorted(p for p in Path(args.image_dir).iterdir()
                   if p.is_file() and p.suffix.lower() in exts)
    if len(files) < args.num:
        raise SystemExit(f"目录内图片不足: {len(files)} < {args.num}")
    picked = random.Random(args.seed).sample(files, args.num)

    bench = (run_file_mode(args, picked) if args.mode == "file" else run_shm_mode(args, picked))
    run_one, keys = bench["run_one"], bench["keys"]

    # 预热 2 张（CUDA 上下文/cuDNN 自动调优），不计入统计
    for i in range(2):
        run_one(picked[i], bench["warmup_pid"](i))
        print(f"[warmup {i + 1}/2] {picked[i].name}")

    stats = {k: [] for k in keys}
    start = time.perf_counter()
    n = len(picked) - 2
    for i, path in enumerate(picked[2:], start=1):
        t = run_one(path, bench["picked_pid"](path))
        for k in keys:
            stats[k].append(t[k])
        if i % 10 == 0 or i == n:
            print(f"[{i}/{n}] {path.name}  total={t.get('realtime', t.get('total')):.3f}s")
    wall = time.perf_counter() - start

    print(f"\n===== {args.mode} mode =====")
    print(f"samples: {n}  dir: {args.image_dir}  seed={args.seed}  device=cuda:0")
    for k in keys:
        v = stats[k]
        avg = sum(v) / n
        extra = ""
        if k in ("total", "infer", "realtime", "engine"):
            vs = sorted(v)
            p95 = vs[int(n * 0.95) - 1]
            extra = f"  min={vs[0]:.3f}s  p95={p95:.3f}s  max={vs[-1]:.3f}s"
        print(f"{k:9s} avg {avg * 1000:8.1f} ms{extra}")
    core = "realtime" if args.mode == "shm" else "total"
    avg_core = sum(stats[core]) / n
    print(f"\ncore ({core}): {avg_core:.3f} s/张（约 {60 / avg_core:.1f} 张/分钟）")
    print(f"wall: {wall:.1f} s")
    shutil.rmtree(bench["cleanup"], ignore_errors=True)


if __name__ == "__main__":
    main()
