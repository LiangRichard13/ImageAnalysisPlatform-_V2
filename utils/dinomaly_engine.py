"""Dinomaly 深度学习异常检测引擎（进程内版本）

从 core/anomaly/dinomaly/dinomaly_service.py（原服务器版）移植的处理链，
供工控软件直接调用。与规则化引擎（utils/rule_based_wrinkle.py）签名一致。

使用方式（懒加载单例，首次调用才加载模型权重）：
    from utils.dinomaly_engine import get_dinomaly_engine
    png, heatmap, json_path = get_dinomaly_engine().process_to_dir(
        image_path, output_dir, process_id, image_type)

依赖 torch 全家桶（完整依赖组）；依赖缺失/权重缺失时抛出带明确提示的异常，
不影响规则引擎与其他功能。
"""

import json
import logging
import math
import os
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from utils.anomaly_config import (
    resolve_anomaly_threshold,
    resolve_crop_exclusions,
)

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
ALGO_DIR = _PROJECT_ROOT / "core" / "anomaly" / "dinomaly"
ALGO_PACKAGES_DIR = _PROJECT_ROOT / "core" / "algo_packages"
DEFAULT_WEIGHT_PATH = ALGO_DIR / "weights" / "coating_anomaly_final.pth"

# 合批推理 chunk 上限：默认单批 25（"very long" 全片子图数，一次前向最快）；
# 小显存现场可经 env DINOMALY_MAX_BATCH 调低（如 13 → 分 13+12 两批）
MAX_INFERENCE_BATCH = int(os.getenv("DINOMALY_MAX_BATCH", "25"))

# transform 并行池：PIL Resize 在 C 层、ToTensor/Normalize 在 ATen CPU kernel，
# 主体计算 GIL 外执行，线程池可并行；每图调用同一 self.transform（无共享可变
# 状态，线程安全），map 保序 → 数值 bitwise 不变。惰性常驻，免每帧建线程。
_TRANSFORM_POOL: Optional["ThreadPoolExecutor"] = None


def _get_transform_pool() -> "ThreadPoolExecutor":
    global _TRANSFORM_POOL
    if _TRANSFORM_POOL is None:
        from concurrent.futures import ThreadPoolExecutor

        _TRANSFORM_POOL = ThreadPoolExecutor(
            max_workers=min(8, os.cpu_count() or 4),
            thread_name_prefix="DinomalyTransform")
    return _TRANSFORM_POOL

# 共享算法包（models/ 由 dinomaly 与 predgru 两链合并而成，必须先于各算法目录挂载，
# 避免同进程内两个引擎的 models 顶层包互相遮蔽）；utils_/dataset 在 ALGO_DIR 下
if str(ALGO_PACKAGES_DIR) not in sys.path:
    sys.path.insert(0, str(ALGO_PACKAGES_DIR))
if str(ALGO_DIR) not in sys.path:
    sys.path.insert(1, str(ALGO_DIR))


def _min_max_norm(arr) -> np.ndarray:
    # 与服务器 utils_.py:min_max_norm 逐行一致
    a_min, a_max = arr.min(), arr.max()
    return (arr - a_min) / (a_max - a_min)


def _cvt2heatmap(gray) -> np.ndarray:
    # 与服务器 utils_.py:cvt2heatmap 逐行一致
    return cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)


class DinomalyEngine:
    """Dinomaly 异常检测引擎：模型加载一次，复用处理全部图片"""

    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        try:
            import torch
            from torchvision import transforms
            from models.dinomaly import Dinomaly
            from utils_ import cal_anomaly_maps, get_gaussian_kernel
        except ImportError as exc:
            raise RuntimeError(
                "Dinomaly 引擎依赖缺失（torch/torchvision/timm/scikit-learn 等），"
                f"请先安装完整依赖组（见 core/requirements.txt）：{exc}"
            ) from exc

        self._torch = torch
        self._cal_anomaly_maps = cal_anomaly_maps

        self.model_path = model_path or os.getenv("ANOMALY_MODEL_PATH") or str(DEFAULT_WEIGHT_PATH)
        available_device = device or "cuda:0"
        self.device = available_device if torch.cuda.is_available() else "cpu"
        self.threshold = resolve_anomaly_threshold()  # ANOMALY_THRESHOLD 可覆盖，默认 0.21

        logger.info("Loading Dinomaly model to device: %s", self.device)
        logger.info("Model checkpoint: %s", self.model_path)

        self.model = Dinomaly().to(self.device)
        if not self.model_path or not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Dinomaly 模型权重不存在: {self.model_path}\n"
                f"请将 coating_anomaly_final.pth 放到 core/anomaly/dinomaly/weights/ 或用 ANOMALY_MODEL_PATH 指定路径"
            )
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        elif "model" in checkpoint:
            self.model.load_state_dict(checkpoint["model"], strict=False)
        elif "state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["state_dict"], strict=False)
        else:
            self.model.load_state_dict(checkpoint, strict=False)

        self.model.eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize((448, 448)),
                transforms.ToTensor(),
                transforms.CenterCrop(392),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(self.device)
        logger.info("Dinomaly model loaded successfully")

    # ---------- 推理 ----------

    def process_pil_image(self, image: Image.Image) -> Dict:
        """单图推理（square 分支用）：薄包装，委托合批入口取 B=1。"""
        return self.process_pil_images([image])[0]

    def process_pil_images(self, images: List[Image.Image]) -> List[Dict]:
        """合批推理：一次前向处理整批子图，逐样本返回（顺序 = 输入顺序）。

        与逐片调用 process_pil_image 数值等价——eval 下 Prompt 逐样本选择
        （core/algo_packages/models/prompt.py 本地补丁）。chunk 上限
        MAX_INFERENCE_BATCH 防 VRAM 峰值，env DINOMALY_MAX_BATCH 可现场调低。
        """
        import torch

        results = []
        for start in range(0, len(images), MAX_INFERENCE_BATCH):
            chunk = images[start:start + MAX_INFERENCE_BATCH]
            # 线程池并行 CPU transform（每图独立、map 保序，bitwise 不变）
            tensors = torch.stack(
                list(_get_transform_pool().map(self.transform, chunk))).to(self.device)
            with torch.no_grad():
                output = self.model(tensors)
                en, de = output[0], output[1]
                anomaly_map, _ = self._cal_anomaly_maps(en, de, tensors.shape[-1])
                anomaly_map = self.gaussian_kernel(anomaly_map)
                scores = torch.max(anomaly_map.flatten(1), dim=1)[0].tolist()
            maps_np = anomaly_map[:, 0].cpu().numpy()
            for i, image in enumerate(chunk):
                results.append({
                    "anomaly_map": maps_np[i],
                    "sample_score": scores[i],
                    "processed_size": maps_np[i].shape,
                    "original_size": image.size,
                })
        return results

    # ---------- 判定（与服务器版公式一致） ----------

    def _get_anomaly_level(self, score: float) -> str:
        if score <= self.threshold:
            return "正常"
        return "异常"

    def _get_overall_anomaly_level(self, scores) -> str:
        if any(score > self.threshold for score in scores):
            return "异常"
        return "正常"

    def _calculate_analog_voltage(self, score: float) -> float:
        if score <= self.threshold:
            return 0.0
        normalized_score = (score - self.threshold) / (1.0 - self.threshold)
        normalized_score = min(max(normalized_score, 0.0), 1.0)
        return normalized_score * 5.0

    # ---------- 裁剪与合并（与服务器版逻辑一致） ----------

    def _crop_fixed(self, image: Image.Image) -> List[Tuple[Image.Image, Tuple[int, int, int, int]]]:
        image_np = np.array(image)
        height, width = image_np.shape[:2]
        channels = 1 if len(image_np.shape) == 2 else image_np.shape[2]

        # 全图 25 等分：不丢弃两端；宽度不能整除时尾片收窄（x_end 兜底到图宽）
        num_sub_images = 25
        sub_image_size = math.ceil(width / num_sub_images)

        crops = []
        for index in range(num_sub_images):
            x_start = index * sub_image_size
            x_end = min(x_start + sub_image_size, width)
            sub_image = image_np[:, x_start:x_end]
            if channels == 1:
                sub_image_pil = Image.fromarray(sub_image, mode="L")
            else:
                sub_image_pil = Image.fromarray(sub_image, mode="RGB")

            position = (x_start, 0, x_end, height)
            crops.append((sub_image_pil, position))

        return crops

    def _crop_adaptive(self, image: Image.Image) -> List[Tuple[Image.Image, Tuple[int, int, int, int]]]:
        image_np = np.array(image)
        height, width = image_np.shape[:2]
        channels = 1 if len(image_np.shape) == 2 else image_np.shape[2]
        sub_image_size = min(width, height)
        crops = []

        if width >= height:
            num_sub_images = math.ceil(width / sub_image_size)
            for index in range(num_sub_images):
                x_start = index * sub_image_size
                x_end = min(x_start + sub_image_size, width)
                if x_end - x_start < sub_image_size and index > 0:
                    x_start = width - sub_image_size
                    x_end = width
                sub_image = image_np[0:height, x_start:x_end]
                if channels == 1:
                    sub_image_pil = Image.fromarray(sub_image, mode="L")
                else:
                    sub_image_pil = Image.fromarray(sub_image, mode="RGB")
                crops.append((sub_image_pil, (x_start, 0, x_end, height)))
        else:
            num_sub_images = math.ceil(height / sub_image_size)
            for index in range(num_sub_images):
                y_start = index * sub_image_size
                y_end = min(y_start + sub_image_size, height)
                if y_end - y_start < sub_image_size and index > 0:
                    y_start = height - sub_image_size
                    y_end = height
                sub_image = image_np[y_start:y_end, 0:width]
                if channels == 1:
                    sub_image_pil = Image.fromarray(sub_image, mode="L")
                else:
                    sub_image_pil = Image.fromarray(sub_image, mode="RGB")
                crops.append((sub_image_pil, (0, y_start, width, y_end)))

        return crops

    def _merge_crop_results(
        self,
        crops_results: List[Tuple[Dict, Tuple[int, int, int, int]]],
        original_size: Tuple[int, int],
        detect_n: int = 0,
        judge_n: int = 0,
        display_n: int = 0,
        include_heatmap: bool = True,
    ) -> Dict:
        """合并子图：判定范围/展示排除按三级排除换算（独立计数，见 _resolve_crop_exclusions）。

        crops_results 已是检测范围内的子图（detect 排除在切片时完成）；判定相对
        排除数 = judge_n - detect_n；展示排除 = 检出列表两端再各收 display_n-detect_n
        份（清单口径），画布保持全尺寸、保留区外置 0——shm RESULT 协议要求
        pred/heat 全分辨率，物理裁窄会撞 write_result 形状守卫丢帧。
        """
        width, height = original_size
        merged_anomaly_map = np.zeros((height, width), dtype=np.float32)
        all_scores = []

        for result, position in crops_results:
            x1, y1, x2, y2 = position
            anomaly_map_resized = cv2.resize(result["anomaly_map"], (x2 - x1, y2 - y1))
            merged_anomaly_map[y1:y2, x1:x2] = anomaly_map_resized
            all_scores.append(result["sample_score"])

        if not all_scores:
            raise ValueError("No crop result available for merge")

        scores_for_overall = list(all_scores)
        rel_judge = judge_n - detect_n
        if rel_judge > 0 and len(all_scores) > 2 * rel_judge:
            scores_for_overall = all_scores[rel_judge:-rel_judge]
        overall_score = max(scores_for_overall)

        # 展示排除：检出列表两端再各收 display_n-detect_n 份（清单口径）；
        # 画布保持全尺寸、窗口外置 0（物理裁窄会违反 shm RESULT 全分辨率协议；
        # 置零须显式做——窗口外仍可能有检出子图的异常值，清零即"不展示"）
        k = display_n - detect_n
        kept = crops_results
        if k > 0 and len(crops_results) > 2 * k:
            kept = crops_results[k:-k]
        fx1, fy1, fx2, _ = kept[0][1]
        _, ly1, lx2, ly2 = kept[-1][1]
        vertical = fx1 == 0 and fx2 == width  # adaptive 竖图：沿 y 置零
        lo, hi = (fy1, ly2) if vertical else (fx1, lx2)

        crop_results_data = []
        for index, (result, position) in enumerate(kept):
            x1, y1, x2, y2 = position
            crop_results_data.append(
                {
                    "crop_id": index,
                    "position": [x1, y1, x2, y2],
                    "sample_score": result["sample_score"],
                    "anomaly_level": self._get_anomaly_level(result["sample_score"]),
                }
            )

        if vertical:
            merged_anomaly_map[:lo, :] = 0.0
            merged_anomaly_map[hi:, :] = 0.0
        else:
            merged_anomaly_map[:, :lo] = 0.0
            merged_anomaly_map[:, hi:] = 0.0

        normalized = _min_max_norm(merged_anomaly_map)
        map_uint8 = (normalized * 255).astype(np.uint8)

        return {
            "anomaly_map_uint8": map_uint8,
            "heatmap": _cvt2heatmap(map_uint8) if include_heatmap else None,
            "overall_score": overall_score,
            "overall_anomaly_level": self._get_overall_anomaly_level(scores_for_overall),
            "crop_results": crop_results_data,
            "original_size": [width, height],
            "num_crops": len(kept),
            "excluded_edge_crops": {"detect": detect_n, "judge": judge_n, "display": display_n},
        }

    # ---------- 保存（输出目录参数化，返回三元组） ----------

    def _write_outputs(self, process_dir: Path, anomaly_map_uint8, heatmap, result_data: Dict) -> Tuple[str, str, str]:
        process_dir.mkdir(parents=True, exist_ok=True)

        anomaly_map_path = process_dir / f"{process_dir.name}.png"
        heatmap_path = process_dir / f"{process_dir.name}_heatmap.png"
        json_path = process_dir / f"{process_dir.name}.json"

        cv2.imwrite(str(anomaly_map_path), anomaly_map_uint8)
        cv2.imwrite(str(heatmap_path), heatmap)

        with open(json_path, "w", encoding="utf-8") as file:
            json.dump(result_data, file, indent=2, ensure_ascii=False)

        return str(anomaly_map_path), str(heatmap_path), str(json_path)

    def _save_square(self, process_dir: Path, results: Dict) -> Tuple[str, str, str]:
        anomaly_map_uint8 = (_min_max_norm(results["anomaly_map"]) * 255).astype(np.uint8)
        heatmap = _cvt2heatmap(anomaly_map_uint8)

        sample_score = results["sample_score"]
        result_data = {
            "process_id": process_dir.name,
            "model_weight": self.model_path,
            "sample_score": sample_score,
            "anomaly_level": self._get_anomaly_level(sample_score),
            "analog_voltage": self._calculate_analog_voltage(sample_score),
            "image_size": list(results["original_size"]),
            "processed_size": list(results["processed_size"]),
            "timestamp": datetime.now().isoformat(),
            "files": {
                "anomaly_map": f"{process_dir.name}.png",
                "heatmap": f"{process_dir.name}_heatmap.png",
            },
        }
        return self._write_outputs(process_dir, anomaly_map_uint8, heatmap, result_data)

    def _save_merged(self, process_dir: Path, results: Dict, processing_mode: str) -> Tuple[str, str, str]:
        overall_score = results["overall_score"]
        result_data = {
            "process_id": process_dir.name,
            "processing_mode": processing_mode,
            "model_weight": self.model_path,
            "sample_score": overall_score,
            "anomaly_level": results["overall_anomaly_level"],
            "analog_voltage": self._calculate_analog_voltage(overall_score),
            "original_size": list(results["original_size"]),
            "num_crops": results["num_crops"],
            "timestamp": datetime.now().isoformat(),
            "crop_results": results["crop_results"],
            "excluded_edge_crops": results["excluded_edge_crops"],
            "files": {
                "anomaly_map": f"{process_dir.name}.png",
                "heatmap": f"{process_dir.name}_heatmap.png",
            },
        }
        return self._write_outputs(process_dir, results["anomaly_map_uint8"], results["heatmap"], result_data)

    # ---------- 对外入口（与规则引擎签名一致） ----------

    def process_array(
        self,
        gray: np.ndarray,
        image_type: str = "very long",
        exclude_edge_crops: Optional[int] = None,
        exclude_detect_crops: Optional[int] = None,
        exclude_display_crops: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """内存直通入口：GRAY8(H×W) → (pred_u8, heat_u8, metrics)。

        pred 与 heat 同源（未上色强度图，归档时才上色）；跳过合并阶段的
        applyColorMap（include_heatmap=False）。metrics 不含 process_id/files。
        三排除参数 None → 环境变量/默认（独立计数，见 _resolve_crop_exclusions）；
        square 无子图概念，排除数忽略。
        """
        # cvtColor（SIMD）替代 np.repeat 逐元素复制：同环境 A/B 长条帧 engine −290ms，
        # 输出逐字节相同（R=G=B=gray）；ascontiguousarray 为未来非连续输入的零成本防御
        rgb = cv2.cvtColor(np.ascontiguousarray(gray), cv2.COLOR_GRAY2RGB)
        image = Image.fromarray(rgb, "RGB")

        if image_type == "square":
            results = self.process_pil_image(image)
            map_uint8 = (_min_max_norm(results["anomaly_map"]) * 255).astype(np.uint8)
            sample_score = results["sample_score"]
            metrics = {
                "model_weight": self.model_path,
                "sample_score": sample_score,
                "anomaly_level": self._get_anomaly_level(sample_score),
                "analog_voltage": self._calculate_analog_voltage(sample_score),
                "image_size": list(results["original_size"]),
                "processed_size": list(results["processed_size"]),
                "timestamp": datetime.now().isoformat(),
            }
            return map_uint8, map_uint8, metrics

        if image_type == "very long":
            crops = self._crop_fixed(image)
            processing_mode = "fixed_crop"
        else:
            # 行为变更：adaptive 此前强制排除 0，现统一尊重配置值（fixed/adaptive 一致）
            crops = self._crop_adaptive(image)
            processing_mode = "adaptive_crop"

        detect_n, judge_n, display_n = resolve_crop_exclusions(
            exclude_detect_crops, exclude_edge_crops, exclude_display_crops)
        if detect_n > 0 and len(crops) > 2 * detect_n:
            crops = crops[detect_n:-detect_n]  # 检测排除：边缘子图不推理（省算力）

        # 合批推理（detect 排除后的剩余子图一次前向；结果顺序 = crops 顺序）
        crops_results = list(zip(self.process_pil_images([crop for crop, _ in crops]),
                                 [position for _, position in crops]))

        merged = self._merge_crop_results(crops_results, image.size,
                                          detect_n=detect_n, judge_n=judge_n,
                                          display_n=display_n, include_heatmap=False)
        overall_score = merged["overall_score"]
        metrics = {
            "processing_mode": processing_mode,
            "model_weight": self.model_path,
            "sample_score": overall_score,
            "anomaly_level": merged["overall_anomaly_level"],
            "analog_voltage": self._calculate_analog_voltage(overall_score),
            "original_size": list(merged["original_size"]),
            "num_crops": merged["num_crops"],
            "timestamp": datetime.now().isoformat(),
            "crop_results": merged["crop_results"],
            "excluded_edge_crops": merged["excluded_edge_crops"],
        }
        return merged["anomaly_map_uint8"], merged["anomaly_map_uint8"], metrics

    def process_to_dir(
        self,
        image_path: str,
        output_dir: str,
        process_id: str,
        image_type: str = "very long",
        exclude_edge_crops: Optional[int] = None,
        exclude_detect_crops: Optional[int] = None,
        exclude_display_crops: Optional[int] = None,
    ) -> Tuple[str, str, str]:
        """处理单张图片，三件套写入 output_dir/process_id/，返回 (预测图, 热力图, JSON) 路径"""
        image = Image.open(image_path).convert("RGB")
        process_dir = Path(output_dir) / process_id

        if image_type == "square":
            results = self.process_pil_image(image)
            return self._save_square(process_dir, results)

        if image_type == "very long":
            crops = self._crop_fixed(image)
            processing_mode = "fixed_crop"
        else:
            crops = self._crop_adaptive(image)
            processing_mode = "adaptive_crop"

        detect_n, judge_n, display_n = resolve_crop_exclusions(
            exclude_detect_crops, exclude_edge_crops, exclude_display_crops)
        if detect_n > 0 and len(crops) > 2 * detect_n:
            crops = crops[detect_n:-detect_n]

        # 合批推理（detect 排除后的剩余子图一次前向；结果顺序 = crops 顺序）
        crops_results = list(zip(self.process_pil_images([crop for crop, _ in crops]),
                                 [position for _, position in crops]))

        merged = self._merge_crop_results(crops_results, image.size,
                                          detect_n=detect_n, judge_n=judge_n,
                                          display_n=display_n)
        return self._save_merged(process_dir, merged, processing_mode)


_ENGINE: Optional[DinomalyEngine] = None
_ENGINE_LOCK = threading.Lock()
_PRELOAD_THREAD: Optional[threading.Thread] = None


def get_dinomaly_engine() -> DinomalyEngine:
    """懒加载单例：首次调用加载模型权重，之后复用（加锁防并发双重加载）"""
    global _ENGINE
    if _ENGINE is None:
        with _ENGINE_LOCK:
            if _ENGINE is None:
                _ENGINE = DinomalyEngine()
    return _ENGINE


def preload_dinomaly_engine_async() -> Optional[threading.Thread]:
    """后台预热模型权重（幂等）。

    已加载或预热进行中返回 None；否则起 daemon 线程执行懒加载并返回该线程。
    失败只记日志、不清单例——首次真实使用时 get_dinomaly_engine 会重试。
    预热中若推理请求到来，双方在 _ENGINE_LOCK 上串行（语义同现状懒加载）。
    """
    global _PRELOAD_THREAD
    if _ENGINE is not None:
        return None
    if _PRELOAD_THREAD is not None and _PRELOAD_THREAD.is_alive():
        return None
    thread = threading.Thread(target=_preload_worker, daemon=True,
                              name="DinomalyPreload")
    _PRELOAD_THREAD = thread
    thread.start()
    return thread


def _preload_worker() -> None:
    try:
        get_dinomaly_engine()
        logger.info("Dinomaly 权重预热完成，首帧/首张图将不再等待加载")
    except Exception:
        logger.exception("Dinomaly 权重预热失败（首次真实使用时将重试）")
