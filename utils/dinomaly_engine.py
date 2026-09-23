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

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
ALGO_DIR = _PROJECT_ROOT / "core" / "anomaly" / "dinomaly"
ALGO_PACKAGES_DIR = _PROJECT_ROOT / "core" / "algo_packages"
DEFAULT_WEIGHT_PATH = ALGO_DIR / "weights" / "coating_anomaly_final.pth"

# 共享算法包（models/ 由 dinomaly 与 predgru 两链合并而成，必须先于各算法目录挂载，
# 避免同进程内两个引擎的 models 顶层包互相遮蔽）；utils_/dataset 在 ALGO_DIR 下
if str(ALGO_PACKAGES_DIR) not in sys.path:
    sys.path.insert(0, str(ALGO_PACKAGES_DIR))
if str(ALGO_DIR) not in sys.path:
    sys.path.insert(1, str(ALGO_DIR))

_THRESHOLD = 0.21


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
        self.threshold = _THRESHOLD

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
        import torch

        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(img_tensor)
            en, de = output[0], output[1]
            anomaly_map, _ = self._cal_anomaly_maps(en, de, img_tensor.shape[-1])
            anomaly_map = self.gaussian_kernel(anomaly_map)
            sample_score = torch.max(anomaly_map.flatten(1), dim=1)[0].item()
            anomaly_map_np = anomaly_map[0, 0, :, :].cpu().numpy()

        return {
            "anomaly_map": anomaly_map_np,
            "sample_score": sample_score,
            "processed_size": anomaly_map_np.shape,
            "original_size": image.size,
        }

    # ---------- 判定（与服务器版公式一致） ----------

    def _get_anomaly_level(self, score: float) -> str:
        if score < self.threshold:
            return "很可能正常"
        return "很可能异常"

    def _get_overall_anomaly_level(self, scores) -> str:
        if any(score >= self.threshold for score in scores):
            return "很可能异常"
        return "很可能正常"

    def _calculate_analog_voltage(self, score: float) -> float:
        if score < self.threshold:
            return 0.0
        normalized_score = (score - self.threshold) / (1.0 - self.threshold)
        normalized_score = min(max(normalized_score, 0.0), 1.0)
        return normalized_score * 5.0

    # ---------- 裁剪与合并（与服务器版逻辑一致） ----------

    def _crop_fixed(self, image: Image.Image) -> List[Tuple[Image.Image, Tuple[int, int, int, int]]]:
        image_np = np.array(image)
        height, width = image_np.shape[:2]
        channels = 1 if len(image_np.shape) == 2 else image_np.shape[2]

        crop_start = 1500
        crop_width = 25000
        center_cropped = image_np[:, crop_start:crop_start + crop_width]
        center_cropped_height, _ = center_cropped.shape[:2]

        crops = []
        sub_image_size = 1000
        num_sub_images = 25
        for index in range(num_sub_images):
            x_start = index * sub_image_size
            x_end = x_start + sub_image_size
            sub_image = center_cropped[:, x_start:x_end]
            if channels == 1:
                sub_image_pil = Image.fromarray(sub_image, mode="L")
            else:
                sub_image_pil = Image.fromarray(sub_image, mode="RGB")

            original_x = crop_start + x_start
            position = (original_x, 0, original_x + sub_image_size, center_cropped_height)
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
        exclude_edge_crops: int,
    ) -> Dict:
        width, height = original_size
        merged_anomaly_map = np.zeros((height, width), dtype=np.float32)
        crop_results_data = []
        all_scores = []

        for index, (result, position) in enumerate(crops_results):
            x1, y1, x2, y2 = position
            crop_width = x2 - x1
            crop_height = y2 - y1
            anomaly_map_resized = cv2.resize(result["anomaly_map"], (crop_width, crop_height))
            merged_anomaly_map[y1:y2, x1:x2] = anomaly_map_resized

            crop_results_data.append(
                {
                    "crop_id": index,
                    "position": list(position),
                    "sample_score": result["sample_score"],
                    "anomaly_level": self._get_anomaly_level(result["sample_score"]),
                }
            )
            all_scores.append(result["sample_score"])

        if not all_scores:
            raise ValueError("No crop result available for merge")

        scores_for_overall = list(all_scores)
        if exclude_edge_crops > 0 and len(all_scores) > 2 * exclude_edge_crops:
            scores_for_overall = all_scores[exclude_edge_crops:-exclude_edge_crops]

        overall_score = max(scores_for_overall)

        normalized = _min_max_norm(merged_anomaly_map)
        map_uint8 = (normalized * 255).astype(np.uint8)

        return {
            "anomaly_map_uint8": map_uint8,
            "heatmap": _cvt2heatmap(map_uint8),
            "overall_score": overall_score,
            "overall_anomaly_level": self._get_overall_anomaly_level(scores_for_overall),
            "crop_results": crop_results_data,
            "original_size": original_size,
            "num_crops": len(crops_results),
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
            "files": {
                "anomaly_map": f"{process_dir.name}.png",
                "heatmap": f"{process_dir.name}_heatmap.png",
            },
        }
        return self._write_outputs(process_dir, results["anomaly_map_uint8"], results["heatmap"], result_data)

    # ---------- 对外入口（与规则引擎签名一致） ----------

    def process_to_dir(
        self,
        image_path: str,
        output_dir: str,
        process_id: str,
        image_type: str = "very long",
        exclude_edge_crops: int = 2,
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
            edge_crops = exclude_edge_crops
        else:
            crops = self._crop_adaptive(image)
            processing_mode = "adaptive_crop"
            edge_crops = 0

        crops_results = []
        for crop_image, position in crops:
            result = self.process_pil_image(crop_image)
            crops_results.append((result, position))

        merged = self._merge_crop_results(crops_results, image.size, edge_crops)
        return self._save_merged(process_dir, merged, processing_mode)


_ENGINE: Optional[DinomalyEngine] = None
_ENGINE_LOCK = threading.Lock()


def get_dinomaly_engine() -> DinomalyEngine:
    """懒加载单例：首次调用加载模型权重，之后复用（加锁防并发双重加载）"""
    global _ENGINE
    if _ENGINE is None:
        with _ENGINE_LOCK:
            if _ENGINE is None:
                _ENGINE = DinomalyEngine()
    return _ENGINE
