#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import logging
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(project_root)), "algo_packages"))

from models.dinomaly import Dinomaly
from utils_ import cal_anomaly_maps, get_gaussian_kernel, min_max_norm, cvt2heatmap

# 默认路径配置（本地运行：脚本目录下的 input/output/weights）
_DEFAULT_DIR = Path(__file__).resolve().parent
DEFAULT_ANOMALY_INPUT_DIR = str(_DEFAULT_DIR / "input")
DEFAULT_ANOMALY_OUTPUT_DIR = str(_DEFAULT_DIR / "output")
DEFAULT_ANOMALY_MODEL_PATH = str(_DEFAULT_DIR / "weights" / "coating_anomaly_final.pth")


LOGGER = logging.getLogger("anomaly_service")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def resolve_configured_path(*env_names: str, default_path: Path) -> Path:
    project_root_path = Path(project_root)
    for env_name in env_names:
        value = os.getenv(env_name)
        if value:
            configured_path = Path(value)
            if not configured_path.is_absolute():
                configured_path = project_root_path / configured_path
            return configured_path
    return default_path


class SharedModelManager:
    """Load the anomaly model once and reuse it across all processing modes."""

    def __init__(self, model_path: Optional[str], device: str = "cuda:0"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_path = model_path
        self.model = None
        self.transform = None
        self.gaussian_kernel = None
        self._load_model()

    def _load_model(self) -> None:
        LOGGER.info("Loading anomaly model to device: %s", self.device)
        LOGGER.info("Model checkpoint: %s", self.model_path)

        self.model = Dinomaly().to(self.device)
        if self.model_path and os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            elif "model" in checkpoint:
                self.model.load_state_dict(checkpoint["model"], strict=False)
            elif "state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["state_dict"], strict=False)
            else:
                self.model.load_state_dict(checkpoint, strict=False)
        else:
            LOGGER.warning("Model checkpoint not found, using randomly initialized weights")

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
        LOGGER.info("Anomaly model loaded successfully")

    def process_pil_image(self, image: Image.Image) -> Dict:
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(img_tensor)
            en, de = output[0], output[1]
            anomaly_map, _ = cal_anomaly_maps(en, de, img_tensor.shape[-1])
            anomaly_map = self.gaussian_kernel(anomaly_map)
            sample_score = torch.max(anomaly_map.flatten(1), dim=1)[0].item()
            anomaly_map_np = anomaly_map[0, 0, :, :].cpu().numpy()

        return {
            "anomaly_map": anomaly_map_np,
            "sample_score": sample_score,
            "processed_size": anomaly_map_np.shape,
            "original_size": image.size,
        }


class AnomalyProcessorService:
    def __init__(
        self,
        model_manager: SharedModelManager,
        input_dir: str,
        output_dir: str,
        poll_interval: float = 2.0,
        threshold: float = 0.21,
        exclude_edge_crops: int = 2,
    ):
        self.model_manager = model_manager
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.poll_interval = poll_interval
        self.threshold = threshold
        self.exclude_edge_crops = max(0, exclude_edge_crops)
        self.last_processed_batch: Optional[str] = None

    def _list_batch_dirs(self) -> List[Path]:
        if not self.input_dir.exists():
            self.input_dir.mkdir(parents=True, exist_ok=True)
        return [
            path
            for path in self.input_dir.iterdir()
            if path.is_dir() and not path.name.startswith(".") and not path.name.endswith(".tmp")
        ]

    def _get_latest_batch_dir(self, batch_dirs: Sequence[Path]) -> Optional[Path]:
        if not batch_dirs:
            return None
        return max(batch_dirs, key=lambda path: (path.stat().st_mtime_ns, path.name.lower()))

    def _load_request_metadata(self, batch_dir: Path) -> Dict:
        request_path = batch_dir / "request.json"
        if not request_path.exists():
            raise FileNotFoundError(f"request.json not found in batch dir: {batch_dir}")
        with open(request_path, "r", encoding="utf-8") as file:
            return json.load(file)

    def _find_batch_image(self, batch_dir: Path) -> Path:
        image_paths = [
            path for path in batch_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if not image_paths:
            raise FileNotFoundError(f"No image file found in batch dir: {batch_dir}")
        return sorted(image_paths, key=lambda path: path.name.lower())[0]

    def _get_processing_mode(self, request_data: Dict) -> str:
        image_type = request_data.get("image_type", "")

        if image_type == "square":
            return "square"
        if image_type == "very long":
            return "fixed_crop"
        return "adaptive_crop"

    def _get_output_paths(self, process_id: str) -> Tuple[Path, Path, Path]:
        process_dir = self.output_dir / process_id
        return (
            process_dir / f"{process_id}.png",
            process_dir / f"{process_id}_heatmap.png",
            process_dir / f"{process_id}.json",
        )

    def _is_result_ready(self, process_id: str) -> bool:
        prediction_path, heatmap_path, json_path = self._get_output_paths(process_id)
        return prediction_path.exists() and heatmap_path.exists() and json_path.exists()

    def _get_anomaly_level(self, score: float) -> str:
        if score < self.threshold:
            return "很可能正常"
        return "很可能异常"

    def _get_overall_anomaly_level(self, scores: Sequence[float]) -> str:
        if any(score >= self.threshold for score in scores):
            return "很可能异常"
        return "很可能正常"

    def _calculate_analog_voltage(self, score: float) -> float:
        if score < self.threshold:
            return 0.0
        normalized_score = (score - self.threshold) / (1.0 - self.threshold)
        normalized_score = min(max(normalized_score, 0.0), 1.0)
        return normalized_score * 5.0

    def _save_square_results(self, process_id: str, results: Dict) -> str:
        process_dir = self.output_dir / process_id
        process_dir.mkdir(parents=True, exist_ok=True)

        anomaly_map = results["anomaly_map"]
        anomaly_map_normalized = min_max_norm(anomaly_map)
        anomaly_map_uint8 = (anomaly_map_normalized * 255).astype(np.uint8)

        anomaly_map_path = process_dir / f"{process_id}.png"
        heatmap_path = process_dir / f"{process_id}_heatmap.png"
        json_path = process_dir / f"{process_id}.json"

        cv2.imwrite(str(anomaly_map_path), anomaly_map_uint8)
        cv2.imwrite(str(heatmap_path), cvt2heatmap(anomaly_map_uint8))

        sample_score = results["sample_score"]
        result_data = {
            "process_id": process_id,
            "model_weight": self.model_manager.model_path,
            "sample_score": sample_score,
            "anomaly_level": self._get_anomaly_level(sample_score),
            "analog_voltage": self._calculate_analog_voltage(sample_score),
            "image_size": results["original_size"],
            "processed_size": results["processed_size"],
            "timestamp": datetime.now().isoformat(),
            "files": {
                "anomaly_map": anomaly_map_path.name,
                "heatmap": heatmap_path.name,
            },
        }

        with open(json_path, "w", encoding="utf-8") as file:
            json.dump(result_data, file, indent=2, ensure_ascii=False)

        return str(process_dir)

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
        crops_results: Sequence[Tuple[Dict, Tuple[int, int, int, int]]],
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
                    "position": position,
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
        return {
            "merged_anomaly_map": merged_anomaly_map,
            "overall_score": overall_score,
            "overall_anomaly_level": self._get_overall_anomaly_level(scores_for_overall),
            "crop_results": crop_results_data,
            "original_size": original_size,
            "num_crops": len(crops_results),
        }

    def _save_merged_results(self, process_id: str, results: Dict, processing_mode: str) -> str:
        process_dir = self.output_dir / process_id
        process_dir.mkdir(parents=True, exist_ok=True)

        merged_anomaly_map = results["merged_anomaly_map"]
        anomaly_map_normalized = min_max_norm(merged_anomaly_map)
        anomaly_map_uint8 = (anomaly_map_normalized * 255).astype(np.uint8)

        anomaly_map_path = process_dir / f"{process_id}.png"
        heatmap_path = process_dir / f"{process_id}_heatmap.png"
        json_path = process_dir / f"{process_id}.json"

        cv2.imwrite(str(anomaly_map_path), anomaly_map_uint8)
        cv2.imwrite(str(heatmap_path), cvt2heatmap(anomaly_map_uint8))

        overall_score = results["overall_score"]
        result_data = {
            "process_id": process_id,
            "processing_mode": processing_mode,
            "model_weight": self.model_manager.model_path,
            "sample_score": overall_score,
            "anomaly_level": results["overall_anomaly_level"],
            "analog_voltage": self._calculate_analog_voltage(overall_score),
            "original_size": results["original_size"],
            "num_crops": results["num_crops"],
            "timestamp": datetime.now().isoformat(),
            "crop_results": results["crop_results"],
            "files": {
                "anomaly_map": anomaly_map_path.name,
                "heatmap": heatmap_path.name,
            },
        }

        with open(json_path, "w", encoding="utf-8") as file:
            json.dump(result_data, file, indent=2, ensure_ascii=False)

        return str(process_dir)

    def _process_square(self, process_id: str, image_path: Path) -> str:
        image = Image.open(image_path).convert("RGB")
        results = self.model_manager.process_pil_image(image)
        return self._save_square_results(process_id, results)

    def _process_cropped(self, process_id: str, image_path: Path, processing_mode: str) -> str:
        image = Image.open(image_path).convert("RGB")
        original_size = image.size
        if processing_mode == "fixed_crop":
            crops = self._crop_fixed(image)
            exclude_edge_crops = self.exclude_edge_crops
        else:
            crops = self._crop_adaptive(image)
            exclude_edge_crops = 0

        crops_results = []
        for crop_image, position in crops:
            result = self.model_manager.process_pil_image(crop_image)
            crops_results.append((result, position))

        merged_results = self._merge_crop_results(
            crops_results,
            original_size,
            exclude_edge_crops=exclude_edge_crops,
        )
        return self._save_merged_results(process_id, merged_results, processing_mode)

    def process_batch(self, batch_dir: Path) -> Dict:
        request_data = self._load_request_metadata(batch_dir)
        process_id = batch_dir.name
        image_path = self._find_batch_image(batch_dir)
        processing_mode = self._get_processing_mode(request_data)

        LOGGER.info("Processing anomaly batch %s with mode=%s image=%s", process_id, processing_mode, image_path)

        if processing_mode == "square":
            output_path = self._process_square(process_id, image_path)
        else:
            output_path = self._process_cropped(process_id, image_path, processing_mode)

        return {
            "status": "success",
            "process_id": process_id,
            "output_path": output_path,
            "processing_mode": processing_mode,
        }

    def run_forever(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Anomaly share-dir service started")
        LOGGER.info("Watching input root directory: %s", self.input_dir)
        LOGGER.info("Result output directory: %s", self.output_dir)

        while True:
            try:
                batch_dirs = self._list_batch_dirs()
                latest_batch_dir = self._get_latest_batch_dir(batch_dirs)
                if latest_batch_dir is None:
                    time.sleep(self.poll_interval)
                    continue

                batch_key = f"{latest_batch_dir.name}:{latest_batch_dir.stat().st_mtime_ns}"
                if batch_key == self.last_processed_batch:
                    time.sleep(self.poll_interval)
                    continue

                process_id = latest_batch_dir.name
                if self._is_result_ready(process_id):
                    self.last_processed_batch = batch_key
                    time.sleep(self.poll_interval)
                    continue

                result = self.process_batch(latest_batch_dir)
                if result.get("status") == "success":
                    self.last_processed_batch = batch_key
                    LOGGER.info("Anomaly batch processed successfully: %s", process_id)
                else:
                    LOGGER.error("Anomaly batch failed: %s", process_id)

            except KeyboardInterrupt:
                LOGGER.info("Anomaly share-dir service stopped by user")
                raise
            except Exception as exc:
                LOGGER.exception("Anomaly service loop failed: %s", exc)

            time.sleep(self.poll_interval)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Anomaly detection shared-folder service")
    parser.add_argument("--input_dir", type=str, default=str(Path(__file__).resolve().parent / "input"))
    parser.add_argument("--output_dir", type=str, default=str(Path(__file__).resolve().parent / "output"))
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--poll_interval", type=float, default=2.0)
    parser.add_argument("--threshold", type=float, default=0.21)
    parser.add_argument(
        "--exclude_edge_crops",
        type=int,
        default=2,
        help="每侧排除参与总评判定的边缘子图份数（仅 fixed_crop 模式生效）",
    )
    return parser


def main() -> None:
    setup_logging()
    parser = build_argument_parser()
    args = parser.parse_args()

    model_path = args.model_path or os.getenv("ANOMALY_MODEL_PATH") or DEFAULT_ANOMALY_MODEL_PATH

    model_manager = SharedModelManager(model_path=model_path, device=args.device)
    service = AnomalyProcessorService(
        model_manager=model_manager,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        poll_interval=args.poll_interval,
        threshold=args.threshold,
        exclude_edge_crops=args.exclude_edge_crops,
    )
    service.run_forever()


if __name__ == "__main__":
    main()
