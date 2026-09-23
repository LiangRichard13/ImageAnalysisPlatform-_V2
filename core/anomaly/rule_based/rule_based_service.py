#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""基于规则的膜面褶皱检测共享目录服务

与 dinomaly_service.py（Dinomaly 深度学习版）的输入输出契约完全一致：
  input/{process_id}/           ← 工控端投递（request.json + 图像）
  output/{process_id}/*.png|json ← 本服务写出三件套
差异仅在推理引擎：预处理 + 轮廓 + 几何规则 + 单遍聚类（仅依赖 cv2 + numpy）。

部署警告：本脚本与 Dinomaly 版监听同一 input 目录，严禁两个进程同时运行，
否则会抢任务并以相同文件名互相覆盖输出。切换顺序：停旧进程 → 启新进程。
"""

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


project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

# 默认路径配置（本地运行：脚本目录下的 input/output）
_DEFAULT_DIR = Path(__file__).resolve().parent
DEFAULT_ANOMALY_INPUT_DIR = str(_DEFAULT_DIR / "input")
DEFAULT_ANOMALY_OUTPUT_DIR = str(_DEFAULT_DIR / "output")

LOGGER = logging.getLogger("anomaly_rule_service")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# JSON 展示用的子图划分：全图宽度 25 等分，与 Dinomaly 版 fixed_crop 的全图划分语义对齐，
# 仅用于 crop_results 字段兼容，不影响实际检测区域
DISPLAY_NUM_CROPS = 25

# 全图评分时剔除的首尾子图数（各 2 片，共 4 片），与 Dinomaly 版 exclude_edge_crops 对齐
_EXCLUDED_EDGE_CROPS = 2

# 合成异常图的高斯平滑核（边缘柔化，接近深度学习版热区观感；1 表示不平滑）
SMOOTH_KERNEL_SIZE = 31

# 规则参数的标定基准：产线原图 31901x1000 的图像高度。
# 检测时按 实际高度/REF_HEIGHT 缩放全部绝对像素参数，使缩小图（如 500x500）同样可用
REF_HEIGHT = 1000


def _odd(value: float, minimum: int = 3) -> int:
    # cv2 的核尺寸/blockSize 必须为正奇数
    v = max(int(round(value)), minimum)
    return v if v % 2 == 1 else v + 1


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


def min_max_norm(image: np.ndarray) -> np.ndarray:
    # 与服务器 utils_.py:min_max_norm 逐行一致；调用方需自行保证 max != min
    a_min, a_max = image.min(), image.max()
    return (image - a_min) / (a_max - a_min)


def cvt2heatmap(gray: np.ndarray) -> np.ndarray:
    # 与服务器 utils_.py:cvt2heatmap 逐行一致
    return cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)


class Cluster:
    """单遍聚类的簇：维护特征质心与成员轮廓点集"""

    def __init__(self, features: Sequence[float], contour: np.ndarray):
        self.centroid = np.array(features, dtype=np.float32)
        self.members: List[Sequence[float]] = [features]
        self.contours: List[np.ndarray] = [contour]
        self.n_samples = 1

    def update_centroid(self, new_features: Sequence[float]) -> None:
        new_point_arr = np.array(new_features, dtype=np.float32)
        self.centroid = (self.centroid * self.n_samples + new_point_arr) / (self.n_samples + 1)
        self.n_samples += 1

    def distance(self, point: Sequence[float], l2: float = 0.5, l3: float = 0.2) -> float:
        pos_diff = np.linalg.norm(np.asarray(point[:2]) - self.centroid[:2])
        area_centroid = self.centroid[2] * self.centroid[3]
        area_point = point[2] * point[3]
        area_diff = abs(area_centroid - area_point) / (max(area_centroid, 1.0))
        angle_diff = abs(point[4] - self.centroid[4])
        return pos_diff * (1 + l2 * area_diff + l3 * angle_diff)


class SinglePassClustering:
    """单遍聚类：合并同一褶皱被分割出的多个轮廓（相对 OptiRoll 原版额外携带轮廓点集）"""

    def __init__(self, threshold: float = 50.0, l2: float = 0.5, l3: float = 0.2):
        self.threshold = threshold
        self.l2 = l2
        self.l3 = l3
        self.clusters: List[Cluster] = []

    def add_point(self, features: Sequence[float], contour: np.ndarray) -> None:
        nearest_cluster: Optional[Cluster] = None
        min_distance = float("inf")

        for cluster in self.clusters:
            dist = cluster.distance(features, self.l2, self.l3)
            if dist < min_distance:
                min_distance = dist
                nearest_cluster = cluster

        if nearest_cluster and min_distance <= self.threshold:
            nearest_cluster.members.append(features)
            nearest_cluster.contours.append(contour)
            nearest_cluster.update_centroid(features)
        else:
            self.clusters.append(Cluster(features, contour))

    def get_results(self) -> List[Tuple[List[float], List[np.ndarray]]]:
        return [(c.centroid.tolist(), c.contours) for c in self.clusters]


class RuleBasedWrinkleCore:
    """规则化褶皱检测核心（移植自 OptiRoll RuleBasedWrinkleDetector，去除项目数据模型依赖）"""

    def __init__(
        self,
        crop_start: int = 0,
        crop_width: int = 10**9,
        min_height: int = 950,
        min_width: int = 20,
        max_width: int = 80,
        min_ratio: float = 5.0,
        max_angle_dev: float = 30.0,
        clustering_threshold: float = 50.0,
    ):
        # 所有构造参数均为 REF_HEIGHT 分辨率下的标定值，detect() 内按图片高度等比缩放。
        # crop_start=0 + 足够大的 crop_width 表示全图检测（detect 内有 min(x2, w) 保护）
        self.crop_start = crop_start
        self.crop_width = crop_width
        self.min_height = min_height
        self.min_width = min_width
        self.max_width = max_width
        self.min_ratio = min_ratio
        self.max_angle_dev = max_angle_dev
        self.clustering_threshold = clustering_threshold

    def preprocess(self, image: np.ndarray, scale: float = 1.0) -> np.ndarray:
        blurred = cv2.medianBlur(image, _odd(5 * scale))
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY, blockSize=_odd(201 * scale, minimum=5), C=25
        )
        binary_inv = cv2.bitwise_not(binary)
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, _odd(15 * scale)))
        refined = cv2.morphologyEx(binary_inv, cv2.MORPH_CLOSE, v_kernel)
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (_odd(3 * scale), _odd(3 * scale)))
        refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, h_kernel)
        return refined

    @staticmethod
    def _calculate_angle_from_min_area_rect(
        rect: Tuple[Tuple[float, float], Tuple[float, float], float]
    ) -> Tuple[float, float, float]:
        (cx, cy), (rw, rh), angle = rect
        if rw > rh:
            rw, rh = rh, rw
            angle += 90

        if angle > 90:
            angle -= 180
        elif angle < -90:
            angle += 180
        return rw, rh, angle

    @staticmethod
    def _calculate_severity(width: float, height: float, aspect_ratio: float, scale: float = 1.0) -> int:
        width_score = 1.0
        if width > 45 * scale: width_score = 5.0
        elif width > 35 * scale: width_score = 4.0
        elif width > 25 * scale: width_score = 3.0
        elif width > 18 * scale: width_score = 2.0

        ratio_score = 1.0
        if aspect_ratio > 12: ratio_score = 5.0
        elif aspect_ratio > 9: ratio_score = 4.0
        elif aspect_ratio > 6: ratio_score = 3.0
        elif aspect_ratio > 4: ratio_score = 2.0

        avg_score = (width_score + ratio_score) / 2
        return min(5, max(1, int(round(avg_score))))

    def _calculate_confidence(self, width, height, angle_error, aspect_ratio, scale: float = 1.0) -> float:
        confidence = 0.5
        if self.min_width * scale <= width <= self.max_width * scale: confidence += 0.2
        elif width > self.max_width * scale: confidence -= 0.1
        if height > self.min_height * scale * 1.5: confidence += 0.15
        elif height > self.min_height * scale: confidence += 0.1
        if angle_error <= 5: confidence += 0.1
        elif angle_error <= self.max_angle_dev: confidence += 0.05
        if aspect_ratio > self.min_ratio * 1.5: confidence += 0.1
        elif aspect_ratio > self.min_ratio: confidence += 0.05
        return min(1.0, max(0.0, confidence))

    def detect(self, gray: np.ndarray) -> List[Dict]:
        """输入全图灰度图，返回全图坐标系的褶皱列表（dict 含合成异常图用的 contours）"""
        h, w = gray.shape[:2]
        scale = h / REF_HEIGHT
        x1 = int(round(self.crop_start * scale))
        x2 = min(x1 + int(round(self.crop_width * scale)), w)
        in_roi = x1 < w
        roi_img = gray[:, x1:x2] if in_roi else gray
        x_offset = x1 if in_roi else 0

        binary_roi = self.preprocess(roi_img, scale)
        contours, _ = cv2.findContours(binary_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_width = self.min_width * scale
        max_width = self.max_width * scale
        min_height = self.min_height * scale

        candidates: List[Tuple[List[float], np.ndarray]] = []
        for cnt in contours:
            rect = cv2.minAreaRect(cnt)
            rw, rh, angle = self._calculate_angle_from_min_area_rect(rect)
            angle_error = abs(angle)
            if angle_error > self.max_angle_dev:
                continue
            if rw < min_width or rw > max_width or rh < min_height:
                continue
            if (rh / (rw + 1e-5)) < self.min_ratio:
                continue

            global_cx = x_offset + rect[0][0]
            cnt_shifted = cnt + np.array([x_offset, 0], dtype=cnt.dtype) if x_offset else cnt
            candidates.append(([global_cx, rect[0][1], rw, rh, angle], cnt_shifted))

        clustering = SinglePassClustering(threshold=self.clustering_threshold * scale)
        for features, cnt in candidates:
            clustering.add_point(features, cnt)

        wrinkles: List[Dict] = []
        for idx, (centroid, member_contours) in enumerate(clustering.get_results()):
            cx, cy, rw, rh, angle = centroid
            width, height = float(rw), float(rh)
            aspect_ratio = height / (width + 1e-5)
            wrinkles.append({
                "id": idx,
                "cx": float(cx),
                "cy": float(cy),
                "width": width,
                "length": height,
                "angle": float(angle),
                "confidence": self._calculate_confidence(width, height, abs(angle), aspect_ratio, scale),
                "severity": self._calculate_severity(width, height, aspect_ratio, scale),
                "bbox": (
                    float(cx - width / 2), float(cy - height / 2),
                    float(cx + width / 2), float(cy + height / 2),
                ),
                "contours": member_contours,
            })
        return wrinkles


class AnomalyProcessorService:
    def __init__(
        self,
        detector: RuleBasedWrinkleCore,
        input_dir: str,
        output_dir: str,
        poll_interval: float = 2.0,
        threshold: float = 0.21,
    ):
        self.detector = detector
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.poll_interval = poll_interval
        self.threshold = threshold
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

    def _is_square_request(self, request_data: Dict) -> bool:
        return request_data.get("image_type", "") == "square"

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
        if score <= self.threshold:
            return "很可能正常"
        return "很可能异常"

    def _calculate_analog_voltage(self, score: float) -> float:
        if score <= self.threshold:
            return 0.0
        normalized_score = (score - self.threshold) / (1.0 - self.threshold)
        normalized_score = min(max(normalized_score, 0.0), 1.0)
        return normalized_score * 5.0

    def _overall_score(self, wrinkles: List[Dict], img_width: int) -> float:
        """全图异常分数：只统计中间 21 片（crop 2..22）内的褶皱，与 Dinomaly 版
        exclude_edge_crops=2 的评分语义对齐；边缘片仍检出并展示，但不驱动全图判定"""
        if not wrinkles:
            return 0.0
        crop_size = math.ceil(img_width / DISPLAY_NUM_CROPS)
        edge = _EXCLUDED_EDGE_CROPS
        center = [
            wrinkle for wrinkle in wrinkles
            if edge <= int(wrinkle["cx"] // crop_size) < DISPLAY_NUM_CROPS - edge
        ]
        return max((wrinkle["confidence"] for wrinkle in center), default=0.0)

    def _compose_anomaly_map(self, map_shape: Tuple[int, int], wrinkles: List[Dict]) -> np.ndarray:
        canvas = np.zeros(map_shape, dtype=np.uint8)
        # severity 高的后画，重叠时覆盖低严重度区域
        for wrinkle in sorted(wrinkles, key=lambda item: item["severity"]):
            gray_val = int(min(255, 100 + (wrinkle["severity"] - 1) * 39))
            cv2.drawContours(canvas, wrinkle["contours"], -1, gray_val, thickness=cv2.FILLED)

        if not wrinkles:
            return canvas

        smoothed = cv2.GaussianBlur(canvas, (SMOOTH_KERNEL_SIZE, SMOOTH_KERNEL_SIZE), 0)
        normalized = min_max_norm(smoothed.astype(np.float32))
        return (normalized * 255).astype(np.uint8)

    def _build_crop_results(self, wrinkles: List[Dict], img_width: int, img_height: int) -> List[Dict]:
        crop_size = math.ceil(img_width / DISPLAY_NUM_CROPS)
        crop_scores = [0.0] * DISPLAY_NUM_CROPS
        for wrinkle in wrinkles:
            crop_index = int(wrinkle["cx"] // crop_size)
            if 0 <= crop_index < DISPLAY_NUM_CROPS:
                crop_scores[crop_index] = max(crop_scores[crop_index], wrinkle["confidence"])

        crop_results = []
        for index, score in enumerate(crop_scores):
            x_start = index * crop_size
            x_end = min(x_start + crop_size, img_width)
            crop_results.append({
                "crop_id": index,
                "position": [x_start, 0, x_end, img_height],
                "sample_score": score,
                "anomaly_level": self._get_anomaly_level(score),
            })
        return crop_results

    @staticmethod
    def _wrinkle_to_json(wrinkle: Dict) -> Dict:
        x_min, y_min, x_max, y_max = wrinkle["bbox"]
        return {
            "id": wrinkle["id"],
            "confidence": wrinkle["confidence"],
            "severity": wrinkle["severity"],
            "angle_degree": wrinkle["angle"],
            "geometry": {
                "width_px": wrinkle["width"],
                "length_px": wrinkle["length"],
                "center_x": wrinkle["cx"],
                "center_y": wrinkle["cy"],
            },
            "bounding_box": {
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max,
            },
        }

    def _save_results(
        self,
        process_id: str,
        img_width: int,
        img_height: int,
        wrinkles: List[Dict],
        is_square: bool,
    ) -> str:
        process_dir = self.output_dir / process_id
        process_dir.mkdir(parents=True, exist_ok=True)

        anomaly_map = self._compose_anomaly_map((img_height, img_width), wrinkles)
        anomaly_map_path = process_dir / f"{process_id}.png"
        heatmap_path = process_dir / f"{process_id}_heatmap.png"
        json_path = process_dir / f"{process_id}.json"

        cv2.imwrite(str(anomaly_map_path), anomaly_map)
        cv2.imwrite(str(heatmap_path), cvt2heatmap(anomaly_map))

        overall_score = self._overall_score(wrinkles, img_width)
        result_data = {
            "process_id": process_id,
            "processing_mode": "rule_based",
            "model_weight": "rule_based",
            "sample_score": overall_score,
            "anomaly_level": self._get_anomaly_level(overall_score),
            "analog_voltage": self._calculate_analog_voltage(overall_score),
            "original_size": [img_width, img_height],
            "timestamp": datetime.now().isoformat(),
            "files": {
                "anomaly_map": anomaly_map_path.name,
                "heatmap": heatmap_path.name,
            },
            "wrinkles": [self._wrinkle_to_json(wrinkle) for wrinkle in wrinkles],
        }

        if is_square:
            result_data["image_size"] = [img_width, img_height]
            result_data["processed_size"] = list(anomaly_map.shape)
        else:
            result_data["num_crops"] = DISPLAY_NUM_CROPS
            result_data["crop_results"] = self._build_crop_results(wrinkles, img_width, img_height)

        with open(json_path, "w", encoding="utf-8") as file:
            json.dump(result_data, file, indent=2, ensure_ascii=False)

        return str(process_dir)

    def _process_image(self, process_id: str, image_path: Path, is_square: bool) -> str:
        gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError(f"Failed to read image: {image_path}")
        img_height, img_width = gray.shape[:2]

        start_time = time.perf_counter()
        wrinkles = self.detector.detect(gray)
        duration_ms = (time.perf_counter() - start_time) * 1000
        LOGGER.info("Rule-based detection: %d wrinkles found in %.1f ms", len(wrinkles), duration_ms)

        return self._save_results(process_id, img_width, img_height, wrinkles, is_square)

    def process_batch(self, batch_dir: Path) -> Dict:
        request_data = self._load_request_metadata(batch_dir)
        process_id = batch_dir.name
        image_path = self._find_batch_image(batch_dir)
        is_square = self._is_square_request(request_data)

        LOGGER.info("Processing anomaly batch %s (rule_based, square=%s) image=%s", process_id, is_square, image_path)
        output_path = self._process_image(process_id, image_path, is_square)

        return {
            "status": "success",
            "process_id": process_id,
            "output_path": output_path,
            "processing_mode": "rule_based",
        }

    def run_forever(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Rule-based anomaly share-dir service started")
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
                LOGGER.info("Rule-based anomaly share-dir service stopped by user")
                raise
            except Exception as exc:
                LOGGER.exception("Anomaly service loop failed: %s", exc)

            time.sleep(self.poll_interval)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rule-based wrinkle detection shared-folder service")
    parser.add_argument("--input_dir", type=str, default=str(Path(__file__).resolve().parent / "input"))
    parser.add_argument("--output_dir", type=str, default=str(Path(__file__).resolve().parent / "output"))
    parser.add_argument("--poll_interval", type=float, default=2.0)
    parser.add_argument("--threshold", type=float, default=0.21)
    parser.add_argument(
        "--process", type=str, default=None,
        help="单图验证模式：直接处理指定图像并输出三件套后退出，不进入目录轮询",
    )
    parser.add_argument("--out_dir", type=str, default="single_test_output", help="单图验证模式的输出目录")
    parser.add_argument("--crop_start", type=int, default=0, help="ROI 左边界（原图 x 坐标），0 表示从全图开始")
    parser.add_argument("--crop_width", type=int, default=10**9, help="ROI 宽度（像素），极大值表示全图检测")
    parser.add_argument("--min_width", type=int, default=20, help="褶皱最小宽度（像素）")
    parser.add_argument("--max_width", type=int, default=80, help="褶皱最大宽度（像素）")
    parser.add_argument("--min_height", type=int, default=950, help="褶皱最小长度（像素）")
    parser.add_argument("--min_ratio", type=float, default=5.0, help="褶皱最小长宽比")
    parser.add_argument("--max_angle_dev", type=float, default=30.0, help="与竖直方向最大角度偏差（度）")
    parser.add_argument("--clustering_threshold", type=float, default=50.0, help="单遍聚类合并距离阈值")
    return parser


def run_single_image(args) -> None:
    """单图验证模式：不搭共享目录、不碰真实 input/output，处理一张图出三件套"""
    detector = build_detector(args)
    image_path = Path(args.process)
    if not image_path.is_file():
        LOGGER.error("Image not found: %s", image_path)
        sys.exit(1)

    gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        LOGGER.error("Failed to read image: %s", image_path)
        sys.exit(1)
    img_height, img_width = gray.shape[:2]
    is_square = 0.9 <= (img_width / img_height) <= 1.1

    service = AnomalyProcessorService(
        detector=detector,
        input_dir=str(image_path.parent),
        output_dir=args.out_dir,
        threshold=args.threshold,
    )
    LOGGER.info("Image %dx%d, is_square=%s", img_width, img_height, is_square)
    output_dir = service._process_image(image_path.stem, image_path, is_square)
    LOGGER.info("Results written to: %s", output_dir)


def build_detector(args) -> RuleBasedWrinkleCore:
    return RuleBasedWrinkleCore(
        crop_start=args.crop_start,
        crop_width=args.crop_width,
        min_width=args.min_width,
        max_width=args.max_width,
        min_height=args.min_height,
        min_ratio=args.min_ratio,
        max_angle_dev=args.max_angle_dev,
        clustering_threshold=args.clustering_threshold,
    )


def main() -> None:
    setup_logging()
    parser = build_argument_parser()
    args = parser.parse_args()

    # 单图验证模式：无需 input/output 共享目录与 request.json
    if args.process:
        run_single_image(args)
        return

    detector = build_detector(args)
    service = AnomalyProcessorService(
        detector=detector,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        poll_interval=args.poll_interval,
        threshold=args.threshold,
    )
    service.run_forever()


if __name__ == "__main__":
    main()
