"""规则化褶皱检测管线（进程内版本）

从 core/anomaly/rule_based/rule_based_service.py 移植的检测核心与产物合成逻辑，
供工控软件直接调用，取消共享目录/网络通信环节。

对外接口：
    pipeline = RuleBasedWrinklePipeline()
    png_path, heatmap_path, json_path = pipeline.process_to_dir(
        image_path, output_dir, process_id)

输出三件套结构与服务端契约一致（JSON 字段同构、JET 热力图同管线），
仅依赖 opencv + numpy。
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# JSON 展示用的子图划分，与服务端 Dinomaly 版 fixed_crop 的 25 份划分语义对齐，
# 仅用于 crop_results 字段兼容，不影响实际检测区域
DISPLAY_CROP_START = 1500
DISPLAY_CROP_SIZE = 1000
DISPLAY_NUM_CROPS = 25

# 合成异常图的高斯平滑核（边缘柔化；1 表示不平滑）
SMOOTH_KERNEL_SIZE = 31


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

    def distance(self, point: Sequence[float], l2: float = 0.5, l3: float = 0.1) -> float:
        pos_diff = np.linalg.norm(np.asarray(point[:2]) - self.centroid[:2])
        area_centroid = self.centroid[2] * self.centroid[3]
        area_point = point[2] * point[3]
        area_diff = abs(area_centroid - area_point) / (max(area_centroid, 1.0))
        angle_diff = abs(point[4] - self.centroid[4])
        return pos_diff * (1 + l2 * area_diff + l3 * angle_diff)


class SinglePassClustering:
    """单遍聚类：合并同一褶皱被分割出的多个轮廓"""

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
    """规则化褶皱检测核心（移植自 OptiRoll RuleBasedWrinkleDetector）"""

    def __init__(
        self,
        crop_start: int = 4000,
        crop_width: int = 22000,
        min_height: int = 950,
        min_width: int = 20,
        max_width: int = 80,
        min_ratio: float = 5.0,
        max_angle_dev: float = 30.0,
        clustering_threshold: float = 50.0,
    ):
        self.crop_start = crop_start
        self.crop_width = crop_width
        self.min_height = min_height
        self.min_width = min_width
        self.max_width = max_width
        self.min_ratio = min_ratio
        self.max_angle_dev = max_angle_dev
        self.clustering_threshold = clustering_threshold

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        blurred = cv2.medianBlur(image, 5)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY, blockSize=201, C=25
        )
        binary_inv = cv2.bitwise_not(binary)
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
        refined = cv2.morphologyEx(binary_inv, cv2.MORPH_CLOSE, v_kernel)
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
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
    def _calculate_severity(width: float, height: float, aspect_ratio: float) -> int:
        width_score = 1.0
        if width > 45: width_score = 5.0
        elif width > 35: width_score = 4.0
        elif width > 25: width_score = 3.0
        elif width > 18: width_score = 2.0

        ratio_score = 1.0
        if aspect_ratio > 12: ratio_score = 5.0
        elif aspect_ratio > 9: ratio_score = 4.0
        elif aspect_ratio > 6: ratio_score = 3.0
        elif aspect_ratio > 4: ratio_score = 2.0

        avg_score = (width_score + ratio_score) / 2
        return min(5, max(1, int(round(avg_score))))

    def _calculate_confidence(self, width, height, angle_error, aspect_ratio) -> float:
        confidence = 0.5
        if self.min_width <= width <= self.max_width: confidence += 0.2
        elif width > self.max_width: confidence -= 0.1
        if height > self.min_height * 1.5: confidence += 0.15
        elif height > self.min_height: confidence += 0.1
        if angle_error <= 5: confidence += 0.1
        elif angle_error <= self.max_angle_dev: confidence += 0.05
        if aspect_ratio > self.min_ratio * 1.5: confidence += 0.1
        elif aspect_ratio > self.min_ratio: confidence += 0.05
        return min(1.0, max(0.0, confidence))

    def detect(self, gray: np.ndarray) -> List[Dict]:
        """输入全图灰度图，返回全图坐标系的褶皱列表（dict 含合成异常图用的 contours）"""
        h, w = gray.shape[:2]
        x1, x2 = self.crop_start, min(self.crop_start + self.crop_width, w)
        in_roi = x1 < w
        roi_img = gray[:, x1:x2] if in_roi else gray
        x_offset = x1 if in_roi else 0

        binary_roi = self.preprocess(roi_img)
        contours, _ = cv2.findContours(binary_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates: List[Tuple[List[float], np.ndarray]] = []
        for cnt in contours:
            rect = cv2.minAreaRect(cnt)
            rw, rh, angle = self._calculate_angle_from_min_area_rect(rect)
            angle_error = abs(angle)
            if angle_error > self.max_angle_dev:
                continue
            if rw < self.min_width or rw > self.max_width or rh < self.min_height:
                continue
            if (rh / (rw + 1e-5)) < self.min_ratio:
                continue

            global_cx = x_offset + rect[0][0]
            cnt_shifted = cnt + np.array([x_offset, 0], dtype=cnt.dtype) if x_offset else cnt
            candidates.append(([global_cx, rect[0][1], rw, rh, angle], cnt_shifted))

        clustering = SinglePassClustering(threshold=self.clustering_threshold)
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
                "confidence": self._calculate_confidence(width, height, abs(angle), aspect_ratio),
                "severity": self._calculate_severity(width, height, aspect_ratio),
                "bbox": (
                    float(cx - width / 2), float(cy - height / 2),
                    float(cx + width / 2), float(cy + height / 2),
                ),
                "contours": member_contours,
            })
        return wrinkles


class RuleBasedWrinklePipeline:
    """检测 + 三件套合成 + 落盘的一体化管线"""

    def __init__(self, threshold: float = 0.21, detector: Optional[RuleBasedWrinkleCore] = None):
        self.threshold = threshold
        self.detector = detector or RuleBasedWrinkleCore()

    def _get_anomaly_level(self, score: float) -> str:
        if score < self.threshold:
            return "很可能正常"
        return "很可能异常"

    def _calculate_analog_voltage(self, score: float) -> float:
        if score < self.threshold:
            return 0.0
        normalized_score = (score - self.threshold) / (1.0 - self.threshold)
        normalized_score = min(max(normalized_score, 0.0), 1.0)
        return normalized_score * 5.0

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

    def _build_crop_results(self, wrinkles: List[Dict], img_height: int) -> List[Dict]:
        crop_scores = [0.0] * DISPLAY_NUM_CROPS
        for wrinkle in wrinkles:
            crop_index = int((wrinkle["cx"] - DISPLAY_CROP_START) // DISPLAY_CROP_SIZE)
            if 0 <= crop_index < DISPLAY_NUM_CROPS:
                crop_scores[crop_index] = max(crop_scores[crop_index], wrinkle["confidence"])

        crop_results = []
        for index, score in enumerate(crop_scores):
            x_start = DISPLAY_CROP_START + index * DISPLAY_CROP_SIZE
            crop_results.append({
                "crop_id": index,
                "position": [x_start, 0, x_start + DISPLAY_CROP_SIZE, img_height],
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

    def process_to_dir(self, image_path: str, output_dir: str, process_id: str) -> Tuple[str, str, str]:
        """处理单张图片，三件套写入 output_dir/process_id/，返回 (预测图, 热力图, JSON) 路径"""
        gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError(f"Failed to read image: {image_path}")
        img_height, img_width = gray.shape[:2]

        start_time = time.perf_counter()
        wrinkles = self.detector.detect(gray)
        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.info("Rule-based detection: %d wrinkles found in %.1f ms", len(wrinkles), duration_ms)

        process_dir = Path(output_dir) / process_id
        process_dir.mkdir(parents=True, exist_ok=True)

        anomaly_map = self._compose_anomaly_map((img_height, img_width), wrinkles)
        anomaly_map_path = process_dir / f"{process_id}.png"
        heatmap_path = process_dir / f"{process_id}_heatmap.png"
        json_path = process_dir / f"{process_id}.json"

        cv2.imwrite(str(anomaly_map_path), anomaly_map)
        cv2.imwrite(str(heatmap_path), cvt2heatmap(anomaly_map))

        overall_score = max((wrinkle["confidence"] for wrinkle in wrinkles), default=0.0)
        result_data = {
            "process_id": process_id,
            "processing_mode": "rule_based",
            "model_weight": "rule_based",
            "sample_score": overall_score,
            "anomaly_level": self._get_anomaly_level(overall_score),
            "analog_voltage": self._calculate_analog_voltage(overall_score),
            "original_size": [img_width, img_height],
            "num_crops": DISPLAY_NUM_CROPS,
            "timestamp": datetime.now().isoformat(),
            "crop_results": self._build_crop_results(wrinkles, img_height),
            "files": {
                "anomaly_map": anomaly_map_path.name,
                "heatmap": heatmap_path.name,
            },
            "wrinkles": [self._wrinkle_to_json(wrinkle) for wrinkle in wrinkles],
        }

        with open(json_path, "w", encoding="utf-8") as file:
            json.dump(result_data, file, indent=2, ensure_ascii=False)

        return str(anomaly_map_path), str(heatmap_path), str(json_path)
