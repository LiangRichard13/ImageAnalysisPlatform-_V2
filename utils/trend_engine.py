"""PredGRU 趋势预测引擎（进程内版本）

从 core/trend/trend_service.py（原服务器版 PredictionAPI）移植，
供工控软件直接调用。懒加载单例：首次调用才 import torch 并加载模型权重。

使用方式：
    from utils.trend_engine import get_trend_engine
    prediction_path, json_path = get_trend_engine().process_to_dir(
        dir_path, output_dir, process_id)

输入：包含 >=16 张膜面序列图的文件夹（按文件名数字排序取最后 16 帧）
输出：<output_dir>/<process_id>/prediction.jpg + {process_id}.json
"""

import json
import logging
import os
import re
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
ALGO_DIR = _PROJECT_ROOT / "core" / "trend"
ALGO_PACKAGES_DIR = _PROJECT_ROOT / "core" / "algo_packages"
DEFAULT_WEIGHT_PATH = ALGO_DIR / "weights" / "checkpoints-ontonet-3-weighted+sobel-model-best-2894.ckpt"

# 共享算法包（models/ 由 dinomaly 与 predgru 两链合并而成，必须先于本算法目录挂载，
# 避免同进程内两个引擎的 models 顶层包互相遮蔽）；layers/dataloader/preprocess/a100_utils 在 ALGO_DIR 下
if str(ALGO_PACKAGES_DIR) not in sys.path:
    sys.path.insert(0, str(ALGO_PACKAGES_DIR))
if str(ALGO_DIR) not in sys.path:
    sys.path.insert(1, str(ALGO_DIR))

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class TrendEngine:
    """PredGRU 趋势预测引擎：模型加载一次，复用处理全部批次"""

    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        try:
            import torch
            from models.model_factory import Model
            from dataloader.wrinkle import InputHandle
            from a100_utils import preprocess
            from a100_utils.eval_bbox import IoU
            from preprocess.tojson import hough, calculate_voltage_from_slope
        except ImportError as exc:
            raise RuntimeError(
                "趋势预测引擎依赖缺失（torch/torchinfo/timm 等），"
                f"请先安装完整依赖组（见 core/requirements.txt）：{exc}"
            ) from exc

        self._torch = torch
        self._Model = Model
        self._InputHandle = InputHandle
        self._preprocess = preprocess
        self._IoU = IoU
        self._hough = hough
        self._calculate_voltage_from_slope = calculate_voltage_from_slope

        self.configs = self._create_configs(model_path=model_path, device=device)

        logger.info("Loading trend model to device: %s", self.configs.device)
        logger.info("Model checkpoint: %s", self.configs.pretrained_model)
        if not os.path.exists(self.configs.pretrained_model):
            raise FileNotFoundError(
                f"趋势预测模型权重不存在: {self.configs.pretrained_model}\n"
                f"请将 checkpoint 放到 core/trend/weights/ 或用 TREND_MODEL_PATH 指定路径"
            )
        self.model = self._Model(self.configs)
        self.model.load(self.configs.pretrained_model)
        logger.info("Trend model loaded successfully")

    def _create_configs(self, model_path: Optional[str], device: Optional[str]):
        import torch

        class Config:
            def __init__(self):
                self.model_name = "predgru_v2"
                self.pretrained_model = model_path or os.getenv("TREND_MODEL_PATH") or str(DEFAULT_WEIGHT_PATH)

                self.input_length = 15
                self.total_length = 16
                self.reverse_input = 1
                requested_device = device or os.getenv("TREND_DEVICE") or "cuda"
                self.device = requested_device if torch.cuda.is_available() else "cpu"
                self.num_hidden = "64,64,64,64"
                self.patch_size = 4
                self.img_channel = 1
                self.img_width = 800
                self.img_height = 100
                self.filter_size = 9
                self.stride = 1
                self.layer_norm = 1
                self.batch_size = 1
                self.lr = 0.0001

                self.gen_frm_dir = "./work_dirs/spinet-3"
                self.save_dir = "./checkpoints/"

        return Config()

    # ---------- 数据准备（与服务器版一致） ----------

    @staticmethod
    def _sort_image_paths(image_paths: Sequence[Path]) -> List[Path]:
        def sort_key(path: Path):
            name = path.stem
            numbers = re.findall(r"\d+", name)
            if numbers:
                return (0, tuple(int(num) for num in numbers), name.lower())
            return (1, path.stat().st_mtime_ns, name.lower())

        return sorted(image_paths, key=sort_key)

    def _load_images_from_paths(self, image_paths: Sequence[Path]) -> Tuple[List[np.ndarray], List[str]]:
        frames_np: List[np.ndarray] = []
        filenames: List[str] = []

        for image_path in image_paths:
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                logger.warning("Failed to read image: %s", image_path)
                continue

            if image.shape[:2] != (self.configs.img_height, self.configs.img_width):
                image = cv2.resize(
                    image,
                    (self.configs.img_width, self.configs.img_height),
                    interpolation=cv2.INTER_AREA,
                )

            frames_np.append(np.array(image, dtype=np.float32) / 255.0)
            filenames.append(image_path.name)

        return frames_np, filenames

    def _create_dummy_masks(self, frames_count: int) -> List[np.ndarray]:
        return [
            np.zeros((self.configs.img_height, self.configs.img_width), dtype=np.float32)
            for _ in range(frames_count)
        ]

    def _create_data_handle(self, frames_np: Sequence[np.ndarray], masks_np: Sequence[np.ndarray]):
        start_index = max(0, len(frames_np) - self.configs.total_length)
        indices = [start_index]
        data = (frames_np, masks_np)

        input_param = {
            "paths": ["", ""],
            "image_width": self.configs.img_width,
            "image_height": self.configs.img_height,
            "minibatch_size": self.configs.batch_size,
            "seq_length": self.configs.total_length,
            "input_data_type": "float32",
            "name": "wrinkle",
        }

        input_handle = self._InputHandle(data, indices, input_param)
        input_handle.begin(do_shuffle=False)
        return input_handle

    def _eval_bbox_logic(self, image: np.ndarray, threshold: int) -> List[int]:
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        sobelx = cv2.convertScaleAbs(sobelx)
        sobely = cv2.convertScaleAbs(sobely)
        edges = cv2.addWeighted(sobelx, 0.75, sobely, 0.75, 0)

        x, y = np.nonzero(edges > threshold)
        if len(x) == 0 or len(y) == 0:
            height, width = image.shape[:2]
            return [0, 0, width - 1, height - 1]

        minx = np.min(x)
        maxx = np.max(x)
        miny = np.min(y)
        maxy = np.max(y)
        return [int(miny), int(minx), int(maxy), int(maxx)]

    # ---------- 推理（与服务器版 _run_prediction 一致） ----------

    def _run_prediction(self, input_handle, frames_np: Sequence[np.ndarray]) -> Tuple[np.ndarray, str, float]:
        test_ims = input_handle.get_batch()
        if test_ims is None:
            raise ValueError("Failed to get data batch")

        if len(test_ims) == 1:
            test_dat = self._preprocess.reshape_patch(test_ims, self.configs.patch_size)
        else:
            test_dat = [
                self._preprocess.reshape_patch(test_ims[0], self.configs.patch_size),
                self._preprocess.reshape_patch(test_ims[1], self.configs.patch_size),
            ]

        test_ims[0] = test_ims[0][:, :, :, :, : self.configs.img_channel]
        test_ims[1] = test_ims[1][:, :, :, :, : self.configs.img_channel]

        real_input_flag = np.zeros(
            (
                self.configs.batch_size,
                self.configs.total_length - self.configs.input_length - 1,
                self.configs.img_height // self.configs.patch_size,
                self.configs.img_width // self.configs.patch_size,
                self.configs.patch_size ** 2 * self.configs.img_channel,
            )
        )

        with self._torch.no_grad():
            next_frames = self.model.test(test_dat, real_input_flag)

        img_gen = self._preprocess.reshape_patch_back(next_frames, self.configs.patch_size)
        img_out = img_gen[:, -1:]

        img_pred = img_out[0, 0, :, :, 0]
        img_pred = np.maximum(img_pred, 0)
        img_pred = np.minimum(img_pred, 1)
        img_pred_orig = np.uint8(img_pred * 255)

        try:
            _, avg_deg = self._hough(img_pred_orig, threshold=50, minLineLength=50, maxLineGap=20)
            analog_voltage = float(self._calculate_voltage_from_slope(avg_deg))
        except Exception as exc:
            logger.warning("Slope calculation failed: %s", exc)
            analog_voltage = 0.0

        bbox = self._eval_bbox_logic(img_pred_orig, threshold=145)
        img_pred_bbox = cv2.rectangle(
            img_pred_orig.copy(),
            (bbox[0], bbox[1]),
            (bbox[2], bbox[3]),
            200,
            2,
        )
        img_gt = np.uint8(test_ims[0][0, -1, :, :, :] * 255).squeeze()
        bbox_gt = self._eval_bbox_logic(img_gt, threshold=140)

        iou = self._IoU(bbox_gt, bbox)
        if iou > 0.9:
            pred_level = "很可能预测正确"
        elif iou > 0.8:
            pred_level = "中等预测异常可能性"
        else:
            pred_level = "很可能预测异常"

        return img_pred_bbox, pred_level, analog_voltage

    # ---------- 对外入口 ----------

    def process_to_dir(self, dir_path: str, output_dir: str, process_id: str) -> Tuple[str, str]:
        """处理一个序列图片文件夹，输出 prediction.jpg + json，返回 (prediction, json) 路径"""
        import time

        folder = Path(dir_path)
        image_paths = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
        image_paths = self._sort_image_paths(image_paths)

        if len(image_paths) < self.configs.total_length:
            raise ValueError(
                f"Insufficient image count, need at least {self.configs.total_length}, got {len(image_paths)}"
            )

        selected_paths = list(image_paths)[-self.configs.total_length:]
        frames_np, filenames = self._load_images_from_paths(selected_paths)
        if len(frames_np) < self.configs.total_length:
            raise ValueError(
                f"Valid images are insufficient after loading, need {self.configs.total_length}, got {len(frames_np)}"
            )

        masks_np = self._create_dummy_masks(len(frames_np))
        input_handle = self._create_data_handle(frames_np, masks_np)

        start_time = time.perf_counter()
        prediction_result, pred_level, analog_voltage = self._run_prediction(input_handle, frames_np)
        logger.info("Trend prediction done in %.1f ms", (time.perf_counter() - start_time) * 1000)

        result_dir = Path(output_dir) / process_id
        result_dir.mkdir(parents=True, exist_ok=True)

        prediction_path = result_dir / "prediction.jpg"
        cv2.imwrite(str(prediction_path), prediction_result)

        result_data = {
            "process_id": process_id,
            "source_images": filenames,
            "pred_level": pred_level,
            "analog_voltage": analog_voltage,
            "timestamp": datetime.now().isoformat(),
        }
        json_path = result_dir / f"{process_id}.json"
        with open(json_path, "w", encoding="utf-8") as file:
            json.dump(result_data, file, indent=2, ensure_ascii=False)

        return str(prediction_path), str(json_path)


_ENGINE: Optional[TrendEngine] = None
_ENGINE_LOCK = threading.Lock()


def get_trend_engine() -> TrendEngine:
    """懒加载单例：首次调用 import torch 并加载模型权重，之后复用（加锁防并发双重加载）"""
    global _ENGINE
    if _ENGINE is None:
        with _ENGINE_LOCK:
            if _ENGINE is None:
                _ENGINE = TrendEngine()
    return _ENGINE
