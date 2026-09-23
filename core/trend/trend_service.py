# -*- coding: utf-8 -*-
import argparse
import datetime
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch


project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
sys.path.append(os.path.join(os.path.dirname(project_root), "algo_packages"))

from models.model_factory import Model
from dataloader.wrinkle import InputHandle
from a100_utils import preprocess
from a100_utils.eval_bbox import IoU
from preprocess.tojson import hough, calculate_voltage_from_slope

# 默认路径配置（本地运行：脚本目录下的 input/output/weights）
_DEFAULT_DIR = Path(__file__).resolve().parent
DEFAULT_TREND_INPUT_DIR = str(_DEFAULT_DIR / "input")
DEFAULT_TREND_OUTPUT_DIR = str(_DEFAULT_DIR / "output")
DEFAULT_TREND_MODEL_PATH = str(_DEFAULT_DIR / "weights" / "checkpoints-ontonet-3-weighted+sobel-model-best-2894.ckpt")

LOGGER = logging.getLogger("trend_api")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


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


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


class PredictionAPI:
    """Single-instance predictor with model preloaded on service startup."""

    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        self.configs = self._create_configs(model_path=model_path, device=device)
        self.model = Model(self.configs)
        self.model_loaded = False
        self.load_model()

    def _create_configs(self, model_path: Optional[str], device: Optional[str]):
        class Config:
            def __init__(self):
                self.model_name = "predgru_v2"
                self.pretrained_model = model_path or os.getenv("TREND_MODEL_PATH") or DEFAULT_TREND_MODEL_PATH

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

    def load_model(self) -> None:
        if self.model_loaded:
            return

        LOGGER.info("Loading model to device: %s", self.configs.device)
        LOGGER.info("Model checkpoint: %s", self.configs.pretrained_model)
        self.model.load(self.configs.pretrained_model)
        self.model_loaded = True
        LOGGER.info("Model loaded successfully")

    def _sort_image_paths(self, image_paths: Sequence[Path]) -> List[Path]:
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
                LOGGER.warning("Failed to read image: %s", image_path)
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
        masks = []
        for _ in range(frames_count):
            masks.append(
                np.zeros((self.configs.img_height, self.configs.img_width), dtype=np.float32)
            )
        return masks

    def _create_data_handle(self, frames_np: Sequence[np.ndarray], masks_np: Sequence[np.ndarray]) -> InputHandle:
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

        input_handle = InputHandle(data, indices, input_param)
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

    def predict_from_paths(
        self,
        process_id: str,
        image_paths: Sequence[Path],
        output_root: Path,
    ) -> dict:
        try:
            image_paths = self._sort_image_paths(image_paths)
            LOGGER.info("Start processing process_id=%s with %d images", process_id, len(image_paths))

            if len(image_paths) < self.configs.total_length:
                raise ValueError(
                    f"Insufficient image count, need at least {self.configs.total_length}, got {len(image_paths)}"
                )

            selected_paths = list(image_paths)[-self.configs.total_length :]
            frames_np, filenames = self._load_images_from_paths(selected_paths)
            if len(frames_np) < self.configs.total_length:
                raise ValueError(
                    f"Valid images are insufficient after loading, need {self.configs.total_length}, got {len(frames_np)}"
                )

            masks_np = self._create_dummy_masks(len(frames_np))
            input_handle = self._create_data_handle(frames_np, masks_np)

            output_dir = output_root / process_id
            output_dir.mkdir(parents=True, exist_ok=True)

            prediction_result, pred_level, analog_voltage = self._run_prediction(
                input_handle=input_handle,
                frames_np=frames_np,
            )

            prediction_path = output_dir / "prediction.jpg"
            cv2.imwrite(str(prediction_path), prediction_result)

            result_data = {
                "process_id": process_id,
                "source_images": filenames,
                "pred_level": pred_level,
                "analog_voltage": analog_voltage,
                "timestamp": datetime.datetime.now().isoformat(),
            }

            json_path = output_dir / f"{process_id}.json"
            with open(json_path, "w", encoding="utf-8") as file:
                json.dump(result_data, file, indent=2, ensure_ascii=False)

            LOGGER.info("Processing completed for %s, output=%s", process_id, prediction_path)
            return {
                "status": "success",
                "process_id": process_id,
                "prediction_path": str(prediction_path),
                "result_json_path": str(json_path),
                "pred_level": pred_level,
                "analog_voltage": analog_voltage,
                "message": "Prediction completed",
            }
        except Exception as exc:
            error_msg = f"Prediction failed: {exc}"
            LOGGER.exception(error_msg)
            return {
                "status": "error",
                "process_id": process_id,
                "error": error_msg,
            }

    def predict(self, process_id: str, folder_path: str, output_root: Optional[str] = None) -> dict:
        folder = Path(folder_path)
        if not folder.exists():
            return {
                "status": "error",
                "process_id": process_id,
                "error": f"Folder does not exist: {folder_path}",
            }

        image_paths = [path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
        resolved_output_root = Path(output_root) if output_root else Path(__file__).resolve().parent / "output"
        return self.predict_from_paths(process_id, image_paths, resolved_output_root)

    def _run_prediction(self, input_handle: InputHandle, frames_np: Sequence[np.ndarray]) -> Tuple[np.ndarray, str, float]:
        test_ims = input_handle.get_batch()
        if test_ims is None:
            raise ValueError("Failed to get data batch")

        if len(test_ims) == 1:
            test_dat = preprocess.reshape_patch(test_ims, self.configs.patch_size)
        else:
            test_dat = [
                preprocess.reshape_patch(test_ims[0], self.configs.patch_size),
                preprocess.reshape_patch(test_ims[1], self.configs.patch_size),
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

        with torch.no_grad():
            next_frames = self.model.test(test_dat, real_input_flag)

        img_gen = preprocess.reshape_patch_back(next_frames, self.configs.patch_size)
        img_out = img_gen[:, -1:]

        img_pred = img_out[0, 0, :, :, 0]
        img_pred = np.maximum(img_pred, 0)
        img_pred = np.minimum(img_pred, 1)
        img_pred_orig = np.uint8(img_pred * 255)

        try:
            _, avg_deg = hough(img_pred_orig, threshold=50, minLineLength=50, maxLineGap=20)
            analog_voltage = float(calculate_voltage_from_slope(avg_deg))
        except Exception as exc:
            LOGGER.warning("Slope calculation failed: %s", exc)
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

        iou = IoU(bbox_gt, bbox)
        if iou > 0.9:
            pred_level = "很可能预测正确"
        elif iou > 0.8:
            pred_level = "中等预测异常可能性"
        else:
            pred_level = "很可能预测异常"

        return img_pred_bbox, pred_level, analog_voltage


class TrendFolderWatcherService:
    """Poll a root folder and process the newest completed image batch folder."""

    def __init__(
        self,
        predictor: PredictionAPI,
        input_dir: str,
        output_dir: str,
        poll_interval: float = 2.0,
        min_images: Optional[int] = None,
    ):
        self.predictor = predictor
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.poll_interval = poll_interval
        self.min_images = min_images or predictor.configs.total_length
        self.last_processed_batch: Optional[str] = None

    def _list_batch_dirs(self) -> List[Path]:
        if not self.input_dir.exists():
            self.input_dir.mkdir(parents=True, exist_ok=True)
        return [
            path for path in self.input_dir.iterdir()
            if path.is_dir() and not path.name.startswith(".") and not path.name.endswith(".tmp")
        ]

    def _list_images_in_batch(self, batch_dir: Path) -> List[Path]:
        return [
            path for path in batch_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]

    def _get_latest_batch_dir(self, batch_dirs: Sequence[Path]) -> Optional[Path]:
        if not batch_dirs:
            return None
        return max(batch_dirs, key=lambda path: (path.stat().st_mtime_ns, path.name.lower()))

    def run_forever(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Trend watcher started")
        LOGGER.info("Watching input root directory: %s", self.input_dir)
        LOGGER.info("Result output directory: %s", self.output_dir)
        LOGGER.info("Minimum images required: %d", self.min_images)

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

                image_paths = self._list_images_in_batch(latest_batch_dir)
                if len(image_paths) < self.min_images:
                    LOGGER.info(
                        "Latest batch folder %s has only %d images, waiting for at least %d",
                        latest_batch_dir,
                        len(image_paths),
                        self.min_images,
                    )
                    time.sleep(self.poll_interval)
                    continue

                process_id = latest_batch_dir.name
                LOGGER.info("Detected new batch folder: %s", latest_batch_dir)
                result = self.predictor.predict_from_paths(
                    process_id=process_id,
                    image_paths=image_paths,
                    output_root=self.output_dir,
                )

                if result.get("status") == "success":
                    self.last_processed_batch = batch_key
                else:
                    LOGGER.error("Processing failed for %s: %s", process_id, result.get("error"))

            except KeyboardInterrupt:
                LOGGER.info("Trend watcher stopped by user")
                raise
            except Exception as exc:
                LOGGER.exception("Watcher loop failed: %s", exc)

            time.sleep(self.poll_interval)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PredGRU trend analysis service")
    parser.add_argument("--mode", choices=["service", "single"], default="service")
    parser.add_argument("--process_id", type=str, help="Process ID for single mode")
    parser.add_argument("--folder_path", type=str, help="Input folder for single mode")
    parser.add_argument("--input_dir", type=str, default=str(Path(__file__).resolve().parent / "input"))
    parser.add_argument("--output_dir", type=str, default=str(Path(__file__).resolve().parent / "output"))
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--poll_interval", type=float, default=2.0)
    return parser


def main() -> None:
    setup_logging()
    parser = build_argument_parser()
    args = parser.parse_args()

    # input/output 默认在脚本目录下；权重解析：参数 > env > 本地默认
    if not args.model_path and not os.getenv("TREND_MODEL_PATH"):
        args.model_path = DEFAULT_TREND_MODEL_PATH

    # Preserve legacy behavior: if old callers pass --process_id and --folder_path
    # without --mode, treat the invocation as one-shot prediction.
    if args.mode == "service" and args.process_id and args.folder_path:
        args.mode = "single"

    predictor = PredictionAPI(model_path=args.model_path, device=args.device)

    if args.mode == "single":
        if not args.process_id or not args.folder_path:
            parser.error("--process_id and --folder_path are required in single mode")
        result = predictor.predict(
            process_id=args.process_id,
            folder_path=args.folder_path,
            output_root=args.output_dir,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if result["status"] != "success":
            sys.exit(1)
        return

    service = TrendFolderWatcherService(
        predictor=predictor,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        poll_interval=args.poll_interval,
    )
    service.run_forever()


if __name__ == "__main__":
    main()
