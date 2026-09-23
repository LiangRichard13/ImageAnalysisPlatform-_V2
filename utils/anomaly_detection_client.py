"""异常检测客户端（本地进程内双引擎版）

原 SSH/共享目录远程方案已废弃，目录结构沿用原有形态以便溯源：
  <根>/anomaly_api/
  ├── input/{process_id}/    ← 原图（原名）+ request.json
  └── output/{process_id}/   ← {process_id}.png / _heatmap.png / .json

根目录默认 download/，env DOWNLOAD_ROOT_DIR 可覆盖。

双引擎：
  - rule_based：规则化褶皱检测（utils/rule_based_wrinkle.py，仅依赖 cv2+numpy，即时可用）
  - dinomaly ：Dinomaly 深度学习检测（utils/dinomaly_engine.py，懒加载模型权重）
"""

import json
import logging
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Tuple

from dotenv import load_dotenv

from utils.file_namer import FileNamer

logger = logging.getLogger(__name__)

ENGINE_RULE_BASED = "rule_based"
ENGINE_DINOMALY = "dinomaly"

_pipeline = None  # 规则引擎单例（无模型权重，仅复用实例）


def _get_rule_pipeline():
    global _pipeline
    if _pipeline is None:
        from utils.rule_based_wrinkle import RuleBasedWrinklePipeline

        _pipeline = RuleBasedWrinklePipeline()
    return _pipeline


class AnomalyDetectionClient:
    """本地进程内异常检测客户端（双引擎）"""

    def __init__(self):
        project_root = Path(__file__).resolve().parent.parent
        load_dotenv(project_root / ".env")

        download_root = Path(os.getenv("DOWNLOAD_ROOT_DIR", str(project_root / "download")))
        self.api_root = download_root / "anomaly_api"
        self.process_id = self.get_process_id()

    def get_process_id(self) -> str:
        return FileNamer.generate_time_based_name()

    @staticmethod
    def _judge_image_type(image_path: str) -> str:
        try:
            from PIL import Image

            with Image.open(image_path) as img:
                width, height = img.size
                ratio = width / height
                if 0.9 <= ratio <= 1.1:
                    return "square"
                if width == 31901 and height == 1000:
                    return "very long"
                return "other"
        except Exception as exc:
            logger.warning("Failed to judge image type: %s", exc)
            return "other"

    def _archive_input(self, image_path: str, image_type: str) -> Path:
        """原图（原名）+ request.json 存档到 input/{process_id}/，返回该目录"""
        source = Path(image_path)
        input_dir = self.api_root / "input" / self.process_id
        input_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy2(source, input_dir / source.name)
        metadata = {
            "process_id": self.process_id,
            "image_type": image_type,
            "source_image": source.name,
            "timestamp": time.time(),
        }
        with open(input_dir / "request.json", "w", encoding="utf-8") as file:
            json.dump(metadata, file, ensure_ascii=False, indent=2)
        return input_dir

    def process_images(self, image_path: str, engine_name: str = ENGINE_RULE_BASED) -> Tuple[str, str, str]:
        """处理单张图片，返回 (预测图, 热力图, JSON) 本地路径"""
        self.process_id = self.get_process_id()

        try:
            image_type = self._judge_image_type(image_path)
            input_dir = self._archive_input(image_path, image_type)
            output_dir = self.api_root / "output"
            logger.info("Start local anomaly processing, engine=%s input=%s", engine_name, input_dir)
            start_time = time.perf_counter()

            if engine_name == ENGINE_DINOMALY:
                from utils.dinomaly_engine import get_dinomaly_engine

                engine = get_dinomaly_engine()
                result = engine.process_to_dir(image_path, str(output_dir), self.process_id, image_type)
            elif engine_name == ENGINE_RULE_BASED:
                result = _get_rule_pipeline().process_to_dir(image_path, str(output_dir), self.process_id)
            else:
                raise ValueError(f"Unknown anomaly engine: {engine_name}")

            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.info("Anomaly processing done in %.1f ms", duration_ms)
            return result
        except Exception as exc:
            logger.error("Anomaly processing failed: %s", exc)
            raise


if __name__ == "__main__":
    client = AnomalyDetectionClient()
    local_result_pre_image, local_result_heat_map, local_result_json = client.process_images(
        image_path="test/gsy_8.jpg"
    )
    print(local_result_pre_image, local_result_heat_map, local_result_json)
