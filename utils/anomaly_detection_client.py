"""异常检测客户端（本地进程内双引擎版）

原共享目录/SSH 远程方案已废弃，类名与 process_images 接口保持不变，
内部直接调用本地引擎：
  - rule_based：规则化褶皱检测（utils/rule_based_wrinkle.py，仅依赖 cv2+numpy，即时可用）
  - dinomaly ：Dinomaly 深度学习检测（utils/dinomaly_engine.py，懒加载模型权重）

结果输出目录结构：<输出根>/anomaly_detection/<YYYY-MM-DD>/<process_id>/三件套
输出根默认 download/，env DOWNLOAD_ROOT_DIR 可覆盖。
"""

import logging
import os
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
    """本地进程内异常检测客户端（保留历史类名，接口与共享目录版兼容）"""

    def __init__(self):
        project_root = Path(__file__).resolve().parent.parent
        load_dotenv(project_root / ".env")

        download_root = Path(os.getenv("DOWNLOAD_ROOT_DIR", str(project_root / "download")))
        self.anomaly_output_root = download_root / "anomaly_detection"
        self.process_id = self.get_process_id()

    def get_process_id(self) -> str:
        return FileNamer.generate_time_based_name()

    def _today_output_dir(self) -> Path:
        # 结果按「检测类型 / 日期 / process_id」三级分组，日期目录幂等创建
        return self.anomaly_output_root / datetime.now().strftime("%Y-%m-%d")

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

    def process_images(self, image_path: str, engine_name: str = ENGINE_RULE_BASED) -> Tuple[str, str, str]:
        """处理单张图片，返回 (预测图, 热力图, JSON) 本地路径"""
        import time

        self.process_id = self.get_process_id()
        output_dir = self._today_output_dir()

        try:
            logger.info("Start local anomaly processing, engine=%s source=%s", engine_name, image_path)
            start_time = time.perf_counter()

            if engine_name == ENGINE_DINOMALY:
                from utils.dinomaly_engine import get_dinomaly_engine

                engine = get_dinomaly_engine()
                image_type = self._judge_image_type(image_path)
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
