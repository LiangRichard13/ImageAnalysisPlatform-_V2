"""趋势预测客户端（本地进程内版）

原 SSH/共享目录远程方案已废弃，目录结构沿用原有形态以便溯源：
  <根>/trend_api/
  ├── input/{process_id}/    ← 本次预测使用的序列原图（原名保存）
  └── output/{process_id}/   ← prediction.jpg + {process_id}.json

根目录默认 download/，env DOWNLOAD_ROOT_DIR 可覆盖。
"""

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

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class TrendAnalysisClient:
    """本地进程内趋势预测客户端"""

    def __init__(self):
        project_root = Path(__file__).resolve().parent.parent
        load_dotenv(project_root / ".env")

        download_root = Path(os.getenv("DOWNLOAD_ROOT_DIR", str(project_root / "download")))
        self.api_root = download_root / "trend_api"
        self.process_id = self.get_process_id()

    def get_process_id(self) -> str:
        return FileNamer.generate_time_based_name()

    def _archive_input(self, dir_path: str) -> Path:
        """本次预测使用的序列原图存档到 input/{process_id}/，返回该目录"""
        input_dir = self.api_root / "input" / self.process_id
        input_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        for source in Path(dir_path).iterdir():
            if source.is_file() and source.suffix.lower() in IMAGE_EXTENSIONS:
                shutil.copy2(source, input_dir / source.name)
                count += 1
        logger.info("Archived %d source images to %s", count, input_dir)
        return input_dir

    def process_images(self, dir_path: str) -> Tuple[str, str]:
        """处理一个序列图片文件夹，返回 (prediction.jpg, json) 本地路径"""
        self.process_id = self.get_process_id()

        try:
            input_dir = self._archive_input(dir_path)
            output_dir = self.api_root / "output"
            logger.info("Start local trend processing, input=%s", input_dir)
            from utils.trend_engine import get_trend_engine

            start_time = time.perf_counter()
            result = get_trend_engine().process_to_dir(dir_path, str(output_dir), self.process_id)
            logger.info("Trend processing done in %.1f ms", (time.perf_counter() - start_time) * 1000)
            return result
        except Exception as exc:
            logger.error("Trend processing failed: %s", exc)
            raise


if __name__ == "__main__":
    client = TrendAnalysisClient()
    pred_file_path, local_result_json = client.process_images(dir_path="test")
    print(pred_file_path, local_result_json)
