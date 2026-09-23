"""趋势预测客户端（本地进程内版）

原共享目录/SSH 远程方案已废弃，类名与 process_images 接口保持不变，
内部直接调用本地 PredGRU 引擎（utils/trend_engine.py，懒加载模型权重）。

结果输出目录结构：<输出根>/trend_prediction/<YYYY-MM-DD>/<process_id>/{prediction.jpg, json}
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


class TrendAnalysisClient:
    """本地进程内趋势预测客户端（保留历史类名，接口与共享目录版兼容）"""

    def __init__(self):
        project_root = Path(__file__).resolve().parent.parent
        load_dotenv(project_root / ".env")

        download_root = Path(os.getenv("DOWNLOAD_ROOT_DIR", str(project_root / "download")))
        self.trend_output_root = download_root / "trend_prediction"
        self.process_id = self.get_process_id()

    def get_process_id(self) -> str:
        return FileNamer.generate_time_based_name()

    def process_images(self, dir_path: str) -> Tuple[str, str]:
        """处理一个序列图片文件夹，返回 (prediction.jpg, json) 本地路径"""
        import time

        self.process_id = self.get_process_id()
        # 结果按「检测类型 / 日期 / process_id」三级分组
        output_dir = self.trend_output_root / datetime.now().strftime("%Y-%m-%d")

        try:
            logger.info("Start local trend processing, source=%s", dir_path)
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
    pred_file_path, local_result_json = client.process_images(
        dir_path="test"
    )
    print(pred_file_path, local_result_json)
