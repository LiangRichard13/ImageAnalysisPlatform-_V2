"""异步归档 worker：推理结果后台落盘，目录结构与现行文件链路完全一致。

线程模型：threading.Thread(daemon=True)——无 GUI 职责，不随 widget 销毁；
cv2.imwrite 在 C 层释放 GIL，1fps 单消费者足够。实时链路只 submit，
满队列丢最旧 + 告警（归档尽力而为，绝不反压推理）。
"""
import json
import logging
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

from utils.ipc import win32
from utils.rule_based_wrinkle import cvt2heatmap

logger = logging.getLogger(__name__)


@dataclass
class ArchiveItem:
    frame_seq: int
    frame_id: int
    process_id: str
    image_type: str
    gray: np.ndarray        # 原始帧 GRAY8（input 存档）
    pred_u8: np.ndarray     # 预测标注图（output {id}.png）
    heat_u8: np.ndarray     # 强度图（output {id}_heatmap.png，落盘时才上色）
    json_dict: Dict         # 含 process_id/files 的完整结果 JSON


class ArchiveWorker(threading.Thread):
    """单消费者归档线程：input/{id}/{frame_id}.png + request.json；
    output/{id}/{id}.png / _heatmap.png / .json（与现行文件链路逐字段一致）。"""

    def __init__(self, api_root: Path, capacity: int = 16):
        super().__init__(daemon=True, name="ArchiveWorker")
        self._api_root = Path(api_root)
        self._queue: "queue.Queue[ArchiveItem]" = queue.Queue(maxsize=capacity)
        self._stop_event = threading.Event()
        self._ctrl = None  # 可选注入 CtrlBlock，更新 archiver_heartbeat
        self.dropped_count = 0
        self.archived_count = 0

    def attach_ctrl(self, ctrl) -> None:
        """注入 CTRL 块后，每归档一件更新 archiver 心跳。"""
        self._ctrl = ctrl

    def submit(self, item: ArchiveItem) -> None:
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            try:
                self._queue.get_nowait()  # 丢最旧
                self.dropped_count += 1
                logger.warning("归档队列已满，丢弃最旧一件（累计 %d）", self.dropped_count)
                self._queue.put_nowait(item)
            except (queue.Empty, queue.Full):
                pass

    def run(self) -> None:
        logger.info("ArchiveWorker 启动，输出根目录 %s", self._api_root)
        while not self._stop_event.is_set() or not self._queue.empty():
            try:
                item = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self.archive(item)
                self.archived_count += 1
            except Exception:
                logger.exception("归档失败 process_id=%s", item.process_id)
            if self._ctrl is not None:
                try:
                    self._ctrl.archiver_heartbeat_ms = win32.GetTickCount64()
                except Exception:
                    logger.exception("更新归档心跳失败")

    def archive(self, item: ArchiveItem) -> None:
        # input/{process_id}/{frame_id}.png + request.json（字段同 _archive_input）
        input_dir = self._api_root / "input" / item.process_id
        input_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(input_dir / f"{item.frame_id}.png"), item.gray)
        request = {
            "process_id": item.process_id,
            "image_type": item.image_type,
            "source_image": f"{item.frame_id}.png",
            "timestamp": time.time(),
        }
        (input_dir / "request.json").write_text(
            json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")

        # output/{process_id}/ 三件套（同现行结构）
        output_dir = self._api_root / "output" / item.process_id
        output_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_dir / f"{item.process_id}.png"), item.pred_u8)
        cv2.imwrite(str(output_dir / f"{item.process_id}_heatmap.png"),
                    cvt2heatmap(item.heat_u8))
        (output_dir / f"{item.process_id}.json").write_text(
            json.dumps(item.json_dict, ensure_ascii=False, indent=2), encoding="utf-8")

    def stop(self) -> None:
        self._stop_event.set()
        self.join(timeout=10)
