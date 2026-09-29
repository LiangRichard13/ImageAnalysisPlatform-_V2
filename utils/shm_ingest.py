"""共享内存实时链路接入线程（A2）。

事件驱动取代文件轮询：FRAME ring latest-wins 取帧 → 引擎推理（内存直通）
→ RESULT ring 写结果 → 归档队列异步落盘。原始帧/结果双链路解耦，
推理慢只影响结果链路，孪生端看原始帧不受影响。

线程模型：QThread（UI 亲和）；事件等待经 ctypes 释放 GIL；
stop() 置标志后 ≤ 一次 fetch/推理周期内退出（禁用 terminate()——
推理中途中断会留下半写状态，属协议语义禁止）。
"""
import logging
import threading
import time
from typing import Optional

import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage

from utils.anomaly_detection_client import ENGINE_RULE_BASED, AnomalyDetectionClient
from utils.archive_worker import ArchiveItem, ArchiveWorker
from utils.ipc import shm_ring
from utils.ipc.config import load_ipc_config
from utils.ipc.layout import CtrlBlock, ProtocolMismatchError, obj_name
from utils.ipc.win32 import GetTickCount64

logger = logging.getLogger(__name__)


def _to_qimage_gray(arr: np.ndarray) -> QImage:
    """GRAY8 ndarray → 独立 QImage（copy 脱离 numpy 生命周期，跨线程传值安全）。"""
    arr = np.ascontiguousarray(arr, dtype=np.uint8)
    h, w = arr.shape[:2]
    return QImage(arr.tobytes(), w, h, w, QImage.Format_Grayscale8).copy()


def _to_qimage_jet(arr: np.ndarray) -> QImage:
    """异常强度图 → JET 伪彩 RGB888 QImage（与归档 _heatmap.png 颜色一致）。"""
    colored = cv2.applyColorMap(np.ascontiguousarray(arr, dtype=np.uint8),
                                cv2.COLORMAP_JET)  # BGR
    rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return QImage(rgb.tobytes(), w, h, w * 3, QImage.Format_RGB888).copy()


class SharedMemoryIngestThread(QThread):
    """FRAME 消费者 + RESULT 生产者 + 归档投递。"""

    # 帧预览/预测图/热力图以 QImage 随结果一并上抛（重活在线程内做完，GUI 只做 scaled）
    result_ready = pyqtSignal(int, dict, QImage, QImage, QImage)
    source_online = pyqtSignal(bool)            # 仅状态翻转时发
    stats = pyqtSignal(int, int, int, int)       # frame_seq, result_seq, dropped_lag, dropped_overwrite（1Hz）
    archive_dropped = pyqtSignal(int)            # 归档累计丢弃数（变化时发）
    error = pyqtSignal(str)

    def __init__(self, engine_name: str = ENGINE_RULE_BASED,
                 namespace: Optional[str] = None,
                 api_root=None, parent=None):
        super().__init__(parent)
        self._engine_name = engine_name
        cfg = load_ipc_config()
        self._namespace = namespace or cfg.namespace
        self._api_root = api_root
        self._stop_flag = threading.Event()
        self._last_seq = 0
        self._client: Optional[AnomalyDetectionClient] = None
        self._archiver: Optional[ArchiveWorker] = None

    # ---- 控制 ----
    def set_engine(self, name: str) -> None:
        """引擎切换（combo 经信号调用），下一帧生效。"""
        self._engine_name = name
        logger.info("实时链路引擎切换为 %s（下一帧生效）", name)

    def stop(self) -> None:
        self._stop_flag.set()

    # ---- 主循环 ----
    def run(self) -> None:  # noqa: C901 - 状态机主循环
        try:
            self._client = AnomalyDetectionClient()
            if self._api_root is not None:
                self._client.api_root = self._api_root
            self._archiver = ArchiveWorker(self._client.api_root)
            self._archiver.start()
            self._loop()
        except ProtocolMismatchError as exc:
            logger.error("实时链路协议不匹配: %s", exc)
            self.error.emit(f"实时链路协议不匹配: {exc}")
        except Exception:
            logger.exception("实时链路线程异常退出")
            self.error.emit("实时链路线程异常，详见日志")
        finally:
            if self._archiver is not None:
                self._archiver.stop()

    def _loop(self) -> None:
        online = False
        last_stats_ts = 0.0
        last_archive_dropped = -1
        cfg = load_ipc_config()

        # RESULT ring 单写者防御：协议 V1 假设单分析端，双开第二个实例立即拒绝
        analyzer_mutex = shm_ring.NamedMutex(obj_name(self._namespace, "ANALYZER_MUTEX"))
        try:
            if not analyzer_mutex.acquire(timeout_ms=3000):
                msg = "检测到另一个分析端实例已占用实时链路（IAP_ANALYZER_MUTEX），本实例不接入"
                logger.error(msg)
                self.error.emit(msg)
                return

            ctrl = None
            consumer = result_producer = None
            evt_frame = evt_result = None
            try:
                # attach 重试：生产者可能晚于本进程启动（源离线语义，非致命）。
                # 以 consumer 为完成判据：半途失败须先清理本轮句柄再整体重试
                while consumer is None and not self._stop_flag.is_set():
                    try:
                        ctrl = CtrlBlock.attach(self._namespace)
                        evt_frame = shm_ring.NamedEvent(obj_name(self._namespace, "EVT_FRAME"))
                        evt_result = shm_ring.NamedEvent(obj_name(self._namespace, "EVT_RESULT"))
                        consumer = shm_ring.FrameRingConsumer.attach(ctrl, self._namespace, evt_frame)
                        result_producer = shm_ring.ResultRingProducer.create_or_attach(
                            ctrl, self._namespace, evt_result)
                        self._archiver.attach_ctrl(ctrl)
                    except ProtocolMismatchError:
                        raise  # 版本不匹配是致命的
                    except Exception as exc:
                        logger.debug("等待生产者/共享内存就绪: %s", exc)
                        for obj in (consumer, result_producer, evt_frame, evt_result, ctrl):
                            if obj is not None:
                                try:
                                    obj.close()
                                except Exception:
                                    pass
                        consumer = result_producer = None
                        evt_frame = evt_result = None
                        ctrl = None
                        self._stop_flag.wait(1.0)

                while not self._stop_flag.is_set():
                    ctrl.analyzer_heartbeat_ms = GetTickCount64()

                    now_online = ctrl.producer_alive(cfg.heartbeat_timeout_ms)
                    if now_online != online:
                        online = now_online
                        self.source_online.emit(online)
                        logger.info("实时链路源状态: %s", "在线" if online else "离线")

                    got = consumer.fetch_latest(last_seq=self._last_seq, wait_ms=50)
                    if got is not None:
                        self._process_frame(ctrl, result_producer, *got)

                    now = time.monotonic()
                    if now - last_stats_ts >= 1.0:
                        last_stats_ts = now
                        self.stats.emit(ctrl.frame_write_seq, ctrl.result_write_seq,
                                        ctrl.dropped_by_lag, ctrl.dropped_by_overwrite)
                        if self._archiver.dropped_count != last_archive_dropped:
                            last_archive_dropped = self._archiver.dropped_count
                            self.archive_dropped.emit(last_archive_dropped)
            finally:
                # 先停归档（它还在更新 ctrl 心跳），再关共享对象——反序会狂刷心跳异常日志
                self._archiver.stop()
                for obj in (consumer, result_producer, evt_frame, evt_result, ctrl):
                    if obj is not None:
                        try:
                            obj.close()
                        except Exception:
                            pass
        finally:
            analyzer_mutex.release()
            analyzer_mutex.close()

    def _process_frame(self, ctrl, result_producer, seq, frame_id, ts_ns, gray) -> None:
        t0 = time.perf_counter()
        try:
            pred, heat, metrics = self._client.process_from_array(
                gray, self._engine_name, frame_id=frame_id)
        except Exception:
            logger.exception("推理失败 frame_seq=%s engine=%s", seq, self._engine_name)
            self._last_seq = seq
            return

        image_type = self._client._judge_shape_type(gray.shape[1], gray.shape[0])
        try:
            result_seq = result_producer.write_result(seq, metrics, pred, heat)
        except Exception:
            logger.exception("写 RESULT ring 失败 frame_seq=%s", seq)
            result_seq = ctrl.result_write_seq

        self._archiver.submit(ArchiveItem(
            frame_seq=seq, frame_id=frame_id, process_id=metrics["process_id"],
            image_type=image_type, gray=gray, pred_u8=pred, heat_u8=heat,
            json_dict=metrics))

        self._last_seq = seq
        self.result_ready.emit(seq, metrics,
                               _to_qimage_gray(gray), _to_qimage_gray(pred),
                               _to_qimage_jet(heat))
        logger.info("实时处理完成 frame_seq=%s → result_seq=%s，耗时 %.0f ms",
                    seq, result_seq, (time.perf_counter() - t0) * 1000)
