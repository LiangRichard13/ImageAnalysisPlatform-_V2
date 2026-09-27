"""命名共享内存环形缓冲（协议 V1 生产者/消费者实现）。

时序遵循 docs/ipc/SHM_PROTOCOL.md §5（生产）/§6（消费 latest-wins）：
正确性靠 seq，事件只做唤醒；生产者永不阻塞（环满覆盖最旧）。
"""
import ctypes
import json
from typing import Dict, Optional, Tuple

import numpy as np

from utils.ipc import win32
from utils.ipc.layout import (
    FRAME_DATA_OFF, FRAME_HDR, HDR_SIZE, JSON_MAX, RESULT_HDR, RESULT_JSON_OFF,
    STATUS_READY, STATUS_WRITING, CtrlBlock, crc32, now_ns, obj_name,
)

_READ_RETRIES = 5


class NamedEvent:
    """manual-reset 命名事件（create-or-open）。"""

    def __init__(self, name: str):
        self.name = name
        self._handle = win32.create_event(name)

    def set(self) -> None:
        win32.set_event(self._handle)

    def reset(self) -> None:
        win32.reset_event(self._handle)

    def wait(self, timeout_ms: int) -> bool:
        return win32.wait_single(self._handle, timeout_ms) == win32.WAIT_OBJECT_0

    def close(self) -> None:
        h, self._handle = self._handle, 0
        if h:
            win32.close_handle(h)

    def __del__(self):  # 兜底
        try:
            self.close()
        except Exception:
            pass


class NamedMutex:
    """命名互斥体；acquire 的 WAIT_ABANDONED 视为获得（前任崩溃接管）。"""

    def __init__(self, name: str):
        self.name = name
        self._handle = win32.create_mutex(name)
        self._held = False

    def acquire(self, timeout_ms: int = 5000) -> bool:
        if self._held:
            return True
        result = win32.wait_single(self._handle, timeout_ms)
        if result in (win32.WAIT_OBJECT_0, win32.WAIT_ABANDONED):
            self._held = True
            return True
        return False

    def release(self) -> None:
        if self._held:
            win32.release_mutex(self._handle)
            self._held = False

    def close(self) -> None:
        self.release()
        h, self._handle = self._handle, 0
        if h:
            win32.close_handle(h)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class _RingMap:
    """命名映射 + 整视图（c_char 数组持有引用；close 先清引用再 Unmap）。"""

    def __init__(self, name: str, size: int):
        handle, _created = win32.create_file_mapping(name, size)
        base = win32.map_viewOfFile(handle, size)
        self._handle = handle
        self._base = base
        self.view = (ctypes.c_char * size).from_address(base)

    def close(self) -> None:
        view, self.view = self.view, None
        handle, self._handle = self._handle, 0
        base, self._base = self._base, 0
        del view
        if base:
            win32.unmap_view(base)
        if handle:
            win32.close_handle(handle)


class _RingBase:
    """槽位读写辅助：header pack/unpack + 数据区零拷贝读取。"""

    def __init__(self, ctrl: CtrlBlock, mapping: _RingMap, *, is_frame: bool):
        self._ctrl = ctrl
        self._map = mapping
        self._view = mapping.view
        self._hdr = FRAME_HDR if is_frame else RESULT_HDR
        self.slot_bytes = ctrl.frame_slot_bytes if is_frame else ctrl.result_slot_bytes
        self.slots = ctrl.frame_slots if is_frame else ctrl.result_slots

    def _slot_off(self, seq: int) -> int:
        return (seq % self.slots) * self.slot_bytes

    def _read_hdr(self, seq: int):
        off = self._slot_off(seq)
        return self._hdr.unpack_from(self._view[off:off + HDR_SIZE])

    def close(self) -> None:
        self._map.close()


def _frame_map(ctrl: CtrlBlock, namespace: str) -> _RingMap:
    return _RingMap(obj_name(namespace, "SHM_FRAME_V1"),
                    ctrl.frame_slots * ctrl.frame_slot_bytes)


def _result_map(ctrl: CtrlBlock, namespace: str) -> _RingMap:
    return _RingMap(obj_name(namespace, "SHM_RESULT_V1"),
                    ctrl.result_slots * ctrl.result_slot_bytes)


class FrameRingProducer(_RingBase):
    """FRAME ring 生产者（须持 IAP_PRODUCER_MUTEX 才可写；协议 §5 六步）。"""

    def __init__(self, ctrl: CtrlBlock, namespace: str, evt: NamedEvent,
                 mutex: NamedMutex):
        super().__init__(ctrl, _frame_map(ctrl, namespace), is_frame=True)
        self._evt = evt
        self._mutex = mutex

    @classmethod
    def create_or_attach(cls, ctrl: CtrlBlock, namespace: str,
                         frame_evt: NamedEvent, mutex: NamedMutex
                         ) -> "FrameRingProducer":
        return cls(ctrl, namespace, frame_evt, mutex)

    def acquire(self, timeout_ms: int = 5000) -> bool:
        return self._mutex.acquire(timeout_ms)

    def release(self) -> None:
        self._mutex.release()

    def write_frame(self, gray: np.ndarray, frame_id: int) -> int:
        """写入一帧 GRAY8(H×W)，返回完成 seq。环满覆盖最旧，永不阻塞。"""
        if gray.dtype != np.uint8 or gray.ndim != 2:
            raise ValueError(f"expect 2-D uint8, got {gray.dtype}/{gray.ndim}-D")
        h, w = gray.shape
        if (w, h) != (self._ctrl.frame_width, self._ctrl.frame_height):
            raise ValueError(
                f"frame shape ({w},{h}) != CTRL "
                f"({self._ctrl.frame_width},{self._ctrl.frame_height})")
        data = gray.tobytes()

        # 1. 心跳
        self._ctrl.producer_heartbeat_ms = win32.GetTickCount64()
        # 2. 定位槽（先写 header seq=0 使旧数据即刻失效）
        next_seq = self._ctrl.frame_write_seq + 1
        off = self._slot_off(next_seq)
        self._view[off:off + HDR_SIZE] = FRAME_HDR.pack(
            0, now_ns(), frame_id, crc32(data), len(data), STATUS_WRITING)
        if next_seq > self.slots:
            self._ctrl.dropped_by_overwrite = self._ctrl.dropped_by_overwrite + 1
        # 3. 数据
        self._view[off + FRAME_DATA_OFF:off + FRAME_DATA_OFF + len(data)] = data
        # 4. seq = 完成标记
        self._view[off:off + 8] = next_seq.to_bytes(8, "little")
        self._view[off + 32:off + 36] = STATUS_READY.to_bytes(4, "little")
        # 5. 全局序号
        self._ctrl.frame_write_seq = next_seq
        # 6. 唤醒
        self._evt.set()
        return next_seq


class FrameRingConsumer(_RingBase):
    """FRAME ring 消费者（latest-wins，协议 §6）。"""

    def __init__(self, ctrl: CtrlBlock, namespace: str, evt: NamedEvent):
        super().__init__(ctrl, _frame_map(ctrl, namespace), is_frame=True)
        self._evt = evt

    @classmethod
    def attach(cls, ctrl: CtrlBlock, namespace: str, frame_evt: NamedEvent
               ) -> "FrameRingConsumer":
        return cls(ctrl, namespace, frame_evt)

    def fetch_latest(self, last_seq: int, wait_ms: int = 50
                     ) -> Optional[Tuple[int, int, int, np.ndarray]]:
        """返回 (seq, frame_id, ts_ns, gray 副本)；无新帧或最新槽不可信 → None。"""
        self._evt.wait(wait_ms)  # 仅唤醒提示，正确性靠 seq
        for _ in range(_READ_RETRIES):
            s = self._ctrl.frame_write_seq
            if s == 0 or s == last_seq:
                return None
            seq, ts_ns, frame_id, crc, data_bytes, _status = self._read_hdr(s)
            if seq != s:  # 已被覆盖/不可信
                self._ctrl.dropped_by_lag = self._ctrl.dropped_by_lag + 1
                continue
            off = self._slot_off(s)
            gray = np.frombuffer(
                self._view, dtype=np.uint8, count=data_bytes,
                offset=off + FRAME_DATA_OFF).copy()  # 立即脱离共享内存
            seq2, _, _, crc2, _, _ = self._read_hdr(s)
            if seq2 != s or crc2 != crc:  # 读取期间被覆盖
                self._ctrl.dropped_by_lag = self._ctrl.dropped_by_lag + 1
                continue
            gray = gray.reshape(self._ctrl.frame_height, self._ctrl.frame_width)
            if s == self._ctrl.frame_write_seq:
                self._evt.reset()  # 追平者重置；竞态由兜底轮询兜底
            return s, frame_id, ts_ns, gray
        return None


class ResultRingProducer(_RingBase):
    """RESULT ring 生产者（分析端写，协议 §4 布局）。"""

    def __init__(self, ctrl: CtrlBlock, namespace: str, evt: NamedEvent):
        super().__init__(ctrl, _result_map(ctrl, namespace), is_frame=False)
        self._evt = evt

    @classmethod
    def create_or_attach(cls, ctrl: CtrlBlock, namespace: str, result_evt: NamedEvent
                         ) -> "ResultRingProducer":
        return cls(ctrl, namespace, result_evt)

    def write_result(self, frame_seq: int, json_dict: Dict,
                     pred_u8: np.ndarray, heat_u8: np.ndarray) -> int:
        json_bytes = json.dumps(json_dict, ensure_ascii=False).encode("utf-8")
        if len(json_bytes) > JSON_MAX:
            raise ValueError(f"json {len(json_bytes)}B > {JSON_MAX}B")
        pred = np.ascontiguousarray(pred_u8)
        heat = np.ascontiguousarray(heat_u8)
        w, h = self._ctrl.frame_width, self._ctrl.frame_height
        if pred.shape != (h, w) or heat.shape != (h, w):
            raise ValueError(f"pred/heat shape {pred.shape}/{heat.shape} != ({h},{w})")
        if pred.dtype != np.uint8 or heat.dtype != np.uint8:
            raise ValueError("pred/heat must be uint8")
        crc = crc32(json_bytes + pred.tobytes() + heat.tobytes())

        # 1. 心跳
        self._ctrl.analyzer_heartbeat_ms = win32.GetTickCount64()
        # 2. 定位槽（header seq=0 即刻失效旧数据；RESULT 头无 status 字段）
        next_seq = self._ctrl.result_write_seq + 1
        off = self._slot_off(next_seq)
        self._view[off:off + HDR_SIZE] = RESULT_HDR.pack(
            0, frame_seq, now_ns(), len(json_bytes), pred.size, heat.size, crc)
        # 3. 数据：json 固定区 + pred + heat
        self._view[off + RESULT_JSON_OFF:off + RESULT_JSON_OFF + len(json_bytes)] = json_bytes
        p_off = off + RESULT_JSON_OFF + JSON_MAX
        self._view[p_off:p_off + pred.size] = pred.tobytes()
        self._view[p_off + pred.size:p_off + pred.size + heat.size] = heat.tobytes()
        # 4. seq 最后写
        self._view[off:off + 8] = next_seq.to_bytes(8, "little")
        # 5. 全局序号；6. 唤醒
        self._ctrl.result_write_seq = next_seq
        self._evt.set()
        return next_seq


class ResultRingConsumer(_RingBase):
    """RESULT ring 消费者（孪生端/工具用，latest-wins）。"""

    def __init__(self, ctrl: CtrlBlock, namespace: str, evt: NamedEvent):
        super().__init__(ctrl, _result_map(ctrl, namespace), is_frame=False)
        self._evt = evt

    @classmethod
    def attach(cls, ctrl: CtrlBlock, namespace: str, result_evt: NamedEvent
               ) -> "ResultRingConsumer":
        return cls(ctrl, namespace, result_evt)

    def fetch_latest(self, last_seq: int, wait_ms: int = 50):
        """返回 (seq, frame_seq, json_dict, pred 副本, heat 副本) 或 None。"""
        self._evt.wait(wait_ms)
        for _ in range(_READ_RETRIES):
            s = self._ctrl.result_write_seq
            if s == 0 or s == last_seq:
                return None
            seq, frame_seq, _ts, json_len, pred_len, heat_len, crc = self._read_hdr(s)
            if seq != s:
                self._ctrl.dropped_by_lag = self._ctrl.dropped_by_lag + 1
                continue
            off = self._slot_off(s)
            raw = bytes(self._view[off + RESULT_JSON_OFF:
                                   off + RESULT_JSON_OFF + json_len])
            p_off = off + RESULT_JSON_OFF + JSON_MAX
            pred = np.frombuffer(self._view, dtype=np.uint8, count=pred_len,
                                 offset=p_off).copy()
            heat = np.frombuffer(self._view, dtype=np.uint8, count=heat_len,
                                 offset=p_off + pred_len).copy()
            seq2, _, _, jl2, pl2, hl2, crc2 = self._read_hdr(s)
            if (seq2, jl2, pl2, hl2, crc2) != (s, json_len, pred_len, heat_len, crc):
                self._ctrl.dropped_by_lag = self._ctrl.dropped_by_lag + 1
                continue
            h, w = self._ctrl.frame_height, self._ctrl.frame_width
            payload = json.loads(raw)
            if s == self._ctrl.result_write_seq:
                self._evt.reset()
            return s, frame_seq, payload, pred.reshape(h, w), heat.reshape(h, w)
        return None
