"""协议 V1 布局：常量、偏移、struct 读写、CTRL 块。

对应 docs/ipc/SHM_PROTOCOL.md §2（CTRL）、§3（FRAME slot）、§4（RESULT slot）。
struct 格式串全 `<` 前缀（小端、无对齐填充），头部 calcsize()==64 由单测锁定。
"""
import ctypes
import struct
import time
import zlib
from typing import Tuple

from utils.ipc import win32

# ---- 协议常量 ----
CTRL_SIZE = 4096
MAGIC = b"IAPS"
VERSION = 1
PIXEL_FORMAT_GRAY8 = 0

FRAME_HDR = struct.Struct("<QQQIII28x")    # seq, ts_ns, frame_id, crc32, data_bytes, status
RESULT_HDR = struct.Struct("<QQQIIII24x")  # seq, frame_seq, ts_ns, json_bytes, pred_bytes, heat_bytes, crc32
HDR_SIZE = 64
FRAME_DATA_OFF = 64
RESULT_JSON_OFF = 64
JSON_MAX = 8192
STATUS_EMPTY, STATUS_WRITING, STATUS_READY = 0, 1, 2

# CTRL 偏移
_OFF_MAGIC = 0
_OFF_VERSION = 4
_OFF_WIDTH = 8
_OFF_HEIGHT = 12
_OFF_PIXEL_FORMAT = 16
_OFF_FRAME_SLOTS = 20
_OFF_FRAME_SLOT_BYTES = 24
_OFF_RESULT_SLOTS = 28
_OFF_RESULT_SLOT_BYTES = 32
_OFF_FRAME_WRITE_SEQ = 40
_OFF_RESULT_WRITE_SEQ = 48
_OFF_PRODUCER_HB = 56
_OFF_ANALYZER_HB = 64
_OFF_ARCHIVER_HB = 72
_OFF_TWIN_HB = 80
_OFF_DROPPED_OVERWRITE = 88
_OFF_DROPPED_LAG = 96


def crc32(data) -> int:
    return zlib.crc32(data) & 0xFFFFFFFF


def obj_name(namespace: str, suffix: str) -> str:
    """namespace + 协议对象名，如 ('Local\\\\IAP', 'SHM_CTRL_V1') -> Local\\IAP_SHM_CTRL_V1。"""
    return f"{namespace}_{suffix}"


class ProtocolMismatchError(RuntimeError):
    """CTRL 校验失败（magic/version/几何不一致），禁止静默错读。"""


def _u32(view, off) -> int:
    return struct.unpack_from("<I", bytes(view[off:off + 4]))[0]


def _u64(view, off) -> int:
    return struct.unpack_from("<Q", bytes(view[off:off + 8]))[0]


def _put_u64(view, off, value: int) -> None:
    view[off:off + 8] = struct.pack("<Q", value)


class CtrlBlock:
    """CTRL 共享块的读写封装（4KB，create-or-open 语义）。"""

    def __init__(self, view, created: bool, handle: int):
        self._view = view      # (c_char * CTRL_SIZE).from_address(base)
        self._created = created
        self._handle = handle
        self._base = ctypes_address(view) if view is not None else 0

    # ---- 生命周期 ----
    @classmethod
    def create_or_open(cls, namespace: str, *, width: int, height: int,
                       frame_slots: int, frame_slot_bytes: int,
                       result_slots: int, result_slot_bytes: int,
                       pixel_format: int = PIXEL_FORMAT_GRAY8
                       ) -> Tuple["CtrlBlock", bool]:
        name = obj_name(namespace, "SHM_CTRL_V1")
        handle, created = win32.create_file_mapping(name, CTRL_SIZE)
        base = win32.map_viewOfFile(handle, CTRL_SIZE)
        view = (ctypes.c_char * CTRL_SIZE).from_address(base)
        blk = cls(view, created, handle)
        try:
            if created:
                for i in range(CTRL_SIZE):
                    view[i] = 0
                view[_OFF_MAGIC:_OFF_MAGIC + 4] = MAGIC
                struct.pack_into("<I", memoryview(view), _OFF_VERSION, VERSION)
                struct.pack_into("<I", memoryview(view), _OFF_WIDTH, width)
                struct.pack_into("<I", memoryview(view), _OFF_HEIGHT, height)
                struct.pack_into("<I", memoryview(view), _OFF_PIXEL_FORMAT, pixel_format)
                struct.pack_into("<I", memoryview(view), _OFF_FRAME_SLOTS, frame_slots)
                struct.pack_into("<I", memoryview(view), _OFF_FRAME_SLOT_BYTES, frame_slot_bytes)
                struct.pack_into("<I", memoryview(view), _OFF_RESULT_SLOTS, result_slots)
                struct.pack_into("<I", memoryview(view), _OFF_RESULT_SLOT_BYTES, result_slot_bytes)
            else:
                # 打开者校验（创建竞态兜底：创建者可能尚未写完 magic，重读 3×50ms）
                import time as _t
                for attempt in range(3):
                    if bytes(view[_OFF_MAGIC:_OFF_MAGIC + 4]) == MAGIC:
                        break
                    _t.sleep(0.05)
                got_magic = bytes(view[_OFF_MAGIC:_OFF_MAGIC + 4])
                if got_magic != MAGIC:
                    raise ProtocolMismatchError(
                        f"CTRL magic mismatch: expect {MAGIC!r} got {got_magic!r}")
                got_version = _u32(view, _OFF_VERSION)
                if got_version != VERSION:
                    raise ProtocolMismatchError(
                        f"CTRL version mismatch: expect {VERSION} got {got_version}")
                mismatches = []
                for label, off, expect in (
                    ("frame_slots", _OFF_FRAME_SLOTS, frame_slots),
                    ("frame_slot_bytes", _OFF_FRAME_SLOT_BYTES, frame_slot_bytes),
                    ("result_slots", _OFF_RESULT_SLOTS, result_slots),
                    ("result_slot_bytes", _OFF_RESULT_SLOT_BYTES, result_slot_bytes),
                ):
                    got = _u32(view, off)
                    if got != expect:
                        mismatches.append(f"{label}: expect {expect} got {got}")
                if mismatches:
                    raise ProtocolMismatchError(
                        "CTRL geometry mismatch: " + "; ".join(mismatches))
        except Exception:
            blk.close()
            raise
        return blk, created

    @classmethod
    def attach(cls, namespace: str) -> "CtrlBlock":
        """不指定几何参数的挂载：纯打开（无创建副作用），仅校验 magic/version。

        不存在时抛 WinAPIError（调用方自行重试等待生产者）。
        不得用 CreateFileMapping 兜底——那会物化一个全零 CTRL，
        与真正的创建者竞态（已由双消费者测试证实）。
        """
        name = obj_name(namespace, "SHM_CTRL_V1")
        handle = win32.open_file_mapping(name)
        base = win32.map_viewOfFile(handle, CTRL_SIZE)
        view = (ctypes.c_char * CTRL_SIZE).from_address(base)
        blk = cls(view, False, handle)
        try:
            import time as _t
            for _ in range(3):
                if bytes(view[_OFF_MAGIC:_OFF_MAGIC + 4]) == MAGIC:
                    break
                _t.sleep(0.05)
            got_magic = bytes(view[_OFF_MAGIC:_OFF_MAGIC + 4])
            if got_magic != MAGIC:
                raise ProtocolMismatchError(
                    f"CTRL magic mismatch: expect {MAGIC!r} got {got_magic!r}")
            got_version = _u32(view, _OFF_VERSION)
            if got_version != VERSION:
                raise ProtocolMismatchError(
                    f"CTRL version mismatch: expect {VERSION} got {got_version}")
        except Exception:
            blk.close()
            raise
        return blk

    def close(self) -> None:
        # 顺序：清 Python 引用 → Unmap → CloseHandle
        view, self._view = self._view, None
        handle, self._handle = self._handle, 0
        if view is not None:
            if self._base:
                win32.unmap_view(self._base)
            view = None
        if handle:
            win32.close_handle(handle)

    # ---- 几何（只读） ----
    @property
    def frame_width(self) -> int:
        return _u32(self._view, _OFF_WIDTH)

    @property
    def frame_height(self) -> int:
        return _u32(self._view, _OFF_HEIGHT)

    @property
    def pixel_format(self) -> int:
        return _u32(self._view, _OFF_PIXEL_FORMAT)

    @property
    def frame_slots(self) -> int:
        return _u32(self._view, _OFF_FRAME_SLOTS)

    @property
    def frame_slot_bytes(self) -> int:
        return _u32(self._view, _OFF_FRAME_SLOT_BYTES)

    @property
    def result_slots(self) -> int:
        return _u32(self._view, _OFF_RESULT_SLOTS)

    @property
    def result_slot_bytes(self) -> int:
        return _u32(self._view, _OFF_RESULT_SLOT_BYTES)

    # ---- 序号 ----
    @property
    def frame_write_seq(self) -> int:
        return _u64(self._view, _OFF_FRAME_WRITE_SEQ)

    @frame_write_seq.setter
    def frame_write_seq(self, v: int) -> None:
        _put_u64(self._view, _OFF_FRAME_WRITE_SEQ, v)

    @property
    def result_write_seq(self) -> int:
        return _u64(self._view, _OFF_RESULT_WRITE_SEQ)

    @result_write_seq.setter
    def result_write_seq(self, v: int) -> None:
        _put_u64(self._view, _OFF_RESULT_WRITE_SEQ, v)

    # ---- 心跳与计数 ----
    def _heartbeat(self, off) -> int:
        return _u64(self._view, off)

    def _set_heartbeat(self, off, v: int) -> None:
        _put_u64(self._view, off, v)

    @property
    def producer_heartbeat_ms(self) -> int:
        return self._heartbeat(_OFF_PRODUCER_HB)

    @producer_heartbeat_ms.setter
    def producer_heartbeat_ms(self, v: int) -> None:
        self._set_heartbeat(_OFF_PRODUCER_HB, v)

    @property
    def analyzer_heartbeat_ms(self) -> int:
        return self._heartbeat(_OFF_ANALYZER_HB)

    @analyzer_heartbeat_ms.setter
    def analyzer_heartbeat_ms(self, v: int) -> None:
        self._set_heartbeat(_OFF_ANALYZER_HB, v)

    @property
    def archiver_heartbeat_ms(self) -> int:
        return self._heartbeat(_OFF_ARCHIVER_HB)

    @archiver_heartbeat_ms.setter
    def archiver_heartbeat_ms(self, v: int) -> None:
        self._set_heartbeat(_OFF_ARCHIVER_HB, v)

    @property
    def twin_heartbeat_ms(self) -> int:
        return self._heartbeat(_OFF_TWIN_HB)

    @twin_heartbeat_ms.setter
    def twin_heartbeat_ms(self, v: int) -> None:
        self._set_heartbeat(_OFF_TWIN_HB, v)

    @property
    def dropped_by_overwrite(self) -> int:
        return _u64(self._view, _OFF_DROPPED_OVERWRITE)

    @dropped_by_overwrite.setter
    def dropped_by_overwrite(self, v: int) -> None:
        _put_u64(self._view, _OFF_DROPPED_OVERWRITE, v)

    @property
    def dropped_by_lag(self) -> int:
        return _u64(self._view, _OFF_DROPPED_LAG)

    @dropped_by_lag.setter
    def dropped_by_lag(self, v: int) -> None:
        _put_u64(self._view, _OFF_DROPPED_LAG, v)

    # ---- 状态判定 ----
    def heartbeat_age_ms(self, which: str = "producer") -> int:
        stored = self._heartbeat({"producer": _OFF_PRODUCER_HB,
                                  "analyzer": _OFF_ANALYZER_HB,
                                  "archiver": _OFF_ARCHIVER_HB,
                                  "twin": _OFF_TWIN_HB}[which])
        if stored == 0:
            return 1 << 62  # 从未签到视为极老
        return max(0, win32.GetTickCount64() - stored)

    def producer_alive(self, timeout_ms: int) -> bool:
        return self.heartbeat_age_ms("producer") <= timeout_ms


def ctypes_address(view) -> int:
    """取 ctypes 数组对象的基地址（供 Unmap 使用）。"""
    import ctypes
    return ctypes.addressof(view)


def now_ns() -> int:
    return time.time_ns()
