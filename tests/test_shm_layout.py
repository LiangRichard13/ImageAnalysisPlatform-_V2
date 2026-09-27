"""utils/ipc/layout.py 协议布局单元测试。

对应协议 V1（docs/ipc/SHM_PROTOCOL.md §2-§4）：
头部 struct 尺寸、字段往返、CRC、CtrlBlock create-or-open 与校验。
"""
import struct

import pytest

from utils.ipc.layout import (
    CTRL_SIZE,
    FRAME_DATA_OFF,
    FRAME_HDR,
    JSON_MAX,
    MAGIC,
    RESULT_HDR,
    RESULT_JSON_OFF,
    VERSION,
    crc32,
)


class TestHeaderLayout:
    def test_frame_header_size_is_64(self):
        assert FRAME_HDR.size == 64

    def test_result_header_size_is_64(self):
        assert RESULT_HDR.size == 64

    def test_ctrl_size_is_4096(self):
        assert CTRL_SIZE == 4096

    def test_data_offsets(self):
        assert FRAME_DATA_OFF == 64
        assert RESULT_JSON_OFF == 64
        assert JSON_MAX == 8192

    def test_frame_header_roundtrip(self):
        fields = (7, 1700000000123456789, 42, 0xDEADBEEF, 31901000, 2)
        packed = FRAME_HDR.pack(*fields)
        assert len(packed) == 64
        assert FRAME_HDR.unpack(packed) == fields

    def test_result_header_roundtrip(self):
        fields = (3, 2, 1700000000987654321, 100, 31901000, 31901000, 0xCAFEBABE)
        packed = RESULT_HDR.pack(*fields)
        assert len(packed) == 64
        assert RESULT_HDR.unpack(packed) == fields

    def test_protocol_constants(self):
        assert MAGIC == b"IAPS"
        assert VERSION == 1


class TestCrc32:
    def test_known_vector(self):
        # CRC-32/ISO-HDLC 标准测试向量
        assert crc32(b"123456789") == 0xCBF43926

    def test_empty(self):
        assert crc32(b"") == 0

    def test_large_payload_rolling_eq_full(self):
        data = bytes(range(256)) * 1000
        assert crc32(data) == crc32(data)


@pytest.mark.skipif(not __import__("sys").platform.startswith("win"),
                    reason="仅 Windows 命名共享内存")
class TestCtrlBlock:
    NS = rf"Local\IAP_TEST_L{__import__('os').getpid()}"

    def _geom(self):
        return dict(width=640, height=480, frame_slots=2, frame_slot_bytes=65536,
                    result_slots=2, result_slot_bytes=131072, pixel_format=0)

    def test_create_then_open(self):
        from utils.ipc.layout import CtrlBlock
        from utils.ipc.win32 import GetTickCount64

        blk, created = CtrlBlock.create_or_open(self.NS, **self._geom())
        assert created is True
        try:
            assert blk.frame_width == 640
            assert blk.frame_height == 480
            assert blk.frame_slots == 2
            assert blk.frame_slot_bytes == 65536
            assert blk.result_slots == 2
            assert blk.result_slot_bytes == 131072
            assert blk.frame_write_seq == 0
            assert blk.result_write_seq == 0

            blk2, created2 = CtrlBlock.create_or_open(self.NS, **self._geom())
            assert created2 is False
            try:
                assert blk2.frame_width == 640
            finally:
                blk2.close()

            blk.frame_write_seq = 5
            assert blk.frame_write_seq == 5
            blk.dropped_by_lag = 3
            assert blk.dropped_by_lag == 3
            blk.analyzer_heartbeat_ms = GetTickCount64()
            assert blk.producer_alive(timeout_ms=10000) is False
        finally:
            blk.close()

    def test_version_mismatch_raises(self):
        from utils.ipc.layout import CtrlBlock, ProtocolMismatchError

        blk, _ = CtrlBlock.create_or_open(self.NS, **self._geom())
        try:
            # 直接把版本位改坏，模拟另一套协议
            blk._view[4:8] = struct.pack("<I", 99)
            with pytest.raises(ProtocolMismatchError) as ei:
                CtrlBlock.create_or_open(self.NS, **self._geom())
            assert "version" in str(ei.value)
        finally:
            blk.close()

    def test_attach_without_geometry(self):
        """attach 只校验 magic/version，几何从块内读（工具/消费者场景）。"""
        import uuid

        from utils.ipc.layout import CtrlBlock

        ns = rf"Local\IAP_TEST_A{__import__('os').getpid()}_{uuid.uuid4().hex[:6]}"
        blk, _ = CtrlBlock.create_or_open(ns, **self._geom())
        try:
            attached = CtrlBlock.attach(ns)
            try:
                assert attached.frame_width == 640
                assert attached.frame_slots == 2
                assert attached.frame_slot_bytes == 65536
            finally:
                attached.close()
        finally:
            blk.close()
