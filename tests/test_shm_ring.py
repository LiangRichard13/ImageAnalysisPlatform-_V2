"""utils/ipc/shm_ring.py 环形缓冲读写单元测试。

使用测试专用小环（独立命名空间 + 小尺寸槽），不占用生产 36MB/64MB 槽。
协议语义：latest-wins、seq 完成判据、事件唤醒/追平重置、覆盖计数。
"""
import os
import sys
import uuid

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(not sys.platform.startswith("win"),
                                reason="仅 Windows 命名共享内存")

W, H = 320, 200                       # 单帧 64000B，适配 64KB 测试槽
FRAME_SLOT = 65536
RESULT_SLOT = 64000 * 2 + 8192 + 64   # 136256
GEOM = dict(width=W, height=H, frame_slots=2, frame_slot_bytes=FRAME_SLOT,
            result_slots=2, result_slot_bytes=RESULT_SLOT, pixel_format=0)


def make_frame(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(H, W), dtype=np.uint8)


@pytest.fixture()
def rig():
    """一个 ctrl + 生产者 + 消费者的最小测试装置（每例独立命名空间）。"""
    from utils.ipc.layout import CtrlBlock, obj_name
    from utils.ipc import shm_ring

    ns = rf"Local\IAP_TEST_R{os.getpid()}_{uuid.uuid4().hex[:8]}"
    ctrl, _ = CtrlBlock.create_or_open(ns, **GEOM)
    evt_frame = shm_ring.NamedEvent(obj_name(ns, "EVT_FRAME"))
    evt_result = shm_ring.NamedEvent(obj_name(ns, "EVT_RESULT"))
    mutex = shm_ring.NamedMutex(obj_name(ns, "PRODUCER_MUTEX"))
    producer = shm_ring.FrameRingProducer.create_or_attach(ctrl, ns, evt_frame, mutex)
    consumer = shm_ring.FrameRingConsumer.attach(ctrl, ns, evt_frame)
    rproducer = shm_ring.ResultRingProducer.create_or_attach(ctrl, ns, evt_result)
    rconsumer = shm_ring.ResultRingConsumer.attach(ctrl, ns, evt_result)
    try:
        yield dict(ctrl=ctrl, producer=producer, consumer=consumer,
                   rproducer=rproducer, rconsumer=rconsumer,
                   evt_frame=evt_frame, evt_result=evt_result, mutex=mutex)
    finally:
        for obj in (producer, consumer, rproducer, rconsumer, evt_frame,
                    evt_result, mutex, ctrl):
            obj.close()


class TestFrameRing:
    def test_write_read_roundtrip(self, rig):
        frame = make_frame(1)
        seq = rig["producer"].write_frame(frame, frame_id=7)
        assert seq == 1
        assert rig["evt_frame"].wait(0) is True, "写帧后事件应置位"

        got = rig["consumer"].fetch_latest(last_seq=0, wait_ms=0)
        assert got is not None
        rseq, frame_id, ts_ns, gray = got
        assert rseq == 1
        assert frame_id == 7
        assert ts_ns > 0
        np.testing.assert_array_equal(gray, frame)
        assert rig["evt_frame"].wait(0) is False, "追平后事件应被重置"

    def test_latest_wins_ten_frames(self, rig):
        for i in range(10):
            rig["producer"].write_frame(make_frame(100 + i), frame_id=i)
        got = rig["consumer"].fetch_latest(last_seq=0, wait_ms=0)
        assert got is not None
        rseq, frame_id, _, gray = got
        assert rseq == 10
        assert frame_id == 9
        np.testing.assert_array_equal(gray, make_frame(109))

    def test_no_new_frame_returns_none(self, rig):
        rig["producer"].write_frame(make_frame(1), frame_id=1)
        rig["consumer"].fetch_latest(last_seq=0, wait_ms=0)
        assert rig["consumer"].fetch_latest(last_seq=1, wait_ms=0) is None

    def test_overwrite_counter_after_wrap(self, rig):
        for i in range(5):  # 2 槽，写 5 帧覆盖 3 次
            rig["producer"].write_frame(make_frame(i), frame_id=i)
        assert rig["ctrl"].dropped_by_overwrite == 3

    def test_corrupt_latest_returns_none_and_counts_lag(self, rig):
        rig["producer"].write_frame(make_frame(1), frame_id=1)
        # 手动把最新槽的 seq 破坏，模拟读取期间被覆盖/数据不可信（在 ring 视图上操作）
        ctrl = rig["ctrl"]
        s = ctrl.frame_write_seq
        off = (s % ctrl.frame_slots) * ctrl.frame_slot_bytes
        ring_view = rig["producer"]._view
        ring_view[off:off + 8] = b"\x00" * 8
        before = ctrl.dropped_by_lag
        assert rig["consumer"].fetch_latest(last_seq=0, wait_ms=0) is None
        assert ctrl.dropped_by_lag >= before + 1

    def test_producer_mutex_excludes_second(self, rig):
        # 命名互斥体线程所有：同线程重入会递归获得，必须跨线程验证互斥
        import queue
        import threading

        from utils.ipc import shm_ring

        assert rig["producer"].acquire(timeout_ms=0) is True
        results = queue.Queue()

        def try_acquire():
            m = shm_ring.NamedMutex(rig["mutex"].name)
            try:
                results.put(m.acquire(timeout_ms=0))
            finally:
                m.close()

        t = threading.Thread(target=try_acquire)
        t.start()
        t.join(timeout=10)
        assert results.get(timeout=5) is False, "持有期间他线程不得获得"

        rig["producer"].release()
        m2 = shm_ring.NamedMutex(rig["mutex"].name)
        try:
            assert m2.acquire(timeout_ms=0) is True, "释放后应可获得"
        finally:
            m2.release()
            m2.close()


class TestResultRing:
    def test_result_roundtrip(self, rig):
        pred, heat = make_frame(2), make_frame(3)
        payload = {"anomaly_level": "很可能异常", "score": 0.5, "nested": {"a": [1, 2]}}
        seq = rig["rproducer"].write_result(1, payload, pred, heat)
        assert seq == 1
        assert rig["evt_result"].wait(0) is True

        got = rig["rconsumer"].fetch_latest(last_seq=0, wait_ms=0)
        assert got is not None
        rseq, frame_seq, js, got_pred, got_heat = got
        assert rseq == 1 and frame_seq == 1
        assert js == payload
        np.testing.assert_array_equal(got_pred, pred)
        np.testing.assert_array_equal(got_heat, heat)

    def test_oversize_json_rejected(self, rig):
        big = {"k": "x" * 9000}
        with pytest.raises(ValueError):
            rig["rproducer"].write_result(1, big, make_frame(1), make_frame(2))
        # 失败不得推进 seq，也不得置位事件
        assert rig["ctrl"].result_write_seq == 0
        assert rig["evt_result"].wait(0) is False
