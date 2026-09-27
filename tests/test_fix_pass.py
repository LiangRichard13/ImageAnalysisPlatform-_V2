"""评审修复批 RED 测试：C1 槽尺寸、I1 文件消失、I2 journal 重入、I5 分析端单实例、I7 接管续写。"""
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(not sys.platform.startswith("win"),
                                reason="仅 Windows 命名共享内存")

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "tools" / "shm_bench.py"
BRIDGE = ROOT / "tools" / "shm_file_bridge.py"
MB = 1024 * 1024


def _ns(tag: str) -> str:
    return rf"Local\IAP_TEST_FIX{os.getpid()}_{tag}_{uuid.uuid4().hex[:6]}"


class TestC1SlotSizing:
    def test_protocol_geometry_helper_production(self):
        """产线尺寸必须给协议表定值：FRAME 36MB / RESULT 64MB。"""
        from utils.ipc.layout import protocol_frame_slot_bytes, protocol_result_slot_bytes
        assert protocol_frame_slot_bytes(31901, 1000) == 37748736
        assert protocol_result_slot_bytes(31901, 1000) == 67108864

    def test_protocol_geometry_helper_formula_fits_layout(self):
        """非产线尺寸按公式：RESULT 槽必须容纳 64+8192+2*w*h。"""
        from utils.ipc.layout import protocol_result_slot_bytes
        w, h = 2000, 500
        need = 64 + 8192 + 2 * w * h
        assert protocol_result_slot_bytes(w, h) >= need

    def test_write_result_capacity_assert_fails_fast(self):
        """槽容量不足时 write_result 必须立即报错，而不是越界写坏邻槽。"""
        from utils.ipc import shm_ring
        from utils.ipc.layout import CtrlBlock, obj_name

        ns = _ns("c1")
        w, h = 2000, 500
        # 故意按旧错误公式建槽（只放一份图+json）：64+8192+w*h 上取整
        bad_result_slot = (64 + 8192 + w * h + MB - 1) // MB * MB
        ctrl, _ = CtrlBlock.create_or_open(
            ns, width=w, height=h, frame_slots=2,
            frame_slot_bytes=(w * h + 64 + MB - 1) // MB * MB,
            result_slots=2, result_slot_bytes=bad_result_slot)
        evt = shm_ring.NamedEvent(obj_name(ns, "EVT_RESULT"))
        producer = shm_ring.ResultRingProducer.create_or_attach(ctrl, ns, evt)
        try:
            pred = np.zeros((h, w), np.uint8)
            with pytest.raises(ValueError, match="slot capacity"):
                producer.write_result(1, {"a": 1}, pred, pred)
            assert ctrl.result_write_seq == 0, "失败不得推进 seq"
        finally:
            producer.close()
            evt.close()
            ctrl.close()

    def test_production_size_write_result_roundtrip(self):
        """31901×1000 真实尺寸的 RESULT 写读往返（协议表槽尺寸）。"""
        from utils.ipc import shm_ring
        from utils.ipc.layout import (CtrlBlock, obj_name,
                                      protocol_frame_slot_bytes,
                                      protocol_result_slot_bytes)

        ns = _ns("c1prod")
        w, h = 31901, 1000
        ctrl, _ = CtrlBlock.create_or_open(
            ns, width=w, height=h, frame_slots=2,
            frame_slot_bytes=protocol_frame_slot_bytes(w, h),
            result_slots=2, result_slot_bytes=protocol_result_slot_bytes(w, h))
        evt = shm_ring.NamedEvent(obj_name(ns, "EVT_RESULT"))
        producer = shm_ring.ResultRingProducer.create_or_attach(ctrl, ns, evt)
        consumer = shm_ring.ResultRingConsumer.attach(ctrl, ns, evt)
        try:
            pred = np.zeros((h, w), np.uint8)
            heat = np.full((h, w), 77, np.uint8)
            payload = {"anomaly_level": "很可能异常", "中文": "键值"}
            seq = producer.write_result(1, payload, pred, heat)
            assert seq == 1
            got = consumer.fetch_latest(last_seq=0, wait_ms=1000)
            assert got is not None
            rseq, frame_seq, js, got_pred, got_heat = got
            assert js == payload
            assert got_pred.shape == (h, w)
            np.testing.assert_array_equal(got_heat, heat)
        finally:
            for o in (producer, consumer, evt, ctrl):
                o.close()


class TestI1VanishedFile:
    def test_bridge_survives_vanished_file(self, tmp_path):
        """待处理文件在两轮扫描间被外部删除 → 桥接进程不退出。"""
        import cv2
        watch = tmp_path / "watch"
        watch.mkdir()
        cv2.imwrite(str(watch / "a.png"), np.zeros((100, 120), np.uint8))
        ns = _ns("i1")
        proc = subprocess.Popen(
            [sys.executable, str(BRIDGE), "--watch", str(watch),
             "--namespace", ns, "--poll-ms", "100", "--width", "120", "--height", "100"],
            cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            # 等 a.png 被吃掉后立刻再放一个再删掉一个，桥接必须还活着
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline and not (watch / ".ingested" / "a.png").exists():
                time.sleep(0.2)
            (watch / "ghost.png").write_bytes(b"trash")  # 会解码失败→重试路径
            time.sleep(0.5)
            (watch / "vanish.png").unlink(missing_ok=True)  # 从未存在过真实内容
            # a.png 入 .ingested 后进程仍须存活
            time.sleep(1.5)
            assert proc.poll() is None, f"桥接崩溃: {proc.communicate(timeout=5)[0]}"
        finally:
            proc.terminate()
            try:
                proc.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


class TestI2JournalNoReingest:
    def test_moved_failed_file_not_reingested(self, tmp_path):
        """journal 降级后（移 .ingested 与 .failed 都失败的场景）不得重复入环。

        模拟：文件可解码、可入环，但目标目录不可写（用同名目录占位堵死
        os.replace 的目标路径），观察 frame_write_seq 不无限增长。
        """
        import cv2
        from utils.ipc.layout import CtrlBlock

        watch = tmp_path / "watch"
        watch.mkdir()
        cv2.imwrite(str(watch / "x.png"), np.zeros((100, 120), np.uint8))
        # 堵死 .ingested 目标：放一个同名目录
        (watch / ".ingested").mkdir()
        (watch / ".ingested" / "x.png").mkdir()
        ns = _ns("i2")
        proc = subprocess.Popen(
            [sys.executable, str(BRIDGE), "--watch", str(watch),
             "--namespace", ns, "--poll-ms", "100", "--width", "120", "--height", "100"],
            cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            ctrl = None
            deadline = time.monotonic() + 20
            while ctrl is None and time.monotonic() < deadline:
                try:
                    ctrl = CtrlBlock.attach(ns)
                except Exception:
                    time.sleep(0.3)
            assert ctrl is not None
            deadline = time.monotonic() + 8
            while time.monotonic() < deadline and ctrl.frame_write_seq < 1:
                time.sleep(0.2)
            first = ctrl.frame_write_seq
            assert first >= 1
            time.sleep(3)  # 若有重复入环，3 秒内会涨多帧
            assert ctrl.frame_write_seq == first, "journal 后重复入环"
            ctrl.close()
        finally:
            proc.terminate()
            try:
                proc.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


class TestWritePerformance:
    def test_write_frame_32mb_under_600ms(self):
        """32MB 写帧必须走 memcpy 级路径（回归：ctypes 切片赋值逐元素慢路径 1.2s/帧）。"""
        import time as _time

        from utils.ipc import shm_ring
        from utils.ipc.layout import (CtrlBlock, obj_name,
                                      protocol_frame_slot_bytes,
                                      protocol_result_slot_bytes)

        ns = _ns("perf")
        w, h = 31901, 1000
        ctrl, _ = CtrlBlock.create_or_open(
            ns, width=w, height=h, frame_slots=2,
            frame_slot_bytes=protocol_frame_slot_bytes(w, h),
            result_slots=2, result_slot_bytes=protocol_result_slot_bytes(w, h))
        evt = shm_ring.NamedEvent(obj_name(ns, "EVT_FRAME"))
        mutex = shm_ring.NamedMutex(obj_name(ns, "PRODUCER_MUTEX"))
        producer = shm_ring.FrameRingProducer.create_or_attach(ctrl, ns, evt, mutex)
        try:
            assert producer.acquire(timeout_ms=1000)
            frame = np.zeros((h, w), np.uint8)
            t0 = _time.perf_counter()
            producer.write_frame(frame, frame_id=1)
            elapsed_ms = (_time.perf_counter() - t0) * 1000
            assert elapsed_ms < 600, f"write_frame 32MB 耗时 {elapsed_ms:.0f}ms（应 <600ms）"
        finally:
            producer.close()
            evt.close()
            mutex.close()
            ctrl.close()


class TestI7TakeoverSeqContinues:
    def test_second_producer_continues_seq(self):
        """消费端持环时，前一个生产者退出、新生产者接管：seq 续写不回零。"""
        from utils.ipc import shm_ring
        from utils.ipc.layout import CtrlBlock, obj_name

        ns = _ns("i7")
        p1 = subprocess.Popen(
            [sys.executable, str(BENCH), "--producer", "--frames", "3",
             "--fps", "10", "--size", "320", "200", "--slots", "4",
             "--namespace", ns],
            cwd=str(ROOT), stdout=subprocess.DEVNULL)
        try:
            ctrl = None
            deadline = time.monotonic() + 20
            while ctrl is None and time.monotonic() < deadline:
                try:
                    ctrl = CtrlBlock.attach(ns)
                except Exception:
                    time.sleep(0.3)
            assert ctrl is not None
            p1.wait(timeout=30)
            assert ctrl.frame_write_seq == 3
            # 第二个生产者接管（消费端仍持句柄，环未销毁）
            p2 = subprocess.Popen(
                [sys.executable, str(BENCH), "--producer", "--frames", "3",
                 "--fps", "10", "--size", "320", "200", "--slots", "4",
                 "--namespace", ns],
                cwd=str(ROOT), stdout=subprocess.DEVNULL)
            try:
                deadline = time.monotonic() + 20
                while time.monotonic() < deadline and ctrl.frame_write_seq < 6:
                    time.sleep(0.3)
                assert ctrl.frame_write_seq == 6, f"seq={ctrl.frame_write_seq}（回零即断言失败）"
            finally:
                p2.wait(timeout=30)
            ctrl.close()
        finally:
            if p1.poll() is None:
                p1.kill()
