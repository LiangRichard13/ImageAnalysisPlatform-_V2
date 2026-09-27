"""跨进程共享内存链路测试：子进程生产者 + 本进程消费者。"""
import os
import subprocess
import sys
import threading
import uuid
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(not sys.platform.startswith("win"),
                                reason="仅 Windows 命名共享内存")

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "tools" / "shm_bench.py"


def _ns(tag: str) -> str:
    return rf"Local\IAP_TEST_MP{os.getpid()}_{tag}_{uuid.uuid4().hex[:6]}"


def _run_bench(args, check=True):
    proc = subprocess.run([sys.executable, str(BENCH), *args],
                          capture_output=True, text=True, cwd=str(ROOT))
    if check and proc.returncode != 0:
        raise AssertionError(f"bench 失败: {proc.stdout}\n{proc.stderr}")
    return proc


def _attach_consumer(ns, frames_expected):
    """本进程作为消费者，读满 frames_expected 或超时。"""
    import time

    from utils.ipc import shm_ring
    from utils.ipc.layout import CtrlBlock, obj_name

    ctrl = None
    t_end = time.monotonic() + 15
    while ctrl is None:
        try:
            ctrl = CtrlBlock.attach(ns)
        except Exception:
            if time.monotonic() > t_end:
                raise
            time.sleep(0.2)
    evt = shm_ring.NamedEvent(obj_name(ns, "EVT_FRAME"))
    consumer = shm_ring.FrameRingConsumer.attach(ctrl, ns, evt)
    try:
        seen, verified, last = 0, 0, 0
        deadline = threading.Event()
        import time
        t_end = time.monotonic() + 30
        while time.monotonic() < t_end:
            got = consumer.fetch_latest(last_seq=last, wait_ms=200)
            if got is None:
                continue
            seq, frame_id, ts_ns, gray = got
            assert seq > last
            last = seq
            seen += 1
            # 帧内容校验：全帧按 seq 取模填充 + 首 8 字节嵌 seq
            flat = gray.reshape(-1)
            assert int.from_bytes(bytes(flat[:8]), "little") == seq, "嵌入 seq 不符"
            assert (flat[8:] == (seq & 0xFF)).all(), "帧内容损坏"
            verified += 1
            if last >= frames_expected:
                break
        return seen, verified, last
    finally:
        consumer.close()
        ctrl.close()


class TestMultiproc:
    def test_producer_subprocess_and_consume(self):
        ns = _ns("prod")
        frames = 6
        proc = subprocess.Popen(
            [sys.executable, str(BENCH), "--producer", "--frames", str(frames),
             "--fps", "20", "--size", "320", "200", "--slots", "4",
             "--namespace", ns],
            cwd=str(ROOT),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            seen, verified, last = _attach_consumer(ns, frames)
            out, err = proc.communicate(timeout=30)
            assert proc.returncode == 0, f"{out}\n{err}"
            assert verified == seen and seen >= 2
            assert last == frames
        finally:
            if proc.poll() is None:
                proc.kill()

    def test_bench_both_quick(self):
        """--both 单命令模式：双进程全链路 + 内容校验 + 时延统计。"""
        ns = _ns("both")
        proc = _run_bench(["--both", "--frames", "5", "--fps", "20",
                           "--size", "320", "200", "--slots", "4",
                           "--namespace", ns])
        assert "BAD=0" in proc.stdout
        assert proc.returncode == 0

    def test_concurrent_first_create(self):
        """两个子进程同时首建 CTRL：恰好一个 created=True。"""
        ns = _ns("race")
        script = (
            "import sys; sys.path.insert(0, '.');"
            "from utils.ipc.layout import CtrlBlock;"
            f"_, created = CtrlBlock.create_or_open(r'{ns}', width=320, height=200,"
            "frame_slots=2, frame_slot_bytes=65536, result_slots=2,"
            "result_slot_bytes=136256);"
            "import time; time.sleep(0.5);"
            "print('CREATED' if created else 'OPENED', flush=True)"
        )
        procs = [subprocess.Popen([sys.executable, "-c", script], cwd=str(ROOT),
                                  stdout=subprocess.PIPE, text=True)
                 for _ in range(2)]
        outs = [p.communicate(timeout=20)[0].strip() for p in procs]
        assert sorted(outs) == ["CREATED", "OPENED"], outs

    def test_two_consumers_concurrent_no_corruption(self):
        """双消费者并发 latest-wins：读到的每一帧都通过内容校验。"""
        ns = _ns("twoc")
        frames = 8
        proc = subprocess.Popen(
            [sys.executable, str(BENCH), "--producer", "--frames", str(frames),
             "--fps", "10", "--size", "320", "200", "--slots", "4",
             "--namespace", ns],
            cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            results = []

            def worker():
                try:
                    results.append(_attach_consumer(ns, frames))
                except Exception as exc:  # noqa: BLE001
                    results.append(("ERROR", str(exc)))

            threads = [threading.Thread(target=worker) for _ in range(2)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=40)
            out, err = proc.communicate(timeout=30)
            assert proc.returncode == 0, f"{out}\n{err}"
            assert len(results) == 2
            for r in results:
                assert not (isinstance(r[0], str) and r[0] == "ERROR"), r
        finally:
            if proc.poll() is None:
                proc.kill()
