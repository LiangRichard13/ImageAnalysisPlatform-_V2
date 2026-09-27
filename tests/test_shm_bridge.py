"""tools/shm_file_bridge.py 文件桥接测试：落图目录 → FRAME ring + .ingested 归位。"""
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(not sys.platform.startswith("win"),
                                reason="仅 Windows 命名共享内存")

ROOT = Path(__file__).resolve().parent.parent
BRIDGE = ROOT / "tools" / "shm_file_bridge.py"
W, H = 320, 200


def _drop_png(directory: Path, name: str, seed: int) -> Path:
    rng = np.random.default_rng(seed)
    path = directory / name
    cv2.imwrite(str(path), rng.integers(0, 256, size=(H, W), dtype=np.uint8))
    return path


def test_bridge_moves_files_into_ring(tmp_path):
    from utils.ipc import shm_ring
    from utils.ipc.layout import CtrlBlock, obj_name

    ns = rf"Local\IAP_TEST_BR{os.getpid()}_{uuid.uuid4().hex[:6]}"
    watch = tmp_path / "watch"
    watch.mkdir()
    img_paths = [_drop_png(watch, f"cam_{i}.png", i) for i in range(3)]
    expected_frames = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in img_paths]

    proc = subprocess.Popen(
        [sys.executable, str(BRIDGE), "--watch", str(watch),
         "--namespace", ns, "--poll-ms", "100",
         "--width", str(W), "--height", str(H)],
        cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    try:
        # 等 CTRL 出现 + 3 帧入环
        ctrl = None
        deadline = time.monotonic() + 30
        while ctrl is None and time.monotonic() < deadline:
            try:
                ctrl = CtrlBlock.attach(ns)
            except Exception:
                time.sleep(0.3)
        assert ctrl is not None, "桥接未创建 CTRL"
        evt = shm_ring.NamedEvent(obj_name(ns, "EVT_FRAME"))
        consumer = shm_ring.FrameRingConsumer.attach(ctrl, ns, evt)
        try:
            deadline = time.monotonic() + 30
            while ctrl.frame_write_seq < 3 and time.monotonic() < deadline:
                time.sleep(0.3)
            assert ctrl.frame_write_seq == 3, f"seq={ctrl.frame_write_seq}"

            got = consumer.fetch_latest(last_seq=0, wait_ms=1000)
            assert got is not None
            seq, frame_id, _ts, gray = got
            assert seq == 3
            np.testing.assert_array_equal(gray, expected_frames[2])

            # 已入 ring 的文件移入 .ingested/
            ingested = watch / ".ingested"
            deadline = time.monotonic() + 10
            while len(list(ingested.glob("*.png"))) < 3 and time.monotonic() < deadline:
                time.sleep(0.2)
            assert sorted(p.name for p in ingested.glob("*.png")) == \
                sorted(p.name for p in img_paths)
            # 监控目录本体已无散图
            assert list(watch.glob("*.png")) == []
        finally:
            consumer.close()
            evt.close()
            ctrl.close()
    finally:
        proc.terminate()
        try:
            proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_bridge_rejects_wrong_size_to_failed(tmp_path):
    """尺寸与 CTRL 几何不符的图移入 .failed/，不入环。"""
    from utils.ipc.layout import CtrlBlock

    ns = rf"Local\IAP_TEST_BRF{os.getpid()}_{uuid.uuid4().hex[:6]}"
    watch = tmp_path / "watch"
    watch.mkdir()
    _drop_png(watch, "good.png", 1)          # 320×200 建几何
    bad = np.zeros((100, 100), dtype=np.uint8)
    cv2.imwrite(str(watch / "bad.png"), bad)

    proc = subprocess.Popen(
        [sys.executable, str(BRIDGE), "--watch", str(watch),
         "--namespace", ns, "--poll-ms", "100",
         "--width", str(W), "--height", str(H)],
        cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        ctrl = None
        deadline = time.monotonic() + 30
        while ctrl is None and time.monotonic() < deadline:
            try:
                ctrl = CtrlBlock.attach(ns)
            except Exception:
                time.sleep(0.3)
        assert ctrl is not None
        assert (ctrl.frame_width, ctrl.frame_height) == (W, H)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and ctrl.frame_write_seq < 1:
            time.sleep(0.3)
        time.sleep(2)  # 给 bad.png 走完判定
        assert ctrl.frame_write_seq == 1, "bad 不应入环"
        failed = watch / ".failed"
        deadline = time.monotonic() + 10
        while not (failed / "bad.png").exists() and time.monotonic() < deadline:
            time.sleep(0.2)
        assert (failed / "bad.png").exists(), "bad 应移入 .failed/"
        assert (watch / ".ingested" / "good.png").exists()
        ctrl.close()
    finally:
        proc.terminate()
        try:
            proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
