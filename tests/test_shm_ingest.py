"""SharedMemoryIngestThread 集成测试：离屏 Qt + bench 子进程真帧全链路。"""
import os
import subprocess
import sys
import threading
import uuid
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytestmark = pytest.mark.skipif(not sys.platform.startswith("win"),
                                reason="仅 Windows 命名共享内存")

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "tools" / "shm_bench.py"


def test_ingest_full_pipeline(tmp_path):
    """bench 生产者 → ingest 线程（rule 引擎）→ RESULT ring + 异步归档三件套。"""
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication

    # 全文件统一 QApplication（offscreen）：若先前留下 QCoreApplication，
    # 在其上创建 QWidget 属未定义行为（实测段错误）
    app = QApplication.instance() or QApplication([])

    from utils.anomaly_detection_client import ENGINE_RULE_BASED
    from utils.shm_ingest import SharedMemoryIngestThread

    ns = rf"Local\IAP_TEST_ING{os.getpid()}_{uuid.uuid4().hex[:6]}"
    # 生产者须常驻至断言完成：短命生产者退出后命名对象被 OS 回收，
    # attach 重试将永远扑空（真实桥接为常驻进程，无此问题）
    frames = 30
    proc = subprocess.Popen(
        [sys.executable, str(BENCH), "--producer", "--frames", str(frames),
         "--fps", "2", "--size", "320", "200", "--slots", "4",
         "--namespace", ns],
        cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    results = []
    got_result = threading.Event()
    thread = SharedMemoryIngestThread(
        engine_name=ENGINE_RULE_BASED, namespace=ns,
        api_root=tmp_path / "anomaly_api")
    # 主线程阻塞等待时事件循环不转：测试用 DirectConnection 直接在工作者线程回调
    thread.result_ready.connect(
        lambda seq, payload: (results.append((seq, payload)), got_result.set()),
        Qt.DirectConnection)
    online_flags = []
    thread.source_online.connect(lambda on: online_flags.append(on),
                                 Qt.DirectConnection)
    thread.start()

    try:
        assert got_result.wait(timeout=30), "超时未产生任何推理结果"
        out, err = proc.communicate(timeout=30)
        assert proc.returncode == 0, f"{out}\n{err}"
        assert len(results) >= 1
        seq, payload = results[-1]
        assert seq >= 1
        assert payload["process_id"]
        assert payload["anomaly_level"] in ("很可能正常", "很可能异常")

        # RESULT ring 里也能读到（孪生端视角）
        from utils.ipc import shm_ring
        from utils.ipc.layout import CtrlBlock, obj_name
        ctrl = CtrlBlock.attach(ns)
        evt = shm_ring.NamedEvent(obj_name(ns, "EVT_RESULT"))
        rconsumer = shm_ring.ResultRingConsumer.attach(ctrl, ns, evt)
        try:
            got = rconsumer.fetch_latest(last_seq=0, wait_ms=2000)
            assert got is not None
            rseq, frame_seq, js, pred, heat = got
            assert js["process_id"]
            assert pred.shape == (200, 320)
        finally:
            rconsumer.close()
            evt.close()
            ctrl.close()

        # 归档三件套落盘（input/{id}/ + output/{id}/）
        deadline = threading.Event()
        import time
        t_end = time.monotonic() + 10
        while time.monotonic() < t_end:
            outs = list((tmp_path / "anomaly_api" / "output").glob("*/"))
            ins = list((tmp_path / "anomaly_api" / "input").glob("*/"))
            if outs and ins:
                break
            time.sleep(0.2)
        out_dir = outs[0]
        pid = out_dir.name
        assert (out_dir / f"{pid}.png").exists()
        assert (out_dir / f"{pid}_heatmap.png").exists()
        assert (out_dir / f"{pid}.json").exists()
        input_dir = ins[0]
        assert list(input_dir.glob("*.png")) and (input_dir / "request.json").exists()
        assert online_flags and online_flags[-1] is True
        app.processEvents()
    finally:
        thread.stop()
        thread.wait(15000)
        if proc.poll() is None:
            proc.kill()


_WIDGET_SMOKE = r"""
import os, sys
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["IPC_MODE"] = "shm"
sys.path.insert(0, ".")
from PyQt5.QtWidgets import QApplication
import anomaly_detection_tab as tab_mod

app = QApplication([])
widget = tab_mod.AnomalyDetectionWidget()
assert widget._ipc_mode == "shm", widget._ipc_mode
assert widget.start_batch_btn.isEnabled() is False
assert widget._ingest_thread is not None
assert hasattr(widget, "shm_status_label")
# 模式守卫：start_batch_processing 直接短路返回（无头环境打桩掉模态弹窗）
tab_mod.QMessageBox.information = lambda *a, **k: None
widget.start_batch_processing()
# 未 show 的 widget 不触发 closeEvent——显式停线程，否则 QThread 携
# ctypes 视图存活到解释器退出会段错误
widget._ingest_thread.stop()
widget._ingest_thread.wait(15000)
widget.close()
print("WIDGET_SMOKE_OK")
sys.exit(0)
"""


def test_widget_shm_mode_disables_batch():
    """UI 接线冒烟（独立进程：Qt+ctypes 状态不跨测试共享）。"""
    import subprocess

    proc = subprocess.run([sys.executable, "-c", _WIDGET_SMOKE],
                          cwd=str(ROOT), capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    assert "WIDGET_SMOKE_OK" in proc.stdout
