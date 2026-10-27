"""tools/camera_simulator.py 相机端模拟器测试：SenderThread 环写入/跳过/接管语义。

离屏 Qt + 唯一命名空间（同 test_shm_ingest.py / test_shm_bridge.py 先例）；
GUI 窗口仅做离屏构造冒烟，视觉/交互走人工冒烟（见 docs/ipc/相机端接入指南.md §8）。
"""
import importlib.util
import os
import sys
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytestmark = pytest.mark.skipif(not sys.platform.startswith("win"),
                                reason="仅 Windows 命名共享内存")

ROOT = Path(__file__).resolve().parent.parent
MODULE_PATH = ROOT / "tools" / "camera_simulator.py"
W, H = 320, 200


def _load_module():
    spec = importlib.util.spec_from_file_location("camera_simulator", MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _drop_image(directory: Path, name: str, fill: int, size=(W, H)) -> Path:
    path = directory / name
    cv2.imwrite(str(path), np.full((size[1], size[0]), fill, dtype=np.uint8))
    return path


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    # 部署 .env 会改写 IPC namespace 等；模拟器显式传参不依赖 env，钉住以防意外
    monkeypatch.setenv("IPC_MODE", "shm")


@pytest.fixture()
def qapp():
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


def _spin_until(app, cond, timeout_s=15.0):
    """QThread 信号是 queued 连接：processEvents+sleep 驱动事件循环直到条件成立。"""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        app.processEvents()
        if cond():
            return True
        time.sleep(0.02)
    return False


def test_sender_sends_frames_cyclically(qapp, tmp_path):
    mod = _load_module()
    ns = rf"Local\IAP_TEST_SIM{os.getpid()}_{uuid.uuid4().hex[:6]}"
    names = ["a_first.png", "b_second.png", "c_third.png"]
    for i, name in enumerate(names):
        _drop_image(tmp_path, name, fill=40 + i * 60)
    sent_records = []
    errors = []
    thread = mod.SenderThread(tmp_path, interval_ms=50, namespace=ns)
    thread.frame_done.connect(
        lambda name, seq, sent, skipped, note: sent_records.append((name, seq, sent, skipped, note)))
    thread.error.connect(errors.append)
    scan_counts = []
    thread.scan_done.connect(scan_counts.append)
    thread.start()
    ctrl_seen = None
    try:
        # >3*2 帧证明回绕循环（sorted 顺序 a→b→c→a…）
        assert _spin_until(qapp, lambda: sent_records and sent_records[-1][2] >= 7), \
            f"未达到 7 帧: {sent_records[-1:]}"
        assert scan_counts == [3]
        seqs = [r[1] for r in sent_records]
        assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs), f"seq 须严格递增: {seqs}"
        got_names = [r[0] for r in sent_records]
        assert got_names[:3] == names, f"首轮须按 sorted 文件序: {got_names[:3]}"
        assert got_names[3] == names[0], f"第 4 帧须回到列表头（循环）: {got_names}"
        assert all(r[4] == "" for r in sent_records), f"正常帧备注须为空: {sent_records}"
        assert errors == [], f"正常发送不应报错: {errors}"
        # 线程释放后无人持引用则 CTRL 被 OS 销毁：stop 前持引用验证建环几何与 seq
        from utils.ipc.layout import CtrlBlock
        ctrl_seen = CtrlBlock.attach(ns)
        assert (ctrl_seen.frame_width, ctrl_seen.frame_height) == (W, H)
        assert ctrl_seen.frame_write_seq >= 7
    finally:
        thread.stop()
        assert _spin_until(qapp, lambda: not thread.isRunning()), "stop 后须快速退出"
        _spin_until(qapp, lambda: False, timeout_s=0.1)  # 排空尾随 queued 信号
        if ctrl_seen is not None:
            ctrl_seen.close()


def test_size_mismatch_and_corrupt_skipped(qapp, tmp_path):
    mod = _load_module()
    ns = rf"Local\IAP_TEST_SIM{os.getpid()}_{uuid.uuid4().hex[:6]}"
    _drop_image(tmp_path, "a_good1.png", fill=10)
    (tmp_path / "b_corrupt.png").write_bytes(b"not a png at all")
    _drop_image(tmp_path, "c_good2.png", fill=20)
    _drop_image(tmp_path, "d_wrong_size.png", fill=30, size=(100, 80))

    records = []
    errors = []
    thread = mod.SenderThread(tmp_path, interval_ms=50, namespace=ns)
    thread.frame_done.connect(
        lambda name, seq, sent, skipped, note: records.append((name, seq, sent, skipped, note)))
    thread.error.connect(errors.append)
    thread.start()
    try:
        # 一整轮（4 文件全处理过）后才可能同时满足 sent≥2 且 skipped≥2
        assert _spin_until(qapp, lambda: records and records[-1][2] >= 2
                           and records[-1][3] >= 2), f"未完成一轮: {records[-1:]}"
        assert errors == [], f"跳过属非致命，不应报错: {errors}"
        skips = [r for r in records if r[1] == 0 and r[4]]
        assert any("解码失败" in r[4] for r in skips), f"损坏文件须记解码失败: {skips}"
        assert any("尺寸不符" in r[4] for r in skips), f"异尺寸须记尺寸不符: {skips}"
        assert any(r[0] == "d_wrong_size.png" and "100x80" in r[4] for r in skips), \
            f"尺寸备注须含实际尺寸: {skips}"
    finally:
        thread.stop()
        assert _spin_until(qapp, lambda: not thread.isRunning())
        _spin_until(qapp, lambda: False, timeout_s=0.1)


def test_empty_folder_errors(qapp, tmp_path):
    mod = _load_module()
    ns = rf"Local\IAP_TEST_SIM{os.getpid()}_{uuid.uuid4().hex[:6]}"
    errors = []
    thread = mod.SenderThread(tmp_path, interval_ms=50, namespace=ns)
    thread.error.connect(errors.append)
    thread.start()
    assert _spin_until(qapp, lambda: bool(errors)), "空目录须报错"
    assert _spin_until(qapp, lambda: not thread.isRunning()), "报错后线程须退出"
    assert "图片" in errors[0]


def test_reuses_existing_ctrl_geometry(qapp, tmp_path):
    mod = _load_module()
    ns = rf"Local\IAP_TEST_SIM{os.getpid()}_{uuid.uuid4().hex[:6]}"
    # 测试进程持引用 → 模拟器释放后命名对象不销毁，模拟第二会话/真实端共存
    from utils.ipc.layout import CtrlBlock, protocol_frame_slot_bytes, protocol_result_slot_bytes
    ctrl_hold, created = CtrlBlock.create_or_open(
        ns, width=W, height=H, frame_slots=8,
        frame_slot_bytes=protocol_frame_slot_bytes(W, H),
        result_slots=8, result_slot_bytes=protocol_result_slot_bytes(W, H))
    assert created
    try:
        # 目录全是 200×150：与现有 CTRL 几何不符 → 全部跳过、不重建环、不报错
        _drop_image(tmp_path, "x_small1.png", fill=11, size=(200, 150))
        _drop_image(tmp_path, "y_small2.png", fill=22, size=(200, 150))
        records = []
        errors = []
        thread = mod.SenderThread(tmp_path, interval_ms=50, namespace=ns)
        thread.frame_done.connect(
            lambda name, seq, sent, skipped, note: records.append((name, seq, sent, skipped, note)))
        thread.error.connect(errors.append)
        thread.start()
        try:
            assert _spin_until(qapp, lambda: records and records[-1][3] >= 2), \
                f"两张异尺寸图都须被跳过: {records[-1:]}"
            assert errors == [], f"沿用几何时不匹配只跳过不报错: {errors}"
            assert all("尺寸不符" in r[4] and "200x150≠320x200" in r[4] for r in records), \
                f"备注须含 现有几何≠图尺寸: {records}"
            assert ctrl_hold.frame_width == W and ctrl_hold.frame_height == H
            assert ctrl_hold.frame_write_seq == 0, f"不得向已有环写入: seq={ctrl_hold.frame_write_seq}"
        finally:
            thread.stop()
            assert _spin_until(qapp, lambda: not thread.isRunning())
            _spin_until(qapp, lambda: False, timeout_s=0.1)
    finally:
        ctrl_hold.close()


def test_window_constructs_offscreen(qapp, tmp_path):
    from PyQt5.QtCore import QSettings
    # 上次目录持久化在注册表：清掉本工具的键，保证"初始无目录"断言确定性
    QSettings("analysis_system", "camera_simulator").clear()
    mod = _load_module()
    win = mod.CameraSimulatorWindow()
    try:
        assert win.start_btn.objectName() == "primaryBtn"
        assert win.stop_btn.objectName() == "dangerBtn"
        assert win.stop_btn.isEnabled() is False, "初始停止钮须禁用"
        assert win.start_btn.isEnabled() is False, "无有效目录时开始钮须禁用"
        # 填入有效目录 → 开始钮启用
        win.dir_edit.setText(str(tmp_path))
        win._validate_dir()
        assert win.start_btn.isEnabled() is True
        win.dir_edit.setText(str(tmp_path / "不存在的目录"))
        win._validate_dir()
        assert win.start_btn.isEnabled() is False
    finally:
        win.close()
