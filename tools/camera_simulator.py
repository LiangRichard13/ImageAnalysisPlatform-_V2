r"""相机端模拟器：选文件夹 → 扫描图片 → 按间隔循环写入 FRAME ring。

用法（项目根，analysis_system 环境）:
    python tools/camera_simulator.py          # GUI
    python tools/camera_simulator.py --smoke  # 打包冒烟（无头构造窗口，SMOKE_PASS/FAILED）
    python tools/build_camera_sim.py          # 打包 exe（onedir，不含 torch，见脚本头注释）

模拟 docs/ipc/相机端接入指南.md 定义的 FRAME 生产者，供无相机/桥接进程时
手工联调分析端/孪生端（自测配合 tools/shm_inspect.py，见指南 §8）。行为语义：

  - "开始发送"时建环并抢 Local\IAP_PRODUCER_MUTEX（单写者，与桥接/真实相机互斥；
    抢不到红字提示不退出）。CTRL 已存在（上一会话/桥接建的）时沿用其几何，
    否则首张成功解码图定几何（utils/ipc/layout.py::protocol_*_slot_bytes 推导槽尺寸）；
  - 尺寸不符/解码失败的图跳过并计数（tools/shm_file_bridge.py 的 .failed 语义之 GUI 版）；
  - 发送等待按 ≤1s 切片刷新 producer_heartbeat_ms（"活着即在线"——间隔长于 10s
    心跳超时也不断链，桥接 :189-191 同款对策）；
  - "停止发送"按 tools/shm_bench.py:57-62 顺序释放全部句柄：CTRL 若无他人持有即销毁，
    分析端按源离线等待（utils/shm_ingest.py attach 重试）自动恢复；seq 由 CTRL 继承，
    重启自动续写不回零。
"""
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
from PyQt5.QtCore import QSettings, QThread, pyqtSignal
from PyQt5.QtWidgets import (QApplication, QFileDialog, QHBoxLayout, QLabel,
                             QLineEdit, QMainWindow, QPushButton, QSpinBox,
                             QVBoxLayout, QWidget)

import ui_theme
from utils.ipc import shm_ring
from utils.ipc.config import load_ipc_config
from utils.ipc.layout import (CtrlBlock, ProtocolMismatchError, obj_name,
                              protocol_frame_slot_bytes,
                              protocol_result_slot_bytes)
from utils.ipc.win32 import GetTickCount64

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
HEARTBEAT_SLICE_S = 1.0   # 心跳续命切片上限（heartbeat_timeout_ms=10000 的 1/10）
_KEEPALIVE_THREADS = []   # closeEvent 兜底保活（anomaly_detection_tab.py:27 先例）


class SenderThread(QThread):
    """扫描 → 建环 → 循环写帧线程（IPC 句柄在线程内创建/销毁，避免跨线程归属）。

    stop() 只置 threading.Event（黄金模板 utils/shm_ingest.py SharedMemoryIngestThread），
    禁 terminate()——写帧中途打断会留半写状态，属协议语义禁止。停止完成复用内建 finished。
    """

    scan_done = pyqtSignal(int)                    # 目录扫描完成，图片张数
    session_started = pyqtSignal(str)              # 环就绪："320x200 slots=8（新建/沿用已存在 CTRL）"
    frame_done = pyqtSignal(str, int, int, int, str)  # (文件名, seq[跳过=0], 已发送, 已跳过, 备注)
    error = pyqtSignal(str)                        # 致命错误；emit 后 run() 收尾退出

    def __init__(self, folder: Path, interval_ms: int, namespace: str, parent=None):
        super().__init__(parent)
        self._folder = Path(folder)
        self._interval_ms = int(interval_ms)
        self._namespace = namespace
        self._stop_flag = threading.Event()
        self._ctrl = self._producer = self._evt = self._mutex = None

    def stop(self) -> None:
        self._stop_flag.set()

    def _scan(self) -> list:
        files = [p for p in sorted(self._folder.iterdir(), key=lambda p: p.name.lower())
                 if p.is_file() and not p.name.startswith(".")
                 and p.suffix.lower() in IMAGE_EXTS]
        return files

    def _ensure_ring(self, gray) -> bool:
        """探测优先 bootstrap：已有 CTRL 沿用其几何（绝不传新几何触发不匹配）。"""
        if self._ctrl is not None:
            return True
        ns = self._namespace
        try:
            ctrl = CtrlBlock.attach(ns)     # 纯打开无创建副作用（layout.py::CtrlBlock.attach）
            created = False
        except Exception:
            h, w = gray.shape[:2]
            ctrl, created = CtrlBlock.create_or_open(
                ns, width=w, height=h, frame_slots=8,
                frame_slot_bytes=protocol_frame_slot_bytes(w, h),
                result_slots=8, result_slot_bytes=protocol_result_slot_bytes(w, h))
        evt = shm_ring.NamedEvent(obj_name(ns, "EVT_FRAME"))
        mutex = shm_ring.NamedMutex(obj_name(ns, "PRODUCER_MUTEX"))
        producer = shm_ring.FrameRingProducer.create_or_attach(ctrl, ns, evt, mutex)
        if not producer.acquire(timeout_ms=5000):
            for obj in (producer, evt, mutex, ctrl):
                try:
                    obj.close()
                except Exception:
                    pass
            self.error.emit("PRODUCER_MUTEX 被占用：桥接进程或另一生产者正在运行")
            return False
        self._ctrl, self._producer, self._evt, self._mutex = ctrl, producer, evt, mutex
        self.session_started.emit(
            f"{ctrl.frame_width}x{ctrl.frame_height} slots={ctrl.frame_slots}"
            f"（{'新建' if created else '沿用已存在 CTRL'}）")
        return True

    def _sleep_with_heartbeat(self) -> None:
        """间隔睡眠切片：每片刷 producer_heartbeat_ms（活着即在线），stop 置位立即醒。"""
        deadline = time.monotonic() + self._interval_ms / 1000.0
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            if self._ctrl is not None:
                self._ctrl.producer_heartbeat_ms = GetTickCount64()
            if self._stop_flag.wait(min(HEARTBEAT_SLICE_S, remaining)):
                return

    def _release(self) -> None:
        """bench:57-62 清理顺序：release → 逆序 close，逐项吞异常（幂等）。"""
        producer, evt, mutex, ctrl = (self._producer, self._evt,
                                      self._mutex, self._ctrl)
        self._ctrl = self._producer = self._evt = self._mutex = None
        if producer is not None:
            try:
                producer.release()
            except Exception:
                pass
            try:
                producer.close()
            except Exception:
                pass
        for obj in (evt, mutex, ctrl):
            if obj is None:
                continue
            try:
                obj.close()
            except Exception:
                pass

    def run(self) -> None:  # noqa: C901 - 状态机主循环
        try:
            files = self._scan()
            if not files:
                self.error.emit("目录下没有可识别的图片文件")
                return
            self.scan_done.emit(len(files))
            frame_id = sent = skipped = 0
            stop = self._stop_flag
            while not stop.is_set():
                for path in files:
                    if stop.is_set():
                        return
                    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                    name = path.name
                    if gray is None:
                        skipped += 1
                        self.frame_done.emit(name, 0, sent, skipped, "解码失败")
                    elif not self._ensure_ring(gray):
                        return  # bootstrap 失败已 emit error
                    elif gray.shape != (self._ctrl.frame_height, self._ctrl.frame_width):
                        skipped += 1
                        self.frame_done.emit(
                            name, 0, sent, skipped,
                            f"尺寸不符 {gray.shape[1]}x{gray.shape[0]}"
                            f"≠{self._ctrl.frame_width}x{self._ctrl.frame_height}")
                    else:
                        frame_id += 1
                        seq = self._producer.write_frame(gray, frame_id)
                        sent += 1
                        self.frame_done.emit(name, seq, sent, skipped, "")
                    # 跳过帧同样切片 pacing：全坏目录不空转打满 CPU
                    self._sleep_with_heartbeat()
                if sent == 0 and self._ctrl is None:
                    self.error.emit("一轮扫描内无任何图片成功解码，无法确定帧几何")
                    return
        except ProtocolMismatchError as exc:
            self.error.emit(f"CTRL 协议/几何不匹配: {exc}")
        except ValueError as exc:
            self.error.emit(f"写帧被拒（协议守卫）: {exc}")
        except Exception as exc:  # noqa: BLE001 - 线程兜底，错误必须上抛 GUI
            self.error.emit(f"发送异常: {exc}")
        finally:
            self._release()


class CameraSimulatorWindow(QMainWindow):
    """选目录 + 间隔 + 启停 + 状态显示；恢复空闲只挂 finished（防互斥体未释放竞态）。"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("相机端模拟器")
        self._thread = None
        self._settings = QSettings("analysis_system", "camera_simulator")
        self._build_ui()
        last = str(self._settings.value("image_dir", "") or "")
        if last:
            self.dir_edit.setText(last)
        self._validate_dir()

    def _build_ui(self) -> None:
        central = QWidget(self)
        layout = QVBoxLayout(central)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("图片文件夹"))
        self.dir_edit = QLineEdit()
        self.dir_edit.setReadOnly(True)
        self.dir_edit.textChanged.connect(self._validate_dir)
        row1.addWidget(self.dir_edit, stretch=1)
        self.browse_btn = QPushButton("浏览...")
        self.browse_btn.clicked.connect(self._browse)
        row1.addWidget(self.browse_btn)
        layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("发送间隔"))
        self.interval_spin = QSpinBox()
        self.interval_spin.setRange(50, 60000)
        self.interval_spin.setValue(1000)
        self.interval_spin.setSuffix(" ms")
        row2.addWidget(self.interval_spin)
        self.count_label = QLabel("图片数: -")
        row2.addWidget(self.count_label)
        row2.addStretch(1)
        layout.addLayout(row2)

        row3 = QHBoxLayout()
        self.start_btn = QPushButton("开始发送")
        self.start_btn.setObjectName("primaryBtn")
        self.start_btn.setEnabled(False)
        self.start_btn.clicked.connect(self._on_start)
        row3.addWidget(self.start_btn)
        self.stop_btn = QPushButton("停止发送")
        self.stop_btn.setObjectName("dangerBtn")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._on_stop)
        row3.addWidget(self.stop_btn)
        layout.addLayout(row3)

        self._state_label = QLabel("待机")
        layout.addWidget(self._state_label)
        self._frame_label = QLabel("")
        layout.addWidget(self._frame_label)
        self._error_label = QLabel("")
        layout.addWidget(self._error_label)
        layout.addStretch(1)

        self.setCentralWidget(central)

    def _browse(self) -> None:
        start = self.dir_edit.text() or self._settings.value("image_dir", "") or ""
        chosen = QFileDialog.getExistingDirectory(self, "选择图片文件夹", str(start))
        if chosen:
            self.dir_edit.setText(chosen)
            self._settings.setValue("image_dir", chosen)

    def _validate_dir(self) -> None:
        try:
            valid = bool(self.dir_edit.text().strip()) and Path(self.dir_edit.text()).is_dir()
        except OSError:
            valid = False
        # 线程引用会话结束后保留（finished 后替换，防析构竞态），
        # 启用与否按 isRunning 判定——按 None 判会让开始钮停止一次后永久禁用
        busy = self._thread is not None and self._thread.isRunning()
        self.start_btn.setEnabled(valid and not busy)

    def _on_start(self) -> None:
        if self._thread is not None and self._thread.isRunning():
            return
        folder = Path(self.dir_edit.text())
        if not folder.is_dir():
            return
        thread = SenderThread(folder, self.interval_spin.value(),
                              load_ipc_config().namespace)
        thread.scan_done.connect(self._on_scan_done)
        thread.session_started.connect(self._on_session_started)
        thread.frame_done.connect(self._on_frame_done)
        thread.error.connect(self._on_error)
        thread.finished.connect(self._on_finished)
        self._thread = thread  # 旧引用在 finished 后才被替换，防析构竞态
        self._set_running(True)
        thread.start()

    def _on_stop(self) -> None:
        if self._thread is None:
            return
        self._thread.stop()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self._state_label.setText("正在停止...")
        self._state_label.setStyleSheet(f"color: {ui_theme.status_color('warn')}")

    def _on_scan_done(self, count: int) -> None:
        self.count_label.setText(f"图片数: {count}")

    def _on_session_started(self, desc: str) -> None:
        self._state_label.setText(f"发送中: {desc}")
        self._state_label.setStyleSheet(f"color: {ui_theme.status_color('ok')}")

    def _on_frame_done(self, name: str, seq: int, sent: int, skipped: int, note: str) -> None:
        text = f"{name} | seq={seq} | 已发送 {sent} 帧 | 跳过 {skipped}"
        if note:
            text += f"（{note}）"
        self._frame_label.setText(text)

    def _on_error(self, msg: str) -> None:
        self._error_label.setText(msg)
        self._error_label.setStyleSheet(f"color: {ui_theme.status_color('alarm')}")
        self._state_label.setText("出错停止")
        self._state_label.setStyleSheet(f"color: {ui_theme.status_color('alarm')}")

    def _on_finished(self) -> None:
        self._set_running(False)
        self._state_label.setText("已停止")
        self._state_label.setStyleSheet(f"color: {ui_theme.status_color('muted')}")

    def _set_running(self, running: bool) -> None:
        self.dir_edit.setEnabled(not running)
        self.browse_btn.setEnabled(not running)
        self.interval_spin.setEnabled(not running)
        if running:
            self._error_label.setText("")
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self._state_label.setText("启动中...")
            self._state_label.setStyleSheet(f"color: {ui_theme.status_color('ok')}")
        else:
            self.stop_btn.setEnabled(False)
            self._validate_dir()  # start 恢复与否取决于目录有效性

    def closeEvent(self, event) -> None:
        thread = self._thread
        if thread is not None and thread.isRunning():
            thread.stop()
            if not thread.wait(5000):
                _KEEPALIVE_THREADS.append(thread)  # 防线程析构崩溃，进程退出由 OS 兜底
        event.accept()


def _run_smoke_check() -> int:
    """打包冒烟：核心导入 + 离屏构造窗口。windowed exe 无 stdout，
    输出双写 exe 旁 _smoke_result.txt（main_window.py 同款约定）。"""
    lines = []

    def say(msg):
        lines.append(msg)
        print(msg, flush=True)

    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        say("[SMOKE] import cv2/PyQt5/utils.ipc ...")
        import numpy as np  # noqa: F401 - 写帧链路依赖，确认随包收集
        from utils.ipc.layout import protocol_frame_slot_bytes
        assert protocol_frame_slot_bytes(31901, 1000) == 37748736
        say(f"[SMOKE] ipc slot bytes OK ({len(IMAGE_EXTS)} 图片扩展名)")
        app = QApplication.instance() or QApplication([])
        ui_theme.apply_theme(app)
        win = CameraSimulatorWindow()
        assert win.start_btn.objectName() == "primaryBtn"
        win.close()
        say("[SMOKE] window construct OK -> SMOKE_PASS")
        code = 0
    except Exception as exc:  # noqa: BLE001 - 冒烟必须报告而非崩溃
        import traceback
        say(f"[SMOKE] FAILED: {exc}")
        lines.extend(traceback.format_exc().splitlines())
        code = 1
    try:
        (Path(sys.executable).parent / "_smoke_result.txt").write_text(
            "\n".join(lines), encoding="utf-8")
    except OSError:
        pass
    return code


def _setup_frozen_environment() -> None:
    """打包(exe)运行适配：配置落 exe 旁（main_window 同款原则）。

    utils/ipc/config.py 以 __file__ 回溯仓库根找 .env，冻结后指向 _internal/；
    这里把 exe 旁 .env 预载进环境变量（dotenv 不覆盖已有 env，无副作用），
    令 SHM_NAMESPACE 等可直接编辑 exe 旁文件生效。开发态无效果。
    """
    if not getattr(sys, "frozen", False):
        return
    from dotenv import load_dotenv
    load_dotenv(Path(sys.executable).resolve().parent / ".env")


def main() -> int:
    _setup_frozen_environment()
    if "--smoke" in sys.argv:
        return _run_smoke_check()
    app = QApplication(sys.argv)
    ui_theme.apply_theme(app)
    win = CameraSimulatorWindow()
    win.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
