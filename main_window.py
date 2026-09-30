import sys
import os
import logging
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QStackedWidget, QLabel,
                             QFrame, QMenuBar, QAction, QMessageBox)
from PyQt5.QtCore import Qt

# 导入两个界面模块
from film_trend_analysis_tab import FilmTrendAnalysisWidget
from anomaly_detection_tab import AnomalyDetectionWidget
from utils.log_format import IapLogHandler
import ui_theme

# 设置日志
logger = logging.getLogger(__name__)

class FilmTrendAnalysisTab(QWidget):
    """镀膜褶皱趋势预测标签页"""
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        """初始化界面"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 创建镀膜褶皱趋势预测界面
        self.film_trend_widget = FilmTrendAnalysisWidget()
        # 移除原有的窗口装饰，只保留内容
        self.film_trend_widget.setParent(self)
        layout.addWidget(self.film_trend_widget)

class AnomalyDetectionTab(QWidget):
    """异常检测标签页"""
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        """初始化界面"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 创建异常检测界面
        self.anomaly_detection_widget = AnomalyDetectionWidget()
        # 移除原有的窗口装饰，只保留内容
        self.anomaly_detection_widget.setParent(self)
        layout.addWidget(self.anomaly_detection_widget)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_tab = 0  # 当前标签页索引
        self.init_ui()
        self.setup_logging()
        
    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle("分析系统 - 主控制台")
        self.setGeometry(100, 100, 1400, 900)
        
        # 创建菜单栏
        self.create_menu_bar()
        
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # 创建导航栏
        self.create_navigation_bar(main_layout)
        
        # 创建标签页堆栈
        self.create_tab_stack(main_layout)
        
        # 设置初始状态
        self.switch_to_tab(0)
        
    def create_menu_bar(self):
        """创建菜单栏"""
        menubar = self.menuBar()
        
        # 文件菜单
        file_menu = menubar.addMenu('文件(&F)')
        
        # 退出动作
        exit_action = QAction('退出(&Q)', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # 工具菜单
        tools_menu = menubar.addMenu('工具(&T)')
        
        # 切换到镀膜褶皱趋势预测
        film_trend_action = QAction('镀膜褶皱趋势预测(&F)', self)
        film_trend_action.setShortcut('Ctrl+1')
        film_trend_action.triggered.connect(lambda: self.switch_to_tab(0))
        tools_menu.addAction(film_trend_action)
        
        # 切换到异常检测
        anomaly_detection_action = QAction('异常检测(&A)', self)
        anomaly_detection_action.setShortcut('Ctrl+2')
        anomaly_detection_action.triggered.connect(lambda: self.switch_to_tab(1))
        tools_menu.addAction(anomaly_detection_action)
        
        # 帮助菜单
        help_menu = menubar.addMenu('帮助(&H)')
        
        # 关于动作
        about_action = QAction('关于(&A)', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
    def create_navigation_bar(self, parent_layout):
        """创建导航栏"""
        # 导航栏容器（样式走全局 QSS：QFrame#navBar）
        nav_frame = QFrame()
        nav_frame.setObjectName("navBar")
        nav_frame.setFrameStyle(QFrame.NoFrame)
        nav_frame.setMaximumHeight(60)

        nav_layout = QHBoxLayout(nav_frame)
        nav_layout.setContentsMargins(20, 10, 20, 10)

        # 系统标题（字号/字重由全局 QSS QLabel#navTitle 规定）
        title_label = QLabel("镀膜状态数字孪生在线分析系统")
        title_label.setObjectName("navTitle")
        nav_layout.addWidget(title_label)

        # 添加弹性空间
        nav_layout.addStretch()

        # 导航按钮（checkable，激活态由全局 QSS :checked 渲染）
        self.film_trend_btn = QPushButton("镀膜褶皱趋势预测")
        self.film_trend_btn.setObjectName("navBtn")
        self.film_trend_btn.setCheckable(True)
        self.film_trend_btn.setChecked(True)  # 初始页（switch_to_tab(0) 会因同页短路）
        self.film_trend_btn.clicked.connect(lambda: self.switch_to_tab(0))
        nav_layout.addWidget(self.film_trend_btn)

        self.anomaly_detection_btn = QPushButton("异常检测")
        self.anomaly_detection_btn.setObjectName("navBtn")
        self.anomaly_detection_btn.setCheckable(True)
        self.anomaly_detection_btn.clicked.connect(lambda: self.switch_to_tab(1))
        nav_layout.addWidget(self.anomaly_detection_btn)

        parent_layout.addWidget(nav_frame)
        
    def create_tab_stack(self, parent_layout):
        """创建标签页堆栈"""
        self.tab_stack = QStackedWidget()
        parent_layout.addWidget(self.tab_stack)
        
        # 创建标签页
        self.film_trend_tab = FilmTrendAnalysisTab()
        self.anomaly_detection_tab = AnomalyDetectionTab()
        
        # 添加到堆栈
        self.tab_stack.addWidget(self.film_trend_tab)
        self.tab_stack.addWidget(self.anomaly_detection_tab)
        
    def switch_to_tab(self, tab_index):
        """切换到指定标签页"""
        if tab_index == self.current_tab:
            return
            
        self.current_tab = tab_index
        self.tab_stack.setCurrentIndex(tab_index)
        
        # 更新导航按钮样式
        self.update_navigation_buttons()
        
        # 记录切换日志
        tab_names = ["镀膜褶皱趋势预测", "异常检测"]
        logger.info(f"切换到 {tab_names[tab_index]} 界面")
        
    def update_navigation_buttons(self):
        """更新导航按钮选中态（视觉由全局 QSS #navBtn:checked 渲染）"""
        self.film_trend_btn.setChecked(self.current_tab == 0)
        self.anomaly_detection_btn.setChecked(self.current_tab == 1)
            
    def show_about(self):
        """显示关于对话框"""
        QMessageBox.about(self, "关于", 
                         "分析系统 v1.0\n\n"
                         "功能模块：\n"
                         "• 镀膜褶皱趋势预测\n"
                         "• 异常检测\n\n"
                         )
        
    def setup_logging(self):
        """设置日志系统"""
        # 统一界面日志处理器（utils/log_format.py：时间戳+emoji 事件标记）
        self.log_handler = IapLogHandler()
        
        # 获取根日志器并添加处理器
        root_logger = logging.getLogger()
        root_logger.addHandler(self.log_handler)
        root_logger.setLevel(logging.INFO)
        
        # 记录启动信息
        logger.info("分析系统主窗口启动")
        
    def closeEvent(self, event):
        """窗口关闭事件"""
        # 关闭日志处理器
        if hasattr(self, 'log_handler'):
            # 先从root logger中移除handler，避免atexit时的错误
            root_logger = logging.getLogger()
            root_logger.removeHandler(self.log_handler)
            self.log_handler.close()
        event.accept()

def _setup_frozen_environment():
    """打包(exe)运行适配：统一 CWD 到 exe 目录、指向外置权重。

    开发态（python main_window.py）无效果；exe 运行时 temp/checkpoint、
    download/、.env、引擎 work_dirs 全部落 exe 旁，权重走可替换的
    weights/ 目录（引擎已支持环境变量覆盖，零引擎改动）。
    """
    if not getattr(sys, "frozen", False):
        return
    exe_dir = Path(sys.executable).resolve().parent
    os.chdir(exe_dir)
    weights = exe_dir / "weights"
    if weights.is_dir():
        anomaly_weight = weights / "coating_anomaly_final.pth"
        if anomaly_weight.is_file():
            os.environ.setdefault("ANOMALY_MODEL_PATH", str(anomaly_weight))
        ckpts = sorted(weights.glob("*.ckpt"))
        if ckpts:
            os.environ.setdefault("TREND_MODEL_PATH", str(ckpts[0]))


def _run_smoke_check() -> int:
    """打包冒烟：核心导入 → dinomaly 权重加载 → 规则引擎推理。

    windowed exe 无 stdout，输出双写 exe 旁 _smoke_result.txt。
    torch 导入失败时输出进程诊断（PATH/已加载模块/逐 DLL 加载），
    用于定位打包环境特有的 DLL 初始化问题。
    """
    lines = []

    def say(msg):
        lines.append(msg)
        print(msg, flush=True)

    def _report_loaded_modules():
        import ctypes.wintypes as wt
        psapi = ctypes.WinDLL("psapi.dll")
        hmods = (wt.HMODULE * 1024)()
        needed = wt.DWORD()
        proc = ctypes.WinDLL("kernel32.dll").GetCurrentProcess()
        if psapi.EnumProcessModules(proc, hmods, ctypes.sizeof(hmods), ctypes.byref(needed)):
            count = min(needed.value // ctypes.sizeof(wt.HMODULE), 1024)
            buf = ctypes.create_unicode_buffer(512)
            for i in range(count):
                if psapi.GetModuleFileNameExW(proc, hmods[i], buf, 512):
                    say(f"  loaded: {buf.value}")

    try:
        say("[SMOKE] import torch/cv2 ...")
        import torch
        import cv2  # noqa: F401
        say(f"[SMOKE] torch {torch.__version__} cuda={torch.cuda.is_available()}")
        from utils.dinomaly_engine import get_dinomaly_engine
        engine = get_dinomaly_engine()
        say(f"[SMOKE] dinomaly loaded from {engine.model_path}")
        from utils.rule_based_wrinkle import RuleBasedWrinklePipeline
        import numpy as np
        _, _, metrics = RuleBasedWrinklePipeline().process_array(
            np.zeros((64, 320), dtype=np.uint8))
        assert metrics.get("anomaly_level"), metrics
        say("[SMOKE] rule engine OK -> SMOKE_PASS")
        code = 0
    except Exception as exc:
        import ctypes
        import glob
        import os
        import traceback
        say(f"[SMOKE] FAILED: {exc}")
        lines.extend(traceback.format_exc().splitlines())
        # ---- 进程诊断：定位打包环境特有的 DLL 初始化问题 ----
        say("[DIAG] PATH=" + os.environ.get("PATH", ""))
        say("[DIAG] loaded modules:")
        try:
            _report_loaded_modules()
        except Exception as e:
            say(f"  (enum failed: {e})")
        lib = os.path.join(getattr(sys, "_MEIPASS", ""), "torch", "lib")
        for p in sorted(glob.glob(os.path.join(lib, "*.dll"))):
            try:
                ctypes.WinDLL(p, winmode=0x1100)
            except OSError as e:
                say(f"[DIAG] DLLFAIL {os.path.basename(p)}: {e}")
        code = 1
    try:
        (Path(sys.executable).parent / "_smoke_result.txt").write_text(
            "\n".join(lines), encoding="utf-8")
    except OSError:
        pass
    return code


def main():
    _setup_frozen_environment()
    if "--smoke" in sys.argv:
        sys.exit(_run_smoke_check())
    app = QApplication(sys.argv)
    ui_theme.apply_theme(app)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
