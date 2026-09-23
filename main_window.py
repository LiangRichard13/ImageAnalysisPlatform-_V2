import sys
import os
import logging
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QStackedWidget, QLabel,
                             QFrame, QMenuBar, QAction, QMessageBox)
from PyQt5.QtCore import Qt, QObject, pyqtSignal
from PyQt5.QtGui import QFont, QIcon

# 导入两个界面模块
from film_trend_analysis_tab import FilmTrendAnalysisWidget
from anomaly_detection_tab import AnomalyDetectionWidget

# 设置日志
logger = logging.getLogger(__name__)

class LogHandler(logging.Handler, QObject):
    """自定义日志处理器，用于将日志显示在界面上"""
    log_signal = pyqtSignal(str)
    
    def __init__(self):
        logging.Handler.__init__(self)
        QObject.__init__(self)
        self._closed = False
    
    def emit(self, record):
        if not self._closed:
            try:
                log_message = self.format(record)
                self.log_signal.emit(log_message)
            except RuntimeError:
                # Qt对象已被删除，忽略错误
                pass
    
    def close(self):
        """关闭处理器"""
        self._closed = True
        super().close()

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
        # 导航栏容器
        nav_frame = QFrame()
        nav_frame.setFrameStyle(QFrame.StyledPanel)
        nav_frame.setStyleSheet("""
            QFrame {
                background-color: #f0f0f0;
                border-bottom: 1px solid #ccc;
            }
        """)
        nav_frame.setMaximumHeight(60)
        
        nav_layout = QHBoxLayout(nav_frame)
        nav_layout.setContentsMargins(20, 10, 20, 10)
        
        # 系统标题
        title_label = QLabel("镀膜状态数字孪生在线分析系统")
        title_label.setFont(QFont("Arial", 16, QFont.Bold))
        title_label.setStyleSheet("color: #2c3e50;")
        nav_layout.addWidget(title_label)
        
        # 添加弹性空间
        nav_layout.addStretch()
        
        # 导航按钮
        self.film_trend_btn = QPushButton("镀膜褶皱趋势预测")
        self.film_trend_btn.setFont(QFont("Arial", 10))
        self.film_trend_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #21618c;
            }
        """)
        self.film_trend_btn.clicked.connect(lambda: self.switch_to_tab(0))
        nav_layout.addWidget(self.film_trend_btn)
        
        self.anomaly_detection_btn = QPushButton("异常检测")
        self.anomaly_detection_btn.setFont(QFont("Arial", 10))
        self.anomaly_detection_btn.setStyleSheet("""
            QPushButton {
                background-color: #95a5a6;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #7f8c8d;
            }
            QPushButton:pressed {
                background-color: #6c7b7d;
            }
        """)
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
        """更新导航按钮样式"""
        if self.current_tab == 0:
            # 镀膜褶皱趋势预测激活
            self.film_trend_btn.setStyleSheet("""
                QPushButton {
                    background-color: #3498db;
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 4px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #2980b9;
                }
                QPushButton:pressed {
                    background-color: #21618c;
                }
            """)
            self.anomaly_detection_btn.setStyleSheet("""
                QPushButton {
                    background-color: #95a5a6;
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 4px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #7f8c8d;
                }
                QPushButton:pressed {
                    background-color: #6c7b7d;
                }
            """)
        else:
            # 异常检测激活
            self.anomaly_detection_btn.setStyleSheet("""
                QPushButton {
                    background-color: #e74c3c;
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 4px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #c0392b;
                }
                QPushButton:pressed {
                    background-color: #a93226;
                }
            """)
            self.film_trend_btn.setStyleSheet("""
                QPushButton {
                    background-color: #95a5a6;
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 4px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #7f8c8d;
                }
                QPushButton:pressed {
                    background-color: #6c7b7d;
                }
            """)
            
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
        # 创建自定义日志处理器
        self.log_handler = LogHandler()
        
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

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
