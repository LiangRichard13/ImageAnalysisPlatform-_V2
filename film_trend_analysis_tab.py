import json
import sys
import os
import logging
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QSplitter, QPushButton, QListWidget, 
                             QLabel, QTextEdit, QFileDialog, QMessageBox, 
                             QScrollArea, QFrame, QProgressBar, QTabWidget)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QObject
from PyQt5.QtGui import QPixmap, QFont
from utils.trend_analysis_client import TrendAnalysisClient
import tempfile
import shutil

# 设置日志
logger = logging.getLogger(__name__)

class ImageProcessingThread(QThread):
    """图片处理线程"""
    finished = pyqtSignal(str, str)  # 处理完成信号，传递结果图片路径
    error = pyqtSignal(str)     # 错误信号
    progress = pyqtSignal(str)  # 进度信号
    
    def __init__(self, image_paths):
        super().__init__()
        self.image_paths = image_paths
        
    def run(self):
        try:
            # 使用项目根目录下的temp文件夹
            project_root = os.path.dirname(os.path.abspath(__file__))
            temp_base_dir = os.path.join(project_root, "temp")
            
            # 确保temp目录存在
            if not os.path.exists(temp_base_dir):
                os.makedirs(temp_base_dir)
            
            # 创建带时间戳的临时子目录
            import time
            timestamp = int(time.time() * 1000)  # 毫秒时间戳
            temp_dir = os.path.join(temp_base_dir, f"processing_{timestamp}")
            os.makedirs(temp_dir)
            
            logger.info(f"创建临时目录: {temp_dir}")
            
            # 复制图片到临时目录
            for i, image_path in enumerate(self.image_paths):
                filename = os.path.basename(image_path)
                dest_path = os.path.join(temp_dir, filename)
                shutil.copy2(image_path, dest_path)
                self.progress.emit(f"正在准备图片 {i+1}/{len(self.image_paths)}: {filename}")
            
            # 调用本地趋势预测客户端处理图片
            self.progress.emit("正在初始化本地趋势预测引擎...")
            client = TrendAnalysisClient()
            self.progress.emit("正在进行趋势预测...")
            result_path, local_result_json = client.process_images(temp_dir)
            
            # 清理临时目录
            shutil.rmtree(temp_dir)
            logger.info(f"清理临时目录: {temp_dir}")
            
            self.finished.emit(result_path, local_result_json)
            
        except Exception as e:
            error_msg = f"图片处理失败: {str(e)}"
            logger.error(error_msg)
            self.error.emit(error_msg)

class FilmTrendAnalysisWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.image_paths = []  # 存储上传的图片路径
        self.current_result_path = None  # 当前处理结果图片路径
        self.processing_thread = None  # 处理线程

        # 预测所需的图片数量
        self.needed_image_count = 16
        
        self.init_ui()
        self.setup_logging()
        
    def init_ui(self):
        """初始化用户界面"""
        # 主布局 - 垂直分割
        main_layout = QVBoxLayout(self)
        
        # 创建上下分割器
        main_splitter = QSplitter(Qt.Vertical)
        main_layout.addWidget(main_splitter)
        
        # 上半部分 - 图片处理区域
        self.create_image_processing_area(main_splitter)
        
        # 下半部分 - 日志区域
        self.create_log_area(main_splitter)
        
        # 设置分割器比例
        main_splitter.setSizes([700, 200])
        
    def create_image_processing_area(self, parent_splitter):
        """创建图片处理区域"""
        # 创建上半部分容器
        image_frame = QFrame()
        image_frame.setFrameStyle(QFrame.StyledPanel)
        
        # 创建水平布局
        image_layout = QHBoxLayout(image_frame)
        
        # 创建左右分割器
        image_splitter = QSplitter(Qt.Horizontal)
        image_layout.addWidget(image_splitter)
        
        # 左侧 - 图片上传区域
        self.create_upload_area(image_splitter)
        
        # 右侧 - 图片展示区域
        self.create_display_area(image_splitter)
        
        # 设置左右分割比例
        image_splitter.setSizes([400, 800])
        
        parent_splitter.addWidget(image_frame)
        
    def create_upload_area(self, parent_splitter):
        """创建图片上传区域"""
        # 左侧容器
        upload_frame = QFrame()
        upload_frame.setFrameStyle(QFrame.StyledPanel)
        upload_layout = QVBoxLayout(upload_frame)
        
        # 标题
        title_label = QLabel("图片上传区域")
        title_label.setFont(QFont("Arial", 12, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        upload_layout.addWidget(title_label)
        
        # 上传按钮
        self.upload_btn = QPushButton("选择图片")
        self.upload_btn.clicked.connect(self.select_images)
        self.upload_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)
        upload_layout.addWidget(self.upload_btn)
        
        # 图片列表
        self.image_list = QListWidget()
        self.image_list.setMaximumHeight(300)
        self.image_list.setMinimumHeight(200)
        self.image_list.setStyleSheet("""
            QListWidget {
                border: 1px solid #ccc;
                border-radius: 4px;
                background-color: white;
                font-size: 10px;
            }
            QListWidget::item {
                padding: 5px;
                border-bottom: 1px solid #eee;
            }
            QListWidget::item:selected {
                background-color: #e3f2fd;
            }
        """)
        upload_layout.addWidget(self.image_list)
        
        # 清空按钮
        self.clear_btn = QPushButton("清空列表")
        self.clear_btn.clicked.connect(self.clear_image_list)
        self.clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
            QPushButton:pressed {
                background-color: #c1170a;
            }
        """)
        upload_layout.addWidget(self.clear_btn)
        
        # 处理按钮
        self.process_btn = QPushButton("开始处理")
        self.process_btn.clicked.connect(self.start_processing)
        self.process_btn.setEnabled(False)
        self.process_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover:enabled {
                background-color: #1976D2;
            }
            QPushButton:pressed:enabled {
                background-color: #1565C0;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        upload_layout.addWidget(self.process_btn)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        upload_layout.addWidget(self.progress_bar)
        
        # 添加弹性空间
        upload_layout.addStretch()
        
        parent_splitter.addWidget(upload_frame)
        
    def create_display_area(self, parent_splitter):
        """创建图片展示区域"""
        # 右侧容器
        display_frame = QFrame()
        display_frame.setFrameStyle(QFrame.StyledPanel)
        display_layout = QVBoxLayout(display_frame)
        
        # 标题和刷新按钮
        header_layout = QHBoxLayout()
        title_label = QLabel("处理结果展示")
        title_label.setFont(QFont("Arial", 12, QFont.Bold))
        header_layout.addWidget(title_label)
        
        self.refresh_btn = QPushButton("刷新页面")
        self.refresh_btn.clicked.connect(self.refresh_page)
        self.refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 10px;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
            QPushButton:pressed {
                background-color: #EF6C00;
            }
        """)
        header_layout.addWidget(self.refresh_btn)
        
        display_layout.addLayout(header_layout)
        
        # 创建标签页显示不同类型的结果
        self.result_tabs = QTabWidget()
        self.result_tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #ccc;
                border-radius: 4px;
                background-color: white;
            }
            QTabBar::tab {
                background-color: #f0f0f0;
                color: #333;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background-color: white;
                color: #2196F3;
                border-bottom: 2px solid #2196F3;
            }
            QTabBar::tab:hover {
                background-color: #e0e0e0;
            }
        """)
        
        # 预测结果标签页
        self.prediction_tab = self.create_image_tab("预测结果")
        self.result_tabs.addTab(self.prediction_tab, "预测结果")
        
        # JSON结果标签页
        self.json_tab = self.create_json_tab()
        self.result_tabs.addTab(self.json_tab, "JSON结果")
        
        display_layout.addWidget(self.result_tabs)
        
        parent_splitter.addWidget(display_frame)
        
    def create_image_tab(self, tab_name):
        """创建图片显示标签页"""
        tab_widget = QWidget()
        layout = QVBoxLayout(tab_widget)
        
        # 图片显示区域
        image_display = QLabel()
        image_display.setAlignment(Qt.AlignCenter)
        image_display.setStyleSheet("""
            QLabel {
                border: 1px solid #ccc;
                border-radius: 4px;
                background-color: white;
                color: #666;
                font-size: 14px;
            }
        """)
        image_display.setMinimumSize(400, 300)
        image_display.setText(f"暂无{tab_name}")
        
        # 使用滚动区域
        scroll_area = QScrollArea()
        scroll_area.setWidget(image_display)
        scroll_area.setWidgetResizable(True)
        layout.addWidget(scroll_area)
        
        return tab_widget
        
    def create_json_tab(self):
        """创建JSON结果显示标签页"""
        tab_widget = QWidget()
        layout = QVBoxLayout(tab_widget)
        
        # JSON文本显示区域
        json_display = QTextEdit()
        json_display.setReadOnly(True)
        json_display.setStyleSheet("""
            QTextEdit {
                border: 1px solid #ccc;
                border-radius: 4px;
                background-color: white;
                font-family: 'Courier New', monospace;
                font-size: 11px;
                color: #333;
            }
        """)
        json_display.setMinimumSize(400, 300)
        json_display.setText("暂无JSON结果")
        layout.addWidget(json_display)
        
        return tab_widget
        
    def create_log_area(self, parent_splitter):
        """创建日志区域"""
        # 日志容器
        log_frame = QFrame()
        log_frame.setFrameStyle(QFrame.StyledPanel)
        log_layout = QVBoxLayout(log_frame)
        
        # 标题
        log_title = QLabel("系统日志")
        log_title.setFont(QFont("Arial", 12, QFont.Bold))
        log_layout.addWidget(log_title)
        
        # 日志文本框
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(200)
        self.log_text.setMinimumHeight(150)
        log_layout.addWidget(self.log_text)
        
        parent_splitter.addWidget(log_frame)
        
    def setup_logging(self):
        """设置日志系统"""
        # 创建自定义日志处理器
        self.log_handler = LogHandler()
        self.log_handler.log_signal.connect(self.append_log)
        
        # 获取根日志器并添加处理器
        root_logger = logging.getLogger()
        root_logger.addHandler(self.log_handler)
        root_logger.setLevel(logging.INFO)
        
        # 记录启动信息
        logger.info("系统启动")
        
    def select_images(self):
        """选择图片文件"""
        file_dialog = QFileDialog()
        file_paths, _ = file_dialog.getOpenFileNames(
            self,
            "选择图片文件",
            "",
            "图片文件 (*.jpg *.jpeg *.png *.bmp *.gif *.tiff)"
        )
        
        if file_paths:
            # 验证图片格式
            valid_paths = []
            for path in file_paths:
                if self.validate_image_format(path):
                    valid_paths.append(path)
                else:
                    QMessageBox.warning(self, "格式错误", f"文件 {os.path.basename(path)} 不是有效的图片格式")
            
            # 添加到列表
            for path in valid_paths:
                if path not in self.image_paths:
                    self.image_paths.append(path)
                    self.image_list.addItem(os.path.basename(path))
            
            # 更新处理按钮状态
            self.update_process_button_state()
            
            logger.info(f"选择了 {len(valid_paths)} 张图片")
            
    def validate_image_format(self, file_path):
        """验证图片格式"""
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']
        _, ext = os.path.splitext(file_path.lower())
        return ext in valid_extensions
        
    def clear_image_list(self):
        """清空图片列表"""
        self.image_paths.clear()
        self.image_list.clear()
        self.update_process_button_state()
        logger.info("已清空图片列表")
        
    def update_process_button_state(self):
        """更新处理按钮状态"""
        if len(self.image_paths) >= self.needed_image_count:
            self.process_btn.setEnabled(True)
            self.process_btn.setText(f"开始处理 ({len(self.image_paths)} 张图片)")
        else:
            self.process_btn.setEnabled(False)
            self.process_btn.setText(f"需要至少{self.needed_image_count}张图片 (当前: {len(self.image_paths)} 张)")
            
    def start_processing(self):
        """开始处理图片"""
        if len(self.image_paths) < self.needed_image_count:
            QMessageBox.warning(self, f"图片数量不足", "至少需要{self.needed_image_count}张图片才能开始处理")
            return
            
        # 禁用处理按钮
        self.process_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # 不确定进度
        
        logger.info(f"开始处理 {len(self.image_paths)} 张图片")
        
        # 创建并启动处理线程
        self.processing_thread = ImageProcessingThread(self.image_paths)
        self.processing_thread.finished.connect(self.on_processing_finished)
        self.processing_thread.error.connect(self.on_processing_error)
        self.processing_thread.progress.connect(self.on_processing_progress)
        self.processing_thread.start()
        
    def on_processing_finished(self, result_path, json_path):
        """处理完成回调"""
        self.progress_bar.setVisible(False)
        self.process_btn.setEnabled(True)

        # 保存结果路径
        self.current_results = {
            'prediction': result_path,
            'json': json_path
        }
        
        # 显示结果图片
        self.display_result_image(result_path)

        # 显示JSON结果
        self.display_json_result()
        
        logger.info(f"图片处理完成，结果保存至: {result_path}和{json_path}")

    def on_processing_error(self, error_msg):
        """处理错误回调"""
        self.progress_bar.setVisible(False)
        self.process_btn.setEnabled(True)
        
        logger.error(error_msg)
        QMessageBox.critical(self, "处理失败", error_msg)
        
    def on_processing_progress(self, progress_msg):
        """处理进度回调"""
        logger.info(progress_msg)
        
    def display_result_image(self, image_path):
        """显示结果图片到标签页"""
        try:
            if os.path.exists(image_path):
                # 获取预测结果标签页中的图片显示组件
                scroll_area = self.prediction_tab.findChild(QScrollArea)
                if scroll_area:
                    image_display = scroll_area.widget()
                    if isinstance(image_display, QLabel):
                        pixmap = QPixmap(image_path)
                        if not pixmap.isNull():
                            # 缩放图片以适应显示区域
                            scaled_pixmap = pixmap.scaled(
                                image_display.size(), 
                                Qt.KeepAspectRatio, 
                                Qt.SmoothTransformation
                            )
                            image_display.setPixmap(scaled_pixmap)
                            self.current_result_path = image_path
                        else:
                            image_display.setText("无法加载图片")
                    else:
                        image_display.setText("无法加载图片")
            else:
                # 获取预测结果标签页中的图片显示组件
                scroll_area = self.prediction_tab.findChild(QScrollArea)
                if scroll_area:
                    image_display = scroll_area.widget()
                    if isinstance(image_display, QLabel):
                        image_display.setText("图片文件不存在")
        except Exception as e:
            logger.error(f"显示图片失败: {str(e)}")
            scroll_area = self.prediction_tab.findChild(QScrollArea)
            if scroll_area:
                image_display = scroll_area.widget()
                if isinstance(image_display, QLabel):
                    image_display.setText("显示图片时出错")

    def display_json_result(self):
        """显示JSON结果到标签页"""
        try:
            if not self.current_results:
                return
            if self.current_results and os.path.exists(self.current_results['json']):
                with open(self.current_results['json'], 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                
                # 检查异常级别并弹出警告
                self.check_pred_level(json_data)
                
                # 格式化JSON数据
                formatted_json = json.dumps(json_data, ensure_ascii=False, indent=2)
                
                # 获取JSON标签页中的文本显示组件
                json_display = self.json_tab.findChild(QTextEdit)
                if json_display:
                    json_display.setText(formatted_json)
            else:
                json_display = self.json_tab.findChild(QTextEdit)
                if json_display:
                    json_display.setText("暂无JSON结果")
        except Exception as e:
            logger.error(f"显示JSON结果失败: {str(e)}")
            json_display = self.json_tab.findChild(QTextEdit)
            if json_display:
                json_display.setText(f"JSON结果显示错误: {str(e)}")
            
    def check_pred_level(self, json_data):
        """检查异常级别并弹出警告窗口"""
        try:
            pred_level = json_data.get('pred_level', '')
            
            # 检查是否为需要弹出警告的异常级别
            if pred_level in ['中等预测异常可能性', '很可能预测异常']:
                self.show_pred_warning(pred_level, json_data)
                logger.info(f"检测到异常级别: {pred_level}，已弹出警告窗口")
                
        except Exception as e:
            logger.error(f"检查异常级别失败: {str(e)}")
    
    def show_pred_warning(self, pred_level, json_data):
        """显示异常警告弹窗"""
        try:
            # 获取模拟电压数值
            analog_voltage = json_data.get('analog_voltage', '未知')
            
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("异常预测警告")
            msg_box.setText(f"模拟电压: {analog_voltage}")
            msg_box.setInformativeText(f"检测到异常级别: {pred_level}")
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setStandardButtons(QMessageBox.Ok)
            
            # 设置弹窗尺寸，使其更宽
            msg_box.resize(400, 150)
            
            # 设置弹窗样式
            msg_box.setStyleSheet("""
                QMessageBox {
                    background-color: white;
                    font-size: 12px;
                    min-width: 400px;
                }
                QMessageBox QLabel {
                    color: #333;
                    font-weight: bold;
                    font-size: 14px;
                }
                QMessageBox QPushButton {
                    background-color: #2196F3;
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 4px;
                    font-weight: bold;
                    min-width: 100px;
                    font-size: 12px;
                }
                QMessageBox QPushButton:hover {
                    background-color: #1976D2;
                }
            """)
            
            msg_box.exec_()
            
        except Exception as e:
            logger.error(f"显示异常警告弹窗失败: {str(e)}")

    def refresh_page(self):
        """刷新页面"""
        # 清空图片列表
        self.clear_image_list()
        
        # 清空结果图片
        self.clear_result_displays()
        self.current_result_path = None
        
        logger.info("页面已刷新")
        
    def clear_result_displays(self):
        """清空结果显示"""
        # 清空预测结果
        self.clear_image_display(self.prediction_tab, "预测结果")
        
        # 清空JSON结果
        json_display = self.json_tab.findChild(QTextEdit)
        if json_display:
            json_display.setText("暂无JSON结果")
            
    def clear_image_display(self, tab_widget, result_type):
        """清空图片显示"""
        scroll_area = tab_widget.findChild(QScrollArea)
        if scroll_area:
            image_display = scroll_area.widget()
            if isinstance(image_display, QLabel):
                image_display.clear()
                image_display.setText(f"暂无{result_type}")
        
    def append_log(self, message):
        """添加日志到界面"""
        self.log_text.append(message)
        # 自动滚动到底部
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def closeEvent(self, event):
        """窗口关闭事件"""
        # 关闭日志处理器
        if hasattr(self, 'log_handler'):
            # 先从root logger中移除handler，避免atexit时的错误
            root_logger = logging.getLogger()
            root_logger.removeHandler(self.log_handler)
            self.log_handler.close()
        event.accept()

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

def main():
    app = QApplication(sys.argv)
    window = QMainWindow()
    window.setWindowTitle("镀膜褶皱趋势预测系统")
    window.setGeometry(100, 100, 1200, 800)
    
    # 创建中央部件
    central_widget = QWidget()
    window.setCentralWidget(central_widget)
    
    # 创建布局
    layout = QVBoxLayout(central_widget)
    layout.setContentsMargins(0, 0, 0, 0)
    
    # 添加镀膜褶皱趋势预测组件
    film_trend_widget = FilmTrendAnalysisWidget()
    layout.addWidget(film_trend_widget)
    
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
