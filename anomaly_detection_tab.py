import sys
import os
import logging
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QSplitter, QPushButton, QLabel,
                             QTextEdit, QFileDialog, QMessageBox,
                             QScrollArea, QFrame, QProgressBar, QTabWidget,
                             QDialog, QListWidget, QListWidgetItem, QDialogButtonBox,
                             QLineEdit, QComboBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QObject, QSettings
from PyQt5.QtGui import QPixmap, QFont
from utils.anomaly_detection_client import AnomalyDetectionClient, ENGINE_RULE_BASED, ENGINE_DINOMALY
from dotenv import load_dotenv
import tempfile
import shutil
import json
import time
import datetime
from pathlib import Path

# 设置日志
logger = logging.getLogger(__name__)

class ImageProcessingThread(QThread):
    """图片处理线程"""
    finished = pyqtSignal(str, str, str)  # 处理完成信号，传递三个结果文件路径
    error = pyqtSignal(str)               # 错误信号
    progress = pyqtSignal(str)            # 进度信号

    def __init__(self, image_path, engine_name=ENGINE_RULE_BASED):
        super().__init__()
        self.image_path = image_path
        self.engine_name = engine_name

    def run(self):
        try:
            self.progress.emit("正在本地处理图片...")
            client = AnomalyDetectionClient()

            # 调用本地客户端处理图片（按界面选择的引擎）
            local_result_pre_image, local_result_heat_map, local_result_json = client.process_images(
                self.image_path, self.engine_name)

            self.finished.emit(local_result_pre_image, local_result_heat_map, local_result_json)

        except Exception as e:
            error_msg = f"图片处理失败: {str(e)}"
            logger.error(error_msg)
            self.error.emit(error_msg)

class CheckpointSelectionDialog(QDialog):
    """检查点选择对话框"""
    def __init__(self, checkpoint_files, checkpoint_dir, parent=None):
        super().__init__(parent)
        self.checkpoint_files = checkpoint_files
        self.checkpoint_dir = checkpoint_dir
        self.selected_checkpoint = None
        self.init_ui()
        
    def init_ui(self):
        """初始化对话框界面"""
        self.setWindowTitle("选择检查点文件")
        self.setModal(True)
        self.resize(600, 400)
        
        layout = QVBoxLayout(self)
        
        # 标题和说明
        title_label = QLabel("请选择要使用的检查点文件：")
        title_label.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(title_label)
        
        info_label = QLabel("检查点文件记录了之前批处理已处理的图片，选择后将继续处理未处理的图片。")
        info_label.setStyleSheet("color: gray; font-size: 10px;")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        # 检查点文件列表
        self.checkpoint_list = QListWidget()
        self.checkpoint_list.setSelectionMode(QListWidget.SingleSelection)
        self.checkpoint_list.itemDoubleClicked.connect(self.on_item_double_clicked)
        layout.addWidget(self.checkpoint_list)
        
        # 加载检查点文件信息
        self.load_checkpoint_info()
        
        # 按钮区域
        button_layout = QHBoxLayout()
        
        self.use_selected_btn = QPushButton("使用选中检查点")
        self.use_selected_btn.clicked.connect(self.use_selected_checkpoint)
        self.use_selected_btn.setEnabled(False)
        button_layout.addWidget(self.use_selected_btn)
        
        self.create_new_btn = QPushButton("创建新检查点")
        self.create_new_btn.clicked.connect(self.create_new_checkpoint)
        button_layout.addWidget(self.create_new_btn)
        
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)
        
        layout.addLayout(button_layout)
        
        # 连接选择变化信号
        self.checkpoint_list.itemSelectionChanged.connect(self.on_selection_changed)
        
    def load_checkpoint_info(self):
        """加载检查点文件信息"""
        for checkpoint_file in self.checkpoint_files:
            file_path = os.path.join(self.checkpoint_dir, checkpoint_file)
            try:
                # 解析文件名中的时间戳
                timestamp_str = checkpoint_file.replace('.json', '')
                try:
                    create_time = datetime.datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                    create_time_str = create_time.strftime("%Y-%m-%d %H:%M:%S")
                except:
                    create_time_str = "未知时间"
                
                # 读取检查点文件信息
                processed_count = 0
                last_update = "未知"
                if os.path.exists(file_path):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        checkpoint_data = json.load(f)
                        processed_count = len(checkpoint_data.get('processed_images', []))
                        last_update_str = checkpoint_data.get('last_update', '')
                        if last_update_str:
                            try:
                                last_update_dt = datetime.datetime.fromisoformat(last_update_str)
                                last_update = last_update_dt.strftime("%Y-%m-%d %H:%M:%S")
                            except:
                                last_update = "未知"
                
                # 创建列表项
                item_text = f"{checkpoint_file}\n创建时间: {create_time_str} | 已处理: {processed_count} 张图片 | 最后更新: {last_update}"
                item = QListWidgetItem(item_text)
                item.setData(Qt.UserRole, checkpoint_file)
                self.checkpoint_list.addItem(item)
                
            except Exception as e:
                logger.error(f"加载检查点文件信息失败 {checkpoint_file}: {str(e)}")
                item = QListWidgetItem(f"{checkpoint_file} (加载失败)")
                item.setData(Qt.UserRole, checkpoint_file)
                self.checkpoint_list.addItem(item)
        
        # 默认选中第一个项目（最新的检查点）
        if self.checkpoint_list.count() > 0:
            self.checkpoint_list.setCurrentRow(0)
    
    def on_selection_changed(self):
        """选择变化回调"""
        selected_items = self.checkpoint_list.selectedItems()
        self.use_selected_btn.setEnabled(len(selected_items) > 0)
    
    def on_item_double_clicked(self, item):
        """双击项目回调"""
        self.use_selected_checkpoint()
    
    def use_selected_checkpoint(self):
        """使用选中的检查点"""
        selected_items = self.checkpoint_list.selectedItems()
        if selected_items:
            selected_file = selected_items[0].data(Qt.UserRole)
            self.selected_checkpoint = os.path.join(self.checkpoint_dir, selected_file)
            self.accept()
    
    def create_new_checkpoint(self):
        """创建新检查点"""
        self.selected_checkpoint = None  # None表示创建新检查点
        self.accept()

class BatchProcessingThread(QThread):
    """批处理线程"""
    progress = pyqtSignal(str)            # 进度信号
    error = pyqtSignal(str)               # 错误信号（静默记录）
    batch_finished = pyqtSignal()         # 批处理完成信号
    image_processed = pyqtSignal(str, str, str)  # 单张图片处理完成信号
    batch_progress = pyqtSignal(int, int) # 批处理进度信号（当前进度，总数量）
    
    def __init__(self, processing_dir, checkpoint_file=None, engine_name=ENGINE_RULE_BASED):
        super().__init__()
        self.processing_dir = processing_dir
        self.checkpoint_file = checkpoint_file
        self.engine_name = engine_name
        self.processed_images = set()
        self.is_running = True
        self.polling_interval = 5  # 轮询间隔5秒
        self.current_batch_images = []  # 当前批次的图片列表
        self.current_batch_processed = 0  # 当前批次已处理的图片数量
        
    def run(self):
        try:
            logger.info(f"批处理启动，监控文件夹: {self.processing_dir}")
            
            # 加载检查点
            if self.checkpoint_file and os.path.exists(self.checkpoint_file):
                self.load_checkpoint()
            
            while self.is_running:
                # 扫描图片
                image_files = self.scan_images()
                
                if image_files:
                    # 过滤掉已处理的图片
                    new_images = [img for img in image_files if img not in self.processed_images]
                    
                    if new_images:
                        logger.info(f"扫描到 {len(new_images)} 张新图片待处理")
                        # 设置当前批次
                        self.current_batch_images = new_images
                        self.current_batch_processed = 0
                        
                        # 发送批次开始信号
                        self.batch_progress.emit(0, len(new_images))
                        
                        # 处理图片
                        for image_path in new_images:
                            if not self.is_running:
                                logger.info("批处理被中断，停止处理图片")
                                break
                                
                            self.process_image(image_path)
                            self.current_batch_processed += 1
                            
                            # 发送进度更新信号
                            self.batch_progress.emit(self.current_batch_processed, len(new_images))
                    else:
                        logger.info("无新图片待处理")
                            
                # 等待下一次轮询，使用更短的间隔并检查停止标志
                for _ in range(self.polling_interval * 10):  # 将5秒拆分为50个0.1秒
                    if not self.is_running:
                        break
                    time.sleep(0.1)
                
            logger.info("批处理已停止")
            self.batch_finished.emit()
            
        except Exception as e:
            error_msg = f"批处理线程异常: {str(e)}"
            logger.error(error_msg)
            self.error.emit(error_msg)
            self.batch_finished.emit()
    
    def scan_images(self):
        """扫描指定目录中的图片文件"""
        try:
            if not os.path.exists(self.processing_dir):
                logger.warning(f"监控文件夹不存在: {self.processing_dir}")
                return []
                
            valid_extensions = ['.png', '.jpg', '.jpeg']
            image_files = []
            
            for filename in os.listdir(self.processing_dir):
                file_path = os.path.join(self.processing_dir, filename)
                if os.path.isfile(file_path):
                    _, ext = os.path.splitext(filename.lower())
                    if ext in valid_extensions:
                        image_files.append(file_path)
            
            # 按创建时间排序（旧的在前，新的在后）
            image_files.sort(key=lambda x: os.path.getctime(x))
            
            return image_files
            
        except Exception as e:
            logger.error(f"扫描图片失败: {str(e)}")
            return []
    
    def process_image(self, image_path):
        """处理单张图片"""
        try:
            logger.info(f"开始处理图片: {os.path.basename(image_path)}")
            self.progress.emit(f"正在处理图片: {os.path.basename(image_path)}")

            # 使用现有的图片处理逻辑（按界面选择的引擎）
            client = AnomalyDetectionClient()
            local_result_pre_image, local_result_heat_map, local_result_json = client.process_images(
                image_path, self.engine_name)
            
            # 记录已处理的图片
            self.processed_images.add(image_path)
            
            # 更新检查点
            if self.checkpoint_file:
                self.update_checkpoint()
            
            logger.info(f"图片处理完成: {os.path.basename(image_path)}")
            self.progress.emit(f"图片处理完成: {os.path.basename(image_path)}")
            
            # 发送处理完成信号
            self.image_processed.emit(local_result_pre_image, local_result_heat_map, local_result_json)
            
        except Exception as e:
            error_msg = f"图片处理失败: {os.path.basename(image_path)} - {str(e)}"
            logger.error(error_msg)
            self.error.emit(error_msg)
            # 批处理状态下不弹出错误提示，继续处理下一张
    
    def load_checkpoint(self):
        """加载检查点文件"""
        try:
            with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
                self.processed_images = set(checkpoint_data.get('processed_images', []))
            logger.info(f"加载检查点，已处理 {len(self.processed_images)} 张图片")
        except Exception as e:
            logger.error(f"加载检查点失败: {str(e)}")
    
    def update_checkpoint(self):
        """更新检查点文件"""
        try:
            checkpoint_dir = os.path.dirname(self.checkpoint_file)
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
                
            checkpoint_data = {
                'processed_images': list(self.processed_images),
                'last_update': datetime.datetime.now().isoformat()
            }
            
            with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.error(f"更新检查点失败: {str(e)}")
    
    def stop(self):
        """停止批处理"""
        self.is_running = False
    
    def terminate(self):
        """强制终止批处理线程"""
        self.is_running = False
        if self.isRunning():
            super().terminate()
            logger.info("强制终止批处理线程")

class AnomalyDetectionWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.image_path = None  # 存储上传的图片路径
        self.current_results = None  # 当前处理结果文件路径
        self.processing_thread = None  # 处理线程
        self.batch_thread = None  # 批处理线程
        self.is_batch_processing = False  # 批处理状态
        self._active_warning_box = None  # 当前警告弹窗引用，非模态模式下防止被垃圾回收
        load_dotenv()  # 项目根 .env：监控路径首次默认值等
        self.settings = QSettings("analysis_system", "anomaly_detection")

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
        image_splitter.setSizes([400, 1000])
        
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

        # 新增输入控件的统一风格（与主色 #2196F3 圆角风格协调）
        input_qss = """
            QComboBox, QLineEdit {
                border: 1px solid #ccc; border-radius: 4px;
                padding: 4px 8px; background-color: white; color: #333;
                font-size: 11px; min-height: 22px;
            }
            QComboBox:hover, QLineEdit:hover { border-color: #2196F3; }
            QComboBox:focus, QLineEdit:focus { border: 1px solid #2196F3; }
            QLineEdit[invalid="true"] { border: 1px solid #f44336; background-color: #FFF3F0; }
            QComboBox QAbstractItemView {
                background-color: white; border: 1px solid #ccc;
                selection-background-color: #e3f2fd; selection-color: #1976D2;
            }
        """
        ghost_btn_qss = """
            QPushButton {
                background-color: white; color: #2196F3;
                border: 1px solid #2196F3; border-radius: 4px;
                padding: 6px 12px; font-weight: bold; font-size: 11px;
            }
            QPushButton:hover { background-color: #e3f2fd; }
            QPushButton:pressed { background-color: #bbdefb; }
        """

        # 检测算法选择行
        engine_row_layout = QHBoxLayout()
        engine_label = QLabel("检测算法:")
        engine_label.setMinimumWidth(70)
        self.engine_combo = QComboBox()
        self.engine_combo.setStyleSheet(input_qss)
        self.engine_combo.addItem("规则化检测（快速）", ENGINE_RULE_BASED)
        self.engine_combo.addItem("深度学习检测（Dinomaly）", ENGINE_DINOMALY)
        saved_engine = self.settings.value("anomaly_engine", ENGINE_RULE_BASED)
        saved_engine_index = self.engine_combo.findData(saved_engine)
        if saved_engine_index >= 0:
            self.engine_combo.setCurrentIndex(saved_engine_index)
        self.engine_combo.currentIndexChanged.connect(self.on_engine_changed)
        engine_row_layout.addWidget(engine_label)
        engine_row_layout.addWidget(self.engine_combo, 1)
        upload_layout.addLayout(engine_row_layout)

        # 按钮行布局 - 选择图片和清空图片在同一行
        button_row_layout = QHBoxLayout()
        
        # 上传按钮
        self.upload_btn = QPushButton("选择图片")
        self.upload_btn.clicked.connect(self.select_image)
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
        button_row_layout.addWidget(self.upload_btn)
        
        # 清空按钮
        self.clear_btn = QPushButton("清空图片")
        self.clear_btn.clicked.connect(self.clear_image)
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
        button_row_layout.addWidget(self.clear_btn)
        
        upload_layout.addLayout(button_row_layout)
        
        # 当前图片显示区域
        image_info_frame = QFrame()
        image_info_frame.setFrameStyle(QFrame.StyledPanel)
        image_info_frame.setStyleSheet("""
            QFrame {
                border: 1px solid #ccc;
                border-radius: 4px;
                background-color: white;
                padding: 5px;
            }
        """)
        image_info_layout = QVBoxLayout(image_info_frame)
        image_info_layout.setContentsMargins(10, 10, 10, 10)
        
        self.current_image_label = QLabel("当前图片:")
        self.current_image_label.setFont(QFont("Arial", 10, QFont.Bold))
        image_info_layout.addWidget(self.current_image_label)
        
        self.image_name_label = QLabel("未选择图片")
        self.image_name_label.setStyleSheet("color: gray; font-style: italic; font-size: 10px;")
        self.image_name_label.setWordWrap(True)
        image_info_layout.addWidget(self.image_name_label)
        
        image_info_frame.setMaximumHeight(120)
        image_info_frame.setMinimumHeight(80)
        upload_layout.addWidget(image_info_frame)
        
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
        
        # 批处理按钮区域
        batch_frame = QFrame()
        batch_frame.setFrameStyle(QFrame.StyledPanel)
        batch_frame.setStyleSheet("""
            QFrame {
                border: 1px solid #ccc;
                border-radius: 4px;
                background-color: #f8f9fa;
                padding: 5px;
            }
        """)
        batch_layout = QVBoxLayout(batch_frame)
        batch_layout.setContentsMargins(10, 10, 10, 10)
        
        batch_title = QLabel("在线批处理")
        batch_title.setFont(QFont("Arial", 10, QFont.Bold))
        batch_layout.addWidget(batch_title)

        # 监控文件夹选择行
        dir_row_layout = QHBoxLayout()
        dir_label = QLabel("监控文件夹:")
        dir_label.setMinimumWidth(70)
        self.batch_dir_edit = QLineEdit()
        self.batch_dir_edit.setStyleSheet(input_qss)
        self.batch_dir_edit.setPlaceholderText("选择或输入要监控的文件夹路径")
        self.batch_dir_edit.textChanged.connect(self._validate_batch_dir)
        saved_dir = self.settings.value("batch_monitor_dir", "") or os.getenv("ONLINE_PROCESSING_AD_DIR", "")
        if saved_dir:
            self.batch_dir_edit.setText(saved_dir)
        browse_btn = QPushButton("浏览...")
        browse_btn.setStyleSheet(ghost_btn_qss)
        browse_btn.clicked.connect(self.browse_batch_dir)
        dir_row_layout.addWidget(dir_label)
        dir_row_layout.addWidget(self.batch_dir_edit, 1)
        dir_row_layout.addWidget(browse_btn)
        batch_layout.addLayout(dir_row_layout)

        # 批处理按钮行布局 - 启动和停止按钮在同一行
        batch_button_row_layout = QHBoxLayout()
        
        # 启动批处理按钮
        self.start_batch_btn = QPushButton("启动在线批处理")
        self.start_batch_btn.clicked.connect(self.start_batch_processing)
        self.start_batch_btn.setStyleSheet("""
            QPushButton {
                background-color: #9C27B0;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #7B1FA2;
            }
            QPushButton:pressed {
                background-color: #6A1B9A;
            }
        """)
        batch_button_row_layout.addWidget(self.start_batch_btn)
        
        # 终止批处理按钮
        self.stop_batch_btn = QPushButton("终止在线批处理")
        self.stop_batch_btn.clicked.connect(self.stop_batch_processing)
        self.stop_batch_btn.setEnabled(False)
        self.stop_batch_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF5722;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover:enabled {
                background-color: #E64A19;
            }
            QPushButton:pressed:enabled {
                background-color: #D84315;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        batch_button_row_layout.addWidget(self.stop_batch_btn)
        
        batch_layout.addLayout(batch_button_row_layout)
        
        # 批处理状态显示
        self.batch_status_label = QLabel("批处理状态: 未启动")
        self.batch_status_label.setStyleSheet("color: gray; font-size: 10px;")
        batch_layout.addWidget(self.batch_status_label)
        
        batch_frame.setMaximumHeight(190)
        batch_frame.setMinimumHeight(150)
        upload_layout.addWidget(batch_frame)
        
        # 图片预览区域
        preview_frame = QFrame()
        preview_frame.setFrameStyle(QFrame.StyledPanel)
        preview_frame.setStyleSheet("""
            QFrame {
                border: 1px solid #ccc;
                border-radius: 4px;
                background-color: white;
            }
        """)
        preview_layout = QVBoxLayout(preview_frame)
        preview_layout.setContentsMargins(5, 5, 5, 5)
        
        preview_title = QLabel("图片预览")
        preview_title.setFont(QFont("Arial", 10, QFont.Bold))
        preview_title.setAlignment(Qt.AlignCenter)
        preview_layout.addWidget(preview_title)
        
        self.image_preview = QLabel()
        self.image_preview.setAlignment(Qt.AlignCenter)
        self.image_preview.setStyleSheet("""
            QLabel {
                border: 1px solid #ddd;
                border-radius: 4px;
                background-color: #f9f9f9;
                color: #999;
                font-size: 12px;
            }
        """)
        self.image_preview.setMinimumSize(200, 150)
        self.image_preview.setMaximumSize(300, 200)
        self.image_preview.setText("暂无图片预览")
        preview_layout.addWidget(self.image_preview)
        
        preview_frame.setMaximumHeight(250)
        preview_frame.setMinimumHeight(200)
        upload_layout.addWidget(preview_frame)
        
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
        
        # 热力图标签页
        self.heatmap_tab = self.create_image_tab("热力图")
        self.result_tabs.addTab(self.heatmap_tab, "热力图")
        
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
        logger.info("异常检测系统启动")
        
    def browse_batch_dir(self):
        """浏览选择批处理监控文件夹"""
        current = self.batch_dir_edit.text().strip()
        start_dir = current if current and os.path.exists(current) else ""
        chosen = QFileDialog.getExistingDirectory(self, "选择监控文件夹", start_dir)
        if chosen:
            self.batch_dir_edit.setText(chosen)
            self.settings.setValue("batch_monitor_dir", chosen)
            logger.info(f"监控文件夹已选择: {chosen}")

    def _validate_batch_dir(self, text):
        """监控路径输入即校验：不存在时红框提示，悬停显示完整路径"""
        ok = os.path.isdir(text.strip()) if text.strip() else True
        self.batch_dir_edit.setProperty("invalid", not ok)
        self.batch_dir_edit.style().unpolish(self.batch_dir_edit)
        self.batch_dir_edit.style().polish(self.batch_dir_edit)
        self.batch_dir_edit.setToolTip("" if ok or len(text) <= 30 else text)

    def on_engine_changed(self):
        """检测算法切换回调"""
        engine_name = self.engine_combo.currentData()
        self.settings.setValue("anomaly_engine", engine_name)
        if engine_name == ENGINE_DINOMALY:
            logger.info("已切换到深度学习检测（Dinomaly）：首次使用时将加载模型权重，耗时较久请耐心等待")
        else:
            logger.info("已切换到规则化检测（快速）")

    def start_batch_processing(self):
        """启动在线批处理"""
        try:
            # 从界面读取监控文件夹
            processing_dir = self.batch_dir_edit.text().strip()
            if not processing_dir:
                QMessageBox.warning(self, "未选择路径", "请先在界面上选择要监控的文件夹")
                return

            # 检查文件夹是否存在
            if not os.path.exists(processing_dir):
                QMessageBox.warning(self, "文件夹不存在", f"指定的监控文件夹不存在: {processing_dir}")
                return

            engine_name = self.engine_combo.currentData()
            self.settings.setValue("batch_monitor_dir", processing_dir)

            # 批处理运行中锁定引擎选择与单张处理，避免并发使用引擎造成困惑
            self.engine_combo.setEnabled(False)
            self.process_btn.setEnabled(False)
            
            # 检查检查点文件
            checkpoint_dir = "temp/batch_processing_checkpoint"
            checkpoint_files = []
            if os.path.exists(checkpoint_dir):
                checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.json')]
            
            checkpoint_file = None
            if checkpoint_files:
                # 使用新的检查点选择对话框
                dialog = CheckpointSelectionDialog(checkpoint_files, checkpoint_dir, self)
                result = dialog.exec_()
                
                if result == QDialog.Accepted:
                    if dialog.selected_checkpoint is None:
                        # 创建新的检查点文件
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        checkpoint_file = os.path.join(checkpoint_dir, f"{timestamp}.json")
                        logger.info(f"创建新的检查点文件: {checkpoint_file}")
                    else:
                        # 使用选中的检查点文件
                        checkpoint_file = dialog.selected_checkpoint
                        logger.info(f"使用选中的检查点文件: {checkpoint_file}")
                else:
                    return  # 用户取消
            else:
                # 创建新的检查点文件
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                checkpoint_file = os.path.join(checkpoint_dir, f"{timestamp}.json")
                logger.info(f"创建新的检查点文件: {checkpoint_file}")
            
            # 启动批处理线程
            self.batch_thread = BatchProcessingThread(processing_dir, checkpoint_file, engine_name)
            self.batch_thread.progress.connect(self.on_batch_progress)
            self.batch_thread.error.connect(self.on_batch_error)
            self.batch_thread.batch_finished.connect(self.on_batch_finished)
            self.batch_thread.image_processed.connect(self.on_batch_image_processed)
            self.batch_thread.batch_progress.connect(self.on_batch_progress_update)
            self.batch_thread.start()
            
            # 更新界面状态
            self.is_batch_processing = True
            self.start_batch_btn.setEnabled(False)
            self.stop_batch_btn.setEnabled(True)
            self.batch_status_label.setText("批处理状态: 运行中")
            self.batch_status_label.setStyleSheet("color: green; font-size: 10px;")
            
            logger.info(f"在线批处理已启动，监控文件夹: {processing_dir}")
            
        except Exception as e:
            error_msg = f"启动批处理失败: {str(e)}"
            logger.error(error_msg)
            QMessageBox.critical(self, "启动失败", error_msg)
    
    def stop_batch_processing(self):
        """停止在线批处理"""
        if self.batch_thread and self.batch_thread.isRunning():
            self.batch_thread.stop()
            logger.info("正在停止批处理...")
            
            # 更新界面状态 - 停止过程中禁用两个按钮
            self.start_batch_btn.setEnabled(False)
            self.stop_batch_btn.setEnabled(False)
            self.batch_status_label.setText("批处理状态: 正在停止...")
            self.batch_status_label.setStyleSheet("color: orange; font-size: 10px;")
            
            # 设置超时检查，如果10秒后线程仍在运行，强制终止
            QTimer.singleShot(10000, self.check_batch_thread_timeout)
    
    def check_batch_thread_timeout(self):
        """检查批处理线程超时，如果仍在运行则强制终止"""
        if self.batch_thread and self.batch_thread.isRunning():
            logger.warning("批处理线程超时未停止，强制终止")
            self.batch_thread.terminate()
            self.on_batch_finished()  # 手动调用完成回调
    
    def on_batch_progress(self, progress_msg):
        """批处理进度回调"""
        logger.info(progress_msg)
    
    def on_batch_error(self, error_msg):
        """批处理错误回调"""
        logger.error(error_msg)
        # 批处理状态下不弹出错误提示，只记录日志
    
    def on_batch_finished(self):
        """批处理完成回调"""
        self.is_batch_processing = False
        self.start_batch_btn.setEnabled(True)
        self.stop_batch_btn.setEnabled(False)
        self.engine_combo.setEnabled(True)
        self.update_process_button_state()
        self.batch_status_label.setText("批处理状态: 已停止")
        self.batch_status_label.setStyleSheet("color: gray; font-size: 10px;")
        logger.info("批处理已停止")
    
    def on_batch_image_processed(self, prediction_path, heatmap_path, json_path):
        """批处理图片处理完成回调"""
        # 保存结果路径
        self.current_results = {
            'prediction': prediction_path,
            'heatmap': heatmap_path,
            'json': json_path
        }
        
        # 显示结果
        self.display_results()
        
        logger.info(f"批处理图片处理完成")
        logger.info(f"预测结果: {prediction_path}")
        logger.info(f"热力图: {heatmap_path}")
        logger.info(f"JSON结果: {json_path}")
    
    def on_batch_progress_update(self, current, total):
        """批处理进度更新回调"""
        if total > 0:
            progress_text = f"当前批处理进度 {current}/{total}"
            self.batch_status_label.setText(f"批处理状态: 运行中 | {progress_text}")
        else:
            self.batch_status_label.setText("批处理状态: 运行中")
        
    def select_image(self):
        """选择图片文件"""
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self,
            "选择图片文件",
            "",
            "图片文件 (*.jpg *.jpeg *.png *.bmp *.gif *.tiff)"
        )
        
        if file_path:
            # 验证图片格式
            if self.validate_image_format(file_path):
                self.image_path = file_path
                self.image_name_label.setText(os.path.basename(file_path))
                self.image_name_label.setStyleSheet("color: black; font-weight: bold;")
                
                # 显示图片预览
                self.display_image_preview(file_path)
                
                self.update_process_button_state()
                logger.info(f"选择了图片: {os.path.basename(file_path)}")
            else:
                QMessageBox.warning(self, "格式错误", f"文件 {os.path.basename(file_path)} 不是有效的图片格式")
            
    def display_image_preview(self, image_path):
        """显示图片预览"""
        try:
            if os.path.exists(image_path):
                pixmap = QPixmap(image_path)
                if not pixmap.isNull():
                    # 计算预览尺寸，保持宽高比
                    preview_size = self.image_preview.size()
                    scaled_pixmap = pixmap.scaled(
                        preview_size.width() - 10,  # 留出边框空间
                        preview_size.height() - 10,
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation
                    )
                    self.image_preview.setPixmap(scaled_pixmap)
                    logger.info(f"显示图片预览: {os.path.basename(image_path)}")
                else:
                    self.image_preview.setText("无法加载图片预览")
                    logger.warning(f"无法加载图片预览: {image_path}")
            else:
                self.image_preview.setText("图片文件不存在")
                logger.error(f"图片文件不存在: {image_path}")
        except Exception as e:
            logger.error(f"显示图片预览失败: {str(e)}")
            self.image_preview.setText("预览加载失败")
            
    def validate_image_format(self, file_path):
        """验证图片格式"""
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']
        _, ext = os.path.splitext(file_path.lower())
        return ext in valid_extensions
        
    def clear_image(self):
        """清空图片"""
        self.image_path = None
        self.image_name_label.setText("未选择图片")
        self.image_name_label.setStyleSheet("color: gray; font-style: italic;")
        
        # 清空图片预览
        self.image_preview.clear()
        self.image_preview.setText("暂无图片预览")
        
        self.update_process_button_state()
        logger.info("已清空图片")
        
    def update_process_button_state(self):
        """更新处理按钮状态"""
        if self.image_path is not None:
            self.process_btn.setEnabled(True)
            self.process_btn.setText("开始处理")
        else:
            self.process_btn.setEnabled(False)
            self.process_btn.setText("请先选择图片")
            
    def start_processing(self):
        """开始处理图片"""
        if self.image_path is None:
            QMessageBox.warning(self, "未选择图片", "请先选择一张图片")
            return
            
        # 禁用处理按钮
        self.process_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # 不确定进度
        
        logger.info(f"开始处理图片: {os.path.basename(self.image_path)}")

        # 创建并启动处理线程（按界面选择的引擎）
        self.processing_thread = ImageProcessingThread(self.image_path, self.engine_combo.currentData())
        self.processing_thread.finished.connect(self.on_processing_finished)
        self.processing_thread.error.connect(self.on_processing_error)
        self.processing_thread.progress.connect(self.on_processing_progress)
        self.processing_thread.start()
        
    def on_processing_finished(self, prediction_path, heatmap_path, json_path):
        """处理完成回调"""
        self.progress_bar.setVisible(False)
        self.process_btn.setEnabled(True)
        
        # 保存结果路径
        self.current_results = {
            'prediction': prediction_path,
            'heatmap': heatmap_path,
            'json': json_path
        }
        
        # 显示结果
        self.display_results()
        
        logger.info(f"图片处理完成")
        logger.info(f"预测结果: {prediction_path}")
        logger.info(f"热力图: {heatmap_path}")
        logger.info(f"JSON结果: {json_path}")
        
    def on_processing_error(self, error_msg):
        """处理错误回调"""
        self.progress_bar.setVisible(False)
        self.process_btn.setEnabled(True)
        
        logger.error(error_msg)
        QMessageBox.critical(self, "处理失败", error_msg)
        
    def on_processing_progress(self, progress_msg):
        """处理进度回调"""
        logger.info(progress_msg)
        
    def display_results(self):
        """显示处理结果"""
        if not self.current_results:
            return
            
        # 显示预测结果
        self.display_image_result(self.prediction_tab, self.current_results['prediction'], "预测结果")
        
        # 显示热力图
        self.display_image_result(self.heatmap_tab, self.current_results['heatmap'], "热力图")
        
        # 显示JSON结果
        self.display_json_result()
        
    def display_image_result(self, tab_widget, image_path, result_type):
        """显示图片结果"""
        try:
            if os.path.exists(image_path):
                # 获取标签页中的图片显示组件
                scroll_area = tab_widget.findChild(QScrollArea)
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
                        else:
                            image_display.setText(f"无法加载{result_type}图片")
                    else:
                        image_display.setText(f"无法加载{result_type}图片")
            else:
                # 获取标签页中的图片显示组件
                scroll_area = tab_widget.findChild(QScrollArea)
                if scroll_area:
                    image_display = scroll_area.widget()
                    if isinstance(image_display, QLabel):
                        image_display.setText(f"{result_type}文件不存在")
        except Exception as e:
            logger.error(f"显示{result_type}失败: {str(e)}")
            
    def display_json_result(self):
        """显示JSON结果"""
        try:
            if self.current_results and os.path.exists(self.current_results['json']):
                with open(self.current_results['json'], 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                
                # 检查异常级别并弹出警告
                self.check_anomaly_level(json_data)
                
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
    
    def check_anomaly_level(self, json_data):
        """检查异常级别并弹出警告窗口"""
        try:
            anomaly_level = json_data.get('anomaly_level', '')
            
            # 检查是否为需要弹出警告的异常级别
            if anomaly_level in ['中等异常可能性', '很可能异常']:
                self.show_anomaly_warning(anomaly_level, json_data)
                logger.info(f"检测到异常级别: {anomaly_level}，已弹出警告窗口")
                
        except Exception as e:
            logger.error(f"检查异常级别失败: {str(e)}")
    
    def _get_alert_auto_close_ms(self):
        """读取警告弹窗自动关闭时长配置，0或负数表示不自动关闭"""
        raw_value = os.getenv('ANOMALY_ALERT_AUTO_CLOSE_MS', '5000')
        try:
            return int(raw_value)
        except ValueError:
            logger.warning(f"ANOMALY_ALERT_AUTO_CLOSE_MS 配置无效: {raw_value}，使用默认值5000")
            return 5000

    def show_anomaly_warning(self, anomaly_level, json_data):
        """显示异常警告弹窗"""
        try:
            # 获取模拟电压数值
            analog_voltage = json_data.get('analog_voltage', '未知')

            auto_close_ms = self._get_alert_auto_close_ms()

            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("异常检测警告")
            msg_box.setText(f"模拟电压: {analog_voltage}")
            if auto_close_ms > 0:
                msg_box.setInformativeText(f"检测到异常级别: {anomaly_level}（{auto_close_ms // 1000}秒后自动关闭）")
            else:
                msg_box.setInformativeText(f"检测到异常级别: {anomaly_level}")
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

            if auto_close_ms > 0:
                # 非模态显示并定时关闭，避免批处理时弹窗堆积阻塞操作
                self._active_warning_box = msg_box
                QTimer.singleShot(auto_close_ms, msg_box.close)
                msg_box.show()
            else:
                # 模态阻塞，等待用户手动关闭
                msg_box.exec_()

        except Exception as e:
            logger.error(f"显示异常警告弹窗失败: {str(e)}")
            
    def refresh_page(self):
        """刷新页面"""
        # 清空图片
        self.clear_image()
        
        # 清空结果
        self.current_results = None
        
        # 清空结果显示
        self.clear_result_displays()
        
        logger.info("页面已刷新")
        
    def clear_result_displays(self):
        """清空结果显示"""
        # 清空预测结果
        self.clear_image_display(self.prediction_tab, "预测结果")
        
        # 清空热力图
        self.clear_image_display(self.heatmap_tab, "热力图")
        
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
    window.setWindowTitle("异常检测系统")
    window.setGeometry(100, 100, 1400, 900)
    
    # 创建中央部件
    central_widget = QWidget()
    window.setCentralWidget(central_widget)
    
    # 创建布局
    layout = QVBoxLayout(central_widget)
    layout.setContentsMargins(0, 0, 0, 0)
    
    # 添加异常检测组件
    anomaly_detection_widget = AnomalyDetectionWidget()
    layout.addWidget(anomaly_detection_widget)
    
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
