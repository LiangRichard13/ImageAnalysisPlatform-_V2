"""界面日志统一格式：`HH:MM:SS emoji 消息`。

emoji 由 Handler 端按 级别优先 → logger 名映射事件域 自动附加，
调用处零改动；未命中映射的模块落默认 📌。
"""
import logging
import time

from PyQt5.QtCore import QObject, pyqtSignal

# 级别映射（优先于模块映射）
_LEVEL_EMOJI = {
    logging.WARNING: "⚠️",
    logging.ERROR: "❌",
    logging.CRITICAL: "💥",
}

# INFO 按事件域映射（logger 名前缀匹配，长前缀优先）
_MODULE_EMOJI = (
    ("utils.shm_ingest", "📡"),
    ("utils.ipc", "📡"),
    ("utils.archive_worker", "💾"),
    ("utils.dinomaly_engine", "🧠"),
    ("anomaly_detection_tab", "🔍"),
    ("film_trend_analysis_tab", "📈"),
    ("main_window", "🚀"),
)
DEFAULT_EMOJI = "📌"


def emoji_for(record: logging.LogRecord) -> str:
    """按级别（优先）与 logger 名前缀选事件 emoji。"""
    emoji = _LEVEL_EMOJI.get(record.levelno)
    if emoji:
        return emoji
    for prefix, mapped in _MODULE_EMOJI:
        if record.name == prefix or record.name.startswith(prefix + "."):
            return mapped
    return DEFAULT_EMOJI


def format_record(record: logging.LogRecord) -> str:
    """统一行格式：HH:MM:SS emoji 消息（等宽字体下时间戳定宽对齐）。"""
    ts = time.strftime("%H:%M:%S", time.localtime(record.created))
    return f"{ts} {emoji_for(record)} {record.getMessage()}"


class IapLogHandler(logging.Handler, QObject):
    """界面日志处理器：格式化后经信号发往日志区（三入口共用）。"""

    log_signal = pyqtSignal(str)

    def __init__(self):
        logging.Handler.__init__(self)
        QObject.__init__(self)
        self._closed = False

    def emit(self, record):
        if not self._closed:
            try:
                self.log_signal.emit(format_record(record))
            except RuntimeError:
                # Qt 对象已被删除，忽略错误
                pass

    def close(self):
        """关闭处理器"""
        self._closed = True
        super().close()
