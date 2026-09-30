"""全局深色工业 HMI 主题（全部色值的唯一出处）。

设计原则：
- 蓝灰分层底：GRAPHITE(窗口底) < SHELL(导航/面板) < PANEL(卡片/输入) < ELEVATED(hover)
- 唯一饱和主色 ACCENT 青蓝只用于主操作；OK/WARN/ALARM 只做状态语义，不做装饰
- 按钮三类：primaryBtn(主操作，每屏至多一个) / dangerBtn(危险操作) / ghost(默认，次要操作)
- 禁用态统一 DISABLED_BG+TEXT_DIM（非白底，满足视觉置灰测试锚点）

用法：QApplication 创建后调用 apply_theme(app)（main_window 与两个 tab
的独立调试入口均调用）。
"""

from PyQt5.QtGui import QFont

# ---- 蓝灰分层底 ----
GRAPHITE = "#14181D"   # 窗口最深底、图像视口底
SHELL = "#1E252B"      # 导航栏/窗口底
PANEL = "#242D35"      # 卡片/输入框/下拉底
ELEVATED = "#2C3640"   # hover 浮起层
LINE = "#3A4650"       # 边框/分隔线

# ---- 文字 ----
TEXT_HI = "#E8EDF1"    # 主文字
TEXT_MID = "#9AA7B2"   # 次要文字
TEXT_DIM = "#5E6B76"   # 禁用/占位

# ---- 功能色 ----
ACCENT = "#00A8CC"         # 工业青蓝：主操作/激活态/focus
ACCENT_HOVER = "#22B9D9"
ACCENT_PRESSED = "#008FB0"
ACCENT_TEXT = "#062A31"    # 实心青蓝上的深字（白字对比不足）

OK = "#3FCF8E"             # 在线/运行中/很可能正常
WARN = "#FFB020"           # 等待/正在停止/中等异常可能性
ALARM = "#FF5252"          # 很可能异常/错误/终止
ALARM_HOVER = "#FF6B6B"
ALARM_PRESSED = "#D94848"
ALARM_TEXT = "#2B0A0A"

DISABLED_BG = "#2A333B"    # 禁用态底（深色，非白底）

_STATUS_COLORS = {
    "ok": OK, "warn": WARN, "alarm": ALARM,
    "muted": TEXT_MID, "dim": TEXT_DIM, "hi": TEXT_HI,
}


def status_color(kind: str) -> str:
    """状态语义色：替换散落的 color: green/orange/gray 命名色字符串。

    kind: ok / warn / alarm / muted(次要) / dim(禁用) / hi(强调)
    """
    return _STATUS_COLORS[kind]


GLOBAL_QSS = f"""
* {{
    color: {TEXT_HI};
    selection-background-color: {ACCENT};
    selection-color: {ACCENT_TEXT};
}}

QMainWindow, QDialog {{
    background-color: {SHELL};
}}

QLabel {{
    background: transparent;
}}
QLabel:disabled {{
    color: {TEXT_DIM};
}}
QLabel#navTitle {{
    font-size: 16px;
    font-weight: bold;
    color: {TEXT_HI};
}}
QLabel#sectionTitle {{
    font-size: 13px;
    font-weight: bold;
    color: {TEXT_HI};
}}

/* ---- 容器 ---- */
QFrame#panelCard {{
    background-color: {PANEL};
    border: 1px solid {LINE};
    border-radius: 6px;
}}
QFrame#navBar {{
    background-color: {SHELL};
    border-bottom: 1px solid {LINE};
}}

/* ---- 按钮：默认 ghost（次要操作），objectName 覆盖为主/危 ---- */
QPushButton {{
    background-color: transparent;
    color: {TEXT_HI};
    border: 1px solid {LINE};
    border-radius: 4px;
    padding: 5px 14px;
    font-size: 12px;
}}
QPushButton:hover {{
    background-color: {ELEVATED};
    border-color: {TEXT_DIM};
}}
QPushButton:pressed {{
    background-color: {PANEL};
}}
QPushButton:disabled {{
    background-color: {DISABLED_BG};
    color: {TEXT_DIM};
    border: 1px solid {LINE};
}}

QPushButton#primaryBtn {{
    background-color: {ACCENT};
    color: {ACCENT_TEXT};
    border: none;
    font-weight: bold;
}}
QPushButton#primaryBtn:hover {{ background-color: {ACCENT_HOVER}; }}
QPushButton#primaryBtn:pressed {{ background-color: {ACCENT_PRESSED}; }}
QPushButton#primaryBtn:disabled {{
    background-color: {DISABLED_BG};
    color: {TEXT_DIM};
    border: none;
}}

QPushButton#dangerBtn {{
    background-color: {ALARM};
    color: {ALARM_TEXT};
    border: none;
    font-weight: bold;
}}
QPushButton#dangerBtn:hover {{ background-color: {ALARM_HOVER}; }}
QPushButton#dangerBtn:pressed {{ background-color: {ALARM_PRESSED}; }}
QPushButton#dangerBtn:disabled {{
    background-color: {DISABLED_BG};
    color: {TEXT_DIM};
    border: none;
}}

/* 导航按钮（checkable，:checked 为激活态） */
QPushButton#navBtn {{
    background: transparent;
    border: none;
    border-bottom: 2px solid transparent;
    border-radius: 0;
    color: {TEXT_MID};
    padding: 6px 18px;
    font-size: 13px;
}}
QPushButton#navBtn:hover {{ color: {TEXT_HI}; }}
QPushButton#navBtn:checked {{
    color: {ACCENT};
    border-bottom: 2px solid {ACCENT};
    font-weight: bold;
}}

/* ---- 输入控件 ---- */
QLineEdit, QComboBox, QTextEdit, QPlainTextEdit, QListWidget {{
    background-color: {PANEL};
    color: {TEXT_HI};
    border: 1px solid {LINE};
    border-radius: 4px;
    padding: 4px 8px;
    font-size: 12px;
}}
QLineEdit:focus, QComboBox:focus {{ border: 1px solid {ACCENT}; }}
QLineEdit:disabled, QComboBox:disabled {{
    background-color: {DISABLED_BG};
    color: {TEXT_DIM};
    border: 1px solid {LINE};
}}
QLineEdit[invalid="true"] {{
    border: 1px solid {ALARM};
    color: {ALARM};
}}

QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{
    background-color: {PANEL};
    color: {TEXT_HI};
    border: 1px solid {LINE};
    selection-background-color: {ELEVATED};
    selection-color: {TEXT_HI};
    outline: none;
}}

QListWidget {{ padding: 0; }}
QListWidget::item {{
    padding: 6px 8px;
    border-bottom: 1px solid {LINE};
}}
QListWidget::item:selected {{ background-color: {ELEVATED}; color: {TEXT_HI}; }}
QListWidget::item:hover:!selected {{ background-color: {ELEVATED}; }}

QTextEdit#jsonView, QPlainTextEdit {{ font-family: "Consolas"; }}
QTextEdit#logView {{ font-family: "Consolas"; font-size: 11px; }}

/* ---- 子页 Tab ---- */
QTabWidget::pane {{
    border: 1px solid {LINE};
    border-radius: 4px;
    background: {PANEL};
    top: -1px;
}}
QTabBar::tab {{
    background: transparent;
    color: {TEXT_MID};
    padding: 6px 16px;
    border: 1px solid {LINE};
    border-bottom: none;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    margin-right: 2px;
    font-size: 12px;
}}
QTabBar::tab:selected {{
    background: {PANEL};
    color: {ACCENT};
    border-bottom: 2px solid {ACCENT};
}}
QTabBar::tab:hover:!selected {{ background: {ELEVATED}; color: {TEXT_HI}; }}
QTabBar::tab:disabled {{ color: {TEXT_DIM}; }}

/* ---- 滚动区/滚动条（深色化，白条在深底上极扎眼） ---- */
QScrollArea {{
    border: 1px solid {LINE};
    border-radius: 4px;
    background: {GRAPHITE};
}}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
QScrollArea::corner {{ background: {GRAPHITE}; }}

QScrollBar:vertical {{ background: {GRAPHITE}; width: 10px; margin: 0; }}
QScrollBar::handle:vertical {{ background: {LINE}; border-radius: 5px; min-height: 30px; }}
QScrollBar::handle:vertical:hover {{ background: {TEXT_DIM}; }}
QScrollBar:horizontal {{ background: {GRAPHITE}; height: 10px; margin: 0; }}
QScrollBar::handle:horizontal {{ background: {LINE}; border-radius: 5px; min-width: 30px; }}
QScrollBar::handle:horizontal:hover {{ background: {TEXT_DIM}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

/* ---- 进度条（批处理 indeterminate 流动） ---- */
QProgressBar {{
    background: {PANEL};
    border: 1px solid {LINE};
    border-radius: 4px;
    min-height: 12px;
    text-align: center;
    color: {TEXT_MID};
    font-size: 11px;
}}
QProgressBar::chunk {{ background-color: {ACCENT}; border-radius: 3px; }}

/* ---- 分割器 ---- */
QSplitter::handle {{ background: {SHELL}; }}
QSplitter::handle:hover {{ background: {ACCENT}; }}
QSplitter::handle:vertical {{ height: 2px; }}
QSplitter::handle:horizontal {{ width: 2px; }}

/* ---- 菜单 ---- */
QMenuBar {{
    background: {SHELL};
    color: {TEXT_HI};
    border-bottom: 1px solid {LINE};
}}
QMenuBar::item {{ background: transparent; padding: 4px 10px; }}
QMenuBar::item:selected {{ background: {ELEVATED}; }}
QMenu {{
    background: {PANEL};
    color: {TEXT_HI};
    border: 1px solid {LINE};
}}
QMenu::item {{ padding: 5px 24px 5px 12px; }}
QMenu::item:selected {{ background: {ELEVATED}; }}
QMenu::separator {{ height: 1px; background: {LINE}; margin: 4px 8px; }}

/* ---- 提示/弹窗 ---- */
QToolTip {{
    background: {PANEL};
    color: {TEXT_HI};
    border: 1px solid {ACCENT};
    padding: 4px 8px;
    font-size: 11px;
}}
QMessageBox {{ background: {SHELL}; }}
QMessageBox QLabel {{ color: {TEXT_HI}; font-size: 13px; background: transparent; }}
QMessageBox QPushButton {{ min-width: 72px; }}
"""


def apply_theme(app) -> None:
    """应用全局主题：统一中文字体 + 全局 QSS。

    Microsoft YaHei UI 为 Win10 标准界面字体（含完整中文，替代原先
    不含中文的 Arial）；字号阶梯 11/12/13/16 由 QSS 按控件细分。
    """
    app.setFont(QFont("Microsoft YaHei UI", 10))
    app.setStyleSheet(GLOBAL_QSS)
