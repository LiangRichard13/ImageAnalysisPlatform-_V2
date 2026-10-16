"""结果图弹性缩放冒烟：预测/热力图随显示区尺寸等比跟随。

回归背景：结果图原先只在设置那一刻按当时区域尺寸缩放一次，
分割条拖动/窗口缩放后不再跟随——长条图放大窗口后不抵左右两侧。
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

_SMOKE = r"""
import os, sys
os.environ["QT_QPA_PLATFORM"] = "offscreen"
sys.path.insert(0, ".")
from PyQt5.QtCore import QSettings
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QApplication, QScrollArea
from utils import dinomaly_engine as de
import anomaly_detection_tab as tab_mod

# 固定 rule_based 引擎，QSettings 持久化到注册表，测后恢复
_qs = QSettings("analysis_system", "anomaly_detection")
_old_engine = _qs.value("anomaly_engine", "")
_qs.setValue("anomaly_engine", "rule_based")
de.preload_dinomaly_engine_async = lambda: None  # 打桩：绝不真加载权重

app = QApplication([])
widget = tab_mod.AnomalyDetectionWidget()
widget.resize(1400, 900)
widget.show()
app.processEvents()

# 长条帧 3200x100（宽高比 32:1，等比缩放后应由宽度决定大小、抵住左右）
img = QImage(3200, 100, QImage.Format_RGB888)
img.fill(200)

widget._display_qimage(widget.prediction_tab, img, "预测结果")
app.processEvents()

scroll = widget.prediction_tab.findChild(QScrollArea)
label = scroll.widget()
pm = label.pixmap()
assert pm is not None and not pm.isNull(), "结果图未显示"
label_w = label.width() - 10  # 预留边框余量，与实现一致
assert abs(pm.width() - label_w) <= 2, \
    f"设置后宽度应抵满: pm={pm.width()} label={label_w}"

# 放大窗口：图必须跟随变大并继续抵满（旧实现只缩放一次，此处必挂）
old_w = pm.width()
widget.resize(2400, 1100)
app.processEvents()
app.processEvents()
pm2 = label.pixmap()
assert pm2.width() > old_w, f"窗口放大后图应跟随变大: {pm2.width()} <= {old_w}"
label_w2 = label.width() - 10
assert abs(pm2.width() - label_w2) <= 2, \
    f"放大后仍应抵满: pm={pm2.width()} label={label_w2}"

# 清空后 resize 不得按残留源复活图像
widget.clear_result_displays()
app.processEvents()
assert label.pixmap() is None or label.pixmap().isNull()
widget.resize(1800, 900)
app.processEvents()
app.processEvents()
assert label.pixmap() is None or label.pixmap().isNull(), "清空后 resize 复活了图像"

widget.close()
if _old_engine:
    _qs.setValue("anomaly_engine", _old_engine)
else:
    _qs.remove("anomaly_engine")
print("RESULT_VIEW_SCALE_OK")
sys.exit(0)
"""


def test_result_view_follows_resize():
    """UI 接线冒烟（独立进程：Qt 状态不跨测试共享）。"""
    proc = subprocess.run([sys.executable, "-c", _SMOKE],
                          cwd=str(ROOT), capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    assert "RESULT_VIEW_SCALE_OK" in proc.stdout
