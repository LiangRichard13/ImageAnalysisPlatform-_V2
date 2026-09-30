"""异常警告弹窗 .env 开关（ANOMALY_ALERT_POPUP_ENABLED）冒烟测试。

独立进程跑 UI（Qt 状态不跨测试共享）；_alerts_popup_enabled 每次调用
都读 os.getenv，单进程内改 os.environ 即可顺序覆盖各取值分支。
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

_SMOKE = r"""
import os, sys
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ.pop("ANOMALY_ALERT_POPUP_ENABLED", None)
sys.path.insert(0, ".")
from PyQt5.QtCore import QSettings
from PyQt5.QtWidgets import QApplication
from utils import dinomaly_engine as de
import anomaly_detection_tab as tab_mod

# 固定记忆引擎为 rule_based（构造期 _maybe_preload_engine 依赖它），
# 测试后恢复——QSettings 持久化到注册表，不得污染用户真实设置
_qs = QSettings("analysis_system", "anomaly_detection")
_old_engine = _qs.value("anomaly_engine", "")
_qs.setValue("anomaly_engine", "rule_based")
de.preload_dinomaly_engine_async = lambda: None  # 打桩：绝不真加载权重

app = QApplication([])
widget = tab_mod.AnomalyDetectionWidget()

calls = []
widget.show_anomaly_warning = lambda level, data: calls.append(level)


def _trigger():
    calls.clear()
    widget.check_anomaly_level(
        {"anomaly_level": "很可能异常", "analog_voltage": 3.3})


# 1) 未设置：默认弹出（向后兼容）
os.environ.pop("ANOMALY_ALERT_POPUP_ENABLED", None)
_trigger()
assert calls == ["很可能异常"], f"默认应弹出, got {calls}"

# 2) =0：禁用弹窗（仅记日志）
os.environ["ANOMALY_ALERT_POPUP_ENABLED"] = "0"
_trigger()
assert calls == [], f"0 应禁用弹窗, got {calls}"

# 3) =false：禁用
os.environ["ANOMALY_ALERT_POPUP_ENABLED"] = "false"
_trigger()
assert calls == [], f"false 应禁用弹窗, got {calls}"

# 4) =1：弹出
os.environ["ANOMALY_ALERT_POPUP_ENABLED"] = "1"
_trigger()
assert calls == ["很可能异常"], f"1 应启用弹窗, got {calls}"

# 5) =true：弹出
os.environ["ANOMALY_ALERT_POPUP_ENABLED"] = "true"
_trigger()
assert calls == ["很可能异常"]

# 6) 无效值：告警并回退默认开启
os.environ["ANOMALY_ALERT_POPUP_ENABLED"] = "abc"
_trigger()
assert calls == ["很可能异常"], f"无效值应回退默认开启, got {calls}"

# 非异常级别任何取值都不弹
os.environ["ANOMALY_ALERT_POPUP_ENABLED"] = "1"
calls.clear()
widget.check_anomaly_level({"anomaly_level": "很可能正常"})
assert calls == [], "正常级别不应弹窗"

widget.close()
if _old_engine:
    _qs.setValue("anomaly_engine", _old_engine)
else:
    _qs.remove("anomaly_engine")
print("ALERT_POPUP_SMOKE_OK")
sys.exit(0)
"""


def test_alert_popup_env_switch():
    """UI 接线冒烟（独立进程：Qt 状态不跨测试共享）。"""
    proc = subprocess.run([sys.executable, "-c", _SMOKE],
                          cwd=str(ROOT), capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    assert "ALERT_POPUP_SMOKE_OK" in proc.stdout
