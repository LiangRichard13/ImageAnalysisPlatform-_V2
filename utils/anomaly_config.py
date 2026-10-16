"""异常判定阈值的环境变量解析（ANOMALY_THRESHOLD）。

仅供 UI 侧进程内引擎（utils/dinomaly_engine.py、utils/rule_based_wrinkle.py）使用；
core/ 下的独立共享目录服务脚本因 sys.path 不含仓库根无法导入本模块，
各自内嵌同逻辑的 _resolve_threshold（保持逐行一致，改这里的判定规则需同步）。

仅依赖标准库——规则引擎链路不得引入 torch。
"""

import logging
import os

logger = logging.getLogger(__name__)

DEFAULT_ANOMALY_THRESHOLD = 0.21


def resolve_anomaly_threshold(default: float = DEFAULT_ANOMALY_THRESHOLD) -> float:
    """读 ANOMALY_THRESHOLD（0-1 浮点，score>阈值 判异常）；无效/越界告警并回退默认值。"""
    raw = os.getenv("ANOMALY_THRESHOLD", "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning(f"ANOMALY_THRESHOLD 配置无效: {raw}，使用默认值 {default}")
        return default
    if not 0.0 < value < 1.0:
        logger.warning(f"ANOMALY_THRESHOLD 超出 (0,1) 区间: {raw}，使用默认值 {default}")
        return default
    return value
