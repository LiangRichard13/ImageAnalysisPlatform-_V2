"""异常判定阈值与边缘子图排除的环境变量解析。

ANOMALY_THRESHOLD：判定阈值（0-1 浮点，score>阈值 判异常）。
ANOMALY_CROP_EXCLUDE_DETECT / _JUDGE / _DISPLAY：膜面长图 25 等分下
"每侧排除 N 份子图"的三级排除（检测/判定/展示），独立计数从原图最边缘
数起，引擎侧生效范围取 max（详见 .env 注释）。

仅供 UI 侧进程内引擎（utils/dinomaly_engine.py、utils/rule_based_wrinkle.py）使用；
core/ 下的独立共享目录服务脚本因 sys.path 不含仓库根无法导入本模块，
各自内嵌同逻辑副本（保持逐行一致，改这里的判定规则需同步）。

仅依赖标准库——规则引擎链路不得引入 torch。
"""

import logging
import os
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_ANOMALY_THRESHOLD = 0.21

DEFAULT_CROP_EXCLUDE_DETECT = 0
DEFAULT_CROP_EXCLUDE_JUDGE = 2
DEFAULT_CROP_EXCLUDE_DISPLAY = 0
MAX_CROP_EXCLUDE = 12  # 25 等分下每侧最多排除 12 份（保留中间 1 份）


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


def _resolve_edge_crop_count(env_name: str, default: int,
                             max_exclude: int = MAX_CROP_EXCLUDE) -> int:
    """读单个边缘排除数（非负整数，≤ 25 等分的一半）；无效告警并回退默认。"""
    raw = os.getenv(env_name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning(f"{env_name} 配置无效: {raw!r}，使用默认值 {default}")
        return default
    if value < 0 or value > max_exclude:
        logger.warning(f"{env_name} 超出 [0,{max_exclude}] 区间: {raw!r}，使用默认值 {default}")
        return default
    return value


def resolve_crop_exclude_detect() -> int:
    return _resolve_edge_crop_count("ANOMALY_CROP_EXCLUDE_DETECT", DEFAULT_CROP_EXCLUDE_DETECT)


def resolve_crop_exclude_judge() -> int:
    return _resolve_edge_crop_count("ANOMALY_CROP_EXCLUDE_JUDGE", DEFAULT_CROP_EXCLUDE_JUDGE)


def resolve_crop_exclude_display() -> int:
    return _resolve_edge_crop_count("ANOMALY_CROP_EXCLUDE_DISPLAY", DEFAULT_CROP_EXCLUDE_DISPLAY)


def resolve_crop_exclusions(detect: Optional[int], judge: Optional[int],
                            display: Optional[int]) -> Tuple[int, int, int]:
    """三排除参数 → 生效值 (detect_n, judge_n, display_n)。

    None → 各自环境变量/默认值（调用时解析，env 可 monkeypatch 测试）；
    独立计数语义：判定/展示生效范围取 max(detect, 各自值)。
    """
    detect_n = detect if detect is not None else resolve_crop_exclude_detect()
    judge_param = judge if judge is not None else resolve_crop_exclude_judge()
    display_param = display if display is not None else resolve_crop_exclude_display()
    return detect_n, max(detect_n, judge_param), max(detect_n, display_param)
