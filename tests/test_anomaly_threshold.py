"""ANOMALY_THRESHOLD 环境变量解析与引擎阈值接线测试。

仅覆盖轻量链路（utils/anomaly_config + 规则引擎默认构造），
不加载 torch 权重——dinomaly 引擎的阈值取值走同一 resolve 函数。
"""
import pytest

from utils.anomaly_config import DEFAULT_ANOMALY_THRESHOLD, resolve_anomaly_threshold
from utils.rule_based_wrinkle import RuleBasedWrinklePipeline


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("ANOMALY_THRESHOLD", raising=False)


def test_default_when_unset():
    assert resolve_anomaly_threshold() == 0.21


def test_env_override(monkeypatch):
    monkeypatch.setenv("ANOMALY_THRESHOLD", "0.35")
    assert resolve_anomaly_threshold() == 0.35


def test_env_blank_and_spaces(monkeypatch):
    monkeypatch.setenv("ANOMALY_THRESHOLD", "  ")
    assert resolve_anomaly_threshold() == DEFAULT_ANOMALY_THRESHOLD


def test_invalid_falls_back(monkeypatch):
    monkeypatch.setenv("ANOMALY_THRESHOLD", "abc")
    assert resolve_anomaly_threshold() == DEFAULT_ANOMALY_THRESHOLD


def test_out_of_range_falls_back(monkeypatch):
    for bad in ("0", "1", "1.5", "-0.1"):
        monkeypatch.setenv("ANOMALY_THRESHOLD", bad)
        assert resolve_anomaly_threshold() == DEFAULT_ANOMALY_THRESHOLD, bad


def test_explicit_param_beats_env(monkeypatch):
    monkeypatch.setenv("ANOMALY_THRESHOLD", "0.35")
    assert RuleBasedWrinklePipeline(threshold=0.5).threshold == 0.5


def test_pipeline_reads_env(monkeypatch):
    monkeypatch.setenv("ANOMALY_THRESHOLD", "0.4")
    assert RuleBasedWrinklePipeline().threshold == 0.4


def test_pipeline_default_without_env():
    assert RuleBasedWrinklePipeline().threshold == DEFAULT_ANOMALY_THRESHOLD
