"""ANOMALY_CROP_EXCLUDE_DETECT/JUDGE/DISPLAY 环境变量解析测试。

沿 test_anomaly_threshold.py 结构：纯配置层，不加载 torch。
"""
import pytest

from utils.anomaly_config import (
    DEFAULT_CROP_EXCLUDE_DETECT,
    DEFAULT_CROP_EXCLUDE_DISPLAY,
    DEFAULT_CROP_EXCLUDE_JUDGE,
    resolve_crop_exclude_detect,
    resolve_crop_exclude_display,
    resolve_crop_exclude_judge,
)

_VARS = {
    "ANOMALY_CROP_EXCLUDE_DETECT": (resolve_crop_exclude_detect, DEFAULT_CROP_EXCLUDE_DETECT),
    "ANOMALY_CROP_EXCLUDE_JUDGE": (resolve_crop_exclude_judge, DEFAULT_CROP_EXCLUDE_JUDGE),
    "ANOMALY_CROP_EXCLUDE_DISPLAY": (resolve_crop_exclude_display, DEFAULT_CROP_EXCLUDE_DISPLAY),
}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in _VARS:
        monkeypatch.delenv(name, raising=False)


def test_defaults_when_unset():
    assert resolve_crop_exclude_detect() == 0
    assert resolve_crop_exclude_judge() == 2
    assert resolve_crop_exclude_display() == 0


def test_env_override(monkeypatch):
    monkeypatch.setenv("ANOMALY_CROP_EXCLUDE_DETECT", "1")
    monkeypatch.setenv("ANOMALY_CROP_EXCLUDE_JUDGE", "4")
    monkeypatch.setenv("ANOMALY_CROP_EXCLUDE_DISPLAY", "3")
    assert resolve_crop_exclude_detect() == 1
    assert resolve_crop_exclude_judge() == 4
    assert resolve_crop_exclude_display() == 3


@pytest.mark.parametrize("bad", ["abc", "2.5", "-1", "13", "1,2", "  "])
def test_invalid_falls_back(bad, monkeypatch):
    for name, (resolver, default) in _VARS.items():
        monkeypatch.setenv(name, bad)
        assert resolver() == default, (name, bad)


def test_boundary_12_valid_13_invalid(monkeypatch):
    monkeypatch.setenv("ANOMALY_CROP_EXCLUDE_JUDGE", "12")
    assert resolve_crop_exclude_judge() == 12
    monkeypatch.setenv("ANOMALY_CROP_EXCLUDE_JUDGE", "13")
    assert resolve_crop_exclude_judge() == DEFAULT_CROP_EXCLUDE_JUDGE


def test_zero_is_valid(monkeypatch):
    monkeypatch.setenv("ANOMALY_CROP_EXCLUDE_JUDGE", "0")
    assert resolve_crop_exclude_judge() == 0
