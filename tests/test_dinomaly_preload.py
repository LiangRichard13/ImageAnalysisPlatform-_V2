"""dinomaly 权重预热助手单元测试（mock，不碰真权重）。"""
import time

import pytest

from utils import dinomaly_engine as de


@pytest.fixture()
def fresh_engine_state(monkeypatch):
    """隔离模块级单例/预热线程状态，测试后还原。"""
    monkeypatch.setattr(de, "_ENGINE", None)
    monkeypatch.setattr(de, "_PRELOAD_THREAD", None)
    yield de


class TestPreloadAsync:
    def test_starts_thread_and_calls_get_once(self, fresh_engine_state, monkeypatch):
        calls = []

        def fake_get():
            time.sleep(0.05)  # 模拟加载耗时，验证幂等窗口
            calls.append(1)
            return object()

        monkeypatch.setattr(de, "get_dinomaly_engine", fake_get)
        t = de.preload_dinomaly_engine_async()
        assert t is not None, "未加载时应起预热线程"
        # 预热窗口内重复调用：幂等，不起第二个线程
        assert de.preload_dinomaly_engine_async() is None
        t.join(timeout=10)
        assert not t.is_alive()
        assert calls == [1]

    def test_skips_when_already_loaded(self, fresh_engine_state, monkeypatch):
        monkeypatch.setattr(de, "_ENGINE", object())  # 已加载占位
        called = []

        def fake_get():
            called.append(1)
            return object()

        monkeypatch.setattr(de, "get_dinomaly_engine", fake_get)
        assert de.preload_dinomaly_engine_async() is None
        time.sleep(0.1)
        assert called == [], "已加载时不得再调 get_dinomaly_engine"

    def test_error_swallowed_and_state_clean(self, fresh_engine_state, monkeypatch):
        def boom():
            raise RuntimeError("no torch")

        monkeypatch.setattr(de, "get_dinomaly_engine", boom)
        t = de.preload_dinomaly_engine_async()
        assert t is not None
        t.join(timeout=10)  # 异常不得逃出线程
        assert de._ENGINE is None, "失败后单例保持 None，首次使用时可重试"
        assert de.preload_dinomaly_engine_async() is not None, "失败后允许再次预热"
