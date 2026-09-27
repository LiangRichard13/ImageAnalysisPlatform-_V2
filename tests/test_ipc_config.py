"""utils/ipc/config.py 配置读取单元测试。"""
from utils.ipc.config import IpcConfig, load_ipc_config


class TestIpcConfig:
    def test_defaults(self):
        cfg = IpcConfig()
        assert cfg.mode == "file"
        assert cfg.namespace == r"Local\IAP"
        assert cfg.frame_slots == 8
        assert cfg.result_slots == 8
        assert cfg.heartbeat_timeout_ms == 10000

    def test_invalid_mode_falls_back_to_file(self, monkeypatch):
        monkeypatch.setenv("IPC_MODE", "nonsense")
        cfg = load_ipc_config()
        assert cfg.mode == "file"

    def test_shm_mode(self, monkeypatch):
        monkeypatch.setenv("IPC_MODE", "shm")
        assert load_ipc_config().mode == "shm"

    def test_env_overrides(self, monkeypatch):
        monkeypatch.setenv("IPC_MODE", "shm")
        monkeypatch.setenv("SHM_NAMESPACE", r"Local\IAPX")
        monkeypatch.setenv("SHM_FRAME_SLOTS", "4")
        monkeypatch.setenv("SHM_RESULT_SLOTS", "6")
        monkeypatch.setenv("SHM_HEARTBEAT_TIMEOUT_MS", "2000")
        cfg = load_ipc_config()
        assert cfg.namespace == r"Local\IAPX"
        assert cfg.frame_slots == 4
        assert cfg.result_slots == 6
        assert cfg.heartbeat_timeout_ms == 2000

    def test_bad_int_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("IPC_MODE", "shm")
        monkeypatch.setenv("SHM_FRAME_SLOTS", "not_a_number")
        cfg = load_ipc_config()
        assert cfg.frame_slots == 8
        assert cfg.mode == "shm"
