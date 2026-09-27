"""IAP 共享内存 IPC 包：协议实现见 docs/ipc/SHM_PROTOCOL.md。"""
from utils.ipc.config import IpcConfig, load_ipc_config

__all__ = ["IpcConfig", "load_ipc_config"]
