"""IPC 配置（.env 读取，沿 utils 客户端 load_dotenv 显式路径先例）。"""
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IpcConfig:
    mode: str = "file"                    # shm | file，非法值回退 file
    namespace: str = r"Local\IAP"
    frame_slots: int = 8
    result_slots: int = 8
    heartbeat_timeout_ms: int = 10000


def _get_int(env: dict, key: str, default: int) -> int:
    raw = env.get(key, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("配置 %s=%r 无效，使用默认值 %s", key, raw, default)
        return default


def load_ipc_config() -> IpcConfig:
    """读 .env + 环境变量（环境变量优先，load_dotenv 默认不覆盖已有 env）。"""
    project_root = Path(__file__).resolve().parent.parent.parent
    load_dotenv(project_root / ".env")

    mode_raw = os.getenv("IPC_MODE", "file").strip().lower()
    if mode_raw not in ("shm", "file"):
        logger.warning("IPC_MODE=%r 非法（shm|file），回退 file", mode_raw)
        mode_raw = "file"

    return IpcConfig(
        mode=mode_raw,
        namespace=os.getenv("SHM_NAMESPACE", r"Local\IAP").strip() or r"Local\IAP",
        frame_slots=_get_int(os.environ, "SHM_FRAME_SLOTS", 8),
        result_slots=_get_int(os.environ, "SHM_RESULT_SLOTS", 8),
        heartbeat_timeout_ms=_get_int(os.environ, "SHM_HEARTBEAT_TIMEOUT_MS", 10000),
    )
