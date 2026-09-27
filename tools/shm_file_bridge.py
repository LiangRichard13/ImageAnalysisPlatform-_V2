"""文件桥接（Phase 3）：IKapExpert 落图目录 → FRAME ring。

相机端封闭软件零改动；本进程常驻：200ms scandir 轮询 + size 稳定判完整
（弃 ReadDirectoryChangesW：纯 ctypes 需 ~150 行 OVERLAPPED/IOCP 脆弱 FFI，
且写入开始即触发、本就需要 size 二次确认——1fps 源 + 2s 预算下轮询代价可忽略）。

语义：
- 首张成功解码的图决定 CTRL 几何（之后尺寸不符 → .failed/）
- 已入 ring 的文件 os.replace 到 .ingested/（移动失败重试 3 次后 journal 降级）
- 解码失败重试 3 次后 → .failed/
- 持 IAP_PRODUCER_MUTEX（WAIT_ABANDONED = 接管前任，seq 续写不回零）

用法（项目根，analysis_system 环境）：
  python tools/shm_file_bridge.py --watch G:\\TestImg\\2026 --namespace Local\\IAP
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from utils.ipc import shm_ring                            # noqa: E402
from utils.ipc.config import load_ipc_config              # noqa: E402
from utils.ipc.layout import CtrlBlock, obj_name          # noqa: E402

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
MB = 1024 * 1024


def _slot_bytes(width: int, height: int) -> int:
    need = width * height + 64
    return (need + MB - 1) // MB * MB


class Bridge:
    def __init__(self, watch_dir: Path, namespace: str, poll_ms: int = 200,
                 stability_ms: int = 100, max_retries: int = 3,
                 expected_size=None):
        self.watch_dir = watch_dir
        self.namespace = namespace
        self.poll_interval = poll_ms / 1000.0
        self.stability_ms = stability_ms
        self.max_retries = max_retries
        self.expected_size = expected_size  # (w, h)：显式几何，混尺寸目录下确定性建环
        self._ctrl = None
        self._producer = None
        self._evt = None
        self._mutex = None
        self._frame_id = 0
        self._pending = {}     # path -> [size_at_first_sight, retries]
        self._journal = watch_dir / ".bridge_journal.txt"

    def _staging_dirs(self):
        (self.watch_dir / ".ingested").mkdir(exist_ok=True)
        (self.watch_dir / ".failed").mkdir(exist_ok=True)

    def _ensure_ring(self, gray: np.ndarray) -> bool:
        """首次成功解码时创建 CTRL/ring 并抢生产者互斥体。"""
        if self._producer is not None:
            return True
        h, w = gray.shape[:2]
        slots = 8
        slot = _slot_bytes(w, h)
        ctrl, _ = CtrlBlock.create_or_open(
            self.namespace, width=w, height=h, frame_slots=slots,
            frame_slot_bytes=slot, result_slots=slots,
            result_slot_bytes=slot + 8256)
        evt = shm_ring.NamedEvent(obj_name(self.namespace, "EVT_FRAME"))
        mutex = shm_ring.NamedMutex(obj_name(self.namespace, "PRODUCER_MUTEX"))
        producer = shm_ring.FrameRingProducer.create_or_attach(ctrl, self.namespace,
                                                               evt, mutex)
        if not producer.acquire(timeout_ms=5000):
            print(f"BRIDGE_MUTEX_BUSY: {self.namespace} 已有生产者，退出", flush=True)
            producer.close()
            evt.close()
            mutex.close()
            ctrl.close()
            return False
        self._ctrl, self._producer, self._evt, self._mutex = ctrl, producer, evt, mutex
        print(f"BRIDGE_STARTED: {self.namespace} geometry={w}x{h} slots={slots} "
              f"slot_bytes={slot}", flush=True)
        return True

    def _decode(self, path: Path):
        return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    def _reject(self, path: Path, reason: str) -> None:
        try:
            os.replace(str(path), str(self.watch_dir / ".failed" / path.name))
            print(f"BRIDGE_REJECT: {path.name} {reason}", flush=True)
        except OSError as exc:
            self._journal.write_text(f"{path}\t{reason}\t{exc}\n")

    def _ingest(self, path: Path, gray: np.ndarray) -> None:
        self._frame_id += 1
        seq = self._producer.write_frame(gray, frame_id=self._frame_id)
        print(f"BRIDGE_FRAME: {path.name} seq={seq}", flush=True)
        for attempt in range(self.max_retries):
            try:
                os.replace(str(path), str(self.watch_dir / ".ingested" / path.name))
                return
            except OSError:
                time.sleep(0.2)
        self._journal.open("a", encoding="utf-8").write(
            f"{path}\tmoved-failed\t{time.time()}\n")

    def run(self) -> int:
        if not self.watch_dir.is_dir():
            print(f"BRIDGE_ERROR: 监控目录不存在 {self.watch_dir}", flush=True)
            return 2
        self._staging_dirs()
        print(f"BRIDGE_WATCHING: {self.watch_dir}", flush=True)
        try:
            while True:
                self._scan_once()
                time.sleep(self.poll_interval)
        except KeyboardInterrupt:
            print("BRIDGE_STOP", flush=True)
            return 0
        finally:
            for obj in (self._producer, self._evt, self._mutex, self._ctrl):
                if obj is not None:
                    try:
                        obj.close()
                    except Exception:
                        pass

    def _scan_once(self) -> None:
        with os.scandir(self.watch_dir) as entries:
            files = [Path(e.path) for e in entries
                     if e.is_file() and e.name[0] != "."
                     and Path(e.name).suffix.lower() in IMAGE_EXTS]
        for path in files:
            state = self._pending.get(path)
            size = path.stat().st_size
            if state is None:
                self._pending[path] = [size, 0]
                continue  # 下一轮比对 size，判写入完成
            if state[0] != size:
                state[0] = size  # 仍在写入
                continue
            gray = self._decode(path)
            if gray is None:
                state[1] += 1
                if state[1] >= self.max_retries:
                    self._pending.pop(path)
                    self._reject(path, "decode-failed")
                continue
            # 显式几何：建环前即确定尺寸不符 → 直接拒绝（混尺寸目录确定性）
            if (self.expected_size is not None and self._ctrl is None
                    and gray.shape != (self.expected_size[1], self.expected_size[0])):
                self._pending.pop(path)
                self._reject(path, f"size-mismatch {gray.shape[1]}x{gray.shape[0]}")
                continue
            if not self._ensure_ring(gray):
                sys.exit(3)
            w, h = self._ctrl.frame_width, self._ctrl.frame_height
            if gray.shape != (h, w):
                self._pending.pop(path)
                self._reject(path, f"size-mismatch {gray.shape[1]}x{gray.shape[0]}")
                continue
            self._pending.pop(path)
            self._ingest(path, gray)


def main() -> int:
    cfg = load_ipc_config()
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--watch", required=True, help="监控目录（.env ONLINE_PROCESSING_AD_DIR 亦可配后省略）")
    parser.add_argument("--namespace", default=cfg.namespace)
    parser.add_argument("--poll-ms", type=int, default=200)
    parser.add_argument("--stability-ms", type=int, default=100)
    parser.add_argument("--width", type=int, default=None,
                        help="显式帧几何（推荐产线固定：混尺寸目录下首图定几何不确定）")
    parser.add_argument("--height", type=int, default=None)
    args = parser.parse_args()
    expected = (args.width, args.height) if args.width and args.height else None
    watch = Path(args.watch or os.getenv("ONLINE_PROCESSING_AD_DIR", ""))
    return Bridge(watch, args.namespace, args.poll_ms, args.stability_ms,
                  expected_size=expected).run()


if __name__ == "__main__":
    sys.exit(main())
