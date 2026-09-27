"""共享内存链路基准：双进程 1fps×32MB 吞吐/时延/完整性验证。

用法（项目根，analysis_system 环境）：
  python tools/shm_bench.py --both                      # 默认 31901×1000×100帧×1fps
  python tools/shm_bench.py --both --frames 5 --fps 20 --size 320 200 --slots 4  # 快速模式
  python tools/shm_bench.py --producer ...              # 仅生产者（供外部消费者）
  python tools/shm_bench.py ...                         # 仅消费者（attach 已有环）

帧内容：全帧填 (seq & 0xFF)，首 8 字节嵌 seq（小端）——消费端逐字节校验。
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from utils.ipc import shm_ring                            # noqa: E402
from utils.ipc.config import load_ipc_config              # noqa: E402
from utils.ipc.layout import (CtrlBlock, obj_name,        # noqa: E402
                              protocol_frame_slot_bytes, protocol_result_slot_bytes)

DEFAULT_W, DEFAULT_H = 31901, 1000


def _make_frame(seq: int, w: int, h: int) -> np.ndarray:
    frame = np.full((h, w), seq & 0xFF, dtype=np.uint8)
    flat = frame.reshape(-1)
    flat[:8] = np.frombuffer(seq.to_bytes(8, "little"), dtype=np.uint8)
    return frame


def run_producer(args) -> int:
    ctrl, _ = CtrlBlock.create_or_open(
        args.namespace, width=args.width, height=args.height,
        frame_slots=args.slots,
        frame_slot_bytes=protocol_frame_slot_bytes(args.width, args.height),
        result_slots=args.slots,
        result_slot_bytes=protocol_result_slot_bytes(args.width, args.height))
    evt = shm_ring.NamedEvent(obj_name(args.namespace, "EVT_FRAME"))
    mutex = shm_ring.NamedMutex(obj_name(args.namespace, "PRODUCER_MUTEX"))
    producer = shm_ring.FrameRingProducer.create_or_attach(ctrl, args.namespace, evt, mutex)
    try:
        if not producer.acquire(timeout_ms=5000):
            print("PRODUCER_MUTEX 被占用，退出")
            return 2
        interval = 1.0 / args.fps
        for seq in range(1, args.frames + 1):
            producer.write_frame(_make_frame(seq, args.width, args.height), frame_id=seq)
            if args.fps > 0:
                time.sleep(interval)
        print(f"PRODUCER_DONE frames={args.frames}")
        return 0
    finally:
        producer.release()
        producer.close()
        evt.close()
        mutex.close()
        ctrl.close()


def run_consumer(args) -> int:
    deadline = time.monotonic() + args.timeout
    ctrl = None
    while ctrl is None:
        try:
            ctrl = CtrlBlock.attach(args.namespace)
        except Exception:
            if time.monotonic() > deadline:
                print("等待生产者超时：CTRL 不存在")
                return 2
            time.sleep(0.2)
    evt = shm_ring.NamedEvent(obj_name(args.namespace, "EVT_FRAME"))
    consumer = shm_ring.FrameRingConsumer.attach(ctrl, args.namespace, evt)
    try:
        last = 0
        seen = bad = 0
        latencies = []
        t_end = time.monotonic() + args.timeout
        while time.monotonic() < t_end and last < args.frames:
            got = consumer.fetch_latest(last_seq=last, wait_ms=200)
            if got is None:
                continue
            seq, _frame_id, ts_ns, gray = got
            if seq <= last:
                continue
            last = seq
            seen += 1
            flat = gray.reshape(-1)
            ok = (int.from_bytes(bytes(flat[:8]), "little") == seq
                  and bool((flat[8:] == (seq & 0xFF)).all()))
            if ok:
                latencies.append((time.time_ns() - ts_ns) / 1e6)
            else:
                bad += 1
                print(f"seq {seq}: 内容校验失败")
        if latencies:
            latencies.sort()
            p50 = latencies[len(latencies) // 2]
            p95 = latencies[int(len(latencies) * 0.95) - 1] if len(latencies) >= 20 else latencies[-1]
            print(f"消费 {seen} 帧（最新 seq={last}），BAD={bad}，"
                  f"时延 ms p50={p50:.1f} p95={p95:.1f} max={latencies[-1]:.1f}")
        else:
            print(f"消费 0 帧，BAD={bad}")
        return 0 if bad == 0 and seen > 0 else 1
    finally:
        consumer.close()
        evt.close()
        ctrl.close()


def main() -> int:
    cfg = load_ipc_config()
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--producer", action="store_true", help="仅运行生产者")
    parser.add_argument("--both", action="store_true", help="子进程生产者 + 本进程消费者")
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--fps", type=float, default=1.0, help="0=全速")
    parser.add_argument("--size", type=int, nargs=2, default=[DEFAULT_W, DEFAULT_H],
                        metavar=("W", "H"))
    parser.add_argument("--slots", type=int, default=8)
    parser.add_argument("--namespace", default=cfg.namespace)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    args.width, args.height = args.size
    # 消费预算默认覆盖整个生产期 + 启动/收尾余量（帧数/fps + 60s）
    if args.timeout == 30.0:
        args.timeout = args.frames / max(args.fps, 0.01) + 60

    if args.producer:
        return run_producer(args)
    if args.both:
        proc = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--producer",
             "--frames", str(args.frames), "--fps", str(args.fps),
             "--size", str(args.width), str(args.height),
             "--slots", str(args.slots), "--namespace", args.namespace],
            stdout=sys.stdout, stderr=sys.stderr)
        try:
            return run_consumer(args)
        finally:
            # 消费者可先于生产者退出（睡眠耗尽 monotonic 预算等）：等不完就杀，不崩
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
    return run_consumer(args)


if __name__ == "__main__":
    sys.exit(main())
