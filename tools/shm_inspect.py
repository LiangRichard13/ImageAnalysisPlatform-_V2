"""共享内存链路诊断工具（联调用）。

用法（项目根运行）：
  python tools/shm_inspect.py status  [--namespace Local\\IAP]
  python tools/shm_inspect.py frame   [--seq latest|N] [--out x.png]
  python tools/shm_inspect.py result  [--seq latest|N] [--out 前缀]
  python tools/shm_inspect.py crc     [--count N]
  python tools/shm_inspect.py watch   [--interval 1]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.ipc import shm_ring                      # noqa: E402
from utils.ipc.config import load_ipc_config        # noqa: E402
from utils.ipc.layout import CtrlBlock, obj_name    # noqa: E402


def _attach(args):
    cfg = load_ipc_config()
    ns = args.namespace or cfg.namespace
    ctrl = CtrlBlock.attach(ns)
    evt = shm_ring.NamedEvent(obj_name(ns, "EVT_FRAME"))
    return ctrl, evt, ns


def cmd_status(args) -> int:
    ctrl, _evt, ns = _attach(args)
    try:
        print(f"namespace        : {ns}")
        print(f"frame geometry   : {ctrl.frame_width}x{ctrl.frame_height} fmt={ctrl.pixel_format}")
        print(f"frame ring       : {ctrl.frame_slots} slots x {ctrl.frame_slot_bytes} B")
        print(f"result ring      : {ctrl.result_slots} slots x {ctrl.result_slot_bytes} B")
        print(f"frame_write_seq  : {ctrl.frame_write_seq}")
        print(f"result_write_seq : {ctrl.result_write_seq}")
        print(f"producer age_ms  : {ctrl.heartbeat_age_ms('producer')}"
              f"  ({'在线' if ctrl.producer_alive(10000) else '离线'})")
        print(f"analyzer age_ms  : {ctrl.heartbeat_age_ms('analyzer')}")
        print(f"archiver age_ms  : {ctrl.heartbeat_age_ms('archiver')}")
        print(f"twin age_ms      : {ctrl.heartbeat_age_ms('twin')}")
        print(f"dropped_overwrite: {ctrl.dropped_by_overwrite}")
        print(f"dropped_by_lag   : {ctrl.dropped_by_lag}")
        return 0
    finally:
        ctrl.close()


def cmd_frame(args) -> int:
    import cv2
    import numpy as np
    ctrl, evt, ns = _attach(args)
    consumer = shm_ring.FrameRingConsumer.attach(ctrl, ns, evt)
    try:
        got = consumer.fetch_latest(last_seq=0, wait_ms=500)
        if got is None:
            print("无可用帧（frame_write_seq=0 或最新槽不可信）")
            return 1
        seq, frame_id, ts_ns, gray = got
        print(f"seq={seq} frame_id={frame_id} ts_ns={ts_ns} shape={gray.shape}")
        if args.seq and args.seq != "latest":
            want = int(args.seq)
            if want != seq:
                print(f"警告：请求 seq={want}，latest-wins 语义下读到最新 {seq}")
        if args.out:
            cv2.imwrite(args.out, gray)
            print(f"已导出 {args.out}")
        return 0
    finally:
        consumer.close()
        ctrl.close()


def cmd_result(args) -> int:
    import cv2
    ctrl, evt, ns = _attach(args)
    consumer = shm_ring.ResultRingConsumer.attach(ctrl, ns, evt)
    try:
        got = consumer.fetch_latest(last_seq=0, wait_ms=500)
        if got is None:
            print("无可用结果（result_write_seq=0）")
            return 1
        seq, frame_seq, payload, pred, heat = got
        print(f"seq={seq} frame_seq={frame_seq}")
        print(json.dumps(payload, ensure_ascii=False, indent=2)[:2000])
        if args.out:
            cv2.imwrite(f"{args.out}_pred.png", pred)
            cv2.imwrite(f"{args.out}_heat.png", heat)
            print(f"已导出 {args.out}_pred.png / {args.out}_heat.png")
        return 0
    finally:
        consumer.close()
        ctrl.close()


def cmd_crc(args) -> int:
    """对最近 N 个可寻址 seq 抽查 FRAME 槽 CRC。"""
    import numpy as np

    from utils.ipc.layout import FRAME_HDR, crc32 as crc_fn

    ctrl, evt, ns = _attach(args)
    ring = shm_ring.FrameRingConsumer.attach(ctrl, ns, evt)
    try:
        latest = ctrl.frame_write_seq
        if latest == 0:
            print("无帧")
            return 1
        n = min(args.count, ctrl.frame_slots)
        bad = ok = 0
        for i in range(n):
            s = latest - i
            if s < 1:
                break
            off = (s % ctrl.frame_slots) * ctrl.frame_slot_bytes
            seq, _ts, _fid, crc, data_bytes, _st = FRAME_HDR.unpack_from(
                ring._view[off:off + 64])
            if seq != s:
                print(f"seq {s}: 槽已被覆盖（当前 seq={seq}）")
                continue
            data = np.frombuffer(ring._view, dtype=np.uint8,
                                 count=data_bytes, offset=off + 64).copy()
            if crc_fn(data.tobytes()) == crc:
                ok += 1
            else:
                bad += 1
                print(f"seq {s}: CRC 不一致！")
        print(f"抽查完成：OK={ok} BAD={bad}")
        return 1 if bad else 0
    finally:
        ring.close()
        ctrl.close()


def cmd_watch(args) -> int:
    import time
    while True:
        try:
            cmd_status(args)
        except Exception as exc:
            print(f"[watch] {exc}")
        print("-" * 40)
        time.sleep(args.interval)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--namespace", default=None, help="默认取 .env SHM_NAMESPACE")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("status", "frame", "result", "crc", "watch"):
        p = sub.add_parser(name)
        p.add_argument("--namespace", default=None)
        if name == "frame":
            p.add_argument("--seq", default="latest")
            p.add_argument("--out", default=None)
        if name == "result":
            p.add_argument("--seq", default="latest")
            p.add_argument("--out", default=None)
        if name == "crc":
            p.add_argument("--count", type=int, default=8)
        if name == "watch":
            p.add_argument("--interval", type=float, default=1.0)
        p.set_defaults(func=globals()[f"cmd_{name}"])
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
