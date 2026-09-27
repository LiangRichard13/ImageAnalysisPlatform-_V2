"""A5 客户端数组直通 + A6 归档 worker 测试。"""
import json
import time

import cv2
import numpy as np

from utils.anomaly_detection_client import AnomalyDetectionClient, ENGINE_RULE_BASED
from utils.archive_worker import ArchiveItem, ArchiveWorker
from utils.rule_based_wrinkle import cvt2heatmap

IMG_DIR = __import__("pathlib").Path(__file__).resolve().parent.parent / "test"


def _real_gray() -> np.ndarray:
    images = sorted(IMG_DIR.glob("gsy_*.jpg"))
    gray = cv2.imread(str(images[0]), cv2.IMREAD_GRAYSCALE)
    assert gray is not None
    return gray


class TestProcessFromArray:
    def test_returns_arrays_with_process_id_and_files(self):
        client = AnomalyDetectionClient()
        gray = _real_gray()
        pred, heat, metrics = client.process_from_array(gray, ENGINE_RULE_BASED, frame_id=42)
        assert pred.shape == gray.shape and pred.dtype == np.uint8
        np.testing.assert_array_equal(heat, pred)
        assert metrics["process_id"] == client.process_id
        assert metrics["files"] == {
            "anomaly_map": f"{client.process_id}.png",
            "heatmap": f"{client.process_id}_heatmap.png",
        }
        assert metrics["anomaly_level"] in ("很可能正常", "很可能异常")


class TestArchiveWorker:
    def _item(self, pid: str, frame_id: int, image_type: str = "other"):
        gray = np.full((16, 32), frame_id & 0xFF, dtype=np.uint8)
        pred = np.full((16, 32), 128, dtype=np.uint8)
        heat = np.full((16, 32), 200, dtype=np.uint8)
        payload = {"process_id": pid, "sample_score": 0.1,
                   "files": {"anomaly_map": f"{pid}.png", "heatmap": f"{pid}_heatmap.png"}}
        return ArchiveItem(frame_seq=frame_id, frame_id=frame_id, process_id=pid,
                           image_type=image_type, gray=gray, pred_u8=pred,
                           heat_u8=heat, json_dict=payload)

    def test_archive_layout_matches_current_structure(self, tmp_path):
        api_root = tmp_path / "anomaly_api"
        worker = ArchiveWorker(api_root)
        worker.start()
        try:
            worker.submit(self._item("pid_a", 7, image_type="very long"))
            deadline = time.monotonic() + 5
            out_dir = api_root / "output" / "pid_a"
            while time.monotonic() < deadline and not (out_dir / "pid_a.json").exists():
                time.sleep(0.05)
        finally:
            worker.stop()

        # input/{id}/{frame_id}.png + request.json（字段同 _archive_input）
        input_dir = api_root / "input" / "pid_a"
        assert (input_dir / "7.png").exists()
        request = json.loads((input_dir / "request.json").read_text(encoding="utf-8"))
        assert request["process_id"] == "pid_a"
        assert request["image_type"] == "very long"
        assert request["source_image"] == "7.png"
        assert "timestamp" in request

        # output/{id}/{id}.png + _heatmap.png + .json（三件套同现行结构）
        assert (out_dir / "pid_a.png").exists()
        assert (out_dir / "pid_a_heatmap.png").exists()
        saved = json.loads((out_dir / "pid_a.json").read_text(encoding="utf-8"))
        assert saved["process_id"] == "pid_a"
        heat_png = cv2.imread(str(out_dir / "pid_a_heatmap.png"))
        np.testing.assert_array_equal(heat_png, cvt2heatmap(np.full((16, 32), 200, np.uint8)))

    def test_overflow_drops_oldest(self, tmp_path):
        api_root = tmp_path / "anomaly_api"
        worker = ArchiveWorker(api_root, capacity=4)
        worker.start()
        try:
            for i in range(8):  # pid_0..pid_7，按提交顺序
                worker.submit(self._item(f"pid_{i}", i))
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline and worker.archived_count < 8:
                time.sleep(0.05)
        finally:
            worker.stop()

        assert worker.dropped_count == 4
        archived = sorted(p.name for p in (api_root / "output").iterdir())
        # 最旧的 pid_0..pid_3 被丢，pid_4..pid_7 落盘
        assert archived == [f"pid_{i}" for i in range(4, 8)]
