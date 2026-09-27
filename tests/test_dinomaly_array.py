"""dinomaly 引擎：process_array 数组直通管线（绕过模型加载，mock 推理）。

沿 tests/test_dinomaly_crop.py 的 object.__new__(DinomalyEngine) 先例。
"""
from pathlib import Path

import cv2
import numpy as np
import pytest

from utils.dinomaly_engine import DinomalyEngine

W, H = 2500, 1000  # "very long" 分支（25 等分，片宽 100）


def _make_engine(monkeypatch) -> DinomalyEngine:
    engine = object.__new__(DinomalyEngine)
    engine.threshold = 0.21
    engine.model_path = "MOCK_WEIGHT"

    def fake_process_pil_image(image):
        arr = np.array(image)
        mean = float(arr.mean()) / 255.0
        return {
            "anomaly_map": np.full((392, 392), mean, dtype=np.float32),
            "sample_score": mean,
            "processed_size": (392, 392),
            "original_size": image.size,
        }

    monkeypatch.setattr(engine, "process_pil_image", fake_process_pil_image)
    return engine


def test_process_array_very_long_matches_file_mode(monkeypatch, tmp_path):
    engine = _make_engine(monkeypatch)
    rng = np.random.default_rng(7)
    gray = rng.integers(0, 256, size=(H, W), dtype=np.uint8)

    pred, heat, metrics = engine.process_array(gray, image_type="very long")
    assert pred.dtype == np.uint8 and pred.shape == (H, W)
    np.testing.assert_array_equal(heat, pred)  # 未上色强度图与预测图同源
    assert metrics["processing_mode"] == "fixed_crop"
    assert metrics["num_crops"] == 25
    assert metrics["model_weight"] == "MOCK_WEIGHT"
    assert "process_id" not in metrics and "files" not in metrics
    assert len(metrics["crop_results"]) == 25

    # 同内容经文件路径模式（mock 同一推理）→ 输出逐项一致
    img_path = tmp_path / "frame.png"
    cv2.imwrite(str(img_path), gray)
    pred_path, heat_path, json_path = engine.process_to_dir(
        str(img_path), str(tmp_path / "out"), "pid_y", image_type="very long")

    got_pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
    np.testing.assert_array_equal(got_pred, pred)

    got_heat = cv2.imread(heat_path)
    from utils.rule_based_wrinkle import cvt2heatmap
    np.testing.assert_array_equal(got_heat, cvt2heatmap(heat))

    import json
    file_json = json.loads(Path(json_path).read_text(encoding="utf-8"))
    for key, value in metrics.items():
        if key == "timestamp":
            continue  # 两次调用时间必然不同
        assert file_json[key] == value, f"字段 {key} 不一致"
    assert file_json["process_id"] == "pid_y"


def test_process_array_gray8_expands_to_rgb(monkeypatch):
    """GRAY8 → RGB 三通道扩展正确性：mock 捕获收到的图验证通道。"""
    engine = _make_engine(monkeypatch)
    seen = {}

    def spy(image):
        arr = np.array(image)
        seen["shape"] = arr.shape
        seen["mode"] = image.mode
        return {
            "anomaly_map": np.zeros((392, 392), dtype=np.float32),
            "sample_score": 0.0,
            "processed_size": (392, 392),
            "original_size": image.size,
        }

    monkeypatch.setattr(engine, "process_pil_image", spy)
    gray = np.full((H, W), 128, dtype=np.uint8)
    engine.process_array(gray, image_type="very long")
    # 每片均为 (H, 片宽, 3) 的 RGB
    assert seen["shape"][0] == H and seen["shape"][2] == 3
    assert seen["mode"] == "RGB"


def test_process_array_square_branch(monkeypatch):
    engine = _make_engine(monkeypatch)
    gray = np.zeros((480, 640), dtype=np.uint8)
    pred, heat, metrics = engine.process_array(gray, image_type="square")
    # 与现状一致：square 分支输出处理尺寸（392×392），不 resize 回原图
    assert pred.shape == (392, 392)
    assert metrics["sample_score"] == 0.0
    assert "image_size" in metrics and "processed_size" in metrics
    assert "crop_results" not in metrics
