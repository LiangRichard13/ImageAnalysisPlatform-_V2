"""dinomaly 引擎三级边缘子图排除（detect/judge/display）管线测试。

沿 test_dinomaly_array.py 的 object.__new__ + mock 推理先例；
W,H=2500×1000 → fixed_crop 25 等分、片宽 100。
"""
import cv2
import numpy as np
import pytest
from pathlib import Path

from utils.dinomaly_engine import DinomalyEngine

W, H = 2500, 1000
CROP_W = 100


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("ANOMALY_CROP_EXCLUDE_DETECT", "ANOMALY_CROP_EXCLUDE_JUDGE",
                 "ANOMALY_CROP_EXCLUDE_DISPLAY", "ANOMALY_THRESHOLD"):
        monkeypatch.delenv(name, raising=False)


def _make_engine(monkeypatch):
    engine = object.__new__(DinomalyEngine)
    engine.threshold = 0.21
    engine.model_path = "MOCK_WEIGHT"
    calls = {"n": 0}

    def fake_process_pil_images(images):
        calls["n"] += len(images)
        results = []
        for image in images:
            mean = float(np.array(image).mean()) / 255.0
            results.append({
                "anomaly_map": np.full((392, 392), mean, dtype=np.float32),
                "sample_score": mean,
                "processed_size": (392, 392),
                "original_size": image.size,
            })
        return results

    monkeypatch.setattr(engine, "process_pil_images", fake_process_pil_images)
    return engine, calls


def _strip_gray(values):
    gray = np.zeros((H, W), np.uint8)
    for i, v in enumerate(values):
        gray[:, i * CROP_W:(i + 1) * CROP_W] = v
    return gray


def test_detect_exclude_skips_inference(monkeypatch):
    engine, calls = _make_engine(monkeypatch)
    pred, heat, metrics = engine.process_array(
        _strip_gray([10] * 25), image_type="very long", exclude_detect_crops=2)
    assert calls["n"] == 21  # 边缘 2+2 份未推理
    # display 生效值 = max(detect, 0) = 2，与检出范围一致（k=0 不再收缩）
    assert metrics["num_crops"] == 21
    # 画布保持全尺寸（shm RESULT 协议要求全分辨率 pred/heat），检出窗口外置 0
    assert metrics["crop_results"][0]["position"] == [200, 0, 300, H]
    assert metrics["original_size"] == [W, H]
    assert pred.shape == (H, W)
    assert not pred[:, :200].any() and not pred[:, 2300:].any()
    assert metrics["excluded_edge_crops"] == {"detect": 2, "judge": 2, "display": 2}


def test_judge_exclude_drops_edge_scores(monkeypatch):
    engine, calls = _make_engine(monkeypatch)
    values = [255] + [10] * 24  # 最高分在 crop 0
    pred, heat, metrics = engine.process_array(
        _strip_gray(values), image_type="very long", exclude_edge_crops=4)
    assert metrics["sample_score"] == pytest.approx(10 / 255)  # 边缘高分被剔除
    assert metrics["excluded_edge_crops"] == {"detect": 0, "judge": 4, "display": 0}
    assert metrics["num_crops"] == 25  # display=0 不裁窄
    assert pred.shape == (H, W)


def test_judge_relative_to_detect(monkeypatch):
    # detect=1（检出 crop1..23）、judge=1 → rel_judge=0：检出边缘 crop1 的高分参与判定
    values = [10] * 25
    values[1] = 255
    engine, calls = _make_engine(monkeypatch)
    pred, heat, metrics = engine.process_array(
        _strip_gray(values), image_type="very long",
        exclude_edge_crops=1, exclude_detect_crops=1)
    assert calls["n"] == 23
    assert metrics["sample_score"] == pytest.approx(1.0)
    # display 生效值 = max(1, 0) = 1，与检出范围一致（rel k=0 不再收缩）
    assert metrics["num_crops"] == 23
    assert metrics["original_size"] == [W, H]


def test_display_exclude_zeroes_margins(monkeypatch, tmp_path):
    engine, calls = _make_engine(monkeypatch)
    pred, heat, metrics = engine.process_array(
        _strip_gray([10] * 25), image_type="very long", exclude_display_crops=3)
    x_lo, x_hi = 3 * CROP_W, 22 * CROP_W  # 保留区 crop3..21
    # 画布保持全尺寸（shm RESULT 协议要求全分辨率 pred/heat），窗口外置 0
    assert pred.shape == (H, W)
    assert metrics["original_size"] == [W, H]
    assert metrics["num_crops"] == 19
    assert metrics["crop_results"][0]["position"] == [x_lo, 0, x_lo + CROP_W, H]
    assert metrics["crop_results"][-1]["position"] == [x_hi - CROP_W, 0, x_hi, H]
    assert not pred[:, :x_lo].any() and not pred[:, x_hi:].any()
    assert metrics["excluded_edge_crops"] == {"detect": 0, "judge": 2, "display": 3}

    # 落盘模式与数组模式同口径：png 全尺寸、JSON 字段一致
    img_path = tmp_path / "frame.png"
    cv2.imwrite(str(img_path), _strip_gray([10] * 25))
    pred_path, heat_path, json_path = engine.process_to_dir(
        str(img_path), str(tmp_path / "out"), "pid_x", image_type="very long",
        exclude_display_crops=3)
    import json
    got = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
    assert got.shape == (H, W)
    assert not got[:, :x_lo].any() and not got[:, x_hi:].any()
    file_json = json.loads(Path(json_path).read_text(encoding="utf-8"))
    for key, value in metrics.items():
        if key == "timestamp":
            continue
        assert file_json[key] == value, f"字段 {key} 不一致"


def test_detect_and_display_combined(monkeypatch):
    engine, calls = _make_engine(monkeypatch)
    pred, heat, metrics = engine.process_array(
        _strip_gray([10] * 25), image_type="very long",
        exclude_detect_crops=1, exclude_display_crops=3)
    assert calls["n"] == 23
    # display 生效值 = 3：k = 3-1 = 2，检出列表（crop1..23）两端再收 2 份 → crop3..21
    assert metrics["num_crops"] == 19
    assert metrics["original_size"] == [W, H]
    assert pred.shape == (H, W)
    assert not pred[:, :300].any() and not pred[:, 2200:].any()


def test_display_exclude_env_full_canvas(monkeypatch):
    # .env 部署口径（DISPLAY=2）经环境变量生效：输出仍全尺寸，
    # write_result 形状守卫（pred.shape == frame 形状）不再丢帧
    monkeypatch.setenv("ANOMALY_CROP_EXCLUDE_DISPLAY", "2")
    engine, calls = _make_engine(monkeypatch)
    pred, heat, metrics = engine.process_array(
        _strip_gray([10] * 25), image_type="very long")
    assert pred.shape == (H, W)
    assert heat.shape == (H, W)
    assert metrics["original_size"] == [W, H]
    assert not pred[:, :200].any() and not pred[:, 2300:].any()
    assert metrics["excluded_edge_crops"]["display"] == 2


def test_square_ignores_exclusions(monkeypatch):
    engine, calls = _make_engine(monkeypatch)
    gray = np.zeros((480, 640), np.uint8)
    pred, heat, metrics = engine.process_array(
        gray, image_type="square",
        exclude_edge_crops=3, exclude_detect_crops=3, exclude_display_crops=3)
    assert pred.shape == (392, 392)
    assert "crop_results" not in metrics
    assert "excluded_edge_crops" not in metrics
    assert calls["n"] == 1


def test_adaptive_honors_judge(monkeypatch):
    # 行为变更锚点：adaptive 此前强制排除 0，现尊重配置值
    gray = np.zeros((1000, 4000), np.uint8)
    gray[:, :1000] = 255  # crop0 高分
    engine, calls = _make_engine(monkeypatch)
    pred, heat, metrics = engine.process_array(
        gray, image_type="other", exclude_edge_crops=1)
    assert metrics["processing_mode"] == "adaptive_crop"
    assert metrics["num_crops"] == 4
    assert metrics["sample_score"] < 1.0  # crop0/3 被剔除出判定
