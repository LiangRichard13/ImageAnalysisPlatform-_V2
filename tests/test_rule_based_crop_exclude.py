"""规则引擎三级边缘子图排除（detect/judge/display）测试。

H×W=1000×8000 → DISPLAY_NUM_CROPS=25、片宽 320；全幅轮廓检测，
检测排除 = cx 段后过滤，展示排除 = 画布物理裁窄。
"""
import cv2
import numpy as np
import pytest

from utils.rule_based_wrinkle import RuleBasedWrinklePipeline

H, W = 1000, 8000
CROP = -(-W // 25)  # 320
BG = 128
STRIPE = 80


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("ANOMALY_CROP_EXCLUDE_DETECT", "ANOMALY_CROP_EXCLUDE_JUDGE",
                 "ANOMALY_CROP_EXCLUDE_DISPLAY", "ANOMALY_THRESHOLD"):
        monkeypatch.delenv(name, raising=False)


def _stripe(img, center_x, width=60):
    x1 = center_x - width // 2
    cv2.rectangle(img, (x1, 0), (x1 + width, H - 1), STRIPE, thickness=-1)
    return img


def _pipeline():
    return RuleBasedWrinklePipeline(threshold=0.21)


def test_default_config_regression_anchor():
    # 默认 0/2/0 ≡ 现行为：全宽输出、25 份、判定剔除前后各 2 份
    img = _stripe(np.full((H, W), BG, dtype=np.uint8), 4000)
    pred, heat, metrics = _pipeline().process_array(img)
    assert pred.shape == (H, W)
    assert metrics["original_size"] == [W, H]
    assert metrics["num_crops"] == 25
    assert metrics["excluded_edge_crops"] == {"detect": 0, "judge": 2, "display": 0}
    assert metrics["sample_score"] > 0  # crop12 在判定范围内


def test_detect_exclude_filters_wrinkles():
    img = np.full((H, W), BG, dtype=np.uint8)
    _stripe(img, 160)   # crop 0
    _stripe(img, 4000)  # crop 12
    pred, heat, metrics = _pipeline().process_array(img, exclude_detect_crops=1)
    cxs = [w["geometry"]["center_x"] for w in metrics["wrinkles"]]
    assert all(cx >= CROP for cx in cxs), "crop0 褶皱应被检测排除丢弃"
    assert any(abs(cx - 4000) < 60 for cx in cxs)


def test_display_exclude_crops_output():
    img = _stripe(np.full((H, W), BG, dtype=np.uint8), 4000)
    pred, heat, metrics = _pipeline().process_array(
        img, exclude_edge_crops=0, exclude_display_crops=2)
    x_lo, x_hi = 2 * CROP, min(23 * CROP, W)  # 640..7360
    assert pred.shape == (H, x_hi - x_lo)
    assert metrics["original_size"] == [x_hi - x_lo, H]
    assert metrics["num_crops"] == 21
    assert metrics["crop_results"][0]["position"] == [0, 0, CROP, H]
    assert metrics["crop_results"][-1]["position"][2] == x_hi - x_lo
    assert metrics["excluded_edge_crops"] == {"detect": 0, "judge": 0, "display": 2}
    assert metrics["sample_score"] > 0  # crop12 仍参与判定
    # wrinkles 清单保持检测范围口径，不随展示排除收缩
    assert any(abs(w["geometry"]["center_x"] - 4000) < 60 for w in metrics["wrinkles"])


def test_judge_exclude_via_env(monkeypatch):
    monkeypatch.setenv("ANOMALY_CROP_EXCLUDE_JUDGE", "4")
    img = _stripe(np.full((H, W), BG, dtype=np.uint8), 1120)  # crop 3
    pred, heat, metrics = _pipeline().process_array(img)
    assert metrics["sample_score"] == 0.0  # crop3 落在判断范围 [4..20] 之外
    assert metrics["excluded_edge_crops"]["judge"] == 4
    # 对照：显式 judge=2 时 crop3 计分
    _, _, metrics2 = _pipeline().process_array(img, exclude_edge_crops=2)
    assert metrics2["sample_score"] > 0


def test_overall_score_edge_param():
    pipeline = _pipeline()
    wrinkles = [{"cx": 160.0, "confidence": 0.9}]  # crop 0
    assert pipeline._overall_score(wrinkles, W) == 0.0  # 默认 edge=2
    assert pipeline._overall_score(wrinkles, W, edge_crops=0) == 0.9
