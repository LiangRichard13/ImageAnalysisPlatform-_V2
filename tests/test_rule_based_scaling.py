"""规则引擎参数自适应缩放测试

业务图源为产线原图 31901x1000，规则参数（min_height=950 等）按该分辨率标定。
缩小图（如 500x500 测试图）上所有绝对像素阈值必须按高度比例缩放，
否则任何轮廓都达不到 min_height，产生 0 wrinkles 的空结果。
"""

import numpy as np
import cv2
import pytest

from utils.rule_based_wrinkle import RuleBasedWrinkleCore, RuleBasedWrinklePipeline

BG = 128
STRIPE = 80  # 暗于背景的竖向褶皱条


def make_image(height: int, width: int, stripe_center_x: int, stripe_width: int) -> np.ndarray:
    img = np.full((height, width), BG, dtype=np.uint8)
    x1 = stripe_center_x - stripe_width // 2
    cv2.rectangle(img, (x1, 0), (x1 + stripe_width, height - 1), STRIPE, thickness=-1)
    return img


def test_detects_wrinkle_on_reference_resolution():
    # 基准分辨率（高 1000）：现有参数直接适用，条在检测 ROI（x>4000）内
    img = make_image(1000, 8000, stripe_center_x=6000, stripe_width=60)
    wrinkles = RuleBasedWrinkleCore().detect(img)
    assert len(wrinkles) >= 1


def test_detects_wrinkle_on_half_scale_image():
    # 同一场景缩到高 500：修复前 min_height=950 一票否决 -> 0 wrinkles
    img = make_image(500, 4000, stripe_center_x=3000, stripe_width=30)
    wrinkles = RuleBasedWrinkleCore().detect(img)
    assert len(wrinkles) >= 1


def test_detects_wrinkle_on_small_square_image():
    # Richard 的场景：500x500 方图 + 贯穿全高的竖条（gsy_15.jpg 形态）
    img = make_image(500, 500, stripe_center_x=250, stripe_width=30)
    wrinkles = RuleBasedWrinkleCore().detect(img)
    assert len(wrinkles) >= 1


def test_no_false_positive_on_plain_small_image():
    # 无褶皱的缩小图必须保持 0 检出，防止缩放把噪声放行
    img = np.full((500, 500), BG, dtype=np.uint8)
    rng = np.random.default_rng(42)
    noisy = np.clip(img.astype(np.int16) + rng.integers(-6, 7, img.shape), 0, 255).astype(np.uint8)
    wrinkles = RuleBasedWrinkleCore().detect(noisy)
    assert len(wrinkles) == 0


def test_detects_wrinkle_in_edge_region():
    # 全图检测：产线原图边缘（原 ROI x<4000 之外）的褶皱也必须检出
    img = make_image(1000, 8000, stripe_center_x=500, stripe_width=60)
    wrinkles = RuleBasedWrinkleCore().detect(img)
    assert len(wrinkles) >= 1


def test_crop_results_cover_full_width():
    # JSON 展示的 25 份划分应覆盖全图宽度（与全图检测语义一致）
    pipeline = RuleBasedWrinklePipeline()
    img = make_image(1000, 8000, stripe_center_x=4000, stripe_width=60)
    wrinkles = RuleBasedWrinkleCore().detect(img)
    crops = pipeline._build_crop_results(wrinkles, img_width=8000, img_height=1000)
    assert crops[0]["position"][0] == 0
    assert crops[-1]["position"][2] == 8000
    assert len(crops) == 25


def test_overall_score_excludes_edge_crops():
    # 全图判定只取中间 21 片（crop 2..22）：边缘褶皱仍检出，但不驱动全图分数
    # img_width=8000 -> crop_size=320：crop 0=[0,320) crop 1=[320,640) crop 24=[7680,8000)
    pipeline = RuleBasedWrinklePipeline()
    wrinkles = [
        {"cx": 100, "confidence": 0.9},   # crop 0（边缘，剔除）
        {"cx": 500, "confidence": 0.95},  # crop 1（边缘，剔除）
        {"cx": 4000, "confidence": 0.7},  # crop 12（中间）
        {"cx": 7900, "confidence": 0.8},  # crop 24（边缘，剔除）
    ]
    score = pipeline._overall_score(wrinkles, img_width=8000)
    assert score == 0.7


def test_overall_score_empty_when_only_edge_wrinkles():
    # 只有边缘褶皱时全图分数为 0（中间区域无异常）
    pipeline = RuleBasedWrinklePipeline()
    wrinkles = [{"cx": 100, "confidence": 0.9}, {"cx": 7900, "confidence": 0.95}]
    assert pipeline._overall_score(wrinkles, img_width=8000) == 0.0


def test_voltage_and_level_agree_at_threshold():
    # 阈值边界必须自洽：score==threshold 归正常且 0V（不再出现"判异常但 0V"）
    pipeline = RuleBasedWrinklePipeline(threshold=0.21)
    assert pipeline._get_anomaly_level(0.21) == "很可能正常"
    assert pipeline._calculate_analog_voltage(0.21) == 0.0
    assert pipeline._get_anomaly_level(0.22) == "很可能异常"
    assert pipeline._calculate_analog_voltage(0.22) > 0.0


def test_cluster_default_l3_matches_clustering():
    from utils.rule_based_wrinkle import Cluster, SinglePassClustering
    import inspect
    d_default = inspect.signature(Cluster.distance).parameters["l3"].default
    c_default = inspect.signature(SinglePassClustering.__init__).parameters["l3"].default
    assert d_default == c_default


def test_geometry_scales_with_height():
    # 检出的几何量应随缩放比例一致（半分辨率图的条宽约为基准的一半）
    img = make_image(500, 4000, stripe_center_x=3000, stripe_width=30)
    wrinkles = RuleBasedWrinkleCore().detect(img)
    assert wrinkles, "expected a wrinkle on half-scale image"
    widths = [w["width"] for w in wrinkles]
    assert all(10 <= w <= 60 for w in widths), f"stripe width out of scaled window: {widths}"
