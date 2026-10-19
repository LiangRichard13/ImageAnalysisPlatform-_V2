"""rule_based 引擎：process_array 与 process_to_dir 输出一致性（文件模式回归锚点）。"""
from pathlib import Path

import cv2
import numpy as np
import pytest

from utils.rule_based_wrinkle import RuleBasedWrinklePipeline, cvt2heatmap

IMG_DIR = Path(__file__).resolve().parent.parent / "test_images"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    # 部署 .env（ANOMALY_CROP_EXCLUDE_* 等）会改变输出尺寸/判定，断言钉在默认
    # 配置上。setenv 而非 delenv：delenv 会让 AnomalyDetectionClient 的 load_dotenv
    # 从部署 .env 重新注入；显式设为代码默认值（0/2/0、0.21），load_dotenv 不覆盖已存在变量
    defaults = {"ANOMALY_CROP_EXCLUDE_DETECT": "0",
                "ANOMALY_CROP_EXCLUDE_JUDGE": "2",
                "ANOMALY_CROP_EXCLUDE_DISPLAY": "0",
                "ANOMALY_THRESHOLD": "0.21"}
    for name, value in defaults.items():
        monkeypatch.setenv(name, value)


def _ensure_image():
    images = sorted(IMG_DIR.glob("gsy_*.jpg"))
    assert images, f"测试图目录无 gsy_*.jpg: {IMG_DIR}"
    return str(images[0])


def test_process_array_matches_process_to_dir(tmp_path):
    image_path = _ensure_image()
    pipe = RuleBasedWrinklePipeline()

    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    assert gray is not None
    pred, heat, metrics = pipe.process_array(gray)

    # 基本契约
    assert pred.dtype == np.uint8 and pred.ndim == 2
    assert pred.shape == gray.shape
    assert heat.shape == pred.shape
    # 本引擎语义：热力图（未上色强度图）与预测图同源
    np.testing.assert_array_equal(heat, pred)
    for key in ("processing_mode", "model_weight", "sample_score", "anomaly_level",
                "analog_voltage", "original_size", "num_crops", "timestamp",
                "crop_results", "wrinkles"):
        assert key in metrics, f"metrics 缺字段 {key}"
    assert "process_id" not in metrics and "files" not in metrics
    assert metrics["original_size"] == [gray.shape[1], gray.shape[0]]

    # 与文件路径模式逐项对比
    out_dir = tmp_path / "out"
    pred_path, heat_path, json_path = pipe.process_to_dir(image_path, str(out_dir), "pid_x")

    got_pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
    np.testing.assert_array_equal(got_pred, pred)

    got_heat = cv2.imread(heat_path)  # BGR 三通道 JET
    np.testing.assert_array_equal(got_heat, cvt2heatmap(heat))

    import json
    file_json = json.loads(Path(json_path).read_text(encoding="utf-8"))
    assert file_json["process_id"] == "pid_x"
    assert set(file_json["files"]) == {"anomaly_map", "heatmap"}
    for key, value in metrics.items():
        if key == "timestamp":
            continue  # 两次调用时间必然不同
        assert file_json[key] == value, f"字段 {key} 不一致"


def _contour(x: int, y: int, w: int, h: int) -> np.ndarray:
    return np.array([[[x, y]], [[x + w, y]], [[x + w, y + h]], [[x, y + h]]], dtype=np.int32)


def test_compose_anomaly_map_matches_legacy_formula():
    """LUT 归一化锚点：输出与旧向量化 float 公式逐像素一致（bitwise）。

    参考链刻意内联公式与核尺寸（不从被测模块导入 min_max_norm /
    SMOOTH_KERNEL_SIZE）——否则实现一改锚点跟着漂，钉不住旧语义。
    """
    pipe = RuleBasedWrinklePipeline()
    wrinkles = [
        {"severity": 2, "contours": [_contour(5, 5, 30, 60)]},
        {"severity": 5, "contours": [_contour(60, 10, 25, 80), _contour(150, 30, 20, 50)]},
        {"severity": 3, "contours": [_contour(200, 0, 15, 90)]},
    ]
    got = pipe._compose_anomaly_map((100, 250), wrinkles)

    # 旧实现独立重算：severity 升序后画轮廓 → 高斯(31) → float32 min-max → ×255 截断
    canvas = np.zeros((100, 250), dtype=np.uint8)
    for wr in sorted(wrinkles, key=lambda item: item["severity"]):
        cv2.drawContours(canvas, wr["contours"], -1,
                         int(min(255, 100 + (wr["severity"] - 1) * 39)),
                         thickness=cv2.FILLED)
    smoothed = cv2.GaussianBlur(canvas, (31, 31), 0)
    legacy = ((smoothed.astype(np.float32) - smoothed.min())
              / (smoothed.max() - smoothed.min()) * 255).astype(np.uint8)
    np.testing.assert_array_equal(got, legacy)
    assert got.min() < got.max()  # 场景非退化（确实经过了归一化）


def test_compose_anomaly_map_constant_canvas_no_warning():
    """max==min 退化画布：返回全 0 且不产生 NaN cast 的 RuntimeWarning。"""
    import warnings

    pipe = RuleBasedWrinklePipeline()
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # 任何 RuntimeWarning 直接失败
        # 轮廓完全在画布外 → canvas 保持全 0 → min==max==0
        wrinkles = [{"severity": 4, "contours": [_contour(5000, 5000, 30, 60)]}]
        got = pipe._compose_anomaly_map((100, 250), wrinkles)
        # severity-5（gray_val=255）整幅覆盖 → min==max==255 同一分支
        full = [{"severity": 5, "contours": [_contour(0, 0, 250, 100)]}]
        got255 = pipe._compose_anomaly_map((100, 250), full)
    assert got.shape == (100, 250) and not got.any()
    assert got255.shape == (100, 250) and not got255.any()


@pytest.mark.parametrize("scale", [1.0, 0.5])
def test_preprocess_matches_legacy_chain(scale):
    """THRESH_BINARY_INV 锚点：preprocess 与旧版 BINARY+bitwise_not 链逐位一致。

    参考链参数以 _odd(...) 表达（连同取奇数语义一起钉住）；scale 覆盖生产
    浮动值（h/1000）的一档非 1.0。
    """
    from utils.rule_based_wrinkle import _odd

    rng = np.random.default_rng(3)
    gray = rng.integers(0, 256, size=(240, 320), dtype=np.uint8)
    gray = cv2.GaussianBlur(gray, (0, 0), 5)  # 造连续灰度，避免退化二值图

    got = RuleBasedWrinklePipeline().detector.preprocess(gray, scale=scale)

    # 旧链独立重算
    blurred = cv2.medianBlur(gray, _odd(5 * scale))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    binary = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
        blockSize=_odd(201 * scale, minimum=5), C=25)
    binary_inv = cv2.bitwise_not(binary)
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, _odd(15 * scale)))
    refined = cv2.morphologyEx(binary_inv, cv2.MORPH_CLOSE, vk)
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (_odd(3 * scale), _odd(3 * scale)))
    legacy = cv2.morphologyEx(refined, cv2.MORPH_OPEN, hk)

    np.testing.assert_array_equal(got, legacy)
    assert set(np.unique(got)) <= {0, 255}
