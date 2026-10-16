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
