"""Dinomaly 引擎长图裁剪测试：全图 25 等分，不丢弃两端"""

from PIL import Image
import pytest

from utils.dinomaly_engine import DinomalyEngine


@pytest.fixture(scope="module")
def engine():
    # 仅使用纯几何方法 _crop_fixed，跳过 __init__ 的模型加载
    return object.__new__(DinomalyEngine)


def test_fixed_crop_covers_full_width(engine):
    img = Image.new("L", (31901, 1000))
    crops = engine._crop_fixed(img)
    assert len(crops) == 25
    positions = [pos for _, pos in crops]
    assert positions[0][0] == 0, "first crop must start at image left edge"
    assert positions[-1][2] == 31901, "last crop must reach image right edge"
    for (_, _, x2_prev, _), (x1_next, _, _, _) in zip(positions, positions[1:]):
        assert x1_next == x2_prev, "crops must tile the image without gaps"


def test_fixed_crop_handles_non_multiple_width(engine):
    # 宽度不能被 25 整除时尾片收窄但仍在图内
    img = Image.new("L", (10000, 500))
    crops = engine._crop_fixed(img)
    assert len(crops) == 25
    assert crops[-1][1][2] == 10000
    assert crops[-1][1][0] < 10000


def test_fixed_crop_ndarray_matches_pil_bitwise(engine):
    """ndarray 入参（shm 热路径直入）与 PIL 入参产出逐位一致的裁剪（①等价锚点）。"""
    import cv2
    import numpy as np

    rng = np.random.default_rng(7)
    gray = rng.integers(0, 256, size=(80, 553), dtype=np.uint8)  # 25 等分尾片收窄
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    from_pil = engine._crop_fixed(Image.fromarray(rgb, "RGB"))
    from_np = engine._crop_fixed(rgb)
    assert [pos for _, pos in from_pil] == [pos for _, pos in from_np]
    for (pil_a, _), (pil_b, _) in zip(from_pil, from_np):
        assert pil_a.mode == pil_b.mode
        assert np.array_equal(np.array(pil_a), np.array(pil_b)), "裁剪像素须逐位一致"


def test_process_array_very_long_skips_full_pil(monkeypatch):
    """① 回归哨兵：very long 热路径不得构造全图 PIL——全图 fromarray +
    np.array 回拷是双 96MB 冗余拷贝（实测 ~211ms，bench 2026-10-16）。

    白盒计数 Image.fromarray：直入 ndarray 后只应有 25 次逐片调用，
    每片宽度 ≤ ceil(W/25)；出现全宽调用即回归。
    """
    import numpy as np
    import utils.dinomaly_engine as de_mod

    calls = []
    real_fromarray = Image.fromarray

    def counting_fromarray(obj, mode=None):
        arr = obj if isinstance(obj, np.ndarray) else None
        calls.append(None if arr is None else arr.shape[1])
        return real_fromarray(obj, mode) if mode is None else real_fromarray(obj, mode)

    monkeypatch.setattr(de_mod.Image, "fromarray", counting_fromarray)

    engine = object.__new__(DinomalyEngine)
    engine.threshold = 0.21
    engine.model_path = "MOCK_WEIGHT"
    W, H = 2500, 1000

    def fake_process_pil_images(images):
        return [{
            "anomaly_map": np.full((392, 392), 0.5, dtype=np.float32),
            "sample_score": 0.5,
            "processed_size": (392, 392),
            "original_size": image.size,
        } for image in images]

    monkeypatch.setattr(engine, "process_pil_images", fake_process_pil_images)
    gray = np.zeros((H, W), dtype=np.uint8)
    engine.process_array(gray, image_type="very long")

    assert len(calls) == 25, f"very long 热路径应只逐片 fromarray 25 次: {len(calls)}"
    assert all(w is not None and w <= -(-W // 25) for w in calls), \
        f"不得出现全宽 fromarray（全图 PIL 往返回归）: {calls[:5]}..."
