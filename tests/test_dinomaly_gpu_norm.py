"""③ GPU 画布归一化 bitwise 等价测试（utils/dinomaly_engine._normalize_canvas_uint8）。

红线：GPU 归一化输出须与现行 CPU 公式（_min_max_norm + *255 + astype）逐位一致；
resize/merge 组装留在 CPU（cv2 与 torch interpolate 不等价，不越红线）。
无 CUDA 环境自动跳过 GPU 断言（CPU 回退路径仍验证）。
"""
import numpy as np
import pytest

from utils.dinomaly_engine import DinomalyEngine


def _cpu_ref(canvas: np.ndarray) -> np.ndarray:
    """现行 CPU 公式参照（_min_max_norm 无守卫，flat 画布 NaN→uint8 截断为本平台 0）。"""
    a_min, a_max = canvas.min(), canvas.max()
    with np.errstate(invalid="ignore"):
        normalized = (canvas - a_min) / (a_max - a_min)
    return (normalized * 255).astype(np.uint8)


def _make_engine(device: str):
    import torch

    engine = object.__new__(DinomalyEngine)
    engine._torch = torch
    engine.device = device
    return engine


def _resolve_device() -> str:
    try:
        import torch
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        pytest.skip("torch 不可用")


def test_gpu_normalize_matches_cpu_bitwise():
    device = _resolve_device()
    engine = _make_engine(device)
    rng = np.random.default_rng(42)
    # 非均匀分布 + 负值 + 非整除尺寸：覆盖 IEEE 逐元素各舍入路径
    canvas = rng.random((500, 3000), dtype=np.float32) * 3.0 - 1.0
    canvas[:, ::7] *= 255.0  # 拉开动态范围
    got = engine._normalize_canvas_uint8(canvas)
    ref = _cpu_ref(canvas)
    assert got.dtype == np.uint8
    assert got.shape == canvas.shape
    assert np.array_equal(got, ref), \
        f"GPU 归一化须与 CPU 公式逐位一致: max|Δ|={np.abs(got.astype(np.int16) - ref.astype(np.int16)).max()}"


def test_flat_canvas_returns_zeros():
    """全零/常数画布：CPU 现行为 0/0→NaN→uint8 截断为全 0（实测），守卫须等价。"""
    device = _resolve_device()
    engine = _make_engine(device)
    for canvas in (np.zeros((40, 500), dtype=np.float32),
                   np.full((40, 500), 7.25, dtype=np.float32)):
        got = engine._normalize_canvas_uint8(canvas)
        assert not got.any(), f"常数画布归一化须全 0（与现行为一致）: max={got.max()}"


def test_cpu_fallback_without_torch():
    """无 _torch 属性（object.__new__ 裸对象/旧调用方）回退 CPU 公式，结果一致。"""
    engine = object.__new__(DinomalyEngine)
    rng = np.random.default_rng(1)
    canvas = rng.random((64, 512), dtype=np.float32)
    got = engine._normalize_canvas_uint8(canvas)
    assert np.array_equal(got, _cpu_ref(canvas))
