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
