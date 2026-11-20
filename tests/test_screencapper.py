"""全局截图回调：注册后所有方法省略 image= 自动取图。"""

import numpy as np
import pytest
from PIL import Image

from imgcolor import (
    Canny,
    Colors,
    CompareColors,
    FindColors,
    FindImages,
    FontLib,
    Ocr,
    OcrX,
    as_image,
    clear_screencapper,
    get_screencapper,
    set_screencapper,
)


@pytest.fixture(autouse=True)
def _clean_capper():
    """每个用例前后都清空全局截图器，避免相互污染。"""
    clear_screencapper()
    yield
    clear_screencapper()


def _scene():
    """带一个红点 + 一个白矩形的 BGR 图。"""
    img = np.zeros((80, 120, 3), np.uint8)
    img[5, 6] = (0, 0, 255)
    import cv2
    cv2.rectangle(img, (30, 30), (70, 60), (255, 255, 255), -1)
    return img


# ---- 注册表基础行为 ------------------------------------------------------ #
def test_register_get_clear():
    assert get_screencapper() is None
    fn = lambda: _scene()
    set_screencapper(fn)
    assert get_screencapper() is fn
    clear_screencapper()
    assert get_screencapper() is None


def test_invalid_capper_raises():
    with pytest.raises(TypeError, match="无参可调用"):
        set_screencapper("not callable")
    with pytest.raises(TypeError, match="无参可调用"):
        set_screencapper(123)


def test_no_image_no_capper_raises():
    with pytest.raises(ValueError, match="不提供截图"):
        FindColors.find_color("#FF0000")


# ---- as_image 走截图器 --------------------------------------------------- #
def test_as_image_uses_capper():
    set_screencapper(lambda: _scene())
    out = as_image(None)
    assert out.shape == (80, 120, 3)
    assert tuple(out[5, 6]) == (0, 0, 255)


def test_capper_may_return_rgb_pil():
    """截图器返回 Pillow RGB 图（u2 screenshot 的典型形态）应被正确转成 BGR。"""
    rgb = Image.fromarray(_scene()[:, :, ::-1])  # BGR -> RGB PIL
    set_screencapper(lambda: rgb)
    out = as_image(None)
    assert tuple(out[5, 6]) == (0, 0, 255)  # 回到 BGR 红


def test_capper_may_return_path(tmp_path):
    import cv2
    p = tmp_path / "screen.png"
    cv2.imwrite(str(p), _scene())
    set_screencapper(lambda: str(p))
    assert FindColors.find_color("#FF0000") == {"x": 6, "y": 5}


# ---- 各模块省略 image ---------------------------------------------------- #
def test_find_colors_no_image():
    set_screencapper(lambda: _scene())
    assert FindColors.find_color("#FF0000") == {"x": 6, "y": 5}
    assert FindColors.find("6,5,#FF0000", image=None) is not None
    assert FindColors.find_all_color("#FF0000") is not None


def test_compare_colors_no_image():
    set_screencapper(lambda: _scene())
    assert CompareColors.compare("6,5,#FF0000") is True
    assert CompareColors.compare_multi(["1,1,#123456", "6,5,#FF0000"]) == 1


def test_colors_no_image():
    set_screencapper(lambda: _scene())
    assert Colors.count("#FFFFFF", rect=[30, 30, 70, 60]) == 40 * 30


def test_canny_no_image():
    set_screencapper(lambda: _scene())
    res = Canny(draw=0).run(None)
    assert len(res.data) >= 1


def test_find_images_no_image():
    set_screencapper(lambda: _scene())
    tpl = np.zeros((6, 6, 3), np.uint8)
    tpl[1:5, 1:5] = (255, 255, 255)
    # 纯色白块模板与白矩形的匹配分约 0.59（同源但无纹理），用较低阈值
    hit = FindImages.find(tpl, confidence=0.5)
    assert hit is not None and 28 <= hit["rect"][0] <= 30


def test_ocrx_no_image():
    set_screencapper(lambda: _scene())
    lib = FontLib()
    lib.add("X", [[1, 1], [1, 1]])
    tiny = np.zeros((10, 10, 3), np.uint8)
    tiny[2:4, 2:4] = (255, 255, 255)
    set_screencapper(lambda: tiny)
    assert OcrX.find_all(lib, color="#FFFFFF") == ["X"]
    assert Ocr.matrix(lib, color="#FFFFFF") == "X"


def test_fontlib_add_from_image_no_image():
    set_screencapper(lambda: _scene())
    g = FontLib(color="#FFFFFF").add_from_image("rect", None, rect=[30, 30, 34, 34])
    assert g.width == 4 and g.height == 4


def test_color_mask_no_image():
    from imgcolor import color_mask, parse_color_spec
    set_screencapper(lambda: _scene())
    m = color_mask(None, parse_color_spec("#FF0000"))
    assert m[5, 6]


# ---- Ocr 便捷方法 -------------------------------------------------------- #
def test_ocr_find_uses_capper(monkeypatch):
    """Ocr 走 _get_frame：无 image 时调用截图器。用假引擎避免真实模型。"""
    captured = {}

    def fake_recognize(image, **kwargs):
        captured["shape"] = image.shape
        return []

    monkeypatch.setattr("imgcolor.ocrx.recognize", fake_recognize)
    set_screencapper(lambda: _scene())
    Ocr.find("登录")
    assert captured["shape"] == (80, 120, 3)


def test_ocr_find_all_uses_capper(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "imgcolor.ocrx.recognize",
        lambda image, **kw: (captured.update(shape=image.shape) or []),
    )
    set_screencapper(lambda: _scene())
    assert Ocr.find_all() is None
    assert captured["shape"] == (80, 120, 3)


# ---- 刷新语义 ------------------------------------------------------------ #
def test_until_refreshes_via_capper():
    """until>0 时每轮重新截图，画面变化后应能命中。"""
    blank = np.zeros((80, 120, 3), np.uint8)  # 第一帧无任何红点
    second = blank.copy()
    second[10, 20] = (0, 0, 255)  # 第二帧出现红点

    calls = {"n": 0}

    def grab():
        calls["n"] += 1
        return blank if calls["n"] == 1 else second

    set_screencapper(grab)
    found = FindColors.find("20,10,#FF0000", until=2000)
    assert calls["n"] >= 2
    assert found is not None and (found.x, found.y) == (20, 10)


def test_ocr_wait_refreshes_via_capper(monkeypatch):
    """Ocr.wait 也应随截图器刷新（用假引擎模拟第 2 帧才出现文字）。"""
    from imgcolor import rapidocr_engine as eng

    calls = {"n": 0}

    def fake_recognize(image, **kw):
        calls["n"] += 1
        if calls["n"] < 2:
            return []
        return [eng.OcrText("登录", [1, 2, 30, 40], 15, 21, 0.99)]

    monkeypatch.setattr("imgcolor.ocrx.recognize", fake_recognize)
    set_screencapper(lambda: _scene())
    hit = Ocr.wait("登录", timeout=2)
    assert calls["n"] >= 2
    assert hit["text"] == "登录"


def test_clear_restores_error():
    set_screencapper(lambda: _scene())
    assert FindColors.find_color("#FF0000") == {"x": 6, "y": 5}
    clear_screencapper()
    with pytest.raises(ValueError, match="不提供截图"):
        FindColors.find_color("#FF0000")


def test_explicit_image_overrides_capper():
    """显式传 image 时优先用显式图，不走截图器。"""
    capper_scene = _scene()  # 红点在 (6,5)
    explicit = np.zeros((30, 30, 3), np.uint8)
    explicit[1, 2] = (0, 0, 255)  # 红点在 (2,1)
    set_screencapper(lambda: capper_scene)
    assert FindColors.find_color("#FF0000", image=explicit) == {"x": 2, "y": 1}
