"""所有接收图像的方法统一支持：文件路径（含中文）、Pillow Image、callable。

Pillow 是库的正式运行依赖，本文件同时验证依赖声明与精确类型识别。
"""

from importlib.metadata import PackageNotFoundError, requires
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from _helpers import DIGITS, dot, draw_text, imwrite_any
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
    binarize,
    color_mask,
    make_font_lib,
    parse_color_spec,
    read_image,
)

PATTERN = "10,10,#FF0000|14,10,#00FF00"


def tpl_pattern():
    t = np.zeros((16, 16, 3), np.uint8)
    t[2:14, 2:14] = (0, 0, 255)
    t[5:11, 5:11] = (0, 0, 0)
    t[7:9, 7:9] = (0, 0, 255)
    return t


def make_scene():
    img = np.zeros((140, 220, 3), np.uint8)
    dot(img, 10, 10, "#FF0000", 2, 2)      # 找色/比色
    dot(img, 14, 10, "#00FF00", 2, 2)
    cv2.rectangle(img, (50, 20), (70, 40), (0x95, 0xD3, 0x05), -1)  # 颜色数量
    cv2.rectangle(img, (100, 60), (160, 100), (255, 255, 255), -1)  # 边缘
    draw_text(img, "12", 20, 80)                                      # 点阵字
    img[20:36, 180:196] = tpl_pattern()                               # 找图
    return img


def to_pil(bgr_img, mode="RGB"):
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb).convert(mode)


def font_lib():
    lib = FontLib(color="#FFFFFF", diff=0.95)
    for ch, m in DIGITS.items():
        lib.add(ch, m)
    return lib


# --------------------------------------------------------------------------- #
# 文件路径
# --------------------------------------------------------------------------- #
def test_path_input_all_methods(tmp_path):
    scene = make_scene()
    path = imwrite_any(tmp_path / "屏幕截图_测试.png", scene)  # 中文路径
    assert Path(path).is_file()

    assert FindColors.find_color("#FF0000", image=path) == {"x": 10, "y": 10}
    assert FindColors.find_all_color("#FF0000", max_num=1, image=path) == [{"x": 10, "y": 10}]
    assert FindColors.find(PATTERN, image=path).as_dict() == {"x": 10, "y": 10}
    assert FindColors.find_all(PATTERN, image=path)
    assert CompareColors.compare(PATTERN, image=path) is True
    assert CompareColors.compare_multi(["1,1,#123456", PATTERN], image=path) == 1
    assert Colors.count("#05D395", image=path) == 441
    assert Colors.count("#05D395", bitmap=path) == 441
    count, dbg = Colors.count_image("#05D395", image=path)
    assert count == 441 and dbg.shape == scene.shape
    assert len(Canny(draw=Canny.DRAW_NONE).run(path).data) >= 1

    lib = font_lib()
    assert OcrX.find_all(lib, rect=[0, 70, 60, 110], image=path) == ["1", "2"]
    assert Ocr.matrix(lib, rect=[0, 70, 60, 110], image=path) == "12"
    g = FontLib(color="#FFFFFF").add_from_image("1", path, rect=[20, 80, 25, 87])
    assert g.width == 3 and g.height == 7

    hit = FindImages.find(tpl_pattern(), image=path, confidence=0.9)
    assert hit is not None and hit["rect"] == [180, 20, 196, 36]
    assert len(FindImages.find_all(tpl_pattern(), image=path, confidence=0.9)) == 1


def test_pathlib_path_input(tmp_path):
    scene = make_scene()
    p = Path(imwrite_any(tmp_path / "s.png", scene))
    assert FindColors.find_color("#FF0000", image=p) == {"x": 10, "y": 10}
    assert Colors.count("#05D395", image=p) == 441

    tpl_path = Path(imwrite_any(tmp_path / "模板_中文.png", tpl_pattern()))
    hit = FindImages.find(str(tpl_path), image=p, confidence=0.9)
    assert hit["rect"] == [180, 20, 196, 36]
    assert FindImages.find(tpl_path, image=p, confidence=0.9) == hit  # Path 对象模板


def test_path_as_template_with_alpha(tmp_path):
    bgra = np.zeros((16, 16, 4), np.uint8)
    bgra[:, :, 0:3] = (0, 0, 255)
    bgra[4:12, 4:12, 3] = 255
    path = imwrite_any(tmp_path / "透明模板_中文.png", bgra)

    img = np.zeros((100, 120, 3), np.uint8)
    img[:] = (0, 255, 0)
    img[34:42, 54:62] = (0, 0, 255)
    hit = FindImages.find(path, image=img, confidence=0.9)
    assert hit is not None and hit["rect"] == [50, 30, 66, 46]


# --------------------------------------------------------------------------- #
# Pillow Image
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mode", ["RGB", "RGBA"])
def test_pillow_modes_keep_colors(mode):
    scene = make_scene()
    pil = to_pil(scene, mode)
    assert FindColors.find_color("#FF0000", image=pil) == {"x": 10, "y": 10}
    assert FindColors.find_color("#0000FF", image=pil) is None  # 通道没被反转错
    assert CompareColors.compare(PATTERN, image=pil) is True
    assert Colors.count("#05D395", image=pil) == 441
    assert len(Canny(draw=Canny.DRAW_NONE).run(pil).data) >= 1
    assert FindImages.find(tpl_pattern(), image=pil, confidence=0.9)["rect"] == [180, 20, 196, 36]


def test_pillow_palette_mode_quantization():
    # P 模式用调色板量化，颜色会偏移几个色阶（#05D395 -> #00CC99），
    # 因此精确比色(sim=1)不适用；结构类方法不受影响。
    scene = make_scene()
    pil = to_pil(scene, "P")
    assert FindColors.find_color("#FF0000", image=pil) == {"x": 10, "y": 10}
    assert FindColors.find_color("#0000FF", image=pil) is None
    assert CompareColors.compare(PATTERN, image=pil) is True
    assert len(Canny(draw=Canny.DRAW_NONE).run(pil).data) >= 1
    assert FindImages.find(tpl_pattern(), image=pil, confidence=0.9)["rect"] == [180, 20, 196, 36]
    assert Colors.count("#05D395", sim=0.9, image=pil) > 400


@pytest.mark.parametrize("mode", ["L", "1"])
def test_pillow_gray_modes_readable(mode):
    scene = make_scene()
    pil = to_pil(scene, mode)
    arr = as_image(pil)
    assert arr.shape == scene.shape and arr.dtype == np.uint8
    assert len(Canny(draw=Canny.DRAW_NONE).run(pil).data) >= 1


def test_pillow_no_double_rgb_flip():
    scene = make_scene()
    pil = to_pil(scene, "RGB")
    # 路径/Pillow 已规范化成 BGR，即使误传 color_order="RGB" 也不应再反转
    assert FindColors.find_color("#FF0000", image=pil, color_order="RGB") == {"x": 10, "y": 10}
    assert FindColors.find_color("#0000FF", image=pil, color_order="RGB") is None


def test_pillow_as_template():
    scene = make_scene()
    tpl_rgb = cv2.cvtColor(tpl_pattern(), cv2.COLOR_BGR2RGB)
    pil_tpl = Image.fromarray(tpl_rgb)
    hit = FindImages.find(pil_tpl, image=to_pil(scene, "RGB"), confidence=0.9)
    assert hit["rect"] == [180, 20, 196, 36]


def test_pillow_rgba_transparent_template():
    rgba = np.zeros((16, 16, 4), np.uint8)
    rgba[:, :, 0:3] = (255, 0, 0)  # RGB 红
    rgba[4:12, 4:12, 3] = 255
    pil = Image.fromarray(rgba)  # 自动识别为 RGBA

    img = np.zeros((100, 120, 3), np.uint8)
    img[:] = (0, 255, 0)
    img[34:42, 54:62] = (0, 0, 255)
    hit = FindImages.find(pil, image=img, confidence=0.9)
    assert hit is not None and hit["rect"] == [50, 30, 66, 46]


def test_pillow_ocr_and_fontlib():
    scene = make_scene()
    pil = to_pil(scene, "RGB")
    lib = font_lib()
    assert Ocr.matrix(lib, rect=[0, 70, 60, 110], image=pil) == "12"
    g = FontLib(color="#FFFFFF").add_from_image("1", pil, rect=[20, 80, 25, 87])
    assert g.width == 3 and g.height == 7


# --------------------------------------------------------------------------- #
# callable（until 重试）+ 错误处理
# --------------------------------------------------------------------------- #
def test_callable_returns_path_or_pillow(tmp_path):
    scene = make_scene()
    path = imwrite_any(tmp_path / "s.png", scene)
    pil = to_pil(scene, "RGB")
    assert FindColors.find_color("#FF0000", image=lambda: path) == {"x": 10, "y": 10}
    assert FindColors.find_color("#FF0000", image=lambda: pil) == {"x": 10, "y": 10}


def test_until_retry_with_path_provider(tmp_path):
    scene = make_scene()
    good = imwrite_any(tmp_path / "good.png", scene)
    blank = imwrite_any(tmp_path / "blank.png", np.zeros_like(scene))
    calls = {"n": 0}

    def provider():
        calls["n"] += 1
        return blank if calls["n"] < 2 else good

    assert FindColors.find(PATTERN, image=provider, until=1000).x == 10
    assert calls["n"] >= 2


def test_read_image_helper(tmp_path):
    scene = make_scene()
    path = imwrite_any(tmp_path / "s.png", scene)
    arr, normalized = read_image(path)
    assert normalized is True and np.array_equal(arr, scene)

    arr2, normalized2 = read_image(scene)
    assert normalized2 is False and np.array_equal(arr2, scene)

    arr3, normalized3 = read_image(to_pil(scene, "RGB"))
    assert normalized3 is True and arr3.shape == (140, 220, 4)


def test_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        as_image(tmp_path / "不存在.png")
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"this is not an image")
    with pytest.raises(ValueError, match="解码失败"):
        as_image(bad)
    with pytest.raises(ValueError, match="不提供截图"):
        FindColors.find_color("#FF0000")
    with pytest.raises(FileNotFoundError):
        FindImages.find(tmp_path / "无模板.png", image=make_scene())


# --------------------------------------------------------------------------- #
# Pillow 作为正式运行依赖 + 底层导出函数的输入支持
# --------------------------------------------------------------------------- #
def test_pillow_is_runtime_dependency():
    """库必须直接 import PIL，而不是靠鸭子类型猜测。"""
    from imgcolor import _core

    assert _core.Image is Image
    assert _core._is_pillow(to_pil(make_scene(), "RGB")) is True
    assert _core._is_pillow(make_scene()) is False

    try:
        reqs = requires("imgcolor") or []
    except PackageNotFoundError:
        pytest.skip("源码树未生成 dist-info")
    assert any(r.lower().startswith("pillow") for r in reqs), reqs


def test_fake_pillow_like_object_gets_clear_error():
    """冒充 PIL 的普通对象不再被误判，应给出明确 TypeError。"""

    class FakePIL:
        mode = "RGB"

        def convert(self, mode):
            raise AssertionError("不应被调用")

    with pytest.raises(TypeError, match="不支持的图像输入类型"):
        as_image(FakePIL())


def test_color_mask_accepts_path_and_pillow(tmp_path):
    scene = make_scene()
    path = imwrite_any(tmp_path / "掩码_中文.png", scene)
    pil = to_pil(scene, "RGB")
    spec = parse_color_spec("#FF0000")

    expected = color_mask(scene, spec)
    # 场景里除了 2x2 红点，红色模板图案的像素同样命中
    assert expected.sum() > 4 and expected[10, 10] and not expected[0, 0]
    assert np.array_equal(color_mask(path, spec), expected)
    assert np.array_equal(color_mask(pil, spec), expected)


def test_binarize_accepts_path_and_pillow(tmp_path):
    scene = make_scene()
    path = imwrite_any(tmp_path / "二值化_中文.png", scene)
    pil = to_pil(scene, "RGB")

    expected = binarize(scene, color="#FFFFFF")
    assert expected.any()
    assert np.array_equal(binarize(path, color="#FFFFFF"), expected)
    assert np.array_equal(binarize(pil, color="#FFFFFF"), expected)
    # 不指定颜色时走 Otsu，输入形式同样支持
    assert np.array_equal(binarize(path), binarize(scene))
    assert np.array_equal(binarize(pil), binarize(scene))


@pytest.mark.parametrize("as_path", [False, True])
def test_make_font_lib_accepts_pillow_or_path(tmp_path, as_path):
    scene = make_scene()
    src = imwrite_any(tmp_path / "字库素材.png", scene) if as_path else to_pil(scene, "RGB")
    out = tmp_path / "字库.json"

    lib = make_font_lib(
        [{"label": "1", "rect": [20, 80, 25, 87]}],
        path=out,
        image=src,
        color="#FFFFFF",
        diff=0.95,
    )
    assert len(lib) == 1 and lib.glyphs[0].width == 3 and lib.glyphs[0].height == 7
    assert out.is_file()
    assert len(FontLib.load(out)) == 1
