"""找图（FindImages）测试。"""

from pathlib import Path

import cv2
import numpy as np
import pytest

from imgcolor import FindImages


def pattern(color=(0, 0, 255), bg=(0, 0, 0)):
    """16x16 带内部结构的图案（BGR），避免纯色模板退化。"""
    t = np.zeros((16, 16, 3), np.uint8)
    t[:] = bg
    t[2:14, 2:14] = color
    t[5:11, 5:11] = bg
    t[7:9, 7:9] = color
    return t


def screen_with_two():
    img = np.zeros((200, 300, 3), np.uint8)
    img[40:56, 30:46] = pattern()
    img[120:136, 150:166] = pattern()
    return img


def test_find_basic_and_fields():
    img = screen_with_two()
    res = FindImages.find(pattern(), image=img, confidence=0.9)
    assert res is not None
    assert set(res) == {"rect", "center_x", "center_y", "confidence", "index"}
    assert res["rect"] == [30, 40, 46, 56]
    assert (res["center_x"], res["center_y"]) == (38, 48)
    assert res["confidence"] >= 0.99
    assert res["index"] == 0


def test_find_all_and_ori():
    img = screen_with_two()
    res = FindImages.find_all(pattern(), image=img, confidence=0.9)
    assert res is not None and len(res) == 2
    assert [r["rect"][0] for r in res] == [30, 150]
    res2 = FindImages.find_all(pattern(), image=img, confidence=0.9, ori=5)
    assert [r["rect"][0] for r in res2] == [150, 30]


def test_find_all_maxcnt():
    img = screen_with_two()
    assert len(FindImages.find_all(pattern(), image=img, confidence=0.9, maxcnt=1)) == 1
    assert len(FindImages.find_all(pattern(), image=img, confidence=0.9, maxcnt=0)) == 2


def test_rect_limits_search():
    img = screen_with_two()
    res = FindImages.find_all(pattern(), rect=[0, 0, 100, 100], image=img, confidence=0.9)
    assert len(res) == 1 and res[0]["rect"] == [30, 40, 46, 56]
    assert FindImages.find(pattern(), rect=[200, 0, 300, 100], image=img,
                           confidence=0.9) is None


def test_confidence_filter_and_no_match():
    img = screen_with_two()
    rng = np.random.default_rng(42)
    other = rng.integers(0, 255, (16, 16, 3), dtype=np.uint8)  # 与源图无关
    assert FindImages.find(other, image=img, confidence=0.9) is None
    assert FindImages.find_all(other, image=img, confidence=0.9) is None


def test_multi_template_index():
    img = np.zeros((100, 150, 3), np.uint8)
    img[30:46, 50:66] = pattern(color=(0, 255, 0))
    wrong = pattern(color=(0, 0, 255))
    right = pattern(color=(0, 255, 0))
    # 用彩色匹配区分红/绿（灰度下同形状会同样高分）
    res = FindImages.find([wrong, right], image=img, confidence=0.9, rgb=True)
    assert res is not None and res["index"] == 1 and res["rect"] == [50, 30, 66, 46]


def test_rgb_vs_gray_matching():
    # 同形状、不同颜色：灰度下是线性亮度变化，CCOEFF 仍高分；彩色下差异大
    img = np.zeros((100, 150, 3), np.uint8)
    img[20:36, 40:56] = pattern(color=(255, 255, 255))
    tpl = pattern(color=(0, 0, 200))
    assert FindImages.find(tpl, image=img, rgb=True, confidence=0.85) is None
    gray = FindImages.find(tpl, image=img, rgb=False, confidence=0.85)
    assert gray is not None and gray["rect"] == [40, 20, 56, 36]


def test_scale_range():
    tpl = pattern()
    big = cv2.resize(tpl, (24, 24), interpolation=cv2.INTER_LINEAR)
    img = np.zeros((120, 180, 3), np.uint8)
    img[40:64, 60:84] = big
    # 单尺度（原尺寸）匹配不到放大版
    assert FindImages.find(tpl, image=img, confidence=0.95) is None
    res = FindImages.find(tpl, image=img, confidence=0.8, scale_range=[1.3, 1.7])
    assert res is not None and res["confidence"] >= 0.85
    assert 22 <= (res["rect"][2] - res["rect"][0]) <= 26
    assert abs(res["center_x"] - 72) <= 2 and abs(res["center_y"] - 52) <= 2


def test_scale_levels_include_one():
    from imgcolor.findimages import _scale_levels

    levels = _scale_levels([0.5, 2.0])
    assert 1.0 in levels and 14 <= len(levels) <= 16
    assert _scale_levels(None) == [1.0]
    with pytest.raises(ValueError):
        _scale_levels([2.0, 0.5])


def test_transparent_png(tmp_path):
    # 模板：整幅红色，只有中心 8x8 不透明
    bgra = np.zeros((16, 16, 4), np.uint8)
    bgra[:, :, 0:3] = (0, 0, 255)
    bgra[4:12, 4:12, 3] = 255
    path = tmp_path / "tpl.png"
    assert cv2.imwrite(str(path), bgra)

    # 源图：绿色背景 + 中心 8x8 红色；外围颜色不同于模板，只能靠 alpha mask 排除
    img = np.zeros((100, 120, 3), np.uint8)
    img[:] = (0, 255, 0)
    img[34:42, 54:62] = (0, 0, 255)
    res = FindImages.find(str(path), image=img, confidence=0.9)
    assert res is not None
    assert res["rect"] == [50, 30, 66, 46]
    assert res["confidence"] >= 0.99


def test_template_path_and_source_img(tmp_path):
    tpl = pattern()
    path = tmp_path / "part.png"
    assert cv2.imwrite(str(path), tpl)
    img = screen_with_two()
    # source_img 是 ascript 参数名，等价于 image
    res = FindImages.find(str(path), source_img=img, confidence=0.9)
    assert res is not None and res["rect"] == [30, 40, 46, 56]
    assert FindImages.find(Path(path), image=img, confidence=0.9) == res


def test_template_aliases():
    img = screen_with_two()
    tpl = pattern()
    kw = dict(image=img, confidence=0.9)
    assert FindImages.find_template(tpl, **kw)["rect"] == [30, 40, 46, 56]
    assert len(FindImages.find_all_template(tpl, **kw)) == 2
    assert FindImages.find_sift(tpl, **kw)["rect"] == [30, 40, 46, 56]
    assert len(FindImages.find_all_sift(tpl, **kw)) == 2


def test_errors():
    img = screen_with_two()
    with pytest.raises(FileNotFoundError):
        FindImages.find("no_such_file.png", image=img)
    with pytest.raises(ValueError, match="不提供截图"):
        FindImages.find(pattern())
    with pytest.raises(ValueError):
        FindImages.find([], image=img)
