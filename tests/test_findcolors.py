import numpy as np

from _helpers import dot
from imgcolor import FindColors, Point, parse_points


def _place(img, x, y):
    """在 (x,y) 放一个 '红点 + 右移2像素绿点' 的图案。"""
    dot(img, x, y, "#FF0000")
    dot(img, x + 2, y, "#00FF00")


PATTERN = "10,10,#FF0000|12,10,#00FF00"


def test_parse_points():
    pts = parse_points("10,20,#FF0000|12,20,#00FF00-#101010")
    assert len(pts) == 2
    assert (pts[0].x, pts[0].y) == (10, 20)
    assert pts[0].spec.bgr == (0, 0, 255)
    assert pts[0].spec.tol is None
    assert pts[1].spec.tol == (16, 16, 16)

    # 偏色的另一种写法 x,y,主色,偏色
    p = parse_points("1,2,#FF0000,#101010")[0]
    assert p.spec.tol == (16, 16, 16)

    # 不带坐标的纯颜色（Colors.count 用）
    c = parse_points("#05D395-#101010")[0]
    assert c.x is None and c.y is None
    assert c.spec.bgr == (0x95, 0xD3, 0x05)


def test_find_color_exact(canvas):
    dot(canvas, 10, 20, "#FF0000", 2, 2)
    assert FindColors.find_color("#FF0000", image=canvas) == {"x": 10, "y": 20}
    assert FindColors.find_color("#FF0000", rect=[0, 0, 5, 5], image=canvas) is None
    assert FindColors.find_color("#00FF00", image=canvas) is None


def test_find_color_diff_and_tol(canvas):
    dot(canvas, 30, 40, "#FE0101")
    assert FindColors.find_color("#FF0000", diff=1.0, image=canvas) is None
    assert FindColors.find_color("#FF0000", diff=0.98, image=canvas) == {"x": 30, "y": 40}
    # 偏色写法：逐通道容差（#101010 表示 RGB 三通道各允许 ±16）
    assert FindColors.find_color("#FF0000-#101010", image=canvas) == {"x": 30, "y": 40}
    # 只在 R 通道给容差时，G/B 有偏差就不应命中
    dot(canvas, 60, 60, "#FE0000")
    assert FindColors.find_color("#FF0000-#100000", image=canvas) == {"x": 60, "y": 60}


def test_find_all_color(canvas):
    for x in (10, 50, 100):
        dot(canvas, x, 20, "#FF0000", 3, 3)
    pts = FindColors.find_all_color("#FF0000", image=canvas)
    assert len(pts) == 27
    assert {"x": 10, "y": 20} in pts
    assert len(FindColors.find_all_color("#FF0000", max_num=5, image=canvas)) == 5
    assert FindColors.find_all_color("#00FF00", image=canvas) is None


def test_find_all_multipoint(canvas):
    _place(canvas, 10, 10)
    _place(canvas, 100, 100)
    pts = FindColors.find_all(PATTERN, image=canvas)
    assert sorted((p.x, p.y) for p in pts) == [(10, 10), (100, 100)]
    assert all(isinstance(p, Point) for p in pts)


def test_find_first_and_ori(canvas):
    _place(canvas, 10, 10)
    _place(canvas, 100, 100)
    assert FindColors.find(PATTERN, image=canvas).as_dict() == {"x": 10, "y": 10}
    assert FindColors.find(PATTERN, ori=5, image=canvas).as_dict() == {"x": 100, "y": 100}
    assert FindColors.find(PATTERN, ori=99, image=canvas).x == 10  # 非法方向回落默认


def test_find_rect_and_p_max_num(canvas):
    _place(canvas, 10, 10)
    _place(canvas, 100, 100)
    pts = FindColors.find_all(PATTERN, rect=[90, 90, 200, 200], image=canvas)
    assert len(pts) == 1 and pts[0].x == 100
    assert len(FindColors.find_all(PATTERN, p_max_num=1, image=canvas)) == 1


def test_find_space_nms(canvas):
    for x in range(20, 40):
        dot(canvas, x, 50, "#FF0000")
    pts = FindColors.find_all("0,0,#FF0000", space=5, image=canvas)
    xs = [p.x for p in pts]
    assert xs[0] == 20
    assert all(b - a > 5 for a, b in zip(xs, xs[1:]))
    assert len(FindColors.find_all("0,0,#FF0000", space=0, image=canvas)) == 20


def test_find_no_match(canvas):
    assert FindColors.find(PATTERN, image=canvas) is None
    assert FindColors.find_all(PATTERN, image=canvas) is None


def test_mode_norm(canvas):
    # 目标 RGB(100,50,50)，实际 RGB(200,100,100)：色比相同但亮度翻倍
    dot(canvas, 60, 60, "#C86464")
    assert FindColors.find_color("#643232", mode=0, diff=0.98, image=canvas) is None
    assert FindColors.find_color("#643232", mode=1, diff=0.98, image=canvas) == {"x": 60, "y": 60}


def test_until_retry_with_callable():
    blank = np.zeros((50, 50, 3), np.uint8)
    target = blank.copy()
    dot(target, 5, 5, "#FF0000")
    calls = {"n": 0}

    def provider():
        calls["n"] += 1
        return blank if calls["n"] < 3 else target

    assert FindColors.find("0,0,#FF0000", image=provider, until=2000) == Point(5, 5)
    assert calls["n"] >= 3


def test_until_static_image_does_not_block(canvas):
    # 静态 ndarray + until>0：只算一次，不会卡住
    assert FindColors.find("0,0,#FF0000", image=canvas, until=500) is None


def test_rgb_input():
    arr = np.zeros((30, 30, 3), np.uint8)
    arr[5, 6] = (255, 0, 0)  # RGB 红
    assert FindColors.find_color("#FF0000", image=arr, color_order="RGB") == {"x": 6, "y": 5}


def test_missing_image_raises():
    import pytest

    with pytest.raises(ValueError, match="不提供截图"):
        FindColors.find_color("#FF0000")
