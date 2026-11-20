import numpy as np

from _helpers import bgr, dot
from imgcolor import Colors


def test_count_exact(canvas):
    dot(canvas, 10, 10, "#05D395", 10, 10)
    assert Colors.count("#05D395", image=canvas) == 100
    assert Colors.count("#05D395", bitmap=canvas) == 100  # ascript 参数名
    assert Colors.count("#FFFFFF", image=canvas) == 0


def test_count_rect(canvas):
    dot(canvas, 10, 10, "#05D395", 10, 10)
    assert Colors.count("#05D395", rect=[10, 10, 15, 20], image=canvas) == 50
    assert Colors.count("#05D395", rect=[0, 0, 5, 5], image=canvas) == 0
    assert Colors.count("#05D395", rect=[10, 10, 0, 0], image=canvas) == 100  # 0 表示到边缘


def test_count_sim(canvas):
    dot(canvas, 5, 5, "#FE0000", 2, 2)
    assert Colors.count("#FF0000", sim=1.0, image=canvas) == 0
    assert Colors.count("#FF0000", sim=0.9, image=canvas) == 4


def test_count_tolerance_and_multi(canvas):
    dot(canvas, 5, 5, "#F00000", 2, 2)
    assert Colors.count("#FF0000-#100000", image=canvas) == 4
    assert Colors.count("#FF0000-#010101", image=canvas) == 0
    # 多颜色取并集，同一像素不重复计数
    assert Colors.count("#FF0000-#100000|#FF0000-#100000", image=canvas) == 4


def test_count_multipoint_spec(canvas):
    # 带坐标的多点串：统计这些颜色在区域内的像素总量（并集）
    dot(canvas, 10, 10, "#FF0000", 3, 3)
    dot(canvas, 50, 50, "#00FF00", 2, 2)
    assert Colors.count("10,10,#FF0000|50,50,#00FF00", image=canvas) == 13


def test_mask_shape(canvas):
    dot(canvas, 10, 10, "#FF0000", 4, 4)
    m = Colors.mask("#FF0000", rect=[10, 10, 14, 14], image=canvas)
    assert m.shape == (4, 4) and m.dtype == bool and m.all()


def test_count_image(canvas):
    dot(canvas, 10, 10, "#05D395", 4, 4)
    count, dbg = Colors.count_image("#05D395", image=canvas, draw_mode=1)
    assert count == 16
    assert dbg.shape == canvas.shape
    assert dbg[10, 10].tolist() == [255, 255, 255]
    assert dbg[0, 0].tolist() == [0, 0, 0]

    count2, dbg2 = Colors.count_image("#05D395", image=canvas, draw_mode=2)
    assert count2 == 16
    assert dbg2[10, 10].tolist() == list(bgr("#05D395"))  # 命中处保留原色
    assert dbg2[0, 0].tolist() == [0, 0, 0]  # 其余涂黑


def test_count_rgb_input():
    arr = np.zeros((20, 20, 3), np.uint8)
    arr[2:4, 3:5] = (255, 0, 0)  # RGB 红
    assert Colors.count("#FF0000", image=arr, color_order="RGB") == 4
