"""端到端：把图存成真实 PNG，再用 cv2.imread 读回来跑全部模块。

验证 BGR 约定与文件读取路径（这是实际业务里最常用的入口）。
"""

import cv2
import numpy as np

from _helpers import DIGITS, dot, draw_text
from imgcolor import Canny, Colors, CompareColors, FindColors, FontLib, Ocr, OcrX


def test_full_pipeline_with_real_png(tmp_path):
    img = np.zeros((120, 200, 3), np.uint8)

    # 多点找色/比色图案
    dot(img, 10, 10, "#FF0000")
    dot(img, 12, 10, "#00FF00")
    # 颜色数量色块
    cv2.rectangle(img, (50, 20), (70, 40), (0x95, 0xD3, 0x05), -1)  # #05D395 的 BGR
    # 边缘检测矩形
    cv2.rectangle(img, (100, 60), (160, 100), (255, 255, 255), -1)
    # 点阵文字
    draw_text(img, "12", 20, 80)

    path = tmp_path / "screen.png"
    assert cv2.imwrite(str(path), img)
    loaded = cv2.imread(str(path))
    assert loaded is not None
    assert loaded.dtype == np.uint8 and loaded.shape == img.shape
    assert np.array_equal(loaded, img)

    pattern = "10,10,#FF0000|12,10,#00FF00"
    p = FindColors.find(pattern, image=loaded)
    assert (p.x, p.y) == (10, 10)
    assert FindColors.find_color("#FF0000", image=loaded) == {"x": 10, "y": 10}
    assert CompareColors.compare(pattern, image=loaded) is True
    assert CompareColors.compare_multi(["1,1,#123456", pattern], image=loaded) == 1
    assert Colors.count("#05D395", image=loaded) == 441

    res = Canny(draw=Canny.DRAW_RECT, width_range=[50, 80]).run(loaded)
    assert any(d["width"] >= 55 and d["height"] >= 35 for d in res.data)

    lib = FontLib(color="#FFFFFF", diff=0.95)
    for ch, m in DIGITS.items():
        lib.add(ch, m)
    assert OcrX.find_all(lib, rect=[0, 75, 60, 100], image=loaded) == ["1", "2"]
    assert Ocr.matrix(lib, rect=[0, 75, 60, 100], image=loaded) == "12"


def test_gray_and_rgba_normalization():
    gray = np.zeros((30, 40), np.uint8)
    gray[5, 6] = 255
    # 灰度图自动转 BGR
    assert FindColors.find_color("#FFFFFF", image=gray) == {"x": 6, "y": 5}

    rgba = np.zeros((30, 40, 4), np.uint8)
    rgba[7, 8] = (255, 0, 0, 255)  # RGBA 红
    assert FindColors.find_color("#FF0000", image=rgba, color_order="RGB") == {"x": 8, "y": 7}


def test_float_image():
    img = np.zeros((20, 20, 3), np.float32)
    img[3, 4] = (1.0, 0.0, 0.0)  # BGR 蓝：#0000FF 对应 (B=255, G=0, R=0)
    assert FindColors.find_color("#0000FF", image=img) == {"x": 4, "y": 3}
