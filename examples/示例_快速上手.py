# -*- coding: utf-8 -*-
"""imgcolor 快速上手示例（全部用合成图，无需手机、无需截图）。

运行：
    uv run python examples/示例_快速上手.py

实际业务里把 ``img`` 换成你的截图 ndarray 即可，例如：
    import cv2
    img = cv2.imread("screen.png")                       # BGR
    # 或 uiautomator2：
    # from PIL import Image
    # import io, numpy as np
    # pil = d.screenshot()
    # img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
"""

import tempfile
from pathlib import Path

import cv2
import numpy as np

from imgcolor import Canny, Colors, CompareColors, FindColors, FindImages, FontLib, Ocr, OcrX


def put(img, x, y, hex_rgb, w=1, h=1):
    """按 #RRGGBB 往 BGR 图上画色块。"""
    s = hex_rgb.lstrip("#")
    img[y : y + h, x : x + w] = (int(s[4:6], 16), int(s[2:4], 16), int(s[0:2], 16))


def main():
    img = np.zeros((300, 500, 3), np.uint8)

    # ---- 1. 多点找色 -------------------------------------------------- #
    # 特征串：锚点(10,10)红 + 右移2像素绿；在两处摆放同样的图案
    for x, y in ((10, 10), (200, 100)):
        put(img, x, y, "#FF0000")
        put(img, x + 2, y, "#00FF00")
    pattern = "10,10,#FF0000|12,10,#00FF00"

    p = FindColors.find(pattern, image=img)
    print("① find       ->", p)
    print("  find_all   ->", FindColors.find_all(pattern, image=img))
    print("  find_color ->", FindColors.find_color("#FF0000", image=img))
    print("  find_all_color(前3) ->",
          FindColors.find_all_color("#FF0000", max_num=3, image=img))

    # ---- 2. 多点比色 -------------------------------------------------- #
    state_a = "10,10,#FF0000|12,10,#00FF00"
    state_b = "200,100,#FF0000|202,100,#00FF00"
    print("② compare    ->", CompareColors.compare(state_a, image=img))
    print("  compare_multi ->",
          CompareColors.compare_multi(["1,1,#123456", state_b], image=img))

    # ---- 3. 颜色数量 -------------------------------------------------- #
    put(img, 300, 50, "#05D395", 20, 20)
    print("③ count      ->", Colors.count("#05D395", rect=[300, 50, 320, 70], image=img))
    n, debug = Colors.count_image("#05D395", image=img, draw_mode=1)
    print("  count_image->", n, "调试图尺寸", debug.shape)

    # ---- 4. 边缘检测 -------------------------------------------------- #
    cv2.rectangle(img, (350, 150), (460, 230), (255, 255, 255), -1)
    res = Canny(draw=Canny.DRAW_RECT, width_range=[50, 200]).run(img)
    print("④ canny 轮廓 ->", res.data)
    # cv2.imwrite("canny_debug.png", res.image)  # 需要时保存绘制结果

    # ---- 5. 点阵字库 -------------------------------------------------- #
    digit_1 = [
        [0, 0, 1, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
    ]
    digit_2 = [
        [0, 1, 1, 1, 0],
        [1, 0, 0, 0, 1],
        [0, 0, 0, 0, 1],
        [0, 0, 1, 1, 0],
        [0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1],
    ]
    lib = FontLib(color="#FFFFFF", diff=0.95)
    lib.add("1", digit_1)
    lib.add("2", digit_2)

    text_img = np.zeros((40, 80, 3), np.uint8)
    for i, matrix in enumerate((digit_1, digit_2)):
        m = np.asarray(matrix, np.uint8)
        x0 = 5 + i * 8
        text_img[10 : 10 + 7, x0 : x0 + 5][m > 0] = (255, 255, 255)

    lib_path = Path(tempfile.gettempdir()) / "imgcolor_demo_font.json"
    lib.save(lib_path)
    print("⑤ OcrX.find_all ->", OcrX.find_all(str(lib_path), image=text_img))
    print("  Ocr.matrix    ->", Ocr.matrix(str(lib_path), image=text_img))
    print("  字库文件      ->", lib_path)

    # 从图片直接制作字模（真实项目里用图色工具框选字符）
    g = FontLib(color="#FFFFFF").add_from_image("1", text_img, rect=[5, 10, 10, 17])
    print("  add_from_image->", g.label, g.width, "x", g.height)

    # ---- 6. 找图（模板匹配）------------------------------------------- #
    # 模板可以直接传 ndarray，也可以传图片路径 / 路径列表（多图搜索）
    tpl = np.zeros((16, 16, 3), np.uint8)
    tpl[2:14, 2:14] = (0, 0, 255)
    tpl[5:11, 5:11] = (0, 0, 0)
    tpl[7:9, 7:9] = (0, 0, 255)
    img[250:266, 400:416] = tpl

    hit = FindImages.find(tpl, image=img, confidence=0.9)
    print("⑥ FindImages.find ->", hit)
    # 模板路径 + 源图路径（Windows 下用 / 或原始字符串，避免 \1 之类的转义坑）
    tpl_png = Path(tempfile.gettempdir()) / "imgcolor_demo_tpl.png"
    src_png = Path(tempfile.gettempdir()) / "imgcolor_demo_src.png"
    cv2.imwrite(str(tpl_png), tpl)
    cv2.imwrite(str(src_png), img)
    print("  find_all(路径) ->",
          FindImages.find_all(str(tpl_png), image=str(src_png), confidence=0.9))

    # 用户素材：小图作模板、大图作源图（Windows 路径用 / 或原始字符串，避免 \1 转义）
    real_tpl, real_src = "examples/1.png", "examples/image.png"
    if Path(real_tpl).is_file() and Path(real_src).is_file():
        # 这两张不是同源截图（模板带不同背景），分数只有 ~0.6，故用较低阈值演示
        loose = FindImages.find_all(real_tpl, image=real_src, confidence=0.6)
        print("  路径找图(跨来源素材) ->", loose[:2] if loose else None)
        # 模板直接从目标截图裁，置信度才接近 1.0（实际项目请这样做）
        src_arr = cv2.imdecode(np.fromfile(real_src, np.uint8), cv2.IMREAD_UNCHANGED)
        same_tpl = src_arr[140:186, 140:186]
        print("  同源模板自截 ->", FindImages.find(same_tpl, image=real_src, confidence=0.9))
    # 跨分辨率：scale_range=[最小缩放比, 最大缩放比]，自动约 15 个步长
    # hit = FindImages.find(tpl, image=img, confidence=0.8, scale_range=[0.8, 1.2])
    # 透明 PNG（alpha 作 mask）与多图搜索：
    # hit = FindImages.find(["a.png", "b.png"], image=img, confidence=0.8)
    # print(hit["index"])  # 匹配的是第几张模板图

    # ---- 7. 直接传图片路径 / Pillow Image ------------------------------ #
    # 所有 image= / bitmap= / source_img= / 模板参数都支持路径（含中文）和 PIL Image
    png = src_png  # 复用第 ⑥ 段保存的源图路径
    print("⑦ 文件路径输入 ->", FindColors.find_color("#FF0000", image=str(png)))
    print("  Path 对象输入 ->", Colors.count("#05D395", image=png))
    try:
        from PIL import Image

        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        print("  Pillow 输入   ->", FindImages.find(tpl, image=pil, confidence=0.9))
    except ImportError:
        print("  （未安装 Pillow，跳过 PIL 演示；库本身不依赖 Pillow）")


if __name__ == "__main__":
    main()
