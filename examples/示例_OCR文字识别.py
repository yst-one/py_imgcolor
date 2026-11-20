"""OCR 便捷方法示例 —— 基于 RapidOCR，对齐 ascript Ocr 便捷 API。

安装可选依赖：pip install "imgcolor[ocr]"
运行：uv run python "examples/示例_OCR文字识别.py"（在项目目录下）
"""

from pathlib import Path
import tempfile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from imgcolor import Ocr, configure_engine


def make_screen():
    """渲染一张模拟界面截图（文字用系统黑体）。"""
    font = ImageFont.truetype(r"C:\Windows\Fonts\simhei.ttf", 48)
    canvas = Image.new("RGB", (640, 360), (245, 245, 245))
    d = ImageDraw.Draw(canvas)
    d.text((60, 60), "登录确认", font=font, fill=(0, 0, 0))
    d.text((60, 160), "欢迎使用", font=font, fill=(0, 0, 0))
    return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)


def main():
    configure_engine()  # 可选：调整 RapidOCR 参数后重建引擎
    img = make_screen()

    print("① find      ->", Ocr.find("登录", confidence=0.5, image=img))
    print("   find(正则)->", Ocr.find("确认|欢迎", confidence=0.5, image=img))
    print("② find_all ->", Ocr.find_all(confidence=0.5, image=img))
    print("③ exists   ->", Ocr.exists("欢迎", confidence=0.5, image=img))

    p = Path(tempfile.gettempdir()) / "imgcolor_ocr_screen.png"
    cv2.imwrite(str(p), img)

    clicked = []
    print("④ click    ->", Ocr.click("登录", confidence=0.5, image=str(p),
                                     clicker=lambda x, y: clicked.append((x, y))),
          "点击坐标:", clicked)

    n = {"v": 0}

    def provider():
        """模拟持续取图（真实场景可接 u2 截图）。"""
        n["v"] += 1
        return img

    print("⑤ wait     ->", Ocr.wait("欢迎", timeout=1, image=provider))
    print("⑥ wait_click ->", Ocr.wait_click("欢迎", timeout=1, image=img,
                                            clicker=lambda x, y: clicked.append((x, y))),
          "点击次数:", len(clicked))


if __name__ == "__main__":
    main()
