"""测试公共工具。"""

from pathlib import Path

import cv2
import numpy as np


def imwrite_any(path, arr: np.ndarray) -> str:
    """写图片，兼容中文/空格路径（cv2.imwrite 在 Windows 上不支持中文路径）。"""
    p = Path(path)
    ext = p.suffix or ".png"
    ok, buf = cv2.imencode(ext, arr)
    assert ok, f"imencode 失败: {ext}"
    buf.tofile(str(p))
    return str(p)


def bgr(rgb_hex: str):
    """'#RRGGBB' -> BGR 元组（写 ndarray 用）。"""
    h = rgb_hex.lstrip("#")
    return int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)


def dot(img: np.ndarray, x: int, y: int, rgb_hex: str, w: int = 1, h: int = 1) -> None:
    img[y : y + h, x : x + w] = bgr(rgb_hex)


DIGITS = {
    "1": [
        [0, 0, 1, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
    ],
    "2": [
        [0, 1, 1, 1, 0],
        [1, 0, 0, 0, 1],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0],
        [1, 1, 1, 1, 1],
    ],
    "3": [
        [1, 1, 1, 1, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1],
        [1, 1, 1, 1, 0],
    ],
}


def draw_text(img: np.ndarray, text: str, x0: int, y0: int,
              gap: int = 3, scale: int = 1) -> None:
    """把 DIGITS 点阵画成白字黑底（BGR 255 白色前景）。"""
    cursor = x0
    for ch in text:
        m = np.asarray(DIGITS[ch], dtype=np.uint8)
        if scale > 1:
            m = np.repeat(np.repeat(m, scale, axis=0), scale, axis=1)
        h, w = m.shape
        block = img[y0 : y0 + h, cursor : cursor + w]
        block[m > 0] = (255, 255, 255)
        cursor += w + gap
