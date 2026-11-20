"""边缘检测 —— 方法命名对齐 ascript ``screen.gp_tasks.Canny``。

ascript 里 Canny 是 GPStack 的插件；这里做成可直接调用的类/函数，
``.run(image)`` 返回 ``CannyResult(image, data, contours, edges)``。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np

from ._core import ImageLike, as_image, crop


@dataclass
class CannyResult:
    """边缘检测结果。

    :param image: 绘制后的 BGR 图像（原图副本）
    :param data: 过滤后的轮廓信息列表，每项含 x/y/width/height/area/perimeter/points
    :param contours: 过滤后的 OpenCV 轮廓
    :param edges: Canny 二值边缘图
    """

    image: np.ndarray
    data: List[Dict[str, Any]] = field(default_factory=list)
    contours: List[np.ndarray] = field(default_factory=list)
    edges: Optional[np.ndarray] = None


def _in_range(value: float, rng: Optional[Sequence[float]]) -> bool:
    if rng is None or len(rng) == 0:
        return True
    lo = float(rng[0])
    hi = float(rng[1]) if len(rng) > 1 else float("inf")
    return lo <= value <= hi


class Canny:
    """Canny 边缘检测 + 轮廓过滤 + 绘制。

    :param low_threshold: 最小梯度差，默认 50
    :param high_threshold: 最大梯度差，默认 150
    :param mode: cv2 轮廓检索模式，默认 ``RETR_EXTERNAL``
    :param draw: 绘制模式，见类常量
    :param desc: 是否在图上绘制轮廓信息文字
    :param left_range / top_range / width_range / height_range: ``[min, max]`` 过滤
    :param blur: 高斯模糊核大小（奇数，0=不模糊），默认 0
    :param min_area: 最小轮廓面积，默认 0（不过滤）
    """

    RETR_EXTERNAL = cv2.RETR_EXTERNAL
    RETR_LIST = cv2.RETR_LIST
    RETR_CCOMP = cv2.RETR_CCOMP
    RETR_TREE = cv2.RETR_TREE

    DRAW_NONE = 0
    DRAW_ALL = 1
    DRAW_RECT = 2
    DRAW_RECT_FILL = 3

    def __init__(
        self,
        low_threshold: int = 50,
        high_threshold: int = 150,
        mode: int = None,
        draw: int = None,
        desc: bool = False,
        left_range: Optional[Sequence[float]] = None,
        top_range: Optional[Sequence[float]] = None,
        width_range: Optional[Sequence[float]] = None,
        height_range: Optional[Sequence[float]] = None,
        blur: int = 0,
        min_area: float = 0,
    ):
        self.low_threshold = int(low_threshold)
        self.high_threshold = int(high_threshold)
        self.mode = self.RETR_EXTERNAL if mode is None else int(mode)
        self.draw = self.DRAW_ALL if draw is None else int(draw)
        self.desc = bool(desc)
        self.left_range = left_range
        self.top_range = top_range
        self.width_range = width_range
        self.height_range = height_range
        self.blur = int(blur)
        self.min_area = float(min_area)

    def run(self, image: ImageLike, rect: Optional[Sequence[int]] = None,
            color_order: str = "BGR") -> CannyResult:
        """对图像做边缘检测。

        :param image: BGR ndarray
        :param rect: 可选，只在该区域内检测（结果坐标为全图坐标）
        """
        img = as_image(image, color_order)
        sub, off_x, off_y = crop(img, rect)
        out = img.copy()
        if sub.size == 0:
            return CannyResult(out, [], [], np.zeros(sub.shape[:2], np.uint8))

        gray = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
        if self.blur >= 3:
            k = self.blur if self.blur % 2 == 1 else self.blur + 1
            gray = cv2.GaussianBlur(gray, (k, k), 0)
        edges = cv2.Canny(gray, self.low_threshold, self.high_threshold)
        contours, _ = cv2.findContours(edges, self.mode, cv2.CHAIN_APPROX_SIMPLE)

        kept: List[np.ndarray] = []
        data: List[Dict[str, Any]] = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = float(cv2.contourArea(cnt))
            if area < self.min_area:
                continue
            if not (
                _in_range(x + off_x, self.left_range)
                and _in_range(y + off_y, self.top_range)
                and _in_range(w, self.width_range)
                and _in_range(h, self.height_range)
            ):
                continue
            gx, gy = x + off_x, y + off_y
            kept.append(cnt)
            data.append({
                "x": gx, "y": gy, "width": w, "height": h,
                "area": round(area, 2),
                "perimeter": round(float(cv2.arcLength(cnt, True)), 2),
                "points": int(len(cnt)),
            })
            self._draw_one(out, cnt, (gx, gy, w, h), off_x, off_y)

        return CannyResult(out, data, kept, edges)

    def _draw_one(self, out: np.ndarray, cnt: np.ndarray, box: Sequence[int],
                  off_x: int, off_y: int) -> None:
        # 轮廓取自裁剪子图，绘制到全图时统一平移 (off_x, off_y)
        if self.draw == self.DRAW_NONE:
            return
        x, y, w, h = box
        if self.draw == self.DRAW_ALL:
            shifted = cnt + np.array([[[off_x, off_y]]], dtype=cnt.dtype)
            cv2.drawContours(out, [shifted], -1, (0, 255, 0), 2)
        elif self.draw == self.DRAW_RECT:
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 255), 2)
        elif self.draw == self.DRAW_RECT_FILL:
            cv2.rectangle(out, (x, y), (x + w, y + h), (255, 255, 255), -1)
        if self.desc:
            cv2.putText(
                out, f"{x},{y} {w}x{h}", (x, max(12, y - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA,
            )

    def __call__(self, image: ImageLike, **kwargs) -> CannyResult:
        return self.run(image, **kwargs)


def canny(image: ImageLike, **kwargs) -> CannyResult:
    """函数式快捷入口：``canny(img, draw=Canny.DRAW_RECT, width_range=[100, 500])``。"""
    return Canny(**kwargs).run(image)


__all__ = ["Canny", "CannyResult", "canny"]
