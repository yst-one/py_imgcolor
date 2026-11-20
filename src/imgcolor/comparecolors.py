"""多点比色 —— 方法命名对齐 ascript ``CompareColors``。

与"找色"的区别：比色校验的是**固定绝对坐标**上的颜色，不搜索位置。
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np

from ._core import ImageLike, color_mask, parse_points, run_until


class CompareColors:
    """多点比色。所有方法通过 ``image=``（或 ``bitmap=``）传入 BGR ndarray。"""

    @staticmethod
    def compare(
        colors: str,
        diff: float = 0.9,
        bitmap: ImageLike = None,
        mode: int = 0,
        until: Union[int, float] = 0,
        sim: float = 1.0,
        image: ImageLike = None,
        color_order: str = "BGR",
    ) -> bool:
        """比色。

        :param colors: 固定点位颜色特征串 ``'339,342,#38D8A8|445,424,#FFFFFF'``
        :param diff: 单点颜色相似度 0-1，默认 0.9
        :param bitmap: 图片（ascript 参数名，等价于 ``image``）
        :param mode: 0=RGB, 1=Norm
        :param until: 超时毫秒数，0=不重试；>0 且 image 为函数时持续重试
        :param sim: 整体相似度阈值 0-1，默认 1.0（所有点都必须通过）。
                    即"通过点数 / 总点数 >= sim"才算比色成功。
        :return: True / False
        """
        src = image if image is not None else bitmap
        pts = parse_points(colors)
        if any(p.x is None for p in pts):
            raise ValueError("多点比色需要绝对坐标，例如 '339,342,#38D8A8|445,424,#FFFFFF'")

        def _run(img: np.ndarray) -> bool:
            height, width = img.shape[:2]
            passed = 0
            total = len(pts)
            for p in pts:
                x, y = int(p.x), int(p.y)  # type: ignore[arg-type]
                if x < 0 or y < 0 or x >= width or y >= height:
                    continue
                patch = img[y : y + 1, x : x + 1]
                if bool(color_mask(patch, p.spec, diff=diff, mode=mode)[0, 0]):
                    passed += 1
            return (passed / total if total else 0.0) >= float(sim) - 1e-9

        return bool(run_until(src, until, _run, color_order))

    @staticmethod
    def compare_multi(
        colors_list: Sequence[str],
        diff: float = 0.9,
        mode: int = 0,
        sim: float = 1.0,
        image: ImageLike = None,
        bitmap: ImageLike = None,
        until: Union[int, float] = 0,
        color_order: str = "BGR",
    ) -> int:
        """多组比色，返回第一个匹配的索引（从 0 开始），全部不匹配返回 -1。

        多组共用同一张图，只做一次图像规范化。
        """
        src = image if image is not None else bitmap

        def _run(img: np.ndarray):
            for index, colors in enumerate(colors_list):
                if CompareColors.compare(
                    colors, diff=diff, mode=mode, sim=sim, image=img, until=0
                ):
                    return index
            return None

        # 索引 0 是 falsy、-1 不能作为"失败"信号，用 None 表示未匹配
        result = run_until(src, until, _run, color_order, done=lambda r: r is not None)
        return -1 if result is None else int(result)


__all__ = ["CompareColors"]
