"""多点找色 / 单点找色 —— 对应 ascript ``FindColors``。

参考: https://www.ascript.cn/docs/android/api/screen/findcolors/

特征串格式（图色助手生成）::

    "x0,y0,#RRGGBB|x1,y1,#RRGGBB|x2,y2,#RRGGBB"

第一个点是**锚点**，其余点按相对锚点的偏移一起验证；
返回的坐标是锚点坐标（与 ascript 一致）。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from ._core import (
    ImageLike,
    Point,
    color_mask,
    crop,
    nms,
    normalize_rect,
    parse_color_spec,
    parse_points,
    run_until,
    shift_mask,
    sort_points,
)


class FindColors:
    """多点找色 / 单点找色。所有方法通过 ``image=`` 传入 BGR ndarray。"""

    # ------------------------------------------------------------------ #
    @staticmethod
    def find(
        colors: str,
        rect: Optional[Sequence[int]] = None,
        space: int = 5,
        ori: int = 2,
        diff: float = 0.98,
        image: ImageLike = None,
        mode: int = 0,
        until: Union[int, float] = 0,
        color_order: str = "BGR",
    ) -> Optional[Point]:
        """找到第一个符合特征点的位置，立刻返回。

        :param colors: 颜色特征串，如 ``'634,2126,#05D395|675,2142,#05D395'``
        :param rect: 搜索范围 ``[x, y, x1, y1]``，None=全图
        :param space: 结果间距，默认 5 像素（``find`` 只取一个，实际不影响结果）
        :param ori: 找色方向 1-8，默认 2（左上→右下，横向优先）
        :param diff: 颜色相似度 0-1，默认 0.98
        :param image: BGR ndarray（本库不截图，必传）；也可传返回 ndarray 的函数配合 until
        :param mode: 0=RGB, 1=Norm（归一化颜色，抗亮度变化）
        :param until: 超时毫秒数，0=不重试；>0 且 image 为函数时持续重试
        :return: :class:`Point` 或 None
        """
        points = FindColors.find_all(
            colors, rect=rect, space=space, ori=ori, diff=diff, image=image,
            p_max_num=1, mode=mode, until=until, color_order=color_order,
        )
        return points[0] if points else None

    # ------------------------------------------------------------------ #
    @staticmethod
    def find_all(
        colors: str,
        rect: Optional[Sequence[int]] = None,
        space: int = 5,
        ori: int = 2,
        diff: float = 0.98,
        image: ImageLike = None,
        p_max_num: int = 999999,
        mode: int = 0,
        until: Union[int, float] = 0,
        color_order: str = "BGR",
    ) -> Optional[List[Point]]:
        """找到全部符合特征点的位置。

        :param p_max_num: 最大返回数量，默认 999999
        :return: ``list[Point]``，未找到返回 None（与 ascript 一致）
        """
        pts = parse_points(colors)
        if pts[0].x is None:
            raise ValueError("多点找色需要带坐标的颜色特征串，例如 '10,20,#FF0000|15,20,#00FF00'")

        anchor = pts[0]
        offsets = [(p.x - anchor.x, p.y - anchor.y, p.spec) for p in pts[1:]]  # type: ignore[operator]

        def _run(img: np.ndarray) -> List[Point]:
            height, width = img.shape[:2]
            cache: Dict[Any, np.ndarray] = {}

            def mask_of(spec):
                key = (spec.bgr, spec.tol)
                if key not in cache:
                    cache[key] = color_mask(img, spec, diff=diff, mode=mode)
                return cache[key]

            # 基于整幅图算颜色掩码：偏移点允许落在 rect 之外
            base = mask_of(anchor.spec)
            if not base.any():
                return []
            for ox, oy, spec in offsets:
                base = base & shift_mask(mask_of(spec), ox, oy)
                if not base.any():
                    return []

            x0, y0, x1, y1 = normalize_rect(rect, width, height)
            roi = base[y0:y1, x0:x1]
            if roi.size == 0:
                return []
            ys, xs = np.nonzero(roi)
            if len(ys) == 0:
                return []

            cand = [(int(x + x0), int(y + y0)) for y, x in zip(ys, xs)]
            cand = sort_points(cand, ori)
            cand = nms(cand, int(space))
            limit = int(p_max_num) if p_max_num else 0
            if limit > 0:
                cand = cand[:limit]
            return [Point(x, y) for x, y in cand]

        result = run_until(image, until, _run, color_order)
        return result or None

    # ------------------------------------------------------------------ #
    @staticmethod
    def find_color(
        color: str,
        rect: Optional[Sequence[int]] = None,
        diff: float = 0.98,
        mode: int = 0,
        image: ImageLike = None,
        color_order: str = "BGR",
    ) -> Optional[Dict[str, int]]:
        """查找第一个匹配指定颜色的像素点（行优先，左上→右下）。

        :param color: 目标颜色 ``"#RRGGBB"``，也支持 ``"#RRGGBB-#101010"`` 偏色写法
        :return: ``{"x": 100, "y": 200}`` 或 None
        """
        from ._core import as_image

        spec = parse_color_spec(color)
        img = as_image(image, color_order)
        sub, x0, y0 = crop(img, rect)
        if sub.size == 0:
            return None
        mask = color_mask(sub, spec, diff=diff, mode=mode)
        ys, xs = np.nonzero(mask)
        if len(ys) == 0:
            return None
        return {"x": int(xs[0] + x0), "y": int(ys[0] + y0)}

    # ------------------------------------------------------------------ #
    @staticmethod
    def find_all_color(
        color: str,
        rect: Optional[Sequence[int]] = None,
        diff: float = 0.98,
        mode: int = 0,
        max_num: int = 999999,
        image: ImageLike = None,
        color_order: str = "BGR",
    ) -> Optional[List[Dict[str, int]]]:
        """查找所有匹配颜色的像素点。

        :param max_num: 最多返回数量，默认 999999
        :return: ``list[dict]``，未找到返回 None
        """
        from ._core import as_image

        spec = parse_color_spec(color)
        img = as_image(image, color_order)
        sub, x0, y0 = crop(img, rect)
        if sub.size == 0:
            return None
        mask = color_mask(sub, spec, diff=diff, mode=mode)
        ys, xs = np.nonzero(mask)
        if len(ys) == 0:
            return None
        limit = int(max_num) if max_num else 0
        if limit > 0:
            ys, xs = ys[:limit], xs[:limit]
        return [{"x": int(x + x0), "y": int(y + y0)} for y, x in zip(ys, xs)]


__all__ = ["FindColors"]
