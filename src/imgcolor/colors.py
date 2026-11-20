"""颜色数量 —— 方法命名对齐 ascript ``Colors``。"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from ._core import ImageLike, as_image, color_mask, crop, parse_points


class Colors:
    """统计图像区域内指定颜色的像素数量。"""

    @staticmethod
    def mask(
        colors: str,
        rect: Optional[Sequence[int]] = None,
        sim: float = 1,
        image: ImageLike = None,
        bitmap: ImageLike = None,
        color_order: str = "BGR",
    ) -> np.ndarray:
        """返回局部区域内颜色匹配的布尔掩码 (h, w)，坐标为区域局部坐标。

        支持 ``"#05D395"``、``"#05D395-#101010"``（偏色），
        也支持 ``|`` 分隔的多个颜色（取并集，同一像素只计一次）。
        """
        src = image if image is not None else bitmap
        img = as_image(src, color_order)
        sub, _, _ = crop(img, rect)
        specs = parse_points(colors)
        if sub.size == 0:
            return np.zeros(sub.shape[:2], dtype=bool)
        combined = np.zeros(sub.shape[:2], dtype=bool)
        for p in specs:
            # 存在偏色值时 sim 失效（与 ascript 文档一致）
            combined |= color_mask(sub, p.spec, diff=sim, mode=0)
        return combined

    @staticmethod
    def count(
        colors: str,
        rect: Optional[Sequence[int]] = None,
        sim: float = 1,
        bitmap: ImageLike = None,
        draw_mode: int = 1,
        image: ImageLike = None,
        color_order: str = "BGR",
    ) -> int:
        """统计区域内指定颜色的像素数量。

        :param colors: 颜色特征串，如 ``"#05D395-#101010"``（偏色）或 ``"#05D395"``
        :param rect: 圈定范围 ``[x, y, x1, y1]``，None=全图
        :param sim: 相似度 0-1，1=100% 匹配；**当颜色串带偏色值时该参数失效**
        :param bitmap: 图片（ascript 参数名，等价于 ``image``）
        :param draw_mode: 绘制模式（ascript 会在屏幕上绘制调试图，本库纯计算，
                          该参数保留以兼容签名；需要调试图请用 :meth:`count_image`）
        :return: int，颜色点的数量
        """
        return int(np.count_nonzero(Colors.mask(
            colors, rect=rect, sim=sim, image=image, bitmap=bitmap,
            color_order=color_order,
        )))

    @staticmethod
    def count_image(
        colors: str,
        rect: Optional[Sequence[int]] = None,
        sim: float = 1,
        bitmap: ImageLike = None,
        draw_mode: int = 1,
        image: ImageLike = None,
        color_order: str = "BGR",
    ) -> Tuple[int, np.ndarray]:
        """与 :meth:`count` 相同，但额外返回调试图（BGR ndarray）。

        - ``draw_mode=1`` 阈值绘制：命中像素画白，其余黑（二值图）
        - ``draw_mode=2`` 黑色背景：命中像素保留原色，其余涂黑
        """
        src = image if image is not None else bitmap
        img = as_image(src, color_order)
        sub, _, _ = crop(img, rect)
        m = Colors.mask(colors, rect=rect, sim=sim, image=img)
        count = int(np.count_nonzero(m))
        if m.size == 0:
            return count, np.zeros((0, 0, 3), dtype=np.uint8)
        if int(draw_mode) == 2:
            out = np.zeros_like(sub)
            out[m] = sub[m]
        else:
            out = cv2.cvtColor((m.astype(np.uint8) * 255), cv2.COLOR_GRAY2BGR)
        return count, out


__all__ = ["Colors"]
