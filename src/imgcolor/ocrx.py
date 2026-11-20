"""点阵字库识别 —— 对应 ascript ``OcrX`` / ``Ocr.matrix``。

参考: https://www.ascript.cn/docs/android/api/screen/ocrx/

适用于游戏数字、定制字体等固定像素字体的小范围精确识别。
识别用 ``cv2.matchTemplate`` 做二值点阵的 XOR 匹配，速度远快于逐像素滑窗。
"""

from __future__ import annotations

import re
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import cv2
import numpy as np

from ._core import ImageLike, as_image, crop
from .fontlib import FontLib, binarize
from .rapidocr_engine import OcrText, recognize

FontLibLike = Union[str, "FontLib"]


class OcrX:
    """点阵字库识别。"""

    @staticmethod
    def find_all(
        font_lib: FontLibLike,
        rect: Optional[Sequence[int]] = None,
        region: float = 0.9,
        image: ImageLike = None,
        color_order: str = "BGR",
        detail: bool = False,
        scales: Sequence[float] = (1.0,),
        max_results: int = 200,
        color: Optional[str] = None,
        diff: Optional[float] = None,
        threshold: Optional[int] = None,
    ) -> Union[List[str], List[Dict[str, Any]]]:
        """给定字库，从图像中识别匹配的文字。

        :param font_lib: JSON 字库路径，或 :class:`FontLib` 实例
        :param rect: 识别范围 ``[x, y, x1, y1]``，None=整图
        :param region: 识别精度 0-1，默认 0.9（点阵匹配率）
        :param image: BGR ndarray（本库不截图，必传）
        :param detail: True 时返回 ``list[dict]``（含 text/x/y/width/height/score）
        :param scales: 模板缩放比例列表，如 ``(0.9, 1.0, 1.1)`` 适配字号微差
        :param color: 覆盖字库前景色；不传则用字库 color，再没有则 Otsu 自动二值化
        :return: 默认 ``list[str]``（按从左到右、从上到下顺序）；``detail=True`` 返回详情列表
        """
        lib = FontLib.load(font_lib)
        if not lib.glyphs:
            raise ValueError("字库为空，请先用 FontLib.add_from_image() 制作字模")

        img = as_image(image, color_order)
        sub, off_x, off_y = crop(img, rect)
        if sub.size == 0:
            return []
        mask = binarize(
            sub, color if color is not None else lib.color,
            lib.diff if diff is None else float(diff),
            lib.threshold if threshold is None else threshold,
        )
        if not mask.any():
            return []

        height, width = mask.shape
        img_f = mask.astype(np.float32)
        candidates: List[Dict[str, Any]] = []

        for glyph in lib.glyphs:
            tpl0 = glyph.ndarray()
            if tpl0.size == 0:
                continue
            for scale in scales:
                if float(scale) == 1.0:
                    tpl = tpl0
                else:
                    nw = max(1, int(round(tpl0.shape[1] * float(scale))))
                    nh = max(1, int(round(tpl0.shape[0] * float(scale))))
                    tpl = cv2.resize(tpl0, (nw, nh), interpolation=cv2.INTER_NEAREST)
                th, tw = tpl.shape[:2]
                if th > height or tw > width:
                    continue
                # 0/1 图像上 TM_SQDIFF 的平方差和 == XOR 像素数
                res = cv2.matchTemplate(img_f, tpl.astype(np.float32), cv2.TM_SQDIFF)
                score_map = np.clip(1.0 - res / float(tpl.size), 0.0, 1.0)
                dilated = cv2.dilate(score_map, np.ones((3, 3), np.float32))
                ys, xs = np.nonzero(
                    (score_map >= float(region) - 1e-6) & (score_map >= dilated - 1e-6)
                )
                if len(ys) == 0:
                    continue
                order = np.argsort(score_map[ys, xs])[::-1]
                for idx in order:
                    y, x = int(ys[idx]), int(xs[idx])
                    candidates.append({
                        "text": glyph.label,
                        "x": int(x + off_x), "y": int(y + off_y),
                        "width": int(tw), "height": int(th),
                        "score": float(score_map[y, x]),
                        "_cx": x + tw / 2.0, "_cy": y + th / 2.0,
                    })
                    if len(candidates) >= max_results * 20:
                        break

        # 非极大抑制：同一字符的重复命中只保留分数最高的一个
        candidates.sort(key=lambda c: -c["score"])
        kept: List[Dict[str, Any]] = []
        for c in candidates:
            overlap = any(
                abs(c["_cx"] - k["_cx"]) < max(c["width"], k["width"]) * 0.5
                and abs(c["_cy"] - k["_cy"]) < max(c["height"], k["height"]) * 0.5
                for k in kept
            )
            if overlap:
                continue
            kept.append(c)
            if len(kept) >= max_results:
                break

        max_h = max((c["height"] for c in kept), default=1)
        kept.sort(key=lambda c: (int(c["_cy"] // max(1, max_h)), c["x"]))
        for c in kept:
            c.pop("_cx", None)
            c.pop("_cy", None)
            c["score"] = round(c["score"], 4)

        if detail:
            return kept
        return [c["text"] for c in kept]

    @staticmethod
    def find_detail(font_lib: FontLibLike, **kwargs) -> List[Dict[str, Any]]:
        """``find_all(detail=True)`` 的便捷入口。"""
        kwargs["detail"] = True
        return OcrX.find_all(font_lib, **kwargs)  # type: ignore[return-value]


def _as_pattern(text: Any) -> Optional[str]:
    """把查找文字转成正则；非法正则退化为字面量匹配。"""
    if text is None:
        return None
    value = str(text)
    if not value:
        return None
    try:
        re.compile(value)
        return value
    except re.error:
        return re.escape(value)


class Ocr:
    """文字识别便捷入口（对齐 ascript ``Ocr``）。

    ``find`` / ``find_all`` / ``exists`` / ``click`` / ``wait`` / ``wait_click``
    底层统一使用 RapidOCR；``matrix`` 为点阵字库识别（无需 RapidOCR）。

    本库不做截图、不做设备点击：

    - 所有方法都必须通过 ``image=`` 传入图片（ndarray / 路径 / Pillow Image）
    - ``click`` / ``wait_click`` 需要先 ``Ocr.set_clicker(fn)``，
      或直接传 ``clicker=lambda x, y: ...``；回调收到文字中心坐标
    - ``timeout > 0`` 且 ``image`` 传「返回新图的函数」时才会真正轮询刷新；
      静态图片只识别一次
    """

    MODE_RAPIDOCR = "rapidocr"
    POLL_INTERVAL = 0.2
    _clicker: Optional[Callable[[int, int], Any]] = None

    # ------------------------------------------------------------------ #
    @staticmethod
    def set_clicker(clicker: Optional[Callable[[int, int], Any]]) -> None:
        """注册全局点击回调 ``fn(x, y)``；传 None 清除。"""
        if clicker is not None and not callable(clicker):
            raise TypeError("clicker 必须是可调用对象 fn(x, y)")
        Ocr._clicker = clicker

    @staticmethod
    def _recognize(image: ImageLike, rect: Optional[Sequence[int]], mode: Optional[int],
                   binary: int, confidence: float, text: Any,
                   color_order: str) -> List[OcrText]:
        # mode 为 ascript 兼容参数：本库只有 RapidOCR 一个引擎，保留签名但不切换
        return recognize(
            image,
            rect=rect,
            binary=binary,
            confidence=confidence,
            pattern=_as_pattern(text),
            color_order=color_order,
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def find(text: Any, rect: Optional[Sequence[int]] = None, mode: Optional[int] = None,
             binary: int = -1, confidence: float = 0.1, image: ImageLike = None,
             color_order: str = "BGR") -> Optional[Dict[str, Any]]:
        """查找文字，返回第一个匹配结果 dict；未找到返回 None。``text`` 支持正则。"""
        src = image() if callable(image) else image
        hits = Ocr._recognize(src, rect, mode, binary, confidence, text, color_order)
        return hits[0].as_dict() if hits else None

    @staticmethod
    def find_all(text: Optional[str] = None, rect: Optional[Sequence[int]] = None,
                 mode: Optional[int] = None, binary: int = -1, confidence: float = 0.1,
                 image: ImageLike = None,
                 color_order: str = "BGR") -> Optional[List[Dict[str, Any]]]:
        """返回所有匹配结果；``text=None`` 返回全部文字。无结果返回 None。"""
        src = image() if callable(image) else image
        hits = Ocr._recognize(src, rect, mode, binary, confidence, text, color_order)
        return [h.as_dict() for h in hits] if hits else None

    @staticmethod
    def exists(text: Any, rect: Optional[Sequence[int]] = None, mode: Optional[int] = None,
               binary: int = -1, confidence: float = 0.1, image: ImageLike = None,
               color_order: str = "BGR") -> bool:
        """判断文字是否存在。"""
        return Ocr.find(text, rect, mode, binary, confidence, image, color_order) is not None

    # ------------------------------------------------------------------ #
    @staticmethod
    def _wait_hit(text: Any, timeout: float, rect: Optional[Sequence[int]], mode: Optional[int],
                  binary: int, confidence: float, image: ImageLike,
                  color_order: str) -> Optional[Dict[str, Any]]:
        deadline = time.monotonic() + max(0.0, float(timeout))
        retryable = callable(image)
        while True:
            src = image() if retryable else image
            hits = Ocr._recognize(src, rect, mode, binary, confidence, text, color_order)
            if hits:
                return hits[0].as_dict()
            if not retryable or time.monotonic() >= deadline:
                return None
            time.sleep(min(Ocr.POLL_INTERVAL, max(0.001, deadline - time.monotonic())))

    @staticmethod
    def click(text: Any, rect: Optional[Sequence[int]] = None, mode: Optional[int] = None,
              binary: int = -1, confidence: float = 0.1, timeout: float = 0,
              image: ImageLike = None, clicker: Optional[Callable[[int, int], Any]] = None,
              color_order: str = "BGR") -> bool:
        """找到文字并点击其中心。成功 True，未找到 False。

        点击能力由 ``clicker(x, y)`` 提供（如 uiautomator2 的 ``d.click``）；
        不传时用 :meth:`set_clicker` 注册的全局回调。未配置时抛 ValueError。
        """
        fn = clicker if clicker is not None else Ocr._clicker
        if fn is None:
            raise ValueError(
                "本库不提供设备点击能力：请先 Ocr.set_clicker(lambda x, y: ...) "
                "或给 click() 传 clicker=lambda x, y: ..."
            )
        if not callable(fn):
            raise TypeError("clicker 必须是可调用对象 fn(x, y)")
        hit = Ocr._wait_hit(text, timeout, rect, mode, binary, confidence, image, color_order)
        if hit is None:
            return False
        fn(hit["center_x"], hit["center_y"])
        return True

    @staticmethod
    def wait(text: Any, timeout: float = 10, rect: Optional[Sequence[int]] = None,
             mode: Optional[int] = None, binary: int = -1, confidence: float = 0.1,
             image: ImageLike = None, color_order: str = "BGR") -> Optional[Dict[str, Any]]:
        """等待文字出现（``timeout`` 单位为秒），超时返回 None。

        只有 ``image`` 传「返回新图的函数」时才会在超时前反复取图；
        静态 ndarray / 路径只识别一次。
        """
        return Ocr._wait_hit(text, timeout, rect, mode, binary, confidence, image, color_order)

    @staticmethod
    def wait_click(text: Any, timeout: float = 10, rect: Optional[Sequence[int]] = None,
                   mode: Optional[int] = None, binary: int = -1, confidence: float = 0.1,
                   image: ImageLike = None,
                   clicker: Optional[Callable[[int, int], Any]] = None,
                   color_order: str = "BGR") -> bool:
        """等待文字出现并点击；超时未找到返回 False。"""
        fn = clicker if clicker is not None else Ocr._clicker
        if fn is None:
            raise ValueError(
                "本库不提供设备点击能力：请先 Ocr.set_clicker(lambda x, y: ...) "
                "或给 wait_click() 传 clicker=lambda x, y: ..."
            )
        if not callable(fn):
            raise TypeError("clicker 必须是可调用对象 fn(x, y)")
        hit = Ocr._wait_hit(text, timeout, rect, mode, binary, confidence, image, color_order)
        if hit is None:
            return False
        fn(hit["center_x"], hit["center_y"])
        return True

    # ------------------------------------------------------------------ #
    @staticmethod
    def matrix(font_lib: FontLibLike, rect: Optional[Sequence[int]] = None,
               region: float = 0.9, image: ImageLike = None, **kwargs) -> str:
        """点阵字库识别，返回拼接好的字符串；识别不到返回 ``""``。"""
        res = OcrX.find_all(font_lib, rect=rect, region=region, image=image, **kwargs)
        return "".join(str(r) for r in res) if res else ""


__all__ = ["OcrX", "Ocr", "OcrText"]
