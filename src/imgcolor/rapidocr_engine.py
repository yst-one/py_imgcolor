"""RapidOCR 引擎封装 —— 供 :class:`imgcolor.Ocr` 的便捷方法使用。

只负责把 ``image`` 规范化、裁剪 ``rect``、可选二值化，然后调用 RapidOCR，
把结果整理为 ascript 风格的 ``OcrText`` 对象。所有坐标都还原到全图。

本库不截图；``image`` 必传，支持 ndarray / 图片路径 / Pillow Image / 无参函数。

RapidOCR 是可选依赖：``pip install "imgcolor[ocr]"``
（等价于 ``pip install rapidocr onnxruntime``）。未安装时给出明确提示。
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ._core import ImageLike, as_image, crop, normalize_rect

_INSTALL_HINT = (
    'OCR 功能需要 RapidOCR：pip install "imgcolor[ocr]" '
    "（或 pip install rapidocr onnxruntime）"
)


@dataclass(frozen=True)
class OcrText:
    """一条 OCR 识别结果，兼容 ascript 的字段与字典式访问。"""

    text: str
    rect: List[int]                    # [left, top, right, bottom]
    center_x: int
    center_y: int
    confidence: float = 1.0

    def __getitem__(self, key: Any) -> Any:
        if key in ("text", 0):
            return self.text
        if key in ("rect", 1):
            return self.rect
        if key in ("center_x", 2):
            return self.center_x
        if key in ("center_y", 3):
            return self.center_y
        if key in ("confidence", 4):
            return self.confidence
        raise KeyError(key)

    def as_dict(self) -> Dict[str, Any]:
        """转成 ascript ``find()`` 返回的 dict 格式。"""
        return {
            "text": self.text,
            "rect": list(self.rect),
            "center_x": self.center_x,
            "center_y": self.center_y,
            "confidence": round(float(self.confidence), 6),
        }

    def __repr__(self) -> str:
        return (
            f"OcrText(text={self.text!r}, rect={self.rect}, "
            f"center=({self.center_x},{self.center_y}), confidence={self.confidence:.4f})"
        )


def _make_engine(params: Optional[Dict[str, Any]] = None):
    """创建 RapidOCR 实例；未安装可选依赖时抛出可操作的提示。"""
    try:
        from rapidocr import RapidOCR
    except ImportError as exc:  # pragma: no cover - 依赖缺失分支
        raise ImportError(_INSTALL_HINT) from exc

    # 关闭 rapidocr 自身的 INFO 日志，避免自动化脚本刷屏。
    # unclip_ratio 提到 2.0：默认 1.6 对"登录""取消"这类 2 字短按钮容易漏检。
    options: Dict[str, Any] = dict(params or {})
    options.setdefault("Global.log_level", "error")
    options.setdefault("Det.unclip_ratio", 2.0)
    try:
        return RapidOCR(params=options)
    except TypeError:
        # 老版本不支持 params 时退回默认配置
        return RapidOCR()


class _Engine:
    """线程安全的惰性单例：首次识别才加载模型。"""

    def __init__(self) -> None:
        self._engine: Any = None
        self._params: Optional[Dict[str, Any]] = None
        self._lock = threading.Lock()

    def configure(self, params: Optional[Dict[str, Any]] = None) -> None:
        """设置 RapidOCR 初始化参数并在下次调用时重建引擎。"""
        with self._lock:
            self._params = dict(params) if params else None
            self._engine = None

    def get(self):
        if self._engine is None:
            with self._lock:
                if self._engine is None:
                    self._engine = _make_engine(self._params)
        return self._engine


_engine = _Engine()


def configure_engine(params: Optional[Dict[str, Any]] = None) -> None:
    """配置 RapidOCR 引擎（如调整 ``Global.text_score``），下次识别生效。

    ``None`` 表示恢复默认参数并重建引擎。
    """
    _engine.configure(params)


def _to_box(item: Any) -> Optional[List[int]]:
    """把 RapidOCR 的 4 点框 ``[[x,y],...]`` 转成 ``[x,y,x1,y1]``。"""
    try:
        pts = np.asarray(item, dtype=np.float32).reshape(4, 2)
    except (TypeError, ValueError):
        return None
    x0, y0 = np.min(pts, axis=0)
    x1, y1 = np.max(pts, axis=0)
    return [int(round(float(x0))), int(round(float(y0))),
            int(round(float(x1))), int(round(float(y1)))]


def _crop_for_ocr(image: ImageLike, rect: Optional[Sequence[int]], binary: int,
                  color_order: str) -> Tuple[np.ndarray, int, int]:
    """裁剪 + 可选二值化，返回 (子图, x 偏移, y 偏移)。"""
    img = as_image(image, color_order)
    sub, off_x, off_y = crop(img, rect)
    if sub.size == 0:
        return sub, off_x, off_y
    b = int(binary)
    if b >= 0:
        gray = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(gray, b, 255, cv2.THRESH_BINARY)
        sub = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
    return sub, off_x, off_y


def recognize(
    image: ImageLike,
    rect: Optional[Sequence[int]] = None,
    binary: int = -1,
    confidence: float = 0.1,
    pattern: Optional[str] = None,
    color_order: str = "BGR",
) -> List[OcrText]:
    """用 RapidOCR 识别图片中的文字，返回按阅读顺序排列的结果列表。

    - ``binary``: -1 不处理；0~255 先按该阈值二值化再识别
    - ``confidence``: 识别置信度阈值
    - ``pattern``: 正则表达式，仅保留匹配的结果
    - 返回的坐标已换算回整图（含 ``rect`` 偏移）
    """
    if image is None:
        raise ValueError("imgcolor 不提供截图功能：请通过 image= 传入图片（ndarray / 路径 / Pillow Image）")

    sub, off_x, off_y = _crop_for_ocr(image, rect, binary, color_order)
    if sub.size == 0:
        return []

    engine = _engine.get()
    try:
        out = engine(sub)
    except TypeError:
        # 兼容部分版本要求显式开关
        out = engine(sub, use_det=True, use_cls=True, use_rec=True)

    boxes = getattr(out, "boxes", None)
    texts = getattr(out, "txts", None)
    scores = getattr(out, "scores", None)
    if boxes is None or texts is None or len(boxes) == 0:
        return []
    if scores is None:
        scores = [1.0] * len(texts)

    compiled: Optional[re.Pattern] = None
    if pattern:
        try:
            compiled = re.compile(str(pattern))
        except re.error as exc:
            raise ValueError(f"非法正则表达式: {pattern!r}") from exc

    thr = float(confidence)
    results: List[OcrText] = []
    for box, text, score in zip(boxes, texts, scores):
        try:
            s = float(score)
        except (TypeError, ValueError):
            s = 1.0
        if s + 1e-9 < thr:
            continue
        value = str(text).strip()
        if not value:
            continue
        if compiled is not None and compiled.search(value) is None:
            continue
        rect4 = _to_box(box)
        if rect4 is None:
            continue
        x0, y0, x1, y1 = rect4
        results.append(OcrText(
            text=value,
            rect=[x0 + off_x, y0 + off_y, x1 + off_x, y1 + off_y],
            center_x=int(round((x0 + x1) / 2.0 + off_x)),
            center_y=int(round((y0 + y1) / 2.0 + off_y)),
            confidence=s,
        ))

    # 按阅读顺序：先按行（以典型行高的 0.6 容差分行），再按列
    if results:
        heights = sorted(r.rect[3] - r.rect[1] for r in results)
        row_h = max(1, heights[len(heights) // 2])
        results.sort(key=lambda r: (int(r.center_y // max(1, int(row_h * 0.6))), r.center_x))
    return results


__all__ = ["OcrText", "recognize", "configure_engine"]
