"""点阵字库的制作与存取 —— 供 :class:`imgcolor.OcrX` 使用。

ascript 的 ``font.t`` 是私有格式，本库使用可读的 JSON 字库，
可用 :meth:`FontLib.add_from_image` 从截图里的字符直接生成模板。

JSON 结构::

    {
      "version": 1,
      "color": "#FFFFFF",          // 可选：前景色；为空时识别用 Otsu 自动二值化
      "diff": 0.9,
      "threshold": null,           // 可选：灰度阈值
      "glyphs": [
        {"label": "0", "matrix": [[0,1,1],[1,0,1],[1,1,0]], "color": null}
      ]
    }
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import cv2
import numpy as np

from ._core import ImageLike, as_image, color_mask, crop, parse_color_spec

PathLike = Union[str, Path]


def binarize(image: ImageLike, color: Optional[str] = None, diff: float = 0.9,
             threshold: Optional[int] = None) -> np.ndarray:
    """把 BGR 图像二值化为 bool 掩码（True = 前景文字像素）。

    - 指定 ``color``：按颜色相似度取前景（最精确，推荐）
    - 否则：灰度 Otsu / 固定阈值，并自动把像素较少的一侧当作前景

    ``image`` 支持 ndarray、图片路径、Pillow Image（统一由 ``as_image`` 规范化）。
    """
    image = as_image(image)
    if image.size == 0:
        return np.zeros(image.shape[:2], dtype=bool)
    if color:
        return color_mask(image, parse_color_spec(color), diff=diff)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if threshold is None:
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = bw > 0
    else:
        mask = gray >= int(threshold)
    if mask.size and float(mask.mean()) > 0.5:
        mask = ~mask
    return mask


@dataclass
class Glyph:
    """一个字符模板。matrix 为 0/1 二维点阵（已裁掉全空行列）。"""

    label: str
    matrix: List[List[int]]
    color: Optional[str] = None

    @property
    def width(self) -> int:
        return len(self.matrix[0]) if self.matrix else 0

    @property
    def height(self) -> int:
        return len(self.matrix)

    def ndarray(self) -> np.ndarray:
        return np.asarray(self.matrix, dtype=np.uint8)

    def to_dict(self) -> Dict[str, Any]:
        return {"label": self.label, "matrix": self.matrix, "color": self.color}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Glyph":
        return cls(str(data["label"]), [[int(v) for v in row] for row in data["matrix"]],
                   data.get("color"))


class FontLib:
    """点阵字库。"""

    def __init__(self, glyphs: Optional[Iterable[Glyph]] = None,
                 color: Optional[str] = None, diff: float = 0.9,
                 threshold: Optional[int] = None):
        self.glyphs: List[Glyph] = list(glyphs or [])
        self.color = color
        self.diff = float(diff)
        self.threshold = threshold

    def __len__(self) -> int:
        return len(self.glyphs)

    def __iter__(self):
        return iter(self.glyphs)

    # ------------------------------------------------------------------ #
    def add(self, label: str, matrix: Sequence[Sequence[int]],
            color: Optional[str] = None) -> Glyph:
        """直接追加一个点阵模板。"""
        g = Glyph(str(label), [[int(v) for v in row] for row in matrix], color or self.color)
        if not g.matrix or g.width == 0:
            raise ValueError("字模点阵不能为空")
        if any(len(row) != g.width for row in g.matrix):
            raise ValueError("字模点阵每行长度必须一致")
        self.glyphs.append(g)
        return g

    def add_from_image(
        self,
        label: str,
        image: ImageLike,
        rect: Optional[Sequence[int]] = None,
        *,
        color: Optional[str] = None,
        diff: Optional[float] = None,
        threshold: Optional[int] = None,
        color_order: str = "BGR",
    ) -> Glyph:
        """从图片中框选一个字符，二值化并裁边后生成字模。

        :param label: 文字代号，如 ``"0"`` / ``"金币"``
        :param image: 源图（ndarray / 图片路径 / Pillow Image）
        :param rect: 字符所在区域 ``[x, y, x1, y1]``，None=整图
        :param color: 前景色 ``"#RRGGBB"``；不给则自动 Otsu 二值化
        :param diff: 取色相似度，默认使用字库的 diff
        :param threshold: 固定灰度阈值（color 为空时生效）
        :return: 新生成的 :class:`Glyph`
        """
        img = as_image(image, color_order)
        sub, _, _ = crop(img, rect)
        if sub.size == 0:
            raise ValueError("框选区域为空")
        fg_color = color or self.color
        mask = binarize(sub, fg_color, self.diff if diff is None else float(diff),
                        self.threshold if threshold is None else threshold)
        ys, xs = np.nonzero(mask)
        if len(ys) == 0:
            raise ValueError(f"字符 {label!r} 未提取到任何前景像素，请调整 rect / color / threshold")
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        matrix = mask[y0:y1, x0:x1].astype(np.uint8).tolist()
        return self.add(label, matrix, fg_color)

    # ------------------------------------------------------------------ #
    def save(self, path: PathLike) -> str:
        """保存为 JSON 字库文件，返回文件路径。"""
        p = Path(path)
        if p.parent and not p.parent.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "color": self.color,
            "diff": self.diff,
            "threshold": self.threshold,
            "glyphs": [g.to_dict() for g in self.glyphs],
        }
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(p)

    @classmethod
    def load(cls, path_or_lib: Union[PathLike, "FontLib"]) -> "FontLib":
        """加载 JSON 字库；传入 FontLib 实例则原样返回。"""
        if isinstance(path_or_lib, FontLib):
            return path_or_lib
        p = Path(path_or_lib)
        if not p.is_file():
            raise FileNotFoundError(f"字库文件不存在: {p}")
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"字库文件不是 JSON: {p}。ascript 的 font.t 为私有格式，"
                "请用 FontLib.add_from_image() 生成本库字库。"
            ) from exc
        if "glyphs" not in data:
            raise ValueError(f"字库缺少 glyphs 字段: {p}")
        return cls(
            [Glyph.from_dict(g) for g in data["glyphs"]],
            data.get("color"), float(data.get("diff", 0.9)), data.get("threshold"),
        )


def make_font_lib(samples: Sequence[Dict[str, Any]], path: PathLike,
                  image: Optional[ImageLike] = None, **kwargs) -> FontLib:
    """批量制作字库。

    ``samples`` 每项: ``{"label": "0", "rect": [x,y,x1,y1], "color": "#FFFFFF"}``，
    也可带自己的 ``"image"``；否则使用公共 ``image``。
    生成后自动保存到 ``path``。
    """
    lib = FontLib(**kwargs)
    for item in samples:
        src = item.get("image", image)
        if src is None:
            raise ValueError(f"字符 {item.get('label')!r} 未提供 image")
        lib.add_from_image(
            item["label"], src, item.get("rect"),
            color=item.get("color"), diff=item.get("diff"),
            threshold=item.get("threshold"),
        )
    lib.save(path)
    return lib


__all__ = ["FontLib", "Glyph", "binarize", "make_font_lib"]
