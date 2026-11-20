"""找图（模板匹配）—— 对应 ascript ``FindImages``。

参考: https://www.ascript.cn/docs/android/api/screen/findimages/

返回 ascript 格式的 dict（单个）或 dict 数组（多个）::

    {'rect': [x, y, x1, y1], 'center_x': 995, 'center_y': 1830,
     'confidence': 0.949, 'index': 0}

支持：透明 PNG（alpha 作 mask）、一次找多张图、``scale_range`` 多尺度、
灰度/彩色匹配、``ori`` 排序方向、``rect`` 缩小搜索区。

``find_sift`` / ``find_all_sift`` 按官方文档说明降级为模板匹配。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from PIL import Image

from ._core import ImageLike, as_image, crop, read_image, sort_points

TemplateLike = Union[str, os.PathLike, np.ndarray, Image.Image]


# --------------------------------------------------------------------------- #
# 模板准备
# --------------------------------------------------------------------------- #
def _prepare_template(part_img: TemplateLike, rgb: bool,
                      color_order: str = "BGR") -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """返回 (模板图, alpha mask)。

    ``part_img`` 可为文件路径（含中文路径）、Pillow Image 或 ndarray。
    路径/Pillow 已由 :func:`read_image` 规范化为 BGR 系。
    """
    arr, normalized = read_image(part_img)
    rgb_input = (not normalized) and str(color_order).upper() == "RGB"

    mask: Optional[np.ndarray] = None
    if arr.ndim == 3 and arr.shape[2] == 4:
        mask = (arr[:, :, 3] > 0).astype(np.uint8) * 255
        color = arr[:, :, :3]
        if rgb_input:
            color = color[:, :, ::-1]
        color = np.ascontiguousarray(color)
    elif arr.ndim == 3 and arr.shape[2] == 3:
        color = np.ascontiguousarray(arr[:, :, ::-1] if rgb_input else arr)
    elif arr.ndim == 2:
        color = arr
    else:
        raise ValueError(f"不支持的模板图维度: {arr.shape}")

    if color.dtype != np.uint8:
        if np.issubdtype(color.dtype, np.floating) and color.size and float(color.max()) <= 1.0:
            color = color * 255.0
        color = np.clip(color, 0, 255).astype(np.uint8)

    if not rgb:
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY) if color.ndim == 3 else color
        return np.ascontiguousarray(gray), mask  # 灰度模板配单通道 mask

    if mask is not None:
        # TM_CCORR_NORMED 的 mask 需与模板同通道数
        mask = cv2.merge([mask, mask, mask])
    return color, mask


def _scale_levels(scale_range: Optional[Sequence[float]]) -> List[float]:
    """生成约 15 个缩放步长；范围含 1.0 时确保包含原尺寸。"""
    if scale_range is None:
        return [1.0]
    if len(scale_range) < 2:
        raise ValueError("scale_range 必须是 [最小缩放比, 最大缩放比]")
    lo, hi = float(scale_range[0]), float(scale_range[1])
    if lo <= 0 or hi <= 0 or hi < lo:
        raise ValueError(f"非法的 scale_range: {scale_range}")
    if lo == hi:
        return [lo]
    values = [lo + (hi - lo) * i / 14.0 for i in range(15)]
    if lo <= 1.0 <= hi:
        values.append(1.0)
    return sorted({round(v, 6) for v in values})


def _scaled(templ: np.ndarray, mask: Optional[np.ndarray],
            scale: float) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if scale == 1.0:
        return templ, mask
    h, w = templ.shape[:2]
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    out_t = cv2.resize(templ, (nw, nh), interpolation=interp)
    if templ.ndim == 2:
        out_t = np.ascontiguousarray(out_t)
    out_m = None
    if mask is not None:
        out_m = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
        if mask.ndim == 2 and out_m.ndim == 3:
            out_m = out_m[:, :, 0]
    return out_t, out_m


# --------------------------------------------------------------------------- #
# 单模板单尺度的匹配
# --------------------------------------------------------------------------- #
def _match_one(search: np.ndarray, templ: np.ndarray, mask: Optional[np.ndarray],
               confidence: float) -> List[Tuple[float, int, int]]:
    """返回 [(score, x, y)]，已做重叠抑制。"""
    sh, sw = search.shape[:2]
    th, tw = templ.shape[:2]
    if th > sh or tw > sw or th == 0 or tw == 0:
        return []

    if mask is not None:
        # 透明 PNG：TM_SQDIFF 支持 mask。CCORR_NORMED 对纯色模板会把任意纯色
        # 背景也判为 1.0，故改用平方差并归一化成"平均色差相似度"。
        # 3 通道 mask 的 count_nonzero 已含通道数，正好对应逐通道平方差求和。
        n = max(1, int(np.count_nonzero(mask)))
        res = cv2.matchTemplate(search, templ, cv2.TM_SQDIFF, mask=mask)
        mean_sq = np.maximum(res.astype(np.float32), 0.0) / float(n)
        score_map = np.clip(1.0 - np.sqrt(mean_sq) / 255.0, 0.0, 1.0).astype(np.float32)
    elif float(np.std(templ)) < 1.0:
        # 纯色模板用 CCOEFF 会得到退化的满屏 1.0，改用 SQDIFF
        res = cv2.matchTemplate(search, templ, cv2.TM_SQDIFF_NORMED)
        score_map = np.clip(1.0 - res, 0.0, 1.0).astype(np.float32)
    else:
        res = cv2.matchTemplate(search, templ, cv2.TM_CCOEFF_NORMED)
        score_map = np.clip(res, 0.0, 1.0).astype(np.float32)

    work = score_map.copy()
    found: List[Tuple[float, int, int]] = []
    while True:
        _, max_val, _, max_loc = cv2.minMaxLoc(work)
        if float(max_val) < confidence:
            break
        x, y = int(max_loc[0]), int(max_loc[1])
        found.append((float(max_val), x, y))
        # 抑制本结果周围，避免同一目标被反复命中
        rx0, ry0 = max(0, x - tw // 2), max(0, y - th // 2)
        rx1, ry1 = min(sw, x + tw + tw // 2), min(sh, y + th + th // 2)
        work[ry0:ry1, rx0:rx1] = -1.0
        if len(found) >= 5000:
            break
    return found


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def _search(
    part_img: Union[TemplateLike, Sequence[TemplateLike]],
    rect: Optional[Sequence[int]],
    confidence: float,
    rgb: bool,
    source_img: ImageLike,
    maxcnt: int,
    ori: int,
    scale_range: Optional[Sequence[float]],
    image: ImageLike,
    color_order: str,
) -> List[Dict[str, Any]]:
    src = image if image is not None else source_img
    img = as_image(src, color_order)
    sub, off_x, off_y = crop(img, rect)
    if sub.size == 0:
        return []
    search = sub if rgb else cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)

    templates = list(part_img) if isinstance(part_img, (list, tuple)) else [part_img]
    if not templates:
        raise ValueError("part_img 不能为空")
    scales = _scale_levels(scale_range)
    threshold = float(np.clip(confidence, 0.0, 1.0))

    candidates: List[Dict[str, Any]] = []
    for index, tpl_src in enumerate(templates):
        templ0, mask0 = _prepare_template(tpl_src, rgb, color_order)
        for scale in scales:
            templ, mask = _scaled(templ0, mask0, scale)
            th, tw = templ.shape[:2]
            for score, x, y in _match_one(search, templ, mask, threshold):
                candidates.append({
                    "score": score, "x": x, "y": y, "w": tw, "h": th,
                    "index": index, "scale": scale,
                    "cx": x + tw / 2.0, "cy": y + th / 2.0,
                })

    # 跨模板/跨尺度去重：重叠的候选只保留置信度最高的
    candidates.sort(key=lambda c: -c["score"])
    kept: List[Dict[str, Any]] = []
    for c in candidates:
        if any(
            abs(c["cx"] - k["cx"]) < max(c["w"], k["w"]) * 0.5
            and abs(c["cy"] - k["cy"]) < max(c["h"], k["h"]) * 0.5
            for k in kept
        ):
            continue
        kept.append(c)
        if maxcnt and len(kept) >= int(maxcnt):
            break

    # 按找色/找图方向 ori 排序（与 FindColors 一致）
    order = sort_points([(k["x"], k["y"]) for k in kept], ori)
    lookup = {(k["x"], k["y"]): k for k in kept}
    ordered = [lookup[p] for p in order if p in lookup]

    results: List[Dict[str, Any]] = []
    for k in ordered:
        gx, gy = int(k["x"] + off_x), int(k["y"] + off_y)
        x1, y1 = gx + k["w"], gy + k["h"]
        results.append({
            "rect": [gx, gy, int(x1), int(y1)],
            "center_x": int((gx + x1) // 2),
            "center_y": int((gy + y1) // 2),
            "confidence": round(float(k["score"]), 4),
            "index": int(k["index"]),
        })
    return results


class FindImages:
    """找图（模板匹配）。源图通过 ``image=`` / ``source_img=`` 传入 BGR ndarray。"""

    @staticmethod
    def find(
        part_img: Union[TemplateLike, Sequence[TemplateLike]],
        rect: Optional[Sequence[int]] = None,
        confidence: float = 0.1,
        rgb: bool = False,
        source_img: ImageLike = None,
        ori: int = 2,
        scale_range: Optional[Sequence[float]] = None,
        image: ImageLike = None,
        color_order: str = "BGR",
    ) -> Optional[Dict[str, Any]]:
        """找一个，返回 dict 或 None。

        多尺度/多图时返回**置信度最高**的结果（文档约定）；
        同分并列时按 ``ori`` 方向取最靠前的一个。
        """
        res = _search(part_img, rect, confidence, rgb, source_img, 0, ori,
                      scale_range, image, color_order)
        if not res:
            return None
        best = max(r["confidence"] for r in res)
        return next(r for r in res if r["confidence"] >= best)

    @staticmethod
    def find_all(
        part_img: Union[TemplateLike, Sequence[TemplateLike]],
        rect: Optional[Sequence[int]] = None,
        confidence: float = 0.1,
        rgb: bool = False,
        source_img: ImageLike = None,
        maxcnt: int = 0,
        ori: int = 2,
        scale_range: Optional[Sequence[float]] = None,
        image: ImageLike = None,
        color_order: str = "BGR",
    ) -> Optional[List[Dict[str, Any]]]:
        """找所有，返回 dict 数组或 None。``maxcnt=0`` 表示不限制数量。"""
        res = _search(part_img, rect, confidence, rgb, source_img, maxcnt, ori,
                      scale_range, image, color_order)
        return res or None

    @staticmethod
    def find_template(part_img, rect=None, confidence: float = 0.5, rgb: bool = False,
                      source_img=None, ori: int = 2, scale_range=None,
                      image=None, color_order: str = "BGR"):
        """模板匹配（单个），默认 confidence=0.5。"""
        return FindImages.find(part_img, rect, confidence, rgb, source_img, ori,
                               scale_range, image, color_order)

    @staticmethod
    def find_all_template(part_img, rect=None, confidence: float = 0.5, rgb: bool = False,
                          maxcnt: int = 0, source_img=None, ori: int = 2,
                          scale_range=None, image=None, color_order: str = "BGR"):
        """模板匹配（所有），默认 confidence=0.5。"""
        return FindImages.find_all(part_img, rect, confidence, rgb, source_img,
                                   maxcnt, ori, scale_range, image, color_order)

    # 官方文档：压缩版 OpenCV 无 SIFT，自动降级为模板匹配
    @staticmethod
    def find_sift(part_img, rect=None, confidence: float = 0.5, rgb: bool = False,
                  source_img=None, ori: int = 2, scale_range=None,
                  image=None, color_order: str = "BGR"):
        """兼容入口：降级为模板匹配。跨分辨率请用 ``scale_range``。"""
        return FindImages.find(part_img, rect, confidence, rgb, source_img, ori,
                               scale_range, image, color_order)

    @staticmethod
    def find_all_sift(part_img, rect=None, confidence: float = 0.5, rgb: bool = False,
                      maxcnt: int = 0, source_img=None, ori: int = 2,
                      scale_range=None, image=None, color_order: str = "BGR"):
        """兼容入口：降级为模板匹配。跨分辨率请用 ``scale_range``。"""
        return FindImages.find_all(part_img, rect, confidence, rgb, source_img,
                                   maxcnt, ori, scale_range, image, color_order)


__all__ = ["FindImages"]
