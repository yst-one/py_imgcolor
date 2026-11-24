"""找图（模板匹配）—— 方法命名对齐 ascript ``FindImages``。

返回 ascript 格式的 dict（单个）或 dict 数组（多个）::

    {'rect': [x, y, x1, y1], 'center_x': 995, 'center_y': 1830,
     'confidence': 0.949, 'index': 0}

支持：透明 PNG（alpha 作 mask）、一次找多张图、``scale_range`` 多尺度、
灰度/彩色匹配、``ori`` 排序方向、``rect`` 缩小搜索区。

``find_sift`` / ``find_all_sift`` 是真正的 SIFT 特征匹配（尺度/旋转不变，
用 RANSAC 单应矩阵定位）；当图像特征不足或 SIFT 不可用时自动回退模板匹配。
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
# SIFT 特征匹配（尺度/旋转不变）
# --------------------------------------------------------------------------- #
_sift_instance = None
_sift_unavailable = False


def _get_sift():
    """惰性创建 SIFT 实例；OpenCV 压缩版无 SIFT 时返回 None（只探测一次）。"""
    global _sift_instance, _sift_unavailable
    if _sift_unavailable:
        return None
    if _sift_instance is None:
        factory = getattr(cv2, "SIFT_create", None)
        if factory is None:
            _sift_unavailable = True
            return None
        try:
            _sift_instance = factory()
        except Exception:
            _sift_unavailable = True
            return None
    return _sift_instance


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def _sift_match_one(
    sift, search_gray: np.ndarray, templ_gray: np.ndarray, threshold: float,
    maxcnt: int, ratio: float = 0.75, reproj: float = 5.0,
) -> List[Dict[str, Any]]:
    """单模板 SIFT 匹配：BFMatcher + Lowe ratio test + 迭代 RANSAC 单应。

    多个相同目标时，一次 RANSAC 会把不同目标的匹配点混在一起拟合出错误小框。
    采用多模型拟合：每拟合出一个实例就移除其 inliers，用剩余匹配点继续找下一个，
    直到不足 4 个或达到 ``maxcnt``。返回候选含 score/x/y/w/h/cx/cy。
    """
    kp1, des1 = sift.detectAndCompute(templ_gray, None)
    kp2, des2 = sift.detectAndCompute(search_gray, None)
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return []

    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn = bf.knnMatch(des1, des2, k=2)
    good = [m for pair in knn if len(pair) == 2
            for m, n in [pair] if m.distance < ratio * n.distance]
    if len(good) < 4:
        return []

    src_all = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_all = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    th, tw = templ_gray.shape[:2]
    corners = np.float32([[0, 0], [tw, 0], [tw, th], [0, th]]).reshape(-1, 1, 2)
    eps = float(np.clip(threshold, 0.0, 1.0))

    out: List[Dict[str, Any]] = []
    remaining = np.ones(len(good), dtype=bool)
    for _ in range(5000):
        if int(remaining.sum()) < 4 or (maxcnt and len(out) >= int(maxcnt)):
            break
        src = src_all[remaining]
        dst = dst_all[remaining]
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, reproj)
        if H is None or mask is None:
            break
        inlier_idx = np.flatnonzero(mask.ravel().astype(bool))
        inliers = len(inlier_idx)
        if inliers < 4:
            break

        inlier_ratio = inliers / len(good)
        strength = min(1.0, inliers / 8.0)
        score = float(inlier_ratio * (0.5 + 0.5 * strength))

        proj = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
        xs, ys = proj[:, 0], proj[:, 1]
        x, y = float(xs.min()), float(ys.min())
        w, h = float(xs.max() - x), float(ys.max() - y)
        # 尺寸合理性：SIFT 尺度变化时框应接近模板等比，允许较大范围但拒绝退化
        if w > 0 and h > 0 and 0.25 <= (w / tw) <= 4.0 and h / w > 0.2:
            if score >= eps:
                out.append({"score": score, "x": x, "y": y, "w": w, "h": h,
                            "scale": 0, "cx": x + w / 2.0, "cy": y + h / 2.0})

        # 移除本轮 inliers，避免下一轮重复拟合同一目标
        removal = np.flatnonzero(remaining)
        remaining[removal[inlier_idx]] = False
    return out


def _search_sift(
    part_img: Union[TemplateLike, Sequence[TemplateLike]],
    rect: Optional[Sequence[int]], confidence: float, source_img: ImageLike,
    maxcnt: int, ori: int, image: ImageLike, color_order: str,
) -> Optional[List[Dict[str, Any]]]:
    """SIFT 匹配主流程。返回 None 表示不可用（调用方应降级），[] 表示特征够但未命中。"""
    sift = _get_sift()
    if sift is None:
        return None
    src = image if image is not None else source_img
    img = as_image(src, color_order)
    sub, off_x, off_y = crop(img, rect)
    if sub.size == 0:
        return []
    search_gray = _to_gray(sub)

    templates = list(part_img) if isinstance(part_img, (list, tuple)) else [part_img]
    if not templates:
        raise ValueError("part_img 不能为空")

    candidates: List[Dict[str, Any]] = []
    any_features = False
    for index, tpl_src in enumerate(templates):
        templ, _mask = _prepare_template(tpl_src, rgb=False, color_order=color_order)
        templ_gray = _to_gray(templ)
        # 探测是否有足够特征：无任何模板有 >=4 关键点时视为 SIFT 不可用
        kp_probe = sift.detectAndCompute(templ_gray, None)[0]
        if kp_probe is not None and len(kp_probe) >= 4:
            any_features = True
        for c in _sift_match_one(sift, search_gray, templ_gray, confidence, maxcnt):
            c["index"] = index
            candidates.append(c)

    if not any_features:
        return None

    # 与模板匹配一致的重叠去重 + ori 排序
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
    order = sort_points([(int(k["x"]), int(k["y"])) for k in kept], ori)
    lookup = {(int(k["x"]), int(k["y"])): k for k in kept}
    ordered = [lookup[p] for p in order if p in lookup]

    results: List[Dict[str, Any]] = []
    for k in ordered:
        gx, gy = int(k["x"] + off_x), int(k["y"] + off_y)
        x1, y1 = int(gx + k["w"]), int(gy + k["h"])
        results.append({
            "rect": [gx, gy, x1, y1],
            "center_x": int((gx + x1) // 2),
            "center_y": int((gy + y1) // 2),
            "confidence": round(float(k["score"]), 4),
            "index": int(k["index"]),
        })
    return results


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

    @staticmethod
    def find_sift(part_img, rect=None, confidence: float = 0.5, rgb: bool = False,
                  source_img=None, ori: int = 2, scale_range=None,
                  image=None, color_order: str = "BGR"):
        """SIFT 特征匹配（单个），天然尺度/旋转不变，无需 ``scale_range``。

        用 RANSAC 单应矩阵把模板四角投影到源图求外接矩形；返回 dict 格式与
        :meth:`find` 相同。特征不足（如纯色块/简单图标）或 OpenCV 无 SIFT 时，
        自动回退模板匹配（此时 ``scale_range`` 生效）。
        """
        res = _search_sift(part_img, rect, confidence, source_img, 0, ori,
                           image, color_order)
        # SIFT 不可用或没命中（简单图标常无特征可匹配）→ 回退模板匹配
        if not res:
            return FindImages.find(part_img, rect, confidence, rgb, source_img,
                                   ori, scale_range, image, color_order)
        best = max(r["confidence"] for r in res)
        return next(r for r in res if r["confidence"] >= best)

    @staticmethod
    def find_all_sift(part_img, rect=None, confidence: float = 0.5, rgb: bool = False,
                      maxcnt: int = 0, source_img=None, ori: int = 2,
                      scale_range=None, image=None, color_order: str = "BGR"):
        """SIFT 特征匹配（全部），返回 dict 数组或 None；``maxcnt=0`` 不限数量。

        特征不足时回退模板匹配，行为同 :meth:`find_sift`。
        """
        res = _search_sift(part_img, rect, confidence, source_img, maxcnt, ori,
                           image, color_order)
        if not res:
            return FindImages.find_all(part_img, rect, confidence, rgb, source_img,
                                       maxcnt, ori, scale_range, image, color_order)
        return res


__all__ = ["FindImages"]
