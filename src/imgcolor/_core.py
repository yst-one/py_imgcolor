"""imgcolor 内部公共工具。

约定：所有图像统一为 **BGR / uint8 / 3 通道** 的 numpy ndarray
（即 ``cv2.imread()`` 的结果）。若你手上是 RGB（PIL / mss / 多数截图库），
请在传入前用 ``imgcolor.from_rgb(arr)`` 转换，或给方法传 ``color_order="RGB"``。

所有接收图像的方法都支持以下输入（由 ``as_image`` 统一规范化）：

- numpy ndarray（BGR / RGB / 灰度 / BGRA / float 0~1）
- **图片文件路径**：``str`` 或 ``pathlib.Path``，支持中文/空格路径
- **Pillow Image 对象**：任意 mode（RGB / RGBA / L / P / 1 等）；
  Pillow 是本库的正式运行依赖，安装 whl 时会自动带上
- 返回上述任意类型的无参函数（配合 ``until>0`` 重试）

本库不提供截图能力，图像一律由调用方通过 ``image=`` / ``bitmap=`` 传入。
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from PIL import Image

# Pillow 为正式依赖，直接用类型联合，不再靠鸭子类型猜测
ImageLike = Union[np.ndarray, str, os.PathLike, Image.Image, Callable[[], Any]]


# --------------------------------------------------------------------------- #
# 基础数据结构
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Point:
    """点位对象，支持 ``p.x`` / ``p.y``、解包、以及 ``p["x"]`` 字典式访问。"""

    x: int
    y: int

    def __iter__(self):
        yield self.x
        yield self.y

    def __getitem__(self, key: Any) -> int:
        if key in (0, "x"):
            return self.x
        if key in (1, "y"):
            return self.y
        raise KeyError(key)

    def as_dict(self) -> Dict[str, int]:
        return {"x": self.x, "y": self.y}

    def __repr__(self) -> str:
        return f"Point(x={self.x}, y={self.y})"


@dataclass(frozen=True)
class ColorSpec:
    """一个颜色特征：BGR 三元组 + 可选逐通道偏色容差。"""

    bgr: Tuple[int, int, int]
    tol: Optional[Tuple[int, int, int]] = None
    raw: str = ""


@dataclass(frozen=True)
class ColorPoint:
    """颜色特征点。x/y 为 None 时表示"不带坐标的纯颜色"（用于 Colors.count）。"""

    x: Optional[int]
    y: Optional[int]
    spec: ColorSpec


# --------------------------------------------------------------------------- #
# 颜色解析
# --------------------------------------------------------------------------- #
_HEX = r"(?:0x|#)?[0-9a-fA-F]{6}"
_POINT_RE = re.compile(
    rf"^\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*({_HEX})"
    rf"(?:\s*-\s*({_HEX}))?(?:\s*,\s*({_HEX}))?\s*$"
)
_COLOR_RE = re.compile(rf"^\s*({_HEX})(?:\s*-\s*({_HEX}))?\s*$")


def _hex_rgb(text: str) -> Tuple[int, int, int]:
    m = re.match(rf"^(?:0x|#)?([0-9a-fA-F]{{6}})$", text.strip())
    if not m:
        raise ValueError(f"非法颜色值: {text!r}，应为 #RRGGBB")
    h = m.group(1)
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _to_bgr(rgb: Tuple[int, int, int]) -> Tuple[int, int, int]:
    return rgb[2], rgb[1], rgb[0]


def _make_spec(main: str, tol: Optional[str]) -> ColorSpec:
    rgb = _hex_rgb(main)
    tol_bgr = _to_bgr(_hex_rgb(tol)) if tol else None
    raw = f"{main}-{tol}" if tol else main
    return ColorSpec(_to_bgr(rgb), tol_bgr, raw)


def parse_color_spec(text: str) -> ColorSpec:
    """解析 ``"#RRGGBB"`` 或 ``"#RRGGBB-#101010"``（后者为偏色容差）。"""
    m = _COLOR_RE.match(str(text).strip())
    if not m:
        raise ValueError(f"无法解析颜色: {text!r}")
    return _make_spec(m.group(1), m.group(2))


def parse_points(colors: Union[str, Sequence[str]]) -> List[ColorPoint]:
    """解析颜色特征串。

    支持（``|`` / 换行 / ``;`` 分隔）：
      - ``"634,2126,#05D395|675,2142,#05D395"``  多点（绝对坐标）
      - ``"339,342,#38D8A8-#101010"``           带偏色
      - ``"339,342,#38D8A8,#101010"``           偏色的另一种写法
      - ``"#05D395-#101010"``                   不带坐标的纯颜色
    """
    if isinstance(colors, (list, tuple)):
        text = "|".join(str(c) for c in colors)
    else:
        text = str(colors)

    out: List[ColorPoint] = []
    for seg in re.split(r"[|\n;]+", text):
        seg = seg.strip()
        if not seg:
            continue
        m = _POINT_RE.match(seg)
        if m:
            out.append(
                ColorPoint(
                    int(m.group(1)),
                    int(m.group(2)),
                    _make_spec(m.group(3), m.group(4) or m.group(5)),
                )
            )
            continue
        m = _COLOR_RE.match(seg)
        if m:
            out.append(ColorPoint(None, None, _make_spec(m.group(1), m.group(2))))
            continue
        raise ValueError(f"无法解析颜色特征串片段: {seg!r}")
    if not out:
        raise ValueError("颜色特征串为空")
    return out


# --------------------------------------------------------------------------- #
# 图像规范化
# --------------------------------------------------------------------------- #
def _imread_path(path: Union[str, os.PathLike]) -> np.ndarray:
    """读图片文件，兼容中文/空格路径（cv2.imread 在 Windows 上读不了中文路径）。

    用 ``IMREAD_UNCHANGED`` 保留 alpha 与灰度信息，交给下游统一处理。
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"图片文件不存在: {p}")
    buf = np.fromfile(str(p), dtype=np.uint8)
    if buf.size == 0:
        raise ValueError(f"图片文件为空: {p}")
    arr = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise ValueError(f"图片解码失败（格式不支持或文件损坏）: {p}")
    return arr


def _is_pillow(obj: Any) -> bool:
    """精确识别 Pillow Image（Pillow 是正式依赖，直接 isinstance）。"""
    return isinstance(obj, Image.Image)


def _pil_to_bgra(obj: Image.Image) -> np.ndarray:
    """Pillow Image -> BGRA ndarray（统一成 cv2 通道顺序）。"""
    if not isinstance(obj, Image.Image):
        raise TypeError(f"需要 Pillow Image 对象，收到 {type(obj).__name__}")
    try:
        rgba = np.asarray(obj.convert("RGBA"))
    except Exception as exc:  # pragma: no cover - 损坏的 PIL 对象
        raise ValueError(f"Pillow 图像转换失败: {exc}") from exc
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError(f"Pillow 图像转换后维度异常: {rgba.shape}")
    if rgba.dtype != np.uint8:
        rgba = np.clip(rgba, 0, 255).astype(np.uint8)
    # RGBA -> BGRA
    return np.ascontiguousarray(rgba[:, :, [2, 1, 0, 3]])


def read_image(source: Any) -> Tuple[np.ndarray, bool]:
    """把任意输入读成 ndarray。

    :return: ``(数组, 是否已规范化为 BGR 系)``
        - 文件路径 / Pillow Image：返回 BGR/BGRA/灰度（OpenCV 顺序），标志为 True
        - numpy ndarray：原样返回，标志为 False（由 ``color_order`` 决定是否需要反转）
        - 无参函数：调用后递归处理
    """
    if source is None:
        raise ValueError("图像输入为 None")
    if callable(source):
        return read_image(source())
    if isinstance(source, (str, os.PathLike)):
        return _imread_path(source), True
    if _is_pillow(source):
        return _pil_to_bgra(source), True
    if isinstance(source, np.ndarray) or hasattr(source, "__array__"):
        return np.asarray(source), False
    raise TypeError(
        f"不支持的图像输入类型: {type(source).__name__}；"
        "支持 numpy ndarray、图片路径(str/Path)、Pillow Image 或返回它们的无参函数。"
    )


def as_image(image: ImageLike, color_order: str = "BGR") -> np.ndarray:
    """把输入统一成 BGR / uint8 / 3 通道 / 连续内存的 ndarray。

    支持 ndarray、文件路径（含中文路径）、Pillow Image、返回上述类型的函数。
    ``color_order="RGB"`` 只对**原始 ndarray** 生效；路径/Pillow 已自动规范化为 BGR。
    """
    if image is None:
        raise ValueError(
            "imgcolor 不提供截图功能：请通过 image= 传入 numpy 图像、图片路径或 Pillow Image。"
            "RGB ndarray 可用 imgcolor.from_rgb(arr) 转换。"
        )

    arr, normalized = read_image(image)
    rgb_input = (not normalized) and str(color_order).upper() == "RGB"

    if arr.ndim == 2:
        arr = np.dstack([arr, arr, arr])
    elif arr.ndim == 3:
        if arr.shape[2] == 4:
            arr = arr[:, :, :3]
        elif arr.shape[2] != 3:
            raise ValueError(f"不支持的通道数: {arr.shape[2]}")
        if rgb_input:
            arr = arr[:, :, ::-1]
    else:
        raise ValueError(f"不支持的图像维度: {arr.ndim}")

    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating) and arr.size and float(arr.max()) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def from_rgb(arr: np.ndarray) -> np.ndarray:
    """RGB ndarray -> BGR ndarray（供 cv2 风格的 API 直接使用）。"""
    return as_image(arr, color_order="RGB")


def normalize_rect(rect: Optional[Sequence[int]], width: int, height: int) -> Tuple[int, int, int, int]:
    """把 ``[x, y, x1, y1]`` 规范成不越界的 (x0, y0, x1, y1)。

    rect 为 None 表示全图；x1/y1 为 0 或小于起点时视为延伸到图像边缘。
    """
    if rect is None:
        return 0, 0, width, height
    if len(rect) != 4:
        raise ValueError(f"rect 必须是 [x, y, x1, y1]，收到 {rect!r}")
    x0, y0, x1, y1 = (int(v) for v in rect)
    x0 = max(0, min(x0, width))
    y0 = max(0, min(y0, height))
    if x1 <= x0 or x1 <= 0:
        x1 = width
    if y1 <= y0 or y1 <= 0:
        y1 = height
    x1 = max(x0, min(x1, width))
    y1 = max(y0, min(y1, height))
    return x0, y0, x1, y1


def crop(image: np.ndarray, rect: Optional[Sequence[int]] = None) -> Tuple[np.ndarray, int, int]:
    """裁剪区域，返回 (子图, 全局 x 偏移, 全局 y 偏移)。"""
    x0, y0, x1, y1 = normalize_rect(rect, image.shape[1], image.shape[0])
    return image[y0:y1, x0:x1], x0, y0


# --------------------------------------------------------------------------- #
# 颜色相似度
# --------------------------------------------------------------------------- #
def color_mask(image: ImageLike, spec: ColorSpec, *, diff: float = 0.98, mode: int = 0) -> np.ndarray:
    """计算颜色匹配的布尔掩码 (H, W)。

    - ``spec.tol`` 存在时：逐通道 ``|实际-目标| <= 偏色``（偏色优先，ascript 规则）
    - ``mode=0``（RGB）：``score = 1 - mean(|ΔR|+|ΔG|+|ΔB|)/255``，``score >= diff``
    - ``mode=1``（Norm）：先把 RGB 按通道和归一化，消除整体亮度影响后再比较
    先由 :func:`as_image` 规范化，因此也可直接传图片路径或 Pillow Image。
    """
    image = as_image(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image 必须是 3 通道 BGR 图像")
    if image.size == 0:
        return np.zeros(image.shape[:2], dtype=bool)

    img = image.astype(np.int16)
    target = np.asarray(spec.bgr, dtype=np.int16).reshape(1, 1, 3)

    if spec.tol is not None:
        tol = np.asarray(spec.tol, dtype=np.int16).reshape(1, 1, 3)
        return np.all(np.abs(img - target) <= tol, axis=2)

    threshold = float(np.clip(diff, 0.0, 1.0)) - 1e-6
    if int(mode) == 1:
        denom = img.sum(axis=2, keepdims=True).astype(np.float32) + 1e-6
        ref = target.astype(np.float32)
        ref = ref / (ref.sum() + 1e-6)
        norm = img.astype(np.float32) / denom
        score = 1.0 - np.abs(norm - ref.reshape(1, 1, 3)).sum(axis=2) / 2.0
    else:
        score = 1.0 - np.abs(img - target).mean(axis=2) / 255.0
    return score >= threshold


def shift_mask(mask: np.ndarray, ox: int, oy: int) -> np.ndarray:
    """返回 out[y, x] = mask[y + oy, x + ox]；越界处为 False。"""
    h, w = mask.shape[:2]
    out = np.zeros((h, w), dtype=bool)
    r0, r1 = max(0, oy), min(h, h + oy)
    c0, c1 = max(0, ox), min(w, w + ox)
    if r1 > r0 and c1 > c0:
        out[r0 - oy : r1 - oy, c0 - ox : c1 - ox] = mask[r0:r1, c0:c1]
    return out


def sort_points(points: Sequence[Tuple[int, int]], ori: int = 2) -> List[Tuple[int, int]]:
    """按找色方向排序。

    1 左上起纵向(列优先) / 2 左上起横向(行优先) / 3 右上起横向 / 4 右上起纵向
    5 右下起横向 / 6 右下起纵向 / 7 左下起横向 / 8 左下起纵向
    """
    keys = {
        1: lambda p: (p[0], p[1]),
        2: lambda p: (p[1], p[0]),
        3: lambda p: (p[1], -p[0]),
        4: lambda p: (-p[0], p[1]),
        5: lambda p: (-p[1], -p[0]),
        6: lambda p: (-p[0], -p[1]),
        7: lambda p: (-p[1], p[0]),
        8: lambda p: (p[0], -p[1]),
    }
    try:
        key = keys.get(int(ori), keys[2])
    except (TypeError, ValueError):
        key = keys[2]
    return sorted(points, key=key)


def nms(points: Sequence[Tuple[int, int]], space: int) -> List[Tuple[int, int]]:
    """结果间距去重：欧氏距离 <= space 的候选点只保留排序最靠前的一个。"""
    space = int(space)
    if space <= 0:
        return list(points)
    cell = space + 1
    grid: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    kept: List[Tuple[int, int]] = []
    limit = space * space
    for x, y in points:
        gx, gy = x // cell, y // cell
        duplicate = False
        for ix in (gx - 1, gx, gx + 1):
            for iy in (gy - 1, gy, gy + 1):
                for kx, ky in grid.get((ix, iy), ()):
                    if (x - kx) ** 2 + (y - ky) ** 2 <= limit:
                        duplicate = True
                        break
                if duplicate:
                    break
            if duplicate:
                break
        if duplicate:
            continue
        kept.append((x, y))
        grid.setdefault((gx, gy), []).append((x, y))
    return kept


def run_until(image: ImageLike, until: Union[int, float], action: Callable[[np.ndarray], Any],
              color_order: str = "BGR", done: Optional[Callable[[Any], bool]] = None) -> Any:
    """``until>0`` 且 ``image`` 为可调用对象时，在超时前反复取图重试。

    image 是静态 ndarray 时只执行一次（无截图能力，无法刷新画面）。
    ``done`` 自定义"结果算成功"的判定（默认 truthy），供返回 0/-1 这类
    索引值的方法使用。
    """
    is_done = done if done is not None else (lambda r: bool(r))
    retryable = callable(image)
    timeout = max(0.0, float(until or 0)) / 1000.0
    deadline = time.monotonic() + timeout if (timeout and retryable) else 0.0
    while True:
        result = action(as_image(image, color_order))
        if is_done(result) or not timeout or not retryable:
            return result
        if time.monotonic() >= deadline:
            return result
        time.sleep(min(0.05, max(0.001, deadline - time.monotonic())))
