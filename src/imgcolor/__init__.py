"""imgcolor —— 基于 OpenCV 的 Python 图色库，方法命名对齐 ascript。

    from imgcolor import FindColors, CompareColors, Colors, Canny, OcrX, Ocr, FontLib, FindImages

库本身只做图像处理，不含设备截图与点击；这两件事通过回调注入，不绑定任何框架。

图片输入：所有 ``image=`` / ``bitmap=`` / ``source_img=`` / 模板参数都支持
numpy ndarray、文件路径（含中文路径）、Pillow Image（任意 mode），以及返回
上述类型的无参函数（配合 ``until`` 重试）。

全局注册 ``set_screencapper(lambda: d.screenshot())`` 后，所有方法省略 ``image=``
都自动取图；回调每次调用，配合 ``until``/``timeout`` 可刷新画面。
``Ocr.set_clicker(lambda x, y: d.click(x, y))`` 注入点击。Pillow 是正式运行依赖。

内部统一为 BGR / uint8 / 3 通道；原始 RGB ndarray 请用 ``from_rgb(arr)``
或给方法传 ``color_order="RGB"``（路径与 Pillow 已自动规范化，无需该参数）。
Pillow 的 ``P``/``1`` 模式有色损，精确比色建议先 ``img.convert("RGB")``。
"""

from ._core import (
    ColorPoint,
    ColorSpec,
    Point,
    as_image,
    clear_screencapper,
    color_mask,
    crop,
    from_rgb,
    get_screencapper,
    parse_color_spec,
    parse_points,
    read_image,
    set_screencapper,
)
from .canny import Canny, CannyResult, canny
from .colors import Colors
from .comparecolors import CompareColors
from .findcolors import FindColors
from .findimages import FindImages
from .fontlib import FontLib, Glyph, binarize, make_font_lib
from .ocrx import Ocr, OcrX, OcrText
from .rapidocr_engine import configure_engine

__version__ = "0.5.3"

__all__ = [
    "FindColors",
    "FindImages",
    "CompareColors",
    "Colors",
    "Canny",
    "CannyResult",
    "canny",
    "OcrX",
    "Ocr",
    "OcrText",
    "configure_engine",
    "FontLib",
    "Glyph",
    "make_font_lib",
    "binarize",
    "Point",
    "ColorSpec",
    "ColorPoint",
    "parse_points",
    "parse_color_spec",
    "color_mask",
    "as_image",
    "read_image",
    "from_rgb",
    "crop",
    "set_screencapper",
    "get_screencapper",
    "clear_screencapper",
]
