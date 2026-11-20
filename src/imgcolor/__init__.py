"""imgcolor —— 基于 OpenCV 的 Python 图色库。

对齐 ascript 图色 API（纯图像处理，不含截图）：

    from imgcolor import FindColors, CompareColors, Colors, Canny, OcrX, Ocr, FontLib, FindImages

图像输入：所有 ``image=`` / ``bitmap=`` / ``source_img=`` / 模板参数都支持
numpy ndarray、**图片文件路径**（含中文路径）、**Pillow Image**（任意 mode），
也可传返回上述类型的无参函数配合 ``until`` 重试。

可全局注册截图回调 ``set_screencapper(lambda: d.screenshot())``，此后所有方法
省略 ``image=`` 时自动取图（回调每次调用，配合 ``until``/``timeout`` 可刷新画面）。
本库本身不含设备截图实现。Pillow 是正式运行依赖。

内部统一为 BGR / uint8 / 3 通道；原始 RGB ndarray 请用 ``from_rgb(arr)``
或给方法传 ``color_order="RGB"``（路径与 Pillow 已自动规范化，无需该参数）。
注意：Pillow 的 ``P``/``1`` 模式存在调色板量化/二值化色损，精确比色
（``sim=1`` / ``diff=1.0``）建议先 ``img.convert("RGB")``。

参考文档：
- 多点找色 https://www.ascript.cn/docs/android/api/screen/findcolors/
- 多点比色 https://www.ascript.cn/docs/android/api/screen/comparecolors/
- 颜色数量 https://www.ascript.cn/docs/android/api/screen/colors/
- 边缘检测 https://www.ascript.cn/docs/android/api/screen/canny/
- 点阵字库 https://www.ascript.cn/docs/android/api/screen/ocrx/
- 找图     https://www.ascript.cn/docs/android/api/screen/findimages/
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

__version__ = "0.5.2"

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
