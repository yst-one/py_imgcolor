# imgcolor

基于 OpenCV 的 Python 图色库，对齐 [ascript 图色 API](https://www.ascript.cn/docs/android/api/screen/findcolors/) 的使用习惯：**多点找色 / 单点找色 / 多点比色 / 颜色数量 / 找图 / 边缘检测 / 文字识别**。

本库只做**纯图像处理**——不截图、不点击设备。图片由调用方通过 `image=` 传入，输出坐标可直接喂给 uiautomator2 等自动化框架。

- Python ≥ 3.9，MIT 协议
- 版本：`0.5.0`
- 仓库：https://github.com/yst-one/py_imgcolor
- PyPI：https://pypi.org/project/imgcolor/

## 安装

```bash
# 基础功能（找色 / 比色 / 找图 / 边缘检测 / 点阵字库）
pip install imgcolor

# 额外启用 RapidOCR 文字识别（Ocr.find / find_all / exists / click / wait / wait_click）
pip install "imgcolor[ocr]"
```

`[ocr]` 会安装 `rapidocr>=3.9,<3.10` 和 `onnxruntime`（首次识别会下载模型）。限定 `<3.10` 是因为 rapidocr 3.10+ 依赖 `opencv-python-headless`，会和本库的 `opencv-python` 产生双 `cv2` 冲突。

源码开发：

```bash
uv sync --extra test --extra ocr
uv run pytest -q
```

## 核心约定（先看这段）

**1. 图像输入——所有 `image=` / `bitmap=` / `source_img=` / 模板参数都支持四种类型：**

| 输入 | 说明 |
|---|---|
| `numpy.ndarray` | BGR（默认）、RGB、灰度、BGRA、float 0~1 均可，自动规范化为 BGR/uint8/3 通道 |
| 文件路径 | `str` 或 `Path`，支持中文/空格路径 |
| `PIL.Image` | 任意 mode；`P`/`1` 有量化色损，精确比色建议先 `convert("RGB")` |
| 无参函数 | 返回上述任意类型，配合 `until` / `timeout` 做轮询取图 |

**2. 颜色是 RGB、图像是 BGR。** 颜色串一律写 `#RRGGBB`，与 OpenCV 的 BGR 内存顺序相反，库内部自动转换。若你手上是 RGB ndarray（PIL / mss / uiautomator2 截图），用 `imgcolor.from_rgb(arr)` 转换，或给方法传 `color_order="RGB"`。路径和 Pillow 输入已自动规范化，无需该参数。

**3. `until` / `timeout` 的重试只在 `image` 传函数时生效。** 传静态 ndarray / 路径时只算一次，不会阻塞等待。

**4. `rect` 一律是 `[x, y, x1, y1]`**（左上、右下，右/下边界不含），`None` 表示全图。

## 快速上手

```python
import cv2
from imgcolor import FindColors, CompareColors, Colors, Canny, FindImages

img = cv2.imread("screen.png")  # BGR ndarray；也可直接传路径 "screen.png"

# 多点找色：锚点(10,10)红 + 右移2像素绿
p = FindColors.find("10,10,#FF0000|12,10,#00FF00", image=img)
print(p.x, p.y)  # Point 对象，也可 p[0]/p[1]、dict(p.as_dict())

# 单点找色 / 统计颜色数量
pt = FindColors.find_color("#FF0000", image=img)          # {"x":..,"y":..} 或 None
n  = Colors.count("#05D395-#101010", rect=[0,0,500,500], image=img)

# 比色：判断固定坐标的颜色是否符合预期
ok = CompareColors.compare("10,10,#FF0000|12,10,#00FF00", image=img)

# 找图：模板可为 ndarray / 路径，返回命中矩形与置信度
hit = FindImages.find("btn.png", image=img, confidence=0.9)
if hit:
    print(hit["rect"], hit["center_x"], hit["center_y"], hit["confidence"])
```

完整可运行示例见 [`examples/示例_快速上手.py`](examples/示例_快速上手.py)（合成图，无需手机）和 [`examples/示例_OCR文字识别.py`](examples/示例_OCR文字识别.py)。

---

## API 参考

### 1. `FindColors` — 找色

特征串格式：`"x,y,#RRGGBB|x,y,#RRGGBB|..."`，第一个点为锚点，其余按相对偏移验证。颜色支持偏色写法 `#RRGGBB-#101010`（逐通道容差，优先级高于 `diff`）。

| 方法 | 签名要点 | 返回 |
|---|---|---|
| `find` | `(colors, rect=None, space=5, ori=2, diff=0.98, image=None, mode=0, until=0)` | 第一个 `Point` 或 `None` |
| `find_all` | 同上 + `p_max_num=999999` | `list[Point]` 或 `None` |
| `find_color` | `(color, rect=None, diff=0.98, mode=0, image=None)` | `{"x":..,"y":..}` 或 `None` |
| `find_all_color` | `(color, rect=None, diff=0.98, mode=0, max_num=999999, image=None)` | `list[dict]` 或 `None` |

- `diff`：颜色相似度 0~1，默认 0.98。
- `mode`：`0`=RGB 空间比较，`1`=Norm 归一化颜色（抗整体亮度变化）。
- `space`：结果间距去重，欧氏距离 ≤ `space` 的候选点只保留排序靠前者。
- `until`：毫秒，`>0` 且 `image` 为函数时持续重试直到命中或超时。

`Point` 支持 `p.x` / `p[0]` / `p.as_dict()`。

**`ori` 找色方向（1-8）：**

| ori | 方向 | ori | 方向 |
|---|---|---|---|
| 1 | 左上起，纵向（列优先） | 2（默认） | 左上起，横向（行优先） |
| 3 | 右上起，横向 | 4 | 右上起，纵向 |
| 5 | 右下起，横向 | 6 | 右下起，纵向 |
| 7 | 左下起，横向 | 8 | 左下起，纵向 |

### 2. `CompareColors` — 比色

颜色串必须是**绝对坐标** `"x,y,#RRGGBB|..."`。

| 方法 | 签名要点 | 返回 |
|---|---|---|
| `compare` | `(colors, diff=0.9, bitmap=None, mode=0, until=0, sim=1.0, image=None)` | `bool` |
| `compare_multi` | `(colors_list, diff=0.9, mode=0, sim=1.0, image=None, until=0)` | 第一个匹配的索引（0 起），全不匹配 `-1` |

`sim` 是整体相似度：`通过点数 / 总点数 ≥ sim` 才算成功。`bitmap` 是 ascript 兼容别名，等价于 `image`。

### 3. `Colors` — 颜色数量

| 方法 | 签名要点 | 返回 |
|---|---|---|
| `mask` | `(colors, rect=None, sim=1, image=None)` | 布尔掩码 ndarray（区域局部坐标） |
| `count` | `(colors, rect=None, sim=1, image=None)` | `int` 像素数量 |
| `count_image` | `(colors, rect=None, sim=1, image=None, draw_mode=1)` | `(数量, 调试图 ndarray)` |

- `colors` 支持 `|` 分隔多色（并集，同一像素只计一次），也支持偏色串。
- 颜色串带偏色值时 `sim` 失效（与 ascript 一致）。
- 调试图：`draw_mode=1` 命中画白/其余黑；`draw_mode=2` 黑底保留命中原色。

### 4. `FindImages` — 找图（模板匹配）

| 方法 | 说明 |
|---|---|
| `find(...)` | 找单个，多图/多尺度时返回置信度最高者 |
| `find_all(..., maxcnt=0)` | 找全部，`maxcnt=0` 不限数量 |
| `find_template` / `find_all_template` | 同上别名，默认 `confidence=0.5` |
| `find_sift` / `find_all_sift` | 兼容入口，降级为模板匹配 |

公共参数：`(part_img, rect=None, confidence=0.1, rgb=False, image=None, ori=2, scale_range=None, color_order="BGR")`。

- `part_img`：模板，可为 ndarray、路径、`Path`，或它们的列表（多图搜索，结果带 `index` 字段）。
- `confidence`：相似度阈值。模板图**必须从目标截图同源裁切**才能接近 1.0；跨来源素材（比如从别处保存的图标）分数会明显偏低，需降低阈值。
- `scale_range=[min, max]`：多尺度搜索（约 15 个步长，含 1.0），用于跨分辨率。
- 透明 PNG 的 alpha 通道会作为 mask 参与匹配。

返回 dict：`{"rect": [x, y, x1, y1], "center_x":.., "center_y":.., "confidence":.., "index":..}`。

### 5. `Canny` — 边缘检测 + 轮廓过滤

```python
res = Canny(width_range=[100, 500], draw=Canny.DRAW_RECT).run(img)
for c in res.data:
    print(c["x"], c["y"], c["width"], c["height"], c["area"])
cv2.imwrite("debug.png", res.image)
```

构造参数：`low_threshold=50, high_threshold=150, mode=RETR_EXTERNAL, draw=DRAW_ALL, desc=False, left_range=None, top_range=None, width_range=None, height_range=None, blur=0, min_area=0`。

- `draw`：`DRAW_NONE=0` 不绘制 / `DRAW_ALL=1` 画轮廓 / `DRAW_RECT=2` 画外接矩形 / `DRAW_RECT_FILL=3` 填充矩形；`desc=True` 附注文字。
- `left/top/width/height_range`：`[min, max]` 过滤轮廓；`blur` 高斯核（奇数，0 不模糊）。
- `run(image, rect=None)` 返回 `CannyResult(image, data, contours, edges)`；`data` 每项含 `x/y/width/height/area/perimeter/points`。
- 函数式入口：`canny(img, draw=Canny.DRAW_RECT)`。

### 6. `Ocr` — 文字识别便捷方法（RapidOCR）

需安装 `[ocr]`。对齐 ascript 便捷方法，`text` 支持正则。

| 方法 | 签名要点 | 返回 |
|---|---|---|
| `find` | `(text, rect=None, binary=-1, confidence=0.1, image=None)` | 第一个结果 dict 或 `None` |
| `find_all` | `(text=None, rect=None, binary=-1, confidence=0.1, image=None)` | 全部结果列表或 `None`；`text=None` 返回全部文字 |
| `exists` | 同 `find` | `bool` |
| `click` | `(text, ..., timeout=0, image=None, clicker=None)` | 找到并点击中心，成功 `True` |
| `wait` | `(text, timeout=10, ..., image=None)` | 等待出现，超时 `None`（超时单位**秒**） |
| `wait_click` | `(text, timeout=10, ..., image=None, clicker=None)` | 等待并点击，超时 `False` |

结果 dict：`{"text", "rect": [x,y,x1,y1], "center_x", "center_y", "confidence"}`。

**点击靠回调注入**（本库不操作设备）：

```python
import uiautomator2 as u2
d = u2.connect()
Ocr.set_clicker(lambda x, y: d.click(x, y))   # 全局注册
Ocr.click("登录", image=img)                   # 或每次传 clicker=lambda x, y: d.click(x, y)
```

**持续等待要传取图函数：**

```python
def grab():
    return cv2.cvtColor(np.asarray(d.screenshot()), cv2.COLOR_RGB2BGR)

Ocr.wait_click("登录成功", timeout=15, image=grab)
```

引擎调优：`from imgcolor import configure_engine; configure_engine({"Det.unclip_ratio": 2.0})`（默认已设为 2.0，改善「登录」这类 2 字短按钮漏检）；传 `None` 恢复默认并重建引擎。

### 7. `OcrX` + `FontLib` — 点阵字库识别（无需 RapidOCR）

适用于颜色固定、字体明确的数字/图标识别（如金币数量）。ascript 的 `font.t` 是私有格式无法直接读，用本库 JSON 字库。

```python
from imgcolor import FontLib, OcrX, Ocr

lib = FontLib(color="#FFFFFF", diff=0.95)
lib.add("1", digit_1_matrix)          # 二维 0/1 点阵
lib.add_from_image("2", img, rect=[10, 20, 15, 27])  # 从图框选生成字模
lib.save("font.json")

found = OcrX.find_all("font.json", image=img)   # ['1', '2', ...]；detail=True 出位置
text  = Ocr.matrix("font.json", image=img)      # "12"
```

- `OcrX.find_all(font_lib, rect=None, region=0.9, image=None, detail=False, scales=(1.0,), ...)`。
- `FontLib.load(path)` 加载字库；`make_font_lib(samples, path, image=...)` 批量制作。
- `binarize(image, color=None, diff=0.9, threshold=None)` 二值化，可单独使用。

---

## 底层工具函数

`as_image` / `read_image` / `from_rgb` / `parse_points` / `parse_color_spec` / `color_mask` / `crop` 均已导出，可按需组合。例如 `color_mask(img_or_path, parse_color_spec("#FF0000"))` 直接拿布尔掩码。

## 常见坑

| 现象 | 原因 / 处理 |
|---|---|
| 找色/比色全不命中 | 颜色串是 RGB、图是 BGR；RGB 图请用 `from_rgb` 或 `color_order="RGB"` |
| 找图分数很低 | 模板与截图不同源；从目标截图同源裁切，或降低 `confidence` / 用 `scale_range` |
| Pillow `P`/`1` 模式比色不准 | 调色板量化色损，先 `convert("RGB")` |
| `click` 报未配置回调 | 先 `Ocr.set_clicker(fn)` 或传 `clicker=` |
| `until`/`timeout` 不等待 | `image` 传的是静态图；要重试必须传返回新图的函数 |
| 中文路径读图失败 | 本库内部用 `np.fromfile + imdecode`，已支持；若自己用 `cv2.imread` 需同样处理 |
| 首次 OCR 慢 | 首次识别下载模型；之后有缓存 |

## 参考文档

- [多点找色](https://www.ascript.cn/docs/android/api/screen/findcolors/) · [多点比色](https://www.ascript.cn/docs/android/api/screen/comparecolors/) · [颜色数量](https://www.ascript.cn/docs/android/api/screen/colors/)
- [边缘检测](https://www.ascript.cn/docs/android/api/screen/canny/) · [找图](https://www.ascript.cn/docs/android/api/screen/findimages/) · [点阵字库](https://www.ascript.cn/docs/android/api/screen/ocrx/) · [OCR 便捷方法](https://www.ascript.cn/docs/android/api/screen/ocr/)
