# imgcolor

基于 OpenCV 的 Python 图色库，方法命名对齐 ascript：**找色 / 比色 / 颜色数量 / 找图 / 边缘检测 / 文字识别**。适合给 uiautomator2 等自动化框架配上"看图、认字"的能力。

- 图片想怎么传都行：**ndarray / 文件路径 / Pillow Image / 返回图片的函数**。
- 注册一次截图回调，之后所有方法都不用再传 `image`。
- 只做图像处理，**不含设备截图与点击**（这两件事交给回调，库不绑定任何框架）。

Python ≥ 3.9 · MIT · 版本 `0.5.4`

- 仓库：https://github.com/yst-one/py_imgcolor
- PyPI：https://pypi.org/project/imgcolor/

## 安装

```bash
pip install imgcolor                 # 找色 / 比色 / 找图 / 边缘检测 / 点阵字库
pip install "imgcolor[ocr]"           # 再加 RapidOCR 文字识别（Ocr 系列方法）
```

`[ocr]` 会装 `rapidocr>=3.9,<3.10` 和 `onnxruntime`，首次识别会自动下载模型，之后走缓存。限定 `<3.10` 是避开 rapidocr 3.10+ 依赖 `opencv-python-headless` 造成的双 `cv2` 冲突。

源码开发：

```bash
uv sync --extra test --extra ocr
uv run pytest -q
```

## 快速上手

接入 uiautomator2 时，把"截图"和"点击"各注册一次，后面就只剩业务代码：

```python
import uiautomator2 as u2
import imgcolor
from imgcolor import FindColors, Ocr

d = u2.connect()
imgcolor.set_screencapper(lambda: d.screenshot())  # 注册取图：返回 Pillow RGB，自动转 BGR
Ocr.set_clicker(lambda x, y: d.click(x, y))        # 注册点击：回调收到屏幕中心坐标

# 找色：拿到第一个红色像素的坐标
p = FindColors.find_color("#FF0000")
print(p)                       # {'x': 120, 'y': 340} 或 None

# OCR：找到「登录」按钮并点击，最多等 10 秒
Ocr.click("登录", timeout=10)

# 等「登录成功」出现，出现后返回其位置，超时返回 None
Ocr.wait("登录成功", timeout=15)
```

不接设备也能跑——直接传图片路径或 ndarray，所有方法都支持 `image=`：

```python
import cv2
from imgcolor import FindColors, Colors, CompareColors, FindImages

img = cv2.imread("screen.png")     # 也可以直接写 image="screen.png"

p = FindColors.find("10,10,#FF0000|12,10,#00FF00", image=img)  # 多点找色 → Point 或 None
n = Colors.count("#05D395", rect=[0, 0, 500, 500], image=img)  # 颜色数量 → int
ok = CompareColors.compare("10,10,#FF0000|12,10,#00FF00", image=img)  # 比色 → bool
hit = FindImages.find("btn.png", image=img, confidence=0.9)     # 找图 → dict 或 None
```

完整可运行示例（合成图，无需手机）：[`示例_快速上手.py`](examples/示例_快速上手.py)、[`示例_OCR文字识别.py`](examples/示例_OCR文字识别.py)。

## 核心约定

**1. 图片输入。** 所有 `image=` / `bitmap=` / `source_img=` / 模板参数都接受：

| 输入 | 说明 |
|---|---|
| `numpy.ndarray` | BGR（默认），也支持 RGB / 灰度 / BGRA / float 0~1，自动规范化为 BGR/uint8/3 通道 |
| 文件路径 | `str` 或 `Path`，兼容中文、空格路径 |
| `PIL.Image` | 任意 mode；`P` / `1` 有量化色损，精确比色建议先 `convert("RGB")` |
| 无参函数 | 返回上述任一类型，配合 `until` / `timeout` 轮询取图 |

**2. 颜色写 RGB，图像内部是 BGR。** 颜色串一律 `#RRGGBB`（如红色 `#FF0000`），与 OpenCV 内存顺序相反，库内自动转换。若手上是 RGB ndarray（PIL / mss / 多数截图库），用 `imgcolor.from_rgb(arr)` 转换，或给方法传 `color_order="RGB"`；路径和 Pillow 输入已自动处理。

**3. 省略 `image` 即自动取图。** 先 `imgcolor.set_screencapper(fn)` 注册一次，之后所有方法的 `image=` 都可省略，库每次调用 `fn()` 取新图。`get_screencapper()` 查看、`clear_screencapper()` 清除；显式传 `image=` 时优先用显式图、不调用回调。

**4. 轮询要传"会更新的图"。** `until`（毫秒，找色/比色）与 `timeout`（秒，OCR）的重试，只在 `image` 传函数、或已注册截图回调且省略 `image` 时生效——每轮重新取图。传静态 ndarray / 路径只算一次，不会真的等待。

**5. `rect` 统一为 `[x, y, x1, y1]`**（左上、右下，右/下边界不含），`None` 表示全图。

---

## API 参考

### 1. `FindColors` — 找色

特征串格式 `"x,y,#RRGGBB|..."`，第一个点是锚点，其余按相对偏移校验；颜色可带偏色 `#RRGGBB-#101010`（逐通道容差，优先级高于 `diff`）。

| 方法 | 签名要点 | 返回 |
|---|---|---|
| `find` | `(colors, rect=None, space=5, ori=2, diff=0.98, image=None, mode=0, until=0)` | 第一个 `Point` 或 `None` |
| `find_all` | 同上 + `p_max_num=999999` | `list[Point]` 或 `None` |
| `find_color` | `(color, rect=None, diff=0.98, mode=0, image=None)` | `{"x":..,"y":..}` 或 `None` |
| `find_all_color` | `(color, rect=None, diff=0.98, mode=0, max_num=999999, image=None)` | `list[dict]` 或 `None` |

- `diff`：相似度 0~1，默认 0.98。`mode`：`0`=RGB 比较，`1`=Norm 归一化（抗整体亮度变化）。
- `space`：结果间距去重，欧氏距离 ≤ `space` 的候选只留排序靠前者。
- `until`：毫秒，`>0` 且能刷新画面时重试到命中或超时。
- `Point` 支持 `p.x` / `p.y` / `p[0]` / `p.as_dict()`。

`ori` 找色方向（1-8）：

| ori | 方向 | ori | 方向 |
|---|---|---|---|
| 1 | 左上起，纵向（列优先） | 2（默认） | 左上起，横向（行优先） |
| 3 | 右上起，横向 | 4 | 右上起，纵向 |
| 5 | 右下起，横向 | 6 | 右下起，纵向 |
| 7 | 左下起，横向 | 8 | 左下起，纵向 |

### 2. `CompareColors` — 比色

校验**固定绝对坐标**上的颜色是否符合预期（不搜索位置）。颜色串必须是绝对坐标 `"x,y,#RRGGBB|..."`。

| 方法 | 签名要点 | 返回 |
|---|---|---|
| `compare` | `(colors, diff=0.9, image=None, mode=0, until=0, sim=1.0)` | `bool` |
| `compare_multi` | `(colors_list, diff=0.9, image=None, mode=0, sim=1.0, until=0)` | 首个匹配的索引（0 起），全不匹配 `-1` |

`sim` 是整体相似度：通过点数 / 总点数 ≥ `sim` 才算成功。`bitmap` 是 `image` 的兼容别名。

### 3. `Colors` — 颜色数量

| 方法 | 签名要点 | 返回 |
|---|---|---|
| `mask` | `(colors, rect=None, sim=1, image=None)` | 布尔掩码 ndarray（区域局部坐标） |
| `count` | `(colors, rect=None, sim=1, image=None)` | `int` 像素数量 |
| `count_image` | `(colors, rect=None, sim=1, image=None, draw_mode=1)` | `(数量, 调试图 ndarray)` |

- `colors` 支持 `|` 分隔多色（并集，同一像素只计一次），也支持偏色串；带偏色时 `sim` 失效（与 ascript 一致）。
- 调试图：`draw_mode=1` 命中白/其余黑，`draw_mode=2` 黑底保留命中原色。

### 4. `FindImages` — 找图（模板匹配）

| 方法 | 说明 |
|---|---|
| `find(...)` | 找单个，多图/多尺度时返回置信度最高者 |
| `find_all(..., maxcnt=0)` | 找全部，`maxcnt=0` 不限数量 |
| `find_template` / `find_all_template` | 同上别名，默认 `confidence=0.5` |
| `find_sift` / `find_all_sift` | **真 SIFT 特征匹配**，尺度/旋转不变，无需 `scale_range` |

公共参数：`(part_img, rect=None, confidence=0.1, rgb=False, image=None, ori=2, scale_range=None)`。

- `part_img`：模板，可为 ndarray、路径、`Path`，或它们的列表（多图搜索，结果带 `index`）。
- `confidence`：阈值。模板**必须从目标截图同源裁切**才能接近 1.0；跨来源素材分数偏低，需降低阈值。
- `scale_range=[min, max]`：多尺度搜索（约 15 步长，含 1.0），用于跨分辨率。透明 PNG 的 alpha 会作为 mask。

**SIFT 匹配（`find_sift` / `find_all_sift`）**：用 SIFT 特征 + RANSAC 单应矩阵定位，**天然尺度/旋转不变**——模板 50×50 匹配截图里 100×100 的目标不用传 `scale_range`：

```python
hit = FindImages.find_sift("icon.png", image=img)   # 缩放、旋转都能命中
```

- 适合纹理丰富的图（图标细节、商品图、场景）；`confidence` 是可信度（内点比例×数量加权），建议 0.3~0.5 起调。
- **简单纯色块/几何图形**没有足够特征点，会自动回退模板匹配（此时 `scale_range` 生效），保证兼容。
- 同一图案在图上出现多次且完全相同时 SIFT 难区分，也可能回退模板匹配；不同图案各自匹配则正常返回 `index`。

返回 dict：`{"rect": [x,y,x1,y1], "center_x":.., "center_y":.., "confidence":.., "index":..}`。

### 5. `Canny` — 边缘检测 + 轮廓过滤

```python
res = Canny(width_range=[100, 500], draw=Canny.DRAW_RECT).run(img)
for c in res.data:
    print(c["x"], c["y"], c["width"], c["height"], c["area"])
```

构造参数：`low_threshold=50, high_threshold=150, mode=RETR_EXTERNAL, draw=DRAW_ALL, desc=False, left_range=None, top_range=None, width_range=None, height_range=None, blur=0, min_area=0`。

- `draw`：`DRAW_NONE=0` 不绘制 / `DRAW_ALL=1` 画轮廓 / `DRAW_RECT=2` 外接矩形 / `DRAW_RECT_FILL=3` 填充矩形；`desc=True` 附注文字。
- `left/top/width/height_range`：`[min, max]` 过滤轮廓；`blur` 高斯核（奇数，0 不模糊）。
- `run(image, rect=None)` 返回 `CannyResult(image, data, contours, edges)`；`data` 每项含 `x/y/width/height/area/perimeter/points`。
- 函数式入口：`canny(img, draw=Canny.DRAW_RECT)`。

### 6. `Ocr` — 文字识别便捷方法（RapidOCR）

需安装 `[ocr]`。底层统一用 RapidOCR，`text` 支持正则。

| 方法 | 签名要点 | 返回 |
|---|---|---|
| `find` | `(text, rect=None, binary=-1, confidence=0.1, image=None)` | 第一个结果 dict 或 `None` |
| `find_all` | `(text=None, rect=None, binary=-1, confidence=0.1, image=None)` | 全部结果列表或 `None`；`text=None` 返回全部文字 |
| `exists` | 同 `find` | `bool` |
| `click` | `(text, ..., timeout=0, image=None, clicker=None)` | 找到并点击中心，成功 `True` |
| `wait` | `(text, timeout=10, ..., image=None)` | 等待出现，超时 `None`（单位**秒**） |
| `wait_click` | `(text, timeout=10, ..., image=None, clicker=None)` | 等待并点击，超时 `False` |

结果 dict：`{"text", "rect": [x,y,x1,y1], "center_x", "center_y", "confidence"}`。

点击回调：`Ocr.set_clicker(fn)` 全局注册一次，或每次传 `clicker=lambda x, y: ...`。截图与点击都注册后，`Ocr.click("登录")`、`Ocr.wait_click("登录成功", timeout=15)` 都无需再传 `image`。

引擎参数可用 `imgcolor.configure_engine({"Det.unclip_ratio": 2.0})` 调整（默认即 2.0，改善两字短文本漏检），传 `None` 恢复默认并重建引擎——一般无需改动。

### 7. `OcrX` + `FontLib` — 点阵字库识别（无需 RapidOCR）

适合颜色固定、字体明确的数字/图标识别（如金币数量）。用本库 JSON 字库（ascript 的 `font.t` 是私有格式，无法直接读）。

```python
from imgcolor import FontLib, OcrX, Ocr

lib = FontLib(color="#FFFFFF", diff=0.95)
lib.add("1", digit_1_matrix)                       # 二维 0/1 点阵
lib.add_from_image("2", img, rect=[10, 20, 15, 27]) # 从图框选生成字模
lib.save("font.json")

found = OcrX.find_all("font.json", image=img)      # ['1', '2', ...]；detail=True 出位置
text  = Ocr.matrix("font.json", image=img)         # "12"
```

- `OcrX.find_all(font_lib, rect=None, region=0.9, image=None, detail=False, scales=(1.0,), ...)`。
- `FontLib.load(path)` 加载字库；`make_font_lib(samples, path, image=...)` 批量制作。
- `binarize(image, color=None, diff=0.9, threshold=None)` 二值化，可单独使用。

---

## 底层工具函数

`as_image` / `read_image` / `from_rgb` / `parse_points` / `parse_color_spec` / `color_mask` / `crop` 均已导出，可按需组合。例如 `color_mask(img_or_path, parse_color_spec("#FF0000"))` 直接拿布尔掩码。

## 常见问题

| 现象 | 原因 / 处理 |
|---|---|
| 找色/比色全不命中 | 颜色串是 RGB、图是 BGR；RGB 图请用 `from_rgb` 或 `color_order="RGB"` |
| 找图分数很低 | 模板与截图不同源；从目标截图同源裁切，或降低 `confidence` / 用 `scale_range` |
| Pillow `P`/`1` 模式比色不准 | 调色板量化色损，先 `convert("RGB")` |
| `click` 报未配置回调 | 先 `Ocr.set_clicker(fn)`，或传 `clicker=` |
| `until`/`timeout` 不等待 | `image` 传的是静态图；改传返回新图的函数，或注册全局截图回调后省略 `image` |
| 省略 `image` 报"不提供截图功能" | 先 `imgcolor.set_screencapper(fn)` 注册取图回调 |
| 中文路径读图失败 | 库内部用 `np.fromfile + imdecode` 已支持；自己用 `cv2.imread` 需同样处理 |
| 首次 OCR 慢 | 首次自动下载模型，之后有缓存 |
