"""RapidOCR 便捷方法测试：假引擎覆盖逻辑 + 真实 OCR 端到端。"""

import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from _helpers import imwrite_any
from imgcolor import Ocr, OcrText, configure_engine
from imgcolor import rapidocr_engine as engine_mod


class FakeEngine:
    """模拟 rapidocr 的 engine(img) 返回结构。"""

    def __init__(self, items):
        self.items = items          # [(box, text, score), ...]
        self.calls = []             # 记录每次送入的图片

    def __call__(self, img, **kwargs):
        self.calls.append(img.copy())
        class Out:
            pass
        out = Out()
        out.boxes = [x[0] for x in self.items]
        out.txts = [x[1] for x in self.items]
        out.scores = [x[2] for x in self.items]
        return out


BOX_A = [[10, 20], [50, 20], [50, 40], [10, 40]]   # center (30, 30)
BOX_B = [[60, 20], [90, 20], [90, 40], [60, 40]]   # center (75, 30)


@pytest.fixture
def fake_engine(monkeypatch):
    engine = FakeEngine([
        (BOX_A, "登录", 0.98),
        (BOX_B, "取消", 0.90),
        ([[10, 50], [60, 50], [60, 70], [10, 70]], "确认登录", 0.85),
    ])
    monkeypatch.setattr(engine_mod._engine, "_engine", engine)
    configure_engine()  # 确保 configure 不会把它清掉
    monkeypatch.setattr(engine_mod._engine, "_engine", engine)
    yield engine
    monkeypatch.setattr(engine_mod._engine, "_engine", None)


def blank():
    return np.zeros((100, 160, 3), np.uint8)


# --------------------------------------------------------------------------- #
# 基础方法
# --------------------------------------------------------------------------- #
def test_find_returns_dict_and_first_match(fake_engine):
    r = Ocr.find("登录", image=blank())
    assert r["text"] == "登录"
    assert r["rect"] == [10, 20, 50, 40]
    assert (r["center_x"], r["center_y"]) == (30, 30)
    assert r["confidence"] == pytest.approx(0.98)


def test_find_all_regex_and_none(fake_engine):
    all_hits = Ocr.find_all(image=blank())
    assert [h["text"] for h in all_hits] == ["登录", "取消", "确认登录"]

    hits = Ocr.find_all("登录", image=blank())
    assert [h["text"] for h in hits] == ["登录", "确认登录"]

    hits2 = Ocr.find_all("^确认", image=blank())
    assert [h["text"] for h in hits2] == ["确认登录"]

    assert Ocr.find_all("不存在", image=blank()) is None


def test_find_invalid_regex_falls_back_to_literal(fake_engine):
    # "登录[" 不是合法正则，退化为字面量，不应抛异常，也不会误匹配"登录"
    assert Ocr.find("登录[", image=blank()) is None
    assert Ocr.find("登录", image=blank())["text"] == "登录"  # 后续调用仍正常


def test_exists(fake_engine):
    assert Ocr.exists("登录", image=blank()) is True
    assert Ocr.exists("没有", image=blank()) is False


def test_confidence_filter(fake_engine):
    assert Ocr.find("登录", confidence=0.95, image=blank()) is not None
    assert Ocr.find("登录", confidence=0.99, image=blank()) is None
    assert Ocr.find("取消", confidence=0.95, image=blank()) is None


def test_rect_offset(fake_engine):
    # rect 必须在图内，否则子图为空（normalize_rect 会裁剪到边界）
    r = Ocr.find("登录", rect=[5, 10, 100, 60], image=blank())
    assert r["rect"] == [15, 30, 55, 50]
    assert (r["center_x"], r["center_y"]) == (35, 40)


def test_binary_and_input_types(fake_engine, tmp_path):
    img = blank()
    p = imwrite_any(tmp_path / "屏幕.png", img)  # 中文路径需用 imencode
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    for src in (img, str(p), p, pil):
        assert Ocr.find("登录", binary=180, image=src)["text"] == "登录"
    # 二值化后的图仍是 3 通道
    assert fake_engine.calls[-1].shape == img.shape


def test_ocrtext_dict_access():
    t = OcrText("a", [1, 2, 3, 4], 2, 3, 0.9)
    assert t["text"] == "a" and t["rect"] == [1, 2, 3, 4]
    assert t["confidence"] == 0.9
    with pytest.raises(KeyError):
        t["nope"]
    assert t.as_dict()["center_x"] == 2


# --------------------------------------------------------------------------- #
# click / wait / wait_click
# --------------------------------------------------------------------------- #
def test_click_with_callback(fake_engine):
    clicks = []
    assert Ocr.click("登录", clicker=lambda x, y: clicks.append((x, y)), image=blank()) is True
    assert clicks == [(30, 30)]
    assert Ocr.click("不存在", clicker=lambda x, y: clicks.append((x, y)), image=blank()) is False
    assert clicks == [(30, 30)]


def test_click_requires_clicker(fake_engine):
    with pytest.raises(ValueError, match="不提供设备点击"):
        Ocr.click("登录", image=blank())


def test_global_clicker(fake_engine):
    clicks = []
    Ocr.set_clicker(lambda x, y: clicks.append((x, y)))
    try:
        assert Ocr.click("取消", image=blank()) is True
        assert Ocr.wait_click("确认", timeout=0, image=blank()) is True
        assert clicks == [(75, 30), (35, 60)]
    finally:
        Ocr.set_clicker(None)


def test_wait_static_image_only_once(fake_engine):
    start = time.monotonic()
    r = Ocr.wait("登录", timeout=2, image=blank())
    assert r["text"] == "登录"
    assert time.monotonic() - start < 1.0  # 静态图不轮询

    assert Ocr.wait("不存在", timeout=0, image=blank()) is None


def test_wait_polls_callable_until_timeout(fake_engine, monkeypatch):
    calls = {"n": 0}

    def provider():
        calls["n"] += 1
        return blank()

    monkeypatch.setattr(Ocr, "POLL_INTERVAL", 0.01)
    assert Ocr.wait("不存在", timeout=0.08, image=provider) is None
    assert calls["n"] >= 2

    calls["n"] = 0
    assert Ocr.wait("登录", timeout=1, image=provider)["text"] == "登录"
    assert calls["n"] == 1


def test_click_polls_callable(fake_engine, monkeypatch):
    monkeypatch.setattr(Ocr, "POLL_INTERVAL", 0.01)
    calls = {"n": 0}
    clicked = []

    def provider():
        calls["n"] += 1
        return blank()

    assert Ocr.click("登录", timeout=1, image=provider,
                     clicker=lambda x, y: clicked.append((x, y))) is True
    assert clicked == [(30, 30)] and calls["n"] == 1


def test_missing_image_raises():
    with pytest.raises(ValueError, match="不提供截图"):
        Ocr.find("登录")


def test_bad_clicker_type(fake_engine):
    with pytest.raises(TypeError, match="clicker"):
        Ocr.click("登录", image=blank(), clicker="not-callable")


# --------------------------------------------------------------------------- #
# 真实 RapidOCR 端到端（未安装时跳过）
# --------------------------------------------------------------------------- #
FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    "/System/Library/Fonts/PingFang.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
]


def _render_text(text, size=56):
    font_path = next((p for p in FONT_CANDIDATES if Path(p).is_file()), None)
    if font_path is None:
        pytest.skip("无中文字体，跳过真实 OCR 测试")
    font = ImageFont.truetype(font_path, size)
    canvas = Image.new("RGB", (max(320, size * len(text) * 2), 140), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((30, 30), text, font=font, fill=(0, 0, 0))
    rgb = np.asarray(canvas)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


@pytest.fixture(scope="module")
def real_engine():
    pytest.importorskip("rapidocr", reason="未安装 ocr 可选依赖")
    pytest.importorskip("onnxruntime", reason="未安装 onnxruntime")
    configure_engine()
    yield
    configure_engine()


def test_real_ocr_find_and_find_all(real_engine):
    img = _render_text("登录确认")
    hit = Ocr.find("登录", confidence=0.5, image=img)
    assert hit is not None, "真实 OCR 未识别到文字"
    assert "登录" in hit["text"]
    assert hit["confidence"] >= 0.5
    assert 0 <= hit["center_x"] < img.shape[1]
    assert 0 <= hit["center_y"] < img.shape[0]

    all_hits = Ocr.find_all(confidence=0.5, image=img)
    assert all_hits and len(all_hits) >= 1
    assert Ocr.exists("确认", confidence=0.5, image=img) is True
    assert Ocr.find("不存在的文字", confidence=0.5, image=img) is None


def test_real_ocr_click_wait_and_path(real_engine, tmp_path):
    # "登录确认" 保证检测稳定；用子串/正则找"登录"验证便捷方法
    img = _render_text("登录确认")
    p = imwrite_any(tmp_path / "真实截图.png", img)

    clicked = []
    assert Ocr.click("登录", confidence=0.5, image=str(p),
                     clicker=lambda x, y: clicked.append((x, y))) is True
    assert len(clicked) == 1

    assert Ocr.wait("登录", timeout=0, confidence=0.5, image=img) is not None
    assert Ocr.wait_click("登录", timeout=0, confidence=0.5, image=img,
                          clicker=lambda x, y: clicked.append((x, y))) is True
    assert len(clicked) == 2


def test_real_ocr_binary_and_pillow(real_engine):
    img = _render_text("登录确认")
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    assert Ocr.find("确认", confidence=0.5, image=pil) is not None
    assert Ocr.find("确认", binary=180, confidence=0.5, image=img) is not None
