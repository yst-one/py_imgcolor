import numpy as np
import pytest

from _helpers import DIGITS, draw_text
from imgcolor import FontLib, Ocr, OcrX, make_font_lib


def _lib():
    lib = FontLib(color="#FFFFFF", diff=0.95)
    for ch, matrix in DIGITS.items():
        lib.add(ch, matrix)
    return lib


def test_font_lib_add_and_roundtrip(tmp_path):
    lib = _lib()
    assert len(lib) == 3
    assert lib.glyphs[0].ndarray().shape == (7, 5)
    path = tmp_path / "font.json"
    assert lib.save(path) == str(path)
    lib2 = FontLib.load(path)
    assert len(lib2) == 3
    assert lib2.color == "#FFFFFF"
    assert lib2.glyphs[1].label == "2"
    assert FontLib.load(lib2) is lib2


def test_font_lib_bad_file(tmp_path):
    p = tmp_path / "font.t"
    p.write_text("not a json", encoding="utf-8")
    with pytest.raises(ValueError, match="不是 JSON"):
        FontLib.load(p)
    with pytest.raises(FileNotFoundError):
        FontLib.load(tmp_path / "missing.json")


def test_add_from_image():
    img = np.zeros((40, 60, 3), np.uint8)
    draw_text(img, "1", 5, 5)
    lib = FontLib(color="#FFFFFF", diff=0.95)
    g = lib.add_from_image("1", img, rect=[5, 5, 10, 12])
    assert g.label == "1"
    assert g.height == 7
    assert g.width == 3  # 全空的首尾列被裁掉
    assert sum(sum(row) for row in g.matrix) > 0
    # 自动 Otsu 二值化路径（不指定前景色）
    lib2 = FontLib()
    g2 = lib2.add_from_image("1", img, rect=[5, 5, 10, 12])
    assert np.array_equal(g.ndarray(), g2.ndarray())


def test_make_font_lib(tmp_path):
    img = np.zeros((40, 40, 3), np.uint8)
    draw_text(img, "1", 5, 5)
    lib = make_font_lib(
        [{"label": "1", "rect": [5, 5, 10, 12]}],
        tmp_path / "lib.json", image=img, color="#FFFFFF",
    )
    assert len(lib) == 1 and (tmp_path / "lib.json").is_file()


def test_ocr_find_all_and_matrix():
    lib = _lib()
    img = np.zeros((40, 80, 3), np.uint8)
    draw_text(img, "12", 5, 10, gap=3)
    assert OcrX.find_all(lib, image=img, region=0.9) == ["1", "2"]
    assert Ocr.matrix(lib, image=img, region=0.9) == "12"


def test_ocr_detail():
    lib = _lib()
    img = np.zeros((40, 80, 3), np.uint8)
    draw_text(img, "3", 5, 10)
    det = OcrX.find_detail(lib, image=img, region=0.9)
    assert len(det) == 1
    assert det[0]["text"] == "3"
    assert det[0]["x"] == 5 and det[0]["y"] == 10
    assert det[0]["width"] == 5 and det[0]["height"] == 7
    assert det[0]["score"] >= 0.99


def test_ocr_region_strict():
    lib = _lib()
    img = np.zeros((40, 80, 3), np.uint8)
    draw_text(img, "12", 5, 10, gap=3)
    img[10, 7] = (0, 0, 0)  # 破坏 "1" 的一个前景像素
    assert OcrX.find_all(lib, image=img, region=1.0) == ["2"]
    assert OcrX.find_all(lib, image=img, region=0.9) == ["1", "2"]


def test_ocr_rect_and_empty():
    lib = _lib()
    img = np.zeros((60, 80, 3), np.uint8)
    draw_text(img, "12", 5, 10, gap=3)
    assert OcrX.find_all(lib, rect=[0, 0, 20, 30], image=img) == ["1", "2"]
    assert OcrX.find_all(lib, rect=[40, 40, 70, 55], image=img) == []
    assert Ocr.matrix(lib, image=np.zeros((20, 20, 3), np.uint8)) == ""


def test_ocr_scales():
    lib = _lib()
    img = np.zeros((60, 100, 3), np.uint8)
    draw_text(img, "1", 5, 10, scale=2)
    assert OcrX.find_all(lib, image=img, region=0.9, scales=(2.0,)) == ["1"]


def test_ocr_empty_lib():
    with pytest.raises(ValueError, match="字库为空"):
        OcrX.find_all(FontLib(), image=np.zeros((10, 10, 3), np.uint8))
