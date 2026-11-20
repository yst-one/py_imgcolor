import cv2
import numpy as np

from imgcolor import Canny, canny


def _rect_image():
    img = np.zeros((200, 300, 3), np.uint8)
    cv2.rectangle(img, (50, 50), (150, 130), (255, 255, 255), -1)
    return img


def test_canny_find_rect():
    img = _rect_image()
    res = Canny(draw=Canny.DRAW_RECT).run(img)
    assert res.image.shape == img.shape
    assert res.edges is not None and res.edges.shape == img.shape[:2]
    big = [d for d in res.data if d["width"] >= 90 and d["height"] >= 70]
    assert len(big) == 1
    d = big[0]
    assert 48 <= d["x"] <= 52 and 48 <= d["y"] <= 52
    assert d["area"] > 0 and d["perimeter"] > 0 and d["points"] > 0
    assert len(res.contours) == len(res.data)
    assert int(res.image[:, :, 2].max()) == 255  # 画了红色矩形框


def test_canny_ranges():
    img = _rect_image()
    assert Canny(draw=Canny.DRAW_NONE, width_range=[200, 300]).run(img).data == []
    assert Canny(draw=Canny.DRAW_NONE, height_range=[200, 300]).run(img).data == []
    assert Canny(draw=Canny.DRAW_NONE, left_range=[0, 40]).run(img).data == []
    assert Canny(draw=Canny.DRAW_NONE, top_range=[0, 40]).run(img).data == []
    assert len(Canny(draw=Canny.DRAW_NONE, width_range=[90, 110]).run(img).data) >= 1


def test_canny_draw_modes():
    img = _rect_image()
    none = Canny(draw=Canny.DRAW_NONE).run(img)
    assert np.array_equal(none.image, img)  # 不绘制时原图不变
    assert len(none.data) >= 1

    all_res = Canny(draw=Canny.DRAW_ALL, desc=True).run(img)
    assert not np.array_equal(all_res.image, img)

    fill = Canny(draw=Canny.DRAW_RECT_FILL).run(img)
    assert not np.array_equal(fill.image, img)


def test_canny_min_area_and_rect_offset():
    img = _rect_image()
    assert Canny(draw=Canny.DRAW_NONE, min_area=10**9).run(img).data == []
    res = Canny(draw=Canny.DRAW_NONE).run(img, rect=[40, 40, 200, 180])
    assert len(res.data) >= 1
    assert all(d["x"] >= 40 and d["y"] >= 40 for d in res.data)


def test_canny_functional_and_retr_modes():
    img = _rect_image()
    assert len(canny(img, draw=Canny.DRAW_RECT).data) >= 1
    assert len(Canny(mode=Canny.RETR_LIST, draw=Canny.DRAW_NONE).run(img).data) >= 1


def test_canny_tiny_region():
    img = _rect_image()
    res = Canny().run(img, rect=[0, 0, 1, 1])
    assert isinstance(res.data, list) and res.image.shape == img.shape
