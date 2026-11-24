"""find_sift / find_all_sift 真实 SIFT 匹配测试。

有结构的纹理图走真 SIFT（尺度/旋转不变 + RANSAC 单应）；
简单方块图特征不足时自动回退模板匹配，保持 ascript 兼容行为。
"""

import cv2
import numpy as np
import pytest

from imgcolor import FindImages


def structured(w=120, h=120, seed=0):
    """有结构的图：随机底噪 + 棋盘 + 圆 + 白框，模拟图标/商品图纹理。"""
    rng = np.random.default_rng(seed)
    img = rng.integers(60, 200, (h, w, 3), dtype=np.uint8)
    for i in range(0, w, 16):
        for j in range(0, h, 16):
            if (i // 16 + j // 16) % 2 == 0:
                img[j:j + 8, i:i + 8] = rng.integers(0, 80, 3, dtype=np.uint8)
    cv2.circle(img, (w // 2, h // 2), w // 4,
               tuple(int(x) for x in rng.integers(0, 255, 3)), -1)
    cv2.rectangle(img, (10, 10), (w - 10, h - 10), (255, 255, 255), 2)
    return img


def square():
    """16x16 简单方块，SIFT 特征区分度低，用于验证回退。"""
    t = np.zeros((16, 16, 3), np.uint8)
    t[2:14, 2:14] = (0, 0, 255)
    t[5:11, 5:11] = (0, 0, 0)
    t[7:9, 7:9] = (0, 0, 255)
    return t


def test_sift_scale_invariant_2x():
    """50x50 模板匹配 100x100 目标，无需 scale_range。"""
    tpl = structured(50, 50, 3)
    big = cv2.resize(tpl, (100, 100), interpolation=cv2.INTER_LINEAR)
    img = np.zeros((300, 400, 3), np.uint8)
    img[60:160, 80:180] = big
    res = FindImages.find_sift(tpl, image=img, confidence=0.3)
    assert res is not None
    assert set(res) == {"rect", "center_x", "center_y", "confidence", "index"}
    x0, y0, x1, y1 = res["rect"]
    assert abs(x0 - 80) <= 3 and abs(y0 - 60) <= 3
    assert abs((x1 - x0) - 100) <= 4 and abs((y1 - y0) - 100) <= 4
    assert res["confidence"] >= 0.3
    assert res["index"] == 0


def test_sift_rotation_invariant():
    tpl = structured(80, 80, 4)
    m = cv2.getRotationMatrix2D((40, 40), 25, 1.0)
    rot = cv2.warpAffine(tpl, m, (80, 80))
    img = np.zeros((200, 300, 3), np.uint8)
    img[40:120, 30:110] = rot
    res = FindImages.find_sift(tpl, image=img, confidence=0.25)
    assert res is not None
    # 旋转后外接矩形变大，但中心应仍在目标附近
    assert abs(res["center_x"] - 70) <= 6
    assert abs(res["center_y"] - 80) <= 6


def test_sift_multi_template_index():
    tpl = structured(50, 50, 3)
    wrong = structured(50, 50, 8)
    wrong[:, :, 0] = 255
    big = cv2.resize(tpl, (100, 100), interpolation=cv2.INTER_LINEAR)
    img = np.zeros((300, 400, 3), np.uint8)
    img[60:160, 80:180] = big
    res = FindImages.find_sift([wrong, tpl], image=img, confidence=0.3)
    assert res is not None and res["index"] == 1


def test_sift_rect_limits_search():
    tpl = structured(50, 50, 3)
    big = cv2.resize(tpl, (100, 100), interpolation=cv2.INTER_LINEAR)
    img = np.zeros((300, 400, 3), np.uint8)
    img[60:160, 80:180] = big
    hit = FindImages.find_sift(tpl, rect=[0, 0, 200, 200], image=img, confidence=0.3)
    assert hit is not None
    miss = FindImages.find_sift(tpl, rect=[250, 200, 400, 300], image=img, confidence=0.3)
    assert miss is None


def test_sift_no_match():
    tpl = structured(60, 60, 3)
    img = np.zeros((200, 300, 3), np.uint8)
    assert FindImages.find_sift(tpl, image=img, confidence=0.3) is None
    assert FindImages.find_all_sift(tpl, image=img, confidence=0.3) is None


def test_sift_fallback_to_template_when_low_texture():
    """简单方块图 SIFT 无有效匹配，回退模板匹配后行为与 find 一致。"""
    t = square()
    img = np.zeros((200, 300, 3), np.uint8)
    img[40:56, 30:46] = t
    img[120:136, 150:166] = t
    hit = FindImages.find_sift(t, image=img, confidence=0.9)
    assert hit is not None and hit["rect"] == [30, 40, 46, 56]
    hits = FindImages.find_all_sift(t, image=img, confidence=0.9)
    assert hits is not None and len(hits) == 2
    assert [h["rect"][0] for h in hits] == [30, 150]


def test_sift_all_distinct_templates_and_maxcnt():
    """多模板：两个不同纹理各自被 SIFT 命中，index 正确。"""
    t1 = structured(50, 50, 3)
    t2 = structured(50, 50, 11)
    big1 = cv2.resize(t1, (100, 100), interpolation=cv2.INTER_LINEAR)
    big2 = cv2.resize(t2, (100, 100), interpolation=cv2.INTER_LINEAR)
    img = np.zeros((300, 400, 3), np.uint8)
    img[20:120, 30:130] = big1
    img[160:260, 200:300] = big2
    hits = FindImages.find_all_sift([t1, t2], image=img, confidence=0.3)
    assert hits is not None and len(hits) == 2
    assert sorted(h["index"] for h in hits) == [0, 1]
    assert len(FindImages.find_all_sift([t1, t2], image=img,
                                        confidence=0.3, maxcnt=1)) == 1


def test_find_sift_via_find_all_equivalence():
    """find_sift 返回的是 find_all_sift 中置信度最高者。"""
    tpl = structured(50, 50, 3)
    big = cv2.resize(tpl, (100, 100), interpolation=cv2.INTER_LINEAR)
    img = np.zeros((300, 400, 3), np.uint8)
    img[60:160, 80:180] = big
    one = FindImages.find_sift(tpl, image=img, confidence=0.3)
    many = FindImages.find_all_sift(tpl, image=img, confidence=0.3)
    assert one in many


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
