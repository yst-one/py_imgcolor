from _helpers import dot
from imgcolor import CompareColors

STATE_A = "10,20,#FF0000|30,40,#00FF00"
STATE_B = "50,60,#0000FF|70,80,#FFFFFF"


def test_compare_all_match(canvas):
    dot(canvas, 10, 20, "#FF0000")
    dot(canvas, 30, 40, "#00FF00")
    assert CompareColors.compare(STATE_A, image=canvas) is True
    assert CompareColors.compare(STATE_A, bitmap=canvas) is True  # ascript 参数名


def test_compare_fail_and_sim(canvas):
    dot(canvas, 10, 20, "#FF0000")
    dot(canvas, 30, 40, "#0000FF")  # 第二个点不匹配
    assert CompareColors.compare(STATE_A, image=canvas) is False
    assert CompareColors.compare(STATE_A, sim=0.5, image=canvas) is True
    assert CompareColors.compare(STATE_A, sim=0.6, image=canvas) is False


def test_compare_diff(canvas):
    dot(canvas, 5, 5, "#FE0000")
    assert CompareColors.compare("5,5,#FF0000", diff=0.99, image=canvas) is True
    assert CompareColors.compare("5,5,#FF0000", diff=1.0, image=canvas) is False


def test_compare_tolerance(canvas):
    dot(canvas, 7, 7, "#F00000")
    assert CompareColors.compare("7,7,#FF0000-#100000", image=canvas) is True
    assert CompareColors.compare("7,7,#FF0000-#010101", image=canvas) is False


def test_compare_out_of_range(canvas):
    assert CompareColors.compare("9999,9999,#000000", image=canvas) is False
    assert CompareColors.compare("-1,-1,#000000", image=canvas) is False


def test_compare_multi(canvas):
    dot(canvas, 50, 60, "#0000FF")
    dot(canvas, 70, 80, "#FFFFFF")
    assert CompareColors.compare_multi([STATE_A, STATE_B], image=canvas) == 1
    assert CompareColors.compare_multi([STATE_A], image=canvas) == -1

    dot(canvas, 10, 20, "#FF0000")
    dot(canvas, 30, 40, "#00FF00")
    assert CompareColors.compare_multi([STATE_A, STATE_B], image=canvas) == 0


def test_compare_mode_norm(canvas):
    dot(canvas, 9, 9, "#C86464")  # RGB(200,100,100)
    spec = "9,9,#643232"  # RGB(100,50,50)
    assert CompareColors.compare(spec, mode=0, diff=0.9, image=canvas) is False
    assert CompareColors.compare(spec, mode=1, diff=0.9, image=canvas) is True


def test_compare_until_callable():
    import numpy as np
    from _helpers import dot as _dot

    blank = np.zeros((30, 30, 3), np.uint8)
    target = blank.copy()
    _dot(target, 3, 3, "#FF0000")
    calls = {"n": 0}

    def provider():
        calls["n"] += 1
        return blank if calls["n"] < 2 else target

    assert CompareColors.compare("3,3,#FF0000", image=provider, until=1000) is True
    assert calls["n"] >= 2


def test_compare_multi_until():
    import numpy as np
    from _helpers import dot as _dot

    blank = np.zeros((30, 30, 3), np.uint8)
    target = blank.copy()
    _dot(target, 3, 3, "#FF0000")
    calls = {"n": 0}

    def provider():
        calls["n"] += 1
        return blank if calls["n"] < 2 else target

    # 匹配到索引 0 时必须立即返回（0 是 falsy，不能被当成"未成功"重试到超时）
    assert CompareColors.compare_multi(
        ["3,3,#FF0000"], image=provider, until=1000
    ) == 0
    assert calls["n"] >= 2
    assert CompareColors.compare_multi(["3,3,#FF0000"], image=blank) == -1
