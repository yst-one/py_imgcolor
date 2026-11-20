import numpy as np
import pytest

from _helpers import dot  # noqa: E402,F401
from imgcolor import Point, parse_points  # noqa: E402,F401


@pytest.fixture
def canvas():
    return np.zeros((200, 300, 3), np.uint8)
