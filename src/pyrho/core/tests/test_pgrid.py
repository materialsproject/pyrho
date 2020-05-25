import pytest
import numpy as np
from pyrho.core.pgrid import PGrid

A, B = 1, 2
NX, NY = 3, 2


@pytest.fixture
def pgrid_example():
    def f(x, y):
        return np.sin(NX * x * 2 * np.pi) + np.cos(NY * y * 2 * np.pi)

    xx = np.linspace(0, A, 20, endpoint=False)
    yy = np.linspace(0, B, 40, endpoint=False)
    X, Y = np.meshgrid(xx, yy, indexing="ij")
    Z = f(X, Y)
    return PGrid(Z, [[A, 0], [0, B]])
