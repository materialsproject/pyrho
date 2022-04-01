import numpy as np
import pytest

from pyrho.core.pgrid import PGrid


@pytest.fixture
def pgrid_example_2d() -> PGrid:
    A, B = 1, 2
    NX, NY = 3, 2

    def f(x, y):
        return np.sin(NX * x * 2 * np.pi) + np.cos(NY * y * 2 * np.pi)

    xx = np.linspace(0, A, 20, endpoint=False)
    yy = np.linspace(0, B, 40, endpoint=False)
    X, Y = np.meshgrid(xx, yy, indexing="ij")
    Z = f(X, Y)
    return PGrid(Z, [[A, 0], [0, B]])


def test_pgrid(pgrid_example_2d: PGrid):
    pgrid = pgrid_example_2d
    assert pgrid.grid_data.shape == (20, 40)
    transformed_data = pgrid.get_transformed_data(
        np.eye(2), [0, 0], [100, 100], up_sample=4
    )
    assert transformed_data.shape == (100, 100)
