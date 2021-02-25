from hypothesis import given, seed, strategies as st
from pyrho.core.utils import pad_arr, interpolate_fourier, get_sc_interp
import numpy as np


def test_pad_arr():
    arr = [
        [5.0, 10.0, 10.0, 7.0],
        [22.0, 12.0, 7.0, 3.0],
        [16.0, 10.0, 3.0, 5.0],
        [16.0, 22.0, 16.0, 6.0],
        [19.0, 20.0, 3.0, 7.0],
    ]
    arr = np.array(arr)
    res = pad_arr(arr_in=arr, shape=[8, 8])
    assert arr.sum() == res.sum()

    res = pad_arr(arr_in=arr, shape=[5, 8])
    assert arr.sum() == res.sum()


def test_interpolate_fourier1():
    # periodic data should be interpolated with near zero loss

    # same data grid out
    def f(x):
        return 20 * np.sin(x) ** 2 * np.cos(x)

    xx = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    res = np.real(interpolate_fourier(f(xx), [len(xx)]))
    ref = f(xx)
    assert abs(max(res - ref)) < 1e-14

    # double the grid size
    res = np.real(interpolate_fourier(f(xx), [len(xx) * 2]))
    ref = f(xx)
    assert abs(max(res[::2] - ref)) < 1e-14


@seed(1337)
@given(
    a=st.integers(min_value=-1000, max_value=1000),
    b=st.integers(min_value=-1000, max_value=1000),
    c=st.integers(min_value=-1000, max_value=1000),
)  # value cannot be
def test_interpolate_fourier_hyp1(a, b, c):
    # same data grid out
    def f(x):
        return a * np.sin(b * x) + np.cos(c * x)

    xx = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    res = np.real(interpolate_fourier(f(xx), [len(xx)]))
    ref = f(xx)
    assert abs(max(res - ref)) < 1e-10

    # double the grid size
    res = np.real(interpolate_fourier(f(xx), [len(xx) * 2]))
    ref = f(xx)
    assert abs(max(res[::2] - ref)) < 1e-10


@seed(1337)
@given(
    a=st.integers(min_value=-10, max_value=10),
    b=st.integers(min_value=-10, max_value=10),
    c=st.integers(min_value=-10, max_value=10),
)
def test_interpolate_fourier_hyp2(a, b, c):
    # same data grid out
    def f(x, y):
        return a * np.sin(b * x) ** 2 + np.cos(c * y) + 1

    xx = np.linspace(0, 2 * np.pi, 500, endpoint=False)
    yy = np.linspace(0, 2 * np.pi, 504, endpoint=False)
    X, Y = np.meshgrid(xx, yy, indexing="ij")
    Z_in = f(X, Y)
    res = np.real(interpolate_fourier(Z_in, [len(xx), len(yy)]))
    ref = f(X, Y)
    assert abs(np.max(res - ref)) / abs(np.max(res)) < 1e-10

    xx2 = np.linspace(0, 2 * np.pi, 800, endpoint=False)
    yy2 = np.linspace(0, 2 * np.pi, 808, endpoint=False)
    X2, Y2 = np.meshgrid(xx2, yy2, indexing="ij")
    Z2 = f(X2, Y2)
    res = np.real(interpolate_fourier(Z_in, [len(xx2), len(yy2)]))
    assert abs(np.max(res - Z2)) / abs(np.max(res)) < 1e-10


def test_get_sc_interp():
    xlist = np.linspace(0, 2 * np.pi, 800, endpoint=False)
    ylist = np.linspace(0, 2 * np.pi, 702, endpoint=False)
    X, Y = np.meshgrid(xlist, ylist, indexing="ij")
    Z_uc = np.sin(X) * np.cos(2 * Y)

    assert_sc_lattice(Z_uc, [60, 43], [[1, -1], [1, 1]])
    assert_sc_lattice(Z_uc, [50, 42], [[2, -2], [1, 1]])
    assert_sc_lattice(Z_uc, [50, 41], [[2, -3], [1, 1]])


# Helper functions


def assert_sc_lattice(Z_uc, target_grid, sc_lat):
    lat_mat = np.array(sc_lat) * 2 * np.pi
    XX, YY = _get_sc_xy(lat_mat, target_grid)
    ZZ_ref = np.sin(XX) * np.cos(2 * YY)
    nX, nY, ZZ_interp = _get_interp_data(Z_uc, sc_lat, target_grid)
    ZZ_interp = ZZ_interp.reshape(target_grid)
    max_diff = np.max(ZZ_interp - ZZ_ref)
    assert max_diff < 1e-4


def _get_interp_data(data_in, sc_mat, grids):
    (nX, nY), data = get_sc_interp(data_in, sc_mat, grids)
    nX = nX.reshape(grids) * 2 * np.pi
    nY = nY.reshape(grids) * 2 * np.pi
    data = data.reshape(grids)
    return nX, nY, data


def _get_sc_xy(lat_mat, grids):
    a_vec = np.linspace(0, 1, grids[0], endpoint=False)
    b_vec = np.linspace(0, 1, grids[1], endpoint=False)
    AA, BB = np.meshgrid(a_vec, b_vec, indexing="ij")
    frac = np.vstack([AA.flatten(), BB.flatten()])
    XX, YY = np.dot(np.array(lat_mat).T, frac)
    XX = XX.reshape(grids)
    YY = YY.reshape(grids)
    return XX, YY
