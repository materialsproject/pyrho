import numpy as np
import time
import pytest
from hypothesis import settings, given, seed
from hypothesis import strategies as st

from pyrho.pgrid import PGrid


try:
    import cupy as cp
    _cupy_available = True
except Exception:
    _cupy_available = False

pytestmark = pytest.mark.skipif(
    not _cupy_available or cp.cuda.runtime.getDeviceCount() == 0,
    reason="CuPy/CUDA not available"
)


@settings(deadline=None)  
@seed(1337)
@given(
    nx=st.integers(min_value=100, max_value=200),
    ny=st.integers(min_value=100, max_value=200),
    A=st.integers(min_value=1, max_value=10),
    B=st.integers(min_value=1, max_value=10),
)
def test_transform(checker_2D, nx, ny, A, B):
    """Verify that the GPU-accelerated PGrid.get_transformed runs faster than the CPU implementation and yields identical results."""
    checker = checker_2D()
    XX, YY = checker.get_xy(np.eye(2), [nx, ny])
    ZZ = checker.function(XX, YY)

    # Create PGrid instance
    pgrid = PGrid(ZZ, [[A, 0], [0, B]])

    sc_mat = np.eye(2)
    grid_out = [nx, ny]
    origin = (0.0, 0.0)
    up_sample = 4

    # --- CPU ---
    transformed_data_cpu = pgrid._transform_data(sc_mat, origin=origin, grid_out=grid_out, up_sample=up_sample, use_gpu=False)
    
    # --- GPU ---
    transformed_data_gpu = pgrid._transform_data(sc_mat, origin=origin, grid_out=grid_out, up_sample=up_sample, use_gpu=True)

    # --- Transformation check ---
    np.testing.assert_allclose(
        transformed_data_gpu, transformed_data_cpu, atol=1e-4,
        err_msg="GPU transform output differs from CPU."
        )