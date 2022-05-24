import numpy as np
from hypothesis import given, seed
from hypothesis import strategies as st

from pyrho.pgrid import PGrid


@seed(1337)
@given(
    nx=st.integers(min_value=40, max_value=80),
    ny=st.integers(min_value=40, max_value=80),
    A=st.integers(min_value=1, max_value=10),
    B=st.integers(min_value=1, max_value=10),
)
def test_pgrid(checker_2D, nx, ny, A, B):
    checker = checker_2D()
    XX, YY = checker.get_xy(np.eye(2), [nx, ny])
    ZZ = checker.function(XX, YY)
    # create PGrid
    pgrid = PGrid(ZZ, [[A, 0], [0, B]])
    assert pgrid.grid_shape == (nx, ny)

    # test reconstructing w/ upsampling
    transformed_data = pgrid._transform_data(
        np.eye(2), origin=(0.0, 0.0), grid_out=[nx, ny], up_sample=4
    )
    assert np.max(np.abs(ZZ - transformed_data)) < 1e-4

    # test reconstructing w/o upsampling
    transformed_data = pgrid._transform_data(
        np.eye(2), origin=[0, 0], grid_out=[nx, ny]
    )
    assert np.max(transformed_data) > 0.1
    assert np.max(np.abs(ZZ - transformed_data)) < 1e-4

    transformed_data = np.real(
        pgrid._transform_data(
            [[1, 1], [1, -1]], origin=[0.5, 0.5], grid_out=[54, 48], up_sample=8
        )
    )
    XX_ref, YY_ref = checker.get_xy([[1, 1], [1, -1]], [54, 48], origin=[0.5, 0.5])
    ZZ_ref = checker.function(XX_ref, YY_ref)
    assert np.max(np.abs(ZZ_ref - transformed_data)) < 1e-4

    # check transformed object
    transformed_obj = pgrid.get_transformed(
        sc_mat=[[1, 1], [1, -1]], grid_out=[54, 48], origin=[0.5, 0.5]
    )
    assert transformed_obj.grid_shape == (54, 48)
