import numpy as np
import pytest
from pymatgen.io.vasp import Chgcar

from pyrho.charge_density import ChargeDensity


def test_charge_density(test_dir):
    chgcar_uc = Chgcar.from_hdf5(test_dir / "Si.uc.hdf5")
    chgcar_sc = Chgcar.from_hdf5(test_dir / "Si.sc.hdf5")
    cden_uc = ChargeDensity.from_pmg(chgcar_uc)
    cden_sc = ChargeDensity.from_pmg(chgcar_sc)
    cden_transformed = cden_uc.get_transformed(
        [[2, 0, 0], [0, 2, 0], [0, 0, 1]],
        grid_out=chgcar_sc.data["total"].shape,
        up_sample=2,
    )

    # check that the UC and SC normalized data looks the same
    assert cden_uc.normalized_data["total"].max() == pytest.approx(
        cden_sc.normalized_data["total"].max()
    )
    # check that the transformed data looks the same as the SC data
    np.testing.assert_allclose(
        cden_transformed.normalized_data["total"],
        cden_sc.normalized_data["total"],
        rtol=0.005,
    )  # 0.5 % relative tolerance

    np.array(
        [
            [5.0, 10.0, 10.0, 7.0],
            [22.0, 12.0, 7.0, 3.0],
            [16.0, 10.0, 3.0, 5.0],
            [16.0, 22.0, 16.0, 6.0],
            [19.0, 20.0, 3.0, 7.0],
        ]
    )
