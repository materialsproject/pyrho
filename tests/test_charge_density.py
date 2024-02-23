import numpy as np
import pytest
from pymatgen.io.vasp import Chgcar

from pyrho.charge_density import ChargeDensity, get_volumetric_like_sc


def test_charge_density(test_dir):
    cden_uc = ChargeDensity.from_file(test_dir / "CHGCAR.uc.vasp")
    cden_sc = ChargeDensity.from_file(test_dir / "CHGCAR.sc1.vasp")
    chgcar_sc = Chgcar.from_file(test_dir / "CHGCAR.sc1.vasp")
    cden_transformed = cden_uc.get_transformed(
        [[1, 1, 0], [1, -1, 0], [0, 0, 1]],
        grid_out=cden_sc.grid_shape,
        up_sample=2,
    )

    # check that the UC and SC normalized data looks the same
    assert cden_uc.normalized_data["total"].max() == pytest.approx(
        cden_sc.normalized_data["total"].max(),
        rel=1e-2,
    )
    # check that the transformed data looks the same as the SC data
    np.testing.assert_allclose(
        cden_transformed.normalized_data["total"],
        cden_sc.normalized_data["total"],
        atol=0.001,
        rtol=0.01,
    )

    chgcar_transformed = cden_transformed.to_Chgcar()

    np.testing.assert_allclose(
        chgcar_sc.data["total"], chgcar_transformed.data["total"], atol=0.1, rtol=0.01
    )  # since the chgcar is scaled, the absolute tolerance is 0.1

def test_mapping(test_dir):
    chgcar_uc = Chgcar.from_file(test_dir / "CHGCAR.uc.vasp")
    chgcar_sc = Chgcar.from_file(test_dir / "CHGCAR.sc1.vasp")
    grid_out = chgcar_sc.dim
    chgcar_out = get_volumetric_like_sc(chgcar_uc, chgcar_sc.structure, grid_out=grid_out, up_sample=2)
    np.testing.assert_allclose(chgcar_out.data["total"], chgcar_sc.data["total"], atol=0.1, rtol=0.01)