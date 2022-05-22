import numpy as np
from pymatgen.io.vasp import Chgcar

from pyrho.core.charge_density import ChargeDensity


def test_charge_density(test_dir):
    chgcar_uc = Chgcar.from_hdf5(test_dir / "Si.uc.hdf5")
    chgcar_sc = Chgcar.from_hdf5(test_dir / "Si.sc.hdf5")
    cden_uc = ChargeDensity.from_pmg(chgcar_uc)
    cden_sc = ChargeDensity.from_pmg(chgcar_sc)
    cden_transformed = cden_uc.get_transformed(
        [[2, 0, 0], [0, 2, 0], [0, 0, 1]], grid_out=chgcar_sc.data["total"].shape
    )
    np.testing.assert_allclose(
        cden_transformed.grid_data, cden_sc.grid_data, rtol=0.005
    )  # 0.5 % relative tolerance
