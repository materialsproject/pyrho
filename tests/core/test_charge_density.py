import numpy as np
from pymatgen.io.vasp import Chgcar

from pyrho.core.chargeDensity import ChargeDensity

global TESTFILE
TESTFILE = "../../test_files/Si.uc.hdf5"

def test_density_copy():
    chgcar: Chgcar = Chgcar.from_hdf5(TESTFILE)
    density = ChargeDensity.from_pmg_volumetric_data(chgcar)

    density_copy = density.from_rho(density.rho, density.structure, normalization=density.normalization)
    assert np.array_equal(density.rho, density_copy.rho)

def test_density_transformed():
    chgcar: Chgcar = Chgcar.from_hdf5(TESTFILE)
    density = ChargeDensity.from_pmg_volumetric_data(chgcar)

    sc_mat = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    density_transformed = density.get_transformed_obj(sc_mat=sc_mat)
    assert density_transformed.structure == density.structure * sc_mat

def test_density_chgcar():
    chgcar: Chgcar = Chgcar.from_hdf5(TESTFILE)
    density = ChargeDensity.from_pmg_volumetric_data(chgcar)

    assert density.to_Chgcar() == chgcar
