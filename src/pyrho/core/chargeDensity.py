# -*- coding: utf-8 -*-

"""Main module."""
import math

from pymatgen import Lattice, Structure
from pymatgen.io.vasp import VolumetricData
from pyrho.core.pgrid import PGrid
import numpy as np


class ChargeDensity(PGrid):
    def __init__(
        self, grid_data: np.ndarray, structure: Structure, normalization: str = "vasp",
    ):
        """
        Class that contains functions to featurize volumetic data with periodic
        boundary conditions.
        Make sure the data being stored is grid-independent

        Args:
            grid_data: Volumetric data to read in
            structure: Atomic structure corresponding to the charge density
            normalization: the normalization scheme:
                - 'vasp' sum of the data / number of grid points == number of electrons
        """
        self.structure = structure.copy()
        self.normalization = normalization
        if normalization[0].lower() == "n":
            """
            No rescaling
            """
            scaled_data = grid_data
        elif normalization[0].lower() == "v":
            """
            the standard charge density from VASP is given as (rho*V) such that:
            sum(rho)/NGRID = NELECT/UC_vol
            so the real rho is:
            rho = (rho*UC_vol)*NGRID/UC_vol/UC_vol
            where the second V account for the different number of electrons in
            different cells
            """
            scaled_data = grid_data / self.structure.volume
        else:
            raise NotImplementedError("Not a valid normalization scheme")

        super().__init__(grid_data=scaled_data, lattice=None)

    @property
    def rho(self) -> np.ndarray:
        """
        Alias for the grid data, which should be the true charge density
        """
        return self.grid_data

    @property
    def lattice(self) -> Lattice:
        """
        Override the lattice definition in PGrid
        """
        return self.structure.lattice.matrix

    @property
    def renormalized_data(self) -> None:
        if self.normalization[0].lower() == "n":
            return self.grid_data
        if self.normalization[0].lower() == "v":
            return self.grid_data * self.structure.volume
        else:
            raise NotImplementedError(
                "Charge density normalization scheme not implemented"
            )

    @classmethod
    def from_pmg_volumetric_data(cls, vdata: VolumetricData, data_key="total"):
        return cls(
            grid_data=vdata.data[data_key],
            structure=vdata.structure,
            normalization="vasp",
        )

    @classmethod
    def from_rho(
        cls, rho: np.ndarray, structure: Structure, normalization: str = "vasp"
    ):
        new_obj = cls(grid_data=rho, structure=structure, normalization="none")
        new_obj.normalization = normalization
        return new_obj

    # @classmethod
    # def from_hdf5(cls, filename):
    #     """
    #     Reads VolumetricData from HDF5 file.
    #     """
    #     import h5py
    #
    #     with h5py.File(filename, "r") as f:
    #         data = {k: np.array(v) for k, v in f["vdata"].items()}
    #         structure = Structure.from_dict(json.loads(f.attrs["structure_json"]))
    #         return cls(structure=structure, data=data)

    def reorient_axis(self) -> None:
        """
        Change the orgientation of the lattice vector so that:
        a points along the x-axis, b is in the xy-plane, c is in the positive-z halve of space
        """
        self.structure.lattice = Lattice.from_parameters(
            *self.structure.lattice.abc, *self.structure.lattice.angles, vesta=True
        )

    # def get_data_in_cube(self, s: float, ngrid: int) -> np.ndarray:
    #     """
    #     Return the charge density data sampled on a cube.
    #
    #     Args:
    #         s: side lengthy in angstroms
    #         ngrid: number of grid points in each direction
    #
    #     Returns:
    #         ndarray: regridded data in a ngrid x ngrid x ngrid array
    #
    #     """
    #     grid_out = [ngrid, ngrid, ngrid]
    #     target_sc_lat_vecs = np.eye(3, 3) * s
    #     sc_mat = np.linalg.inv(self.structure.lattice.matrix) @ target_sc_lat_vecs
    #     _, res = get_sc_interp(self.rho, sc_mat, grid_out)
    #     return res.reshape(grid_out)
    #
    def get_reshaped_cell(self, sc_mat, frac_shift, new_grid=int(1e9)):
        """
        Motify the structure and data and return a new object containing the reshaped
        data
        Args:
            sc_mat: Matrix to create the new cell
            frac_shift: translation to be applied on the cell after the matrix
            transformation
            new_grid: density of the new grid, can also just take the desired
            dimension as a list.

        Returns:

        """
        new_structure = self.structure * sc_mat
        new_structure.translate_sites(
            list(range(len(new_structure))), -np.array(frac_shift)
        )

        # determine the output grid
        lengths = new_structure.lattice.abc
        if isinstance(new_grid, int):
            ngrid = new_grid / new_structure.volume
            mult = (np.prod(lengths) / ngrid) ** (1 / 3)
            grid_out = [int(math.floor(max(l / mult, 1))) for l in lengths]
        else:
            grid_out = new_grid

        new_rho = self.get_transformed_data(sc_mat, frac_shift, grid_out=grid_out)
        return self.from_rho(new_rho, new_structure, self.normalization)

    #
    #     _, new_rho = get_sc_interp(self.rho, sc_mat, grid_sizes=grid_out)
    #     new_rho = new_rho.reshape(grid_out)
    #
    #     grid_shifts = [
    #         int(t * g) for t, g in zip(translation - np.round(translation), grid_out)
    #     ]
    #
    #     new_rho = roll_array(new_rho, grid_shifts)
    #     return self.__class__.from_rho(new_rho, new_structure)
