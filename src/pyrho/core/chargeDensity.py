# -*- coding: utf-8 -*-

"""Chang Density Objects: Periodic Grid + Lattice / Atoms"""
import math
from abc import ABCMeta, abstractmethod
from typing import Dict, List, Union

import numpy as np
from monty.json import MSONable
from pymatgen import Lattice, Structure
from pymatgen.io.vasp import VolumetricData, Chgcar, Poscar
from pyrho.core.pgrid import PGrid
import numpy.typing as npt


class ChargeABC(metaclass=ABCMeta):
    @abstractmethod
    def get_reshaped_cell(
        self,
        sc_mat: npt.ArrayLike = ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        frac_shift: npt.ArrayLike = (0.0, 0.0, 0.0),
        grid_out: Union[List, int] = int(1e9),
    ) -> "ChargeABC":
        pass

    @abstractmethod
    def reorient_axis(self) -> None:
        pass


class ChargeDensity(PGrid, ChargeABC):
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
    def lattice(self) -> np.ndarray:  # type: ignore
        """
        Override the lattice definition in PGrid, this makes getting the reoriented charge density easier.
        """
        return self.structure.lattice.matrix

    @property
    def renormalized_data(self) -> np.ndarray:
        if self.normalization[0].lower() == "n":
            return self.grid_data
        if self.normalization[0].lower() == "v":
            return self.grid_data * self.structure.volume
        else:
            raise NotImplementedError("Charge density normalization scheme not implemented")

    @classmethod
    def from_pmg_volumetric_data(cls, vdata: VolumetricData, data_key="total") -> "ChargeDensity":
        """
        Read a single key from the data field of a VolumetricData object
        Args:
            vdata: The volumetric data object
            data_key: The key to read from in the data field

        Returns:
            ChargeDensity object
        """
        return cls(grid_data=vdata.data[data_key], structure=vdata.structure, normalization="vasp",)

    @classmethod
    def from_rho(cls, rho: np.ndarray, structure: Structure, normalization: str = "vasp"):
        new_obj = cls(grid_data=rho, structure=structure, normalization="none")
        new_obj.normalization = normalization
        return new_obj

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

    def get_transformed_obj(
        self,
        sc_mat: npt.ArrayLike = ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        frac_shift: npt.ArrayLike = (0.0, 0.0, 0.0),
        grid_out: Union[List[int], int] = int(1e9),
        up_sample: int = 1,
    ) -> "ChargeDensity":
        """
        Modify the structure and data and return a new object containing the reshaped
        data
        Args:
            sc_mat: Matrix to create the new cell
            frac_shift: translation to be applied on the cell after the matrix
            transformation
            grid_out: density of the new grid, can also just take the desired
            dimension as a list.

        Returns:

        """
        new_structure = self.structure.copy()
        new_structure.translate_sites(list(range(len(new_structure))), -np.array(frac_shift))
        new_structure = new_structure * sc_mat

        # determine the output grid
        lengths = new_structure.lattice.abc
        if isinstance(grid_out, int):
            ngrid = grid_out / new_structure.volume
            mult = (np.prod(lengths) / ngrid) ** (1 / 3)
            grid_out = [int(math.floor(max(l_ / mult, 1))) for l_ in lengths]
        else:
            grid_out = grid_out

        new_rho = self.get_transformed_data(sc_mat, frac_shift, grid_out=grid_out, up_sample=up_sample)
        return ChargeDensity.from_rho(new_rho, new_structure, self.normalization)

    def get_reshaped_cell(
        self,
        sc_mat: npt.ArrayLike = ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        frac_shift: npt.ArrayLike = (0.0, 0.0, 0.0),
        grid_out: Union[List, int] = int(1e9),
        up_sample: int = 1,
    ) -> "ChargeDensity":
        return self.get_transformed_obj(sc_mat=sc_mat, frac_shift=frac_shift, grid_out=grid_out, up_sample=up_sample)

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


class SpinChargeDensity(MSONable, ChargeABC):
    def __init__(self, chargeden_dict: Dict, aug_charge: Dict = None):
        """
        Wrapper class that parses multiple sets of grid data on the same lattice

        Args:
            chargeden_dict: A dictionary containing multiple charge density objects
                        typically in the format {'total' : ChargeDen1, 'diff' : ChargeDen2}
        """
        self.chargeden_dict = chargeden_dict
        self.aug_charge = aug_charge
        self._tmp_key = next(
            iter(self.chargeden_dict)
        )  # get one key in the dictionary to make writing the subsequent code easier

    @classmethod
    def from_pmg_volumetric_data(cls, vdata: VolumetricData, data_keys=("total", "diff")):
        chargeden_dict = {}
        data_aug = getattr(vdata, "data_aug", None)
        for k in data_keys:
            chargeden_dict[k] = ChargeDensity.from_pmg_volumetric_data(vdata, data_key=k)
        return cls(chargeden_dict, aug_charge=data_aug)

    @property
    def lattice(self) -> Lattice:
        return self.chargeden_dict[self._tmp_key].lattice

    def to_Chgcar(self) -> Chgcar:
        struct = self.chargeden_dict[self._tmp_key].structure
        data_ = {k: v.renormalized_data for k, v in self.chargeden_dict.items()}
        return Chgcar(Poscar(struct), data_, data_aug=self.aug_charge)

    def to_VolumetricData(self) -> VolumetricData:
        key_ = next(iter(self.chargeden_dict))
        struct = self.chargeden_dict[key_].structure
        data_ = {k: v.renormalized_data for k, v in self.chargeden_dict.items()}
        return VolumetricData(struct, data_)

    def get_reshaped_cell(
        self,
        sc_mat: npt.ArrayLike = ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        frac_shift: npt.ArrayLike = (0.0, 0.0, 0.0),
        grid_out: Union[List, int] = int(1e9),
        up_sample: int = 1,
    ) -> "SpinChargeDensity":
        new_spin_charge = {}
        for k, v in self.chargeden_dict.items():
            new_spin_charge[k] = v.get_reshaped_cell(sc_mat, frac_shift, grid_out)
        factor = int(
            new_spin_charge[self._tmp_key].structure.num_sites / self.chargeden_dict[self._tmp_key].structure.num_sites
        )
        new_aug = {}
        if self.aug_charge is not None:
            for k, v in self.aug_charge.items():
                new_aug[k] = multiply_aug(v, factor)
        return SpinChargeDensity(new_spin_charge, new_aug)

    def reorient_axis(self) -> None:
        for k, v in self.chargeden_dict:
            v.reorient_axis()


def multiply_aug(data_aug: List[str], factor: int) -> List[str]:
    """
    The original idea here was to use to to speed up some vasp calculations for
    supercells by initializing the entire CHGCAR file.
    The current code does not deal with transformation of the Augemetation charges after regridding.

    This is a naive way to multiply the Augmentation data in the CHGCAR,
    a real working implementation will require analysis of the PAW projection operators.
    However, even with such an implementation, the speed up will be minimal due to VASP's interal
    minimization algorithms.
    Args:
        data_aug: The original augmentation data from a CHGCAR
        factor: The multiplication factor (some integer number of times it gets repeated)
    Returns:
        List of strings for each line of the Augmentation data.
    """
    res = []  # type: List[str]
    cur_block = []  # type: List[str]
    cnt = 0
    for ll in data_aug:
        if "augmentation" in ll:
            if cur_block:
                for j in range(factor):
                    cnt += 1
                    cur_block[0] = f"augmentation occupancies{cnt:>4}{cur_block[0].split()[-1]:>4}\n"
                    res.extend(cur_block)
            cur_block = [ll]
        else:
            cur_block.append(ll)
    else:
        for j in range(factor):
            cnt += 1
            cur_block[0] = f"augmentation occupancies{cnt:>4}{cur_block[0].split()[-1]:>4}\n"
            res.extend(cur_block)
    return res
