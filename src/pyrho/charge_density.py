"""Chang Density Objects: Periodic Grid + Lattice / Atoms."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import numpy as np
import numpy.typing as npt
from monty.dev import deprecated
from monty.json import MSONable
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.io.vasp import Chgcar, Poscar, VolumetricData

from pyrho.pgrid import PGrid
from pyrho.utils import get_sc_interp

__all__ = ["ChargeDensity"]


@dataclass
class ChargeDensity(MSONable):
    """Charge density object.

    Defines a charge density with a PGrid object along with the atomic structure.

    Attributes
    ----------
    pgrids: Dict[str, PGrid]
        Dictionaries whose values are periodic ``PGrid`` objects
        representing some periodic scalar field (typically the keys are ``total`` and ``diff``
        for spin-polarized charge densities.)
    structure: Structure
        The atomic structure for the electronic charge density.
    normalization: str | None = "vasp"
        The normalzation scheme (``vasp`` or ``None``)

    """

    pgrids: Dict[str, PGrid]
    structure: Structure
    normalization: str | None = "vasp"

    def __post_init__(self):
        """Post initialization.

        Steps:
            - Make sure all the lattices are identical

        """
        lattices = [self.pgrids[key].lattice for key in self.pgrids.keys()]
        if not all(
            np.allclose(self.structure.lattice._matrix, lattice) for lattice in lattices
        ):
            raise ValueError("Lattices are not identical")

    @property
    def normalized_data(self) -> dict[str, npt.NDArray]:
        """Get the normalized data.

        Since different codes use different normalization methods for
        volumetric data we should convert them to the same units (electrons / Angstrom^3)

        Returns
        -------
        dict[str, NDArray]:
            The normalized data in units of (electrons / Angstrom^3)

        """
        return {
            k: _normalize_data(
                grid_data=v.grid_data,
                lattice=self.structure.lattice,
                normalization=self.normalization,
            )
            for k, v in self.pgrids.items()
        }

    @property
    def grid_shape(self) -> Tuple[int, int, int]:
        """Return the shape of the charge density."""
        return self.pgrids["total"].grid_shape

    @property
    def normalized_pgrids(self) -> dict[str, PGrid]:
        """Get the normalized pgrids.

        Since different codes use different normalization methods for
        volumetric data we should convert them to the same units (electrons / Angstrom^3)

        Returns
        -------
        dict[str, PGrid]:
            The normalized pgrids in units of (electrons / Angstrom^3)

        """
        return {
            k: PGrid(
                grid_data=_normalize_data(
                    grid_data=v.grid_data,
                    lattice=self.structure.lattice,
                    normalization=self.normalization,
                ),
                lattice=v.lattice,
            )
            for k, v in self.pgrids.items()
        }

    @property
    def lattice(self) -> npt.NDArray:
        """Lattice represented as an NDArray."""
        return self.structure.lattice.matrix

    @classmethod
    def from_pmg(
        cls, vdata: VolumetricData, normalization: str = "vasp"
    ) -> "ChargeDensity":
        """Get data from pymatgen object.

        Read a single key from the data field of a VolumetricData object

        Parameters
        ----------
        vdata:
            The volumetric data object
        normalization:
            The normalization scheme
            - 'vasp' sum of the data / number of grid points == number of electrons
            - None/"none" no normalization

        Returns
        -------
        ChargeDensity:
            The charge density object

        """
        pgrids = {
            k: PGrid(v, vdata.structure.lattice._matrix) for k, v in vdata.data.items()
        }
        return cls(
            pgrids=pgrids, structure=vdata.structure, normalization=normalization
        )

    def reorient_axis(self) -> None:
        """Rorient the lattices.

        Change the orientation of the lattice vector so that: ``a`` points along the x-axis, ``b`` is in the xy-plane,
        ``c`` is in the positive-z halve of space

        """
        args: Tuple[float, float, float, float, float, float] = (
            self.structure.lattice.abc + self.structure.lattice.angles
        )
        self.structure.lattice = Lattice.from_parameters(*args, vesta=True)

    def get_data_in_cube(self, s: float, ngrid: int, key: str = "total") -> npt.NDArray:
        """Return the charge density data sampled on a cube.

        Obtain a cubic basic cubic crop of the normalized charge density data.

        Parameters
        ----------
        s:
            The side length of the cube
        ngrid:
            Number of grid points in each direction
        key:
            The key to read from ``self.normalized_data``

        Returns
        -------
        NDArray:
            Regridded data in a ngrid x ngrid x ngrid array

        """
        grid_out = [ngrid, ngrid, ngrid]
        target_sc_lat_vecs = np.eye(3, 3) * s
        sc_mat = np.linalg.inv(self.structure.lattice.matrix) @ target_sc_lat_vecs
        _, res = get_sc_interp(self.normalized_data[key], sc_mat, grid_out)
        return res.reshape(grid_out)

    def get_transformed(
        self,
        sc_mat: npt.NDArray,
        grid_out: Union[List[int], int],
        origin: npt.ArrayLike = (0, 0, 0),
        up_sample: int = 1,
    ) -> "ChargeDensity":
        """Modify the structure and data and return a new object containing the reshaped data.

        Parameters
        ----------
        sc_mat:
            The transformation matrix to apply to the lattice vectors
        grid_out:
            The dimensions of the transformed grid
        origin:
            Origin of the new lattice in fractional coordinates of the input cell
        up_sample:
            The factor to scale up the sampling of the grid data using Fourier interpolation

        Returns
        -------
        ChargeDensity:
            The transformed ChargeDensity object

        """
        # warning if the sc_mat is not integer valued
        if not np.allclose(np.round(sc_mat), sc_mat):
            warnings.warn(
                "The `sc_mat` is not integer valued.\n"
                "Non-integer valued transformations are valid but will not product periodic structures, thus we cannot define a new Structure object.\n"
                "We will round the sc_mat to integer values for now but can implement functionality that returns a Molecule object in the future.",
            )
        sc_mat = np.round(sc_mat).astype(int)
        new_structure = self.structure.copy()
        new_structure.translate_sites(
            list(range(len(new_structure))), -np.array(origin)
        )
        new_structure = new_structure * sc_mat

        # determine the output grid
        lengths = new_structure.lattice.abc
        if isinstance(grid_out, int):
            ngrid = grid_out / new_structure.volume
            mult = (np.prod(lengths) / ngrid) ** (1 / 3)
            grid_out = [int(math.floor(max(l_ / mult, 1))) for l_ in lengths]
        else:
            grid_out = grid_out

        pgrids = {}
        for k, pgrid in self.normalized_pgrids.items():
            new_pgrid = pgrid.get_transformed(
                sc_mat=sc_mat, grid_out=grid_out, origin=origin, up_sample=up_sample
            )
            pgrids[k] = _scaled_data(
                grid_data=new_pgrid,
                lattice=new_structure.lattice,
                normalization=self.normalization,
            )

        return ChargeDensity(
            pgrids=pgrids,
            structure=new_structure,
            normalization=self.normalization,
        )

    def to_Chgcar(self) -> Chgcar:
        """Convert the charge density to a ``pymatgen.io.vasp.outputs.Chgcar`` object.

        Scale and convert each key in the pgrids dictionary and create a ``Chgcar`` object

        Returns
        -------
        Chgcar:
            The charge density object

        """
        struct = self.structure.copy()
        data_dict = {}
        for k, v in self.normalized_data.items():
            data_dict[k] = _scaled_data(
                v, lattice=self.structure.lattice, normalization="vasp"
            )
        return Chgcar(Poscar(struct), data=data_dict)

    @classmethod
    def from_file(
        cls, filename: str, pmg_obj: VolumetricData = Chgcar
    ) -> "ChargeDensity":
        """Read a ChargeDensity object from a file.

        Parameters
        ----------
        filename:
            The filename of the ChargeDensity object
        pmg_obj:
            The pymatgen object to read from the file (default: Chgcar).
            the `from_file` method from this class will be called to read the file.

        Returns
        -------
            ChargeDensity: The ChargeDensity object

        """
        return cls.from_pmg(pmg_obj.from_file(filename))

    @classmethod
    def from_hdf5(
        cls, filename: str, pmg_obj: VolumetricData = Chgcar
    ) -> "ChargeDensity":
        """Read a ChargeDensity object from a hdf5 file.

        Parameters
        ----------
        filename:
            The filename of the ChargeDensity object
        pmg_obj:
            The pymatgen object to read from the file (default: Chgcar).
            the `from_file` method from this class will be called to read the file.

        Returns
        -------
            ChargeDensity: The ChargeDensity object

        """
        return cls.from_pmg(pmg_obj.from_hdf5(filename))

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


# class SpinChargeDensity(MSONable, ChargeABC):
#     def __init__(self, chargeden_dict: Dict, aug_charge: Dict = None):
#         """
#         Wrapper class that parses multiple sets of grid data on the same lattice

#         Args:
#             chargeden_dict: A dictionary containing multiple charge density objects
#                         typically in the format {'total' : ChargeDen1, 'diff' : ChargeDen2}
#         """
#         self.chargeden_dict = chargeden_dict
#         self.aug_charge = aug_charge
#         self._tmp_key = next(
#             iter(self.chargeden_dict)
#         )  # get one key in the dictionary to make writing the subsequent code easier

#     @classmethod
#     def from_pmg_volumetric_data(
#         cls, vdata: VolumetricData, data_keys=("total", "diff")
#     ):
#         chargeden_dict = {}
#         data_aug = getattr(vdata, "data_aug", None)
#         for k in data_keys:
#             chargeden_dict[k] = ChargeDensity.from_pmg(vdata, data_key=k)
#         return cls(chargeden_dict, aug_charge=data_aug)

#     @property
#     def lattice(self) -> Lattice:
#         return self.chargeden_dict[self._tmp_key].lattice

#     def to_Chgcar(self) -> Chgcar:
#         struct = self.chargeden_dict[self._tmp_key].structure
#         data_ = {k: v.renormalized_data for k, v in self.chargeden_dict.items()}
#         return Chgcar(Poscar(struct), data_, data_aug=self.aug_charge)

#     def to_VolumetricData(self) -> VolumetricData:
#         key_ = next(iter(self.chargeden_dict))
#         struct = self.chargeden_dict[key_].structure
#         data_ = {k: v.renormalized_data for k, v in self.chargeden_dict.items()}
#         return VolumetricData(struct, data_)

#     def get_reshaped(
#         self,
#         sc_mat: npt.ArrayLike,
#         grid_out: Union[List, int],
#         origin: npt.ArrayLike = (0.0, 0.0, 0.0),
#         up_sample: int = 1,
#     ) -> "SpinChargeDensity":
#         new_spin_charge = {}
#         for k, v in self.chargeden_dict.items():
#             new_spin_charge[k] = v.get_reshaped_cell(sc_mat, frac_shift, grid_out)
#         factor = int(
#             new_spin_charge[self._tmp_key].structure.num_sites
#             / self.chargeden_dict[self._tmp_key].structure.num_sites
#         )
#         new_aug = {}
#         if self.aug_charge is not None:
#             for k, v in self.aug_charge.items():
#                 new_aug[k] = multiply_aug(v, factor)
#         return self.__class__(new_spin_charge, new_aug)

#     def reorient_axis(self) -> None:
#         for k, v in self.chargeden_dict:
#             v.reorient_axis()


@deprecated
def multiply_aug(data_aug: List[str], factor: int) -> List[str]:
    """Update the data in the augmentation charge.

    The original idea here was to use to to speed up some vasp calculations for
    supercells by initializing the entire CHGCAR file.
    The current code does not deal with transformation of the Augementation charges after regridding.

    This is a naive way to multiply the Augmentation data in the CHGCAR,
    a real working implementation will require analysis of the PAW projection operators.
    However, even with such an implementation, the speed up will be minimal due to VASP's internal
    minimization algorithms.

    Parameters
    ----------
    data_aug:
        The original augmentation data from a CHGCAR
    factor:
        The multiplication factor (some integer number of times it gets repeated)

    Returns
    -------
    List[str]:
        Each line of the augmentation data.

    """
    res: List[str] = []
    cur_block: List[str] = []
    cnt = 0
    for ll in data_aug:
        if "augmentation" in ll:
            if cur_block:
                for _ in range(factor):
                    cnt += 1
                    cur_block[
                        0
                    ] = f"augmentation occupancies{cnt:>4}{cur_block[0].split()[-1]:>4}\n"
                    res.extend(cur_block)
            cur_block = [ll]
        else:
            cur_block.append(ll)
    else:
        for _ in range(factor):
            cnt += 1
            cur_block[
                0
            ] = f"augmentation occupancies{cnt:>4}{cur_block[0].split()[-1]:>4}\n"
            res.extend(cur_block)
    return res


def _normalize_data(
    grid_data: npt.NDArray, lattice: Lattice, normalization: str | None = "vasp"
) -> npt.NDArray:
    """Normalize the data to the number of electrons.

    Since different codes use different normalization methods for
    volumetric data we should convert them to the same units (electrons / Angstrom^3)

    Parameters
    ----------
    grid_data:
        The grid data to normalize
    lattice:
        The lattice that the grid data is represented on
    normalization:
        The normalization method defaults to vasp
        - None: no normalization
        - vasp:
            The standard charge density from VASP is given as (rho*V) such that:
                sum(rho)/NGRID = NELECT/vol
            so the real normalized rho should be:
                rho = (rho*vol)*NGRID/vol/vol
            where the second `/vol` account for the different number of electrons in
            different cells

    Returns
    -------
    NDArray:
        The normalized grid data

    """
    if normalization is None or normalization[0].lower() == "n":
        return grid_data
    elif normalization[0].lower() == "v":
        return grid_data / lattice.volume
    else:
        raise NotImplementedError("Not a valid normalization scheme")


def _scaled_data(
    grid_data: npt.NDArray, lattice: Lattice, normalization: str | None = "vasp"
) -> npt.NDArray:
    """Undo the normalization of the data.

    Parameters
    ----------
    grid_data:
        The grid data to unnormalize
    lattice:
        The lattice that the grid data is represented on
    normalization:
        The normalization method defaults to vasp

    Returns
    -------
    NDArray:
        The un-normalized grid data

    """
    if normalization is None or normalization[0].lower() == "n":
        return grid_data
    elif normalization[0].lower() == "v":
        return grid_data * lattice.volume
    else:
        raise NotImplementedError("Not a valid normalization scheme")
