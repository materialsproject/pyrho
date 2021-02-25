# -*- coding: utf-8 -*-

"""Python class for ND grid data volumetric data"""
from typing import List, Union, Tuple
import numpy as np
from monty.json import MSONable
from pyrho.core.utils import (
    get_sc_interp,
    interpolate_fourier,
    get_ucell_frac_fit_sphere,
)
from scipy.ndimage import convolve
import numpy.typing as npt


class PGrid(MSONable):
    def __init__(self, grid_data: np.ndarray, lattice_vecs: np.ndarray = None):
        """
        Base class for N-dimensional Regular period grid data.
        The core code should be valid in N-dimensions and not depend on pymatgen

        Args:
            grid_data: Data stored on the regular rid
            lattice_vecs: list of lattice vectors
        """
        if lattice_vecs is not None:  # Some children will set the lattice
            self.lattice_vecs = np.array(lattice_vecs)
        self.grid_data = grid_data
        self._dim = len(self.grid_data.shape)
        self.grid_shape = self.grid_data.shape
        self.ngridpts = np.prod(self.grid_shape)

    def get_transformed_data(
        self, sc_mat: npt.ArrayLike, frac_shift: npt.ArrayLike, grid_out: List[int], up_sample: int = 1,
    ) -> np.ndarray:
        """
        Apply a transformation to the grid data
        This function assumes that the data is fixed in place and the transformation
        is applied to the lattice vectors.
        Args:
            sc_mat: matrix transformation applied to the lattice vectors
            frac_shift: shift the lattice in fractional coordinates of the input cell
            up_sample: the factor to scale up the sampling of the grid data

        sc_mat  --->  [2, 1]    trans  --->  [0.1, 0.3]
                      [0, 1]
        new lattice vectors:
            a = [0.2, 0.4] --> [2.2, 1.4]
            b = [0.2, 0.4] --> [0.2, 1.1]

        Returns:
            transformed data

        """
        if up_sample == 1:
            interp_grid_data = self.grid_data
        else:
            interp_grid_data = interpolate_fourier(
                arr_in=self.grid_data, shape=[g_dim_ * up_sample for g_dim_ in self.grid_data.shape],
            )
        _, new_rho = get_sc_interp(interp_grid_data, sc_mat, grid_sizes=grid_out, origin=frac_shift)  # type: ignore
        new_rho = new_rho.reshape(grid_out)

        # TODO make this part of the original transformation
        # grid_shifts = [
        #     int(t * g) for t, g in zip(frac_shift - np.round(frac_shift), grid_out)
        # ]
        #
        # new_rho = roll_array(new_rho, grid_shifts)

        return new_rho

    def get_transformed_obj(
        self,
        sc_mat: Union[List[List[int]], np.ndarray],
        frac_shift: Union[np.ndarray, List[float], Tuple[float]],
        grid_out: List[int],
        up_sample: int = 1,
    ) -> "PGrid":
        """
        Get a new PGrid object for the new transformed data
        Args:
            sc_mat: matrix transformation applied to the lattice vectors
            frac_shift: shift the lattice in fractional coordinates of the output cell
            grid_out: The size of the output grid to interpolate on
            up_sample: the factor to scale up the sampling of the grid data

        Returns:
            New PGrid object
        """
        new_data = self.get_transformed_data(sc_mat, frac_shift, grid_out=grid_out, up_sample=up_sample)
        new_lattice = np.dot(sc_mat, self.lattice)
        return PGrid(grid_data=new_data, lattice_vecs=new_lattice)

    def gaussian_smear(self, sigma: float = 0.2, arr_in: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applies an isotropic Gaussian smear of width (standard deviation) r to
        the potential field. This is necessary to avoid finding paths through
        narrow minima or nodes that may exist in the field (although any
        potential or charge distribution generated from GGA should be
        relatively smooth anyway). The smearing obeys periodic
        boundary conditions at the edges of the cell.

        Args:
            sigma: Smearing width in cartesian coordinates, in the same units as the
            structure lattice vectors
            arr_in: input data array to smear, if None: smear self.grid_data
        """
        # get the dimension of the filter needed for 1 std dev of gaussian mask
        # Go 4 standard deviations away
        if arr_in is None:
            arr = self.grid_data
        else:
            arr = arr_in
        r_frac = get_ucell_frac_fit_sphere(lattice=self.lattice, r=sigma * 5)
        filter_shape = [
            int(np.ceil(itr_rf * itr_dim / 2) * 2)  # dimension of the filter should be even
            for itr_rf, itr_dim in zip(r_frac, arr.shape)
        ]

        filter_latt = np.array([(filter_shape[_] + 1) / (arr.shape[_] + 1) * self.lattice[_] for _ in range(self._dim)])

        # Get the fractional positions
        filter_frac_c = [np.linspace(0, 1, _, endpoint=False) for _ in filter_shape]
        frac_pos = np.meshgrid(*filter_frac_c, indexing="ij")
        frac_pos = [_.flatten() for _ in frac_pos]

        # convert to cartesian
        cart_pos = np.matmul(filter_latt.T, np.vstack(frac_pos))

        # Distance the center if 1d we make  this iterable
        tmp = sum(filter_latt)  # type: Union[float, np.ndarray]
        if isinstance(tmp, np.ndarray):
            mid_point = tmp / 2  # type: Union[List, np.ndarray]
        else:
            mid_point = [tmp / 2]
        disp2mid2 = [(i_coord.reshape(filter_shape) - mp_coord) ** 2 for mp_coord, i_coord in zip(mid_point, cart_pos)]
        dist2mid = np.sqrt(sum(disp2mid2))
        # make sure the mask is zero?
        mm = dist2mid <= sigma * 4
        gauss = np.exp(-1 / 2 * (dist2mid / sigma) ** 2) * mm
        gauss = gauss / gauss.sum()
        return convolve(input=arr, weights=gauss, mode="wrap"), gauss

    def lossy_smooth_compression(self, grid_out: List, smear_std: float = 0.2) -> np.ndarray:
        """
        Perform Fourier interpolation then Gaussian smoothing.
        the smoothing makes sure that simple operation like max and min filters still
        give the same results.

        Args:
            grid_out: desired output grid of the compressed data,

        Returns:
            Smoothed array
        """
        arr_interp = np.absolute(interpolate_fourier(self.grid_data, grid_out))
        if smear_std > 0:
            arr_interp, _ = self.gaussian_smear(arr_in=arr_interp, sigma=smear_std)
        return arr_interp
