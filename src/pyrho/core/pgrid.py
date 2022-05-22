from __future__ import annotations

"""Python class for ND grid data volumetric data"""
from typing import List, Tuple, Union

import numpy as np
import numpy.typing as npt
from monty.json import MSONable
from scipy.ndimage import convolve

from pyrho.core.utils import (
    get_sc_interp,
    get_ucell_frac_fit_sphere,
    interpolate_fourier,
)


class PGrid(MSONable):
    def __init__(self, grid_data: npt.NDArray, lattice: npt.NDArray | None = None):
        """Base class for N-dimensional Regular period grid data.

        The core code is valid for any N-dimensional periodic data

        Parameters
        ----------
        grid_data:
            Data stored on the regular rid
        lattice:
            Lattice vectors of the grid
        """
        if lattice is not None:  # type: ignore
            self.lattice = np.array(lattice)
        self.grid_data = grid_data
        self._dim = len(self.grid_data.shape)
        self.grid_shape = self.grid_data.shape
        self.ngridpts = np.prod(self.grid_shape)

    def _transform_data(
        self,
        sc_mat: npt.ArrayLike,
        grid_out: List[int],
        origin: npt.ArrayLike = (0, 0, 0),
        up_sample: int = 1,
    ) -> npt.NDArray:
        """Apply a supercell transformation to the grid data

        This function assumes that the data is fixed in place and the transformation
        is applied to the lattice vectors.

        Parameters
        ----------
        sc_mat:
            Matrix transformation applied to the lattice vectors
        grid_out:
            The dimensions of the output grid
        origin:
            Origin of the new lattice in fractional coordinates of the input cell
        up_sample:
            The factor to scale up the sampling of the grid data using Fourier interpolation

        Returns
        -------
        NDArray:
            The transformed data

        """
        if up_sample == 1:
            interp_grid_data = self.grid_data
        else:
            interp_grid_data = interpolate_fourier(
                arr_in=self.grid_data,
                shape=[g_dim_ * up_sample for g_dim_ in self.grid_data.shape],
            )
        _, new_data = get_sc_interp(interp_grid_data, sc_mat, grid_sizes=grid_out, origin=origin)  # type: ignore
        new_data = new_data.reshape(grid_out)
        return new_data

    def get_transformed(
        self,
        sc_mat: Union[List[List[int]], npt.NDArray],
        grid_out: List[int],
        origin: npt.NDArray = (0, 0, 0),
        up_sample: int = 1,
    ) -> PGrid:
        """Get a new PGrid object for the new transformed data

        Parameters
        ----------
        sc_mat:
            Matrix transformation applied to the lattice vectors
        grid_out:
            The dimensions of the output grid
        origin:
            Origin of the new lattice in fractional coordinates of the input cell
        up_sample:
            The factor to scale up the sampling of the grid data using Fourier interpolation

        Returns
        -------
        PGrid:
            The transformed PGrid object

        """
        new_data = self._transform_data(
            sc_mat=sc_mat, grid_out=grid_out, origin=origin, up_sample=up_sample
        )
        new_lattice = np.dot(sc_mat, self.lattice)
        return PGrid(grid_data=new_data, lattice=new_lattice)

    def gaussian_smear(
        self,
        arr_in: npt.NDArray | None = None,
        sigma: float = 0.2,
    ) -> Tuple[npt.NDArray, npt.NDArray]:
        """Applies an isotropic Gaussian smear

        Apply a Gaussian smearing of width (standard deviation) `sigma` to
        the periodic field.  The smearing obeys periodic boundary conditions at
        the edges of the cell.

        Parameters
        ----------
        arr_in:
            input data array to smear, if None: smear self.grid_data
        sigma:
            Smearing width in cartesian coordinates, in the same units as the lattice vectors
        """
        # get the dimension of the filter needed for 1 std dev of gaussian mask
        # Go 5 standard deviations away
        if arr_in is None:
            arr = self.grid_data
        else:
            arr = arr_in

        r_frac = get_ucell_frac_fit_sphere(lattice=self.lattice, r=sigma * 5)
        filter_shape = [
            int(
                np.ceil(itr_rf * itr_dim / 2) * 2
            )  # dimension of the filter should be even
            for itr_rf, itr_dim in zip(r_frac, arr.shape)
        ]

        filter_latt = np.array(
            [
                (filter_shape[_] + 1) / (arr.shape[_] + 1) * self.lattice[_]
                for _ in range(self._dim)
            ]
        )

        # Get the fractional positions
        filter_frac_c = [np.linspace(0, 1, _, endpoint=False) for _ in filter_shape]
        frac_pos = np.meshgrid(*filter_frac_c, indexing="ij")
        frac_pos = [_.flatten() for _ in frac_pos]

        # convert to cartesian
        cart_pos = np.matmul(filter_latt.T, np.vstack(frac_pos))

        # Distance the center if 1d we make  this iterable
        tmp: Union[float, npt.NDArray] = sum(filter_latt)
        if isinstance(tmp, np.ndarray):
            mid_point = tmp / 2
        else:
            mid_point = [tmp / 2]
        disp2mid2 = [
            (i_coord.reshape(filter_shape) - mp_coord) ** 2
            for mp_coord, i_coord in zip(mid_point, cart_pos)
        ]
        dist2mid = np.sqrt(sum(disp2mid2))
        # make sure the mask is zero?
        mm = dist2mid <= sigma * 4
        gauss = np.exp(-1 / 2 * (dist2mid / sigma) ** 2) * mm
        gauss = gauss / gauss.sum()
        return convolve(input=arr, weights=gauss, mode="wrap"), gauss

    def lossy_smooth_compression(
        self, grid_out: List, smear_std: float = 0.2
    ) -> npt.NDArray:
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
