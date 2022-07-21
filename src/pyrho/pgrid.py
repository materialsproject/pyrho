"""Python class for ND grid data volumetric data."""
from __future__ import annotations

from typing import List, Union

import numpy as np
import numpy.typing as npt
from monty.json import MSONable

from pyrho.utils import gaussian_smear, get_sc_interp, interpolate_fourier


class PGrid(MSONable):
    """Class representing of _periodic_ grid data.

    Represents a periodic scalar field on a regular grid. The data is defined by the lattice vectors and the grid data.
    The grid points are implicitly defined by the lattice vectors and the grid shape.

    """

    def __init__(self, grid_data: npt.NDArray, lattice: npt.NDArray):
        """Initialize the PGrid object.

        Attributes
        ----------
        grid_data:
            Data stored on the regular rid
        lattice:
            Lattice vectors of the grid

        """
        self.grid_data = grid_data
        self.lattice = np.array(lattice)
        self._dim = len(self.grid_data.shape)
        self.grid_shape = self.grid_data.shape
        self.ngridpts = np.prod(self.grid_shape)

    def _transform_data(
        self,
        sc_mat: npt.ArrayLike,
        grid_out: List[int],
        origin: npt.ArrayLike | None = None,
        up_sample: int = 1,
    ) -> npt.NDArray:
        """Apply a supercell transformation to the grid data.

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
        origin = np.array(origin) if origin is not None else np.zeros(self._dim)
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

    def __mul__(self, factor: float) -> PGrid:
        """Multiply the grid data by a factor.

        Parameters
        ----------
        factor:
            The factor to multiply the grid data by

        Returns
        -------
        PGrid:
            The new PGrid object

        """
        return PGrid(grid_data=self.grid_data * factor, lattice=self.lattice)

    def __truediv__(self, factor: float) -> PGrid:
        """Divide the grid data by a factor.

        Parameters
        ----------
        factor:
            The factor to divide the grid data by

        Returns
        -------
        PGrid:
            The new PGrid object

        """
        return PGrid(grid_data=self.grid_data / factor, lattice=self.lattice)

    def get_transformed(
        self,
        sc_mat: Union[List[List[int]], npt.NDArray],
        grid_out: List[int],
        origin: npt.NDArray | None = None,
        up_sample: int = 1,
    ) -> PGrid:
        """Get a new PGrid object for the new transformed data.

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
        origin = np.array(origin) if origin is not None else np.zeros(self._dim)
        new_data = self._transform_data(
            sc_mat=sc_mat, grid_out=grid_out, origin=origin, up_sample=up_sample
        )
        new_lattice = np.dot(sc_mat, self.lattice)
        return PGrid(grid_data=new_data, lattice=new_lattice)

    def lossy_smooth_compression(
        self, grid_out: List, smear_std: float = 0.2
    ) -> npt.NDArray:
        """Perform Fourier interpolation then Gaussian smoothing.

        The smoothing makes sure that simple operation like max and min filters still
        give the same results.

        Parameters
        ----------
        grid_out:
            desired output grid of the compressed data.
        smear_std:
            standard deviation of the Gaussian smoothing

        Returns
        -------
        NDArray:
            Smoothed array

        """
        arr_interp = np.absolute(interpolate_fourier(self.grid_data, grid_out))
        if smear_std > 0:
            arr_interp, _ = gaussian_smear(
                arr=arr_interp, lattice=self.lattice, sigma=smear_std
            )
        return arr_interp
