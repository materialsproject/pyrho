# -*- coding: utf-8 -*-

"""Fourier analysis functions."""

# Calculate the fourier transform
from typing import List, Callable
import numpy.typing as npt
from monty.json import MSONable
import numpy as np


class PFourier(MSONable):
    def __init__(self, fourier_data: np.ndarray, lattice: npt.ArrayLike):
        """
        Fourier transform defined for a periodic function. The representation must be
        SC agnostic, (i.e. if we double the size of the unit cell)

        Much of the analysis relies on the fact that numpy.ndarray.flatten puts the array in a particular order.
        The correct ordering is used but not strictly enforced by the present code so use with care.
        Also make sure the fourier data is generated in the correct way: with index='ij'

        Args:
            fourier_data: Complex data for each point in fourier_pos
            lattice: The corresponding real space lattice parameters
        """
        self.lattice = np.array(lattice)
        self.fourier_data = fourier_data
        self.shape = fourier_data.shape

    @property
    def fractional_reciprocal_pos(self) -> np.ndarray:
        """
        Return fft position  structured grid
        Assuming standard fft format A[N-1] = A[-1]
        """
        grid_vec = [np.linspace(0, 1, isize, endpoint=False) for isize in self.fourier_data.shape]
        frac_coords = np.meshgrid(*grid_vec, indexing="ij")
        return np.vstack([icoord.flatten() for icoord in frac_coords])

    @property
    def fft_pos_centered(self) -> np.ndarray:
        """
        Return the fft positions where the N-k is changed to -k
        """
        return np.array([ipos - np.round(ipos) for ipos in self.fractional_reciprocal_pos])

    @property
    def fft_pos_centered_s(self) -> np.ndarray:
        """
        Return the fft positions where the N-k is changed to -k
        """
        return np.array([ipos * self.fourier_data.shape[itr] for itr, ipos in enumerate(self.fft_pos_centered)])

    @property
    def fft_pos_centered_cartesian(self) -> List:
        """
        Return the fft positions where the N-k is changed to -k
        """
        return np.dot(self.reciprocal_lattice.T, self.fft_pos_centered)

    @property
    def fft_pos_centered_cartesian_s(self) -> List:
        """
        Return the fft positions where the N-k is changed to -k
        """
        return np.dot(self.reciprocal_lattice.T, self.fft_pos_centered_s)

    @property
    def reciprocal_lattice(self) -> np.ndarray:
        """
        Return the reciprocal lattice. Note that this is the standard
        reciprocal lattice used for solid state physics with a factor of 2 *
        pi.
        """
        return np.linalg.inv(self.lattice).T * 2 * np.pi

    @property
    def cartesian_reciprocal_pos(self) -> np.ndarray:
        """
        Get the list of reciprocal lattice points in cartesian coordinates
        """
        return np.dot(self.reciprocal_lattice.T, self.fractional_reciprocal_pos)

    # Helper functions open to input
    def get_points_dict(self, filter_pos: Callable = None, filter_val: Callable = None):
        """
        Return filtered data as a (position, fft_coefficient) pair
        Args:
            filter_val: Filter function applied to the fourier data values (Example: Only keep large Fourier weights)
            ftiler_pos: Filter function applied to the reciprocal positions (Example: low pass filter)
        """
        for cart_pos, val in zip(self.cartesian_reciprocal_pos.T, self.fourier_data.flatten()):
            if filter_val is not None and not filter_val(val):
                continue
            if filter_pos is not None and not filter_pos(cart_pos):
                continue
            yield cart_pos, val
