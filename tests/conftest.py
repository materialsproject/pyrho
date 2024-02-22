"""Shared fixtures for tests."""

from dataclasses import dataclass

import numpy as np
import pytest
from matplotlib import pyplot as plt


@pytest.fixture(scope="session")
def test_dir():
    from pathlib import Path

    module_dir = Path(__file__).resolve().parent.parent
    test_dir = module_dir / "test_files"
    return test_dir.resolve()


@dataclass
class Checker2D:
    """Check 2D regridding using an analytic 2D function."""

    mX: int = 1
    mY: int = 1

    def function(self, x, y):
        """Analytical function in 2D."""
        return np.sin(self.mX * x * 2 * np.pi)  # + np.cos(self.mY * y * 2 * np.pi)

    def get_xy(self, lat_mat, grids, origin=(0, 0)):
        """Get the x and y coordinates for a given pair of lattice vectors and grid size.

        Args:
        ----
            lat_mat: lattice vectors
            grids: grid size
            origin_cart: origin of the lattice vectors in cartesian coordinates

        Returns:
        -------
            XX: x coordinates for the grid in the shape of the grid
            YY: y coordinates for the grid in the shape of the grid

        """
        a_vec = np.linspace(0, 1, grids[0], endpoint=False)
        b_vec = np.linspace(0, 1, grids[1], endpoint=False)
        AA, BB = np.meshgrid(a_vec, b_vec, indexing="ij")
        frac = np.vstack([AA.flatten(), BB.flatten()])
        XX, YY = np.dot(np.array(lat_mat).T, frac)
        XX += origin[0]
        YY += origin[1]
        XX = XX.reshape(grids)
        YY = YY.reshape(grids)
        return XX, YY

    def check_z(self, ZZ, lat_mat, grids):
        """Check the regridded data against the analytic function."""
        XX, YY = self.get_xy(lat_mat, grids)
        ZZ_check = self.function(XX, YY)
        max_diff = np.max(ZZ - ZZ_check)
        assert max_diff < 1e-4

    def plot(self, lat_mat, grids, origin=(0, 0), ZZ=None):
        XX, YY = self.get_xy(lat_mat, grids, origin=origin)
        fig, ax = plt.subplots()
        if ZZ is None:
            ZZ = self.function(XX, YY)
        ax.pcolormesh(XX, YY, ZZ)
        ax.set_aspect("equal")
        plt.show()


@pytest.fixture(scope="session")
def checker_2D():
    return Checker2D
