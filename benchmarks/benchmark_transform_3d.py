import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from pyrho.pgrid import PGrid

try:
    import cupy as cp

    _cupy_available = True
except ImportError:
    _cupy_available = False

# Check CuPy + CUDA availability
if not _cupy_available or cp.cuda.runtime.getDeviceCount() == 0:
    raise RuntimeError("CuPy/CUDA not available. Cannot run GPU code.")


@dataclass
class Checker3D:
    """Check 3D regridding using an analytic 3D function."""

    mX: int = 1
    mY: int = 1
    mZ: int = 1

    def function(self, x, y, z):
        """Analytical function in 3D."""
        return (
            np.sin(self.mX * x * 2 * np.pi)
            + np.cos(self.mY * y * 2 * np.pi)
            + np.sin(self.mZ * z * 2 * np.pi)
        )

    def get_xyz(self, lat_mat, grids, origin=(0, 0, 0)):
        """Get the x, y, z coordinates for a given set of lattice vectors and grid size.

        Args:
        ----
            lat_mat: lattice vectors (3x3)
            grids: grid size (nx, ny, nz)
            origin: origin of the lattice vectors in cartesian coordinates

        Returns
        -------
            XX, YY, ZZ: coordinates in the shape of the grid
        """
        a_vec = np.linspace(0, 1, grids[0], endpoint=False)
        b_vec = np.linspace(0, 1, grids[1], endpoint=False)
        c_vec = np.linspace(0, 1, grids[2], endpoint=False)
        AA, BB, CC = np.meshgrid(a_vec, b_vec, c_vec, indexing="ij")
        frac = np.vstack([AA.flatten(), BB.flatten(), CC.flatten()])
        coords = np.dot(np.array(lat_mat).T, frac)
        XX = coords[0].reshape(grids) + origin[0]
        YY = coords[1].reshape(grids) + origin[1]
        ZZ = coords[2].reshape(grids) + origin[2]
        return XX, YY, ZZ


def gpu_warmup(
    num_warups: int = 10,
    grid_out: list = [2, 2, 2],
    origin: list = [0.0, 0.0, 0.0],
    up_sample: int = 1,
):
    nx, ny, nz = np.random.randint(10, 11, size=3)
    A, B, C = np.random.randint(1, 11, size=3)
    checker = Checker3D()
    WW, XX, YY = checker.get_xyz(np.eye(3), [nx, ny, nz])
    ZZ = checker.function(WW, XX, YY)

    # Create PGrid instance
    pgrid = PGrid(ZZ, [[A, 0, 0], [0, B, 0], [0, 0, C]])
    sc_mat = np.eye(3)

    # --- GPU warm-up
    for _ in range(num_warups):
        pgrid._transform_data(
            sc_mat, origin=origin, grid_out=grid_out, up_sample=up_sample, use_gpu=True
        )


def plot_timing(num_runs: int = 51, origin: list = [0.0, 0.0, 0.0], up_sample: int = 4):
    nx, ny, nz = np.random.randint(10, 101, size=3)
    A, B, C = np.random.randint(1, 11, size=3)
    checker = Checker3D()
    WW, XX, YY = checker.get_xyz(np.eye(3), [nx, ny, nz])
    ZZ = checker.function(WW, XX, YY)

    # Create PGrid instance
    pgrid = PGrid(ZZ, [[A, 0, 0], [0, B, 0], [0, 0, C]])

    sc_mat = np.eye(3)
    grid_out = [nx, ny, nz]

    # --- GPU ---
    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    for _ in range(num_runs):
        transformed_data_fast = pgrid._transform_data(
            sc_mat, origin=origin, grid_out=grid_out, up_sample=up_sample, use_gpu=True
        )
    cp.cuda.Stream.null.synchronize()
    t_gpu = time.perf_counter() - start

    # --- CPU ---
    start = time.perf_counter()
    for _ in range(num_runs):
        transformed_data = pgrid._transform_data(
            sc_mat, origin=origin, grid_out=grid_out, up_sample=up_sample, use_gpu=False
        )
    t_cpu = time.perf_counter() - start

    # --- Transformation check ---
    assert np.max(np.abs(transformed_data - transformed_data_fast)) < 1e-4, (
        "Fast transform output differs from original."
    )
    plt.subplot(1, 2, 1)
    plt.plot([np.prod(grid_out)], [1000 * t_cpu / num_runs], "ro", markersize=3)
    plt.plot([np.prod(grid_out)], [1000 * t_gpu / num_runs], "bo", markersize=3)

    plt.subplot(1, 2, 2)
    plt.plot([np.prod(grid_out)], [t_cpu / t_gpu], "go", markersize=3)


if __name__ == "__main__":
    gpu_warmup()
    plt.figure(figsize=(10, 5))
    for i in range(50):
        plot_timing()
    plt.subplot(1, 2, 1)
    plt.xscale("log")
    plt.xlabel("Grid size (log scale)")
    ax = plt.gca()
    ax.get_xaxis().set_major_formatter(
        plt.ScalarFormatter()
    )  # show all labels as numbers
    plt.ylabel("Time (ms)")
    plt.title("Benchmark timing - 3D grids")
    plt.grid(True, which="both", ls="--", lw=0.5)
    legend_elements = [
        Line2D([0], [0], marker="o", color="r", label="CPU", linestyle=""),
        Line2D([0], [0], marker="o", color="b", label="GPU", linestyle=""),
    ]
    plt.legend(handles=legend_elements)

    plt.subplot(1, 2, 2)
    plt.xscale("log")
    plt.xlabel("Grid size (log scale)")
    ax = plt.gca()
    ax.get_xaxis().set_major_formatter(
        plt.ScalarFormatter()
    )  # show all labels as numbers
    plt.ylabel("Speedup (GPU / CPU)")
    plt.title("Benchmark timing - 3D grids")
    plt.grid(True, which="both", ls="--", lw=0.5)

    plt.tight_layout(pad=3.0)
    plt.savefig("results/timing_3d.png", dpi=300)
