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
class Checker2D:
    """Check 2D regridding using an analytic 2D function."""

    mX: int = 1
    mY: int = 1

    def function(self, x, y):
        """Analytical function in 2D."""
        return np.sin(self.mX * x * 2 * np.pi)

    def get_xy(self, lat_mat, grids, origin=(0, 0)):
        """Get the x and y coordinates for a given pair of lattice vectors and grid size.

        Args:
        ----
            lat_mat: lattice vectors
            grids: grid size
            origin_cart: origin of the lattice vectors in cartesian coordinates

        Returns
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


def gpu_warmup(
    num_warups: int = 10,
    grid_out: list = [2, 2],
    origin: list = [0.0, 0.0],
    up_sample: int = 1,
):
    nx, ny = np.random.randint(10, 11, size=2)
    A, B = np.random.randint(1, 11, size=2)
    checker = Checker2D()
    XX, YY = checker.get_xy(np.eye(2), [nx, ny])
    ZZ = checker.function(XX, YY)

    # Create PGrid instance
    pgrid = PGrid(ZZ, [[A, 0], [0, B]])
    sc_mat = np.eye(2)

    # --- GPU warm-up
    for _ in range(num_warups):
        pgrid._transform_data(
            sc_mat, origin=origin, grid_out=grid_out, up_sample=up_sample, use_gpu=True
        )


def plot_timing(num_runs: int = 51, origin: list = [0.0, 0.0], up_sample: int = 4):
    nx, ny = np.random.randint(10, 501, size=2)
    A, B = np.random.randint(1, 11, size=2)
    checker = Checker2D()
    XX, YY = checker.get_xy(np.eye(2), [nx, ny])
    ZZ = checker.function(XX, YY)

    # Create PGrid instance
    pgrid = PGrid(ZZ, [[A, 0], [0, B]])

    sc_mat = np.eye(2)
    grid_out = [nx, ny]

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
    plt.title("Benchmark timing - 2D grids")
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
    plt.title("Benchmark timing - 2D grids")
    plt.grid(True, which="both", ls="--", lw=0.5)

    plt.tight_layout(pad=3.0)
    plt.savefig("results/timing_2d.png", dpi=300)
