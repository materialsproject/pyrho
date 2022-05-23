# mp-pyrho
[![.github/workflows/testing.yml](https://github.com/materialsproject/pyrho/actions/workflows/testing.yml/badge.svg)](https://github.com/materialsproject/pyrho/actions/workflows/testing.yml)
[![codecov](https://codecov.io/gh/materialsproject/pyrho/branch/main/graph/badge.svg?token=YoFMXzpeKI)](https://codecov.io/gh/materialsproject/pyrho)

Tools for re-griding volumetric quantum chemistry data for machine-learning purposes.

- Free software: BSD license
- Documentation: https://materialsproject.github.io/pyrho/

## Features

- Restructuring of periodic grid data (mapping from one cell representation to another)
- Up-sampling of periodic grid data using Fourier interpolation
- Helper functions to plot and examine the data

# Example Usage

## Basic usage of the PGrid class

The `PGrid` object is defined by an N-dimensional numpy array `grid_data` and a N lattice vectors given as a matrix `lattice`.
The input array is a scalar field that is defined on a regularly spaced set of grid points starting at the origin.
For example, you can construct a periodic field as follows:
```python
import numpy as np
from pyrho.core.pgrid import PGrid
from pyrho.vis.plotly import get_plotly_scatter_plot

def func(X,Y):
    return np.sin(X) * np.cos(2*Y)
a = np.linspace(0, np.pi, 27,endpoint=True)
b = np.linspace(0, np.pi, 28,endpoint=True)
arg_ = np.meshgrid(a, b, indexing='ij')
data = func(*arg_)
pg = PGrid(grid_data=data, lattice=[[np.pi,0], [0,np.pi]])
get_plotly_scatter_plot(pg.grid_data, pg.lattice, skips=1, opacity=1, marker_size=15)
```
The data can be examined using the helper plotting function.
![2d_pgrid_ex1](https://raw.github.com/materialsproject/pyrho/master/docs/_images/2d_pgrid_ex1.png)

The PGrid object has no concept of normalization so if you half the number of points in the domain, the range of the data will stay the same.
This is different from how the charge density is stored in codes like VASP where the values at each point depends on the number of grid points used to store the data.

The regridding capabilities allow the user to obtain the data in any arbitrary representation.
For example, if we want to shift to the middle of the unit-cell and create a ((1,1), (1,-1)) super-cell,
with a 30 by 32 grid, we can run:

```python
pg_2x = pg.get_transformed_obj([[1,1], [1,-1]], frac_shift=[0.5, 0.5], grid_out=[30,32])
get_plotly_scatter_plot(pg_2x.grid_data, pg_2x.lattice, skips=1, opacity=1, marker_size=10)
```
Which looks like this:
![2d_pgrid_ex2](https://raw.github.com/materialsproject/pyrho/master/docs/_images/2d_pgrid_ex2.png)

## The ChargeDensity class

The `ChargeDensity` object inherits from `PGrid` but also understands `pymatgen`'s definition of `VolumetricData`.
This allows us to actually read in CHGCAR objects either directly using `Chgcar.from_file` or read in the same data stored via `hdf5` as shown below:

```python
from pymatgen.io.vasp import Chgcar
from pyrho.core.chargeDensity import ChargeDensity
chgcar = Chgcar.from_hdf5("./test_files/Si.uc.hdf5")
chgcar = ChargeDensity.from_pmg_volumetric_data(chgcar)
get_plotly_scatter_plot(chgcar.grid_data,
                        lat_mat=chgcar.lattice,
                        skips=4,
                        mask=chgcar.grid_data > 0.3)
```

Here the plotting function slices the data using `[::4]` in each direction and filters out the data points below 0.3.
This makes the final plot less busy, so we can examine the data.

![chgcar_ex1](https://raw.github.com/materialsproject/pyrho/master/docs/_images/chgcar_ex1.png)

This charge density can also be transformed:
```python
chgcar_x2 = chgcar.get_transformed_obj(sc_mat = [[1,1,0],[1,-1,0],[0,0,1]], frac_shift=[0.5,0.5,0.5], grid_out=[120,120,60])
get_plotly_scatter_plot(chgcar_x2.grid_data, lat_mat=chgcar_x2.lattice, skips=4, mask=chgcar_x2.grid_data > 0.5, marker_size=10)
```

Note that we have shifted the origin to the center of unit cell which should be empty after the filering.
In the final transformed supercell, the new origin is indicated with a star.

![chgcar_ex2](https://raw.github.com/materialsproject/pyrho/master/docs/_images/chgcar_ex2.png)

## Credits

Jimmy-Xuan Shen: Project lead

Wennie Wang: For naming the package
