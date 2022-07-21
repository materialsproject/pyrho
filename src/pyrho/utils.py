"""Util Functions."""
from __future__ import annotations

from itertools import combinations
from typing import Iterable, List, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import convolve

__all__ = [
    "pad_arr",
    "interpolate_fourier",
    "roll_array",
    "get_sc_interp",
    "get_padded_array",
    "get_plane_spacing",
    "get_ucell_frac_fit_sphere",
    "get_ucell_frac_fit_sphere",
    "gaussian_smear",
]


def pad_arr(arr_in: NDArray, shape: List[int]) -> NDArray:
    """Pad a function on a hypercube.

    .. note::
        We basically need to move the data at the corners a hypercube to the corners of a bigger/smaller
        hypercube. Each corner of the hypercube can be represented as a binary number with length equal
        to the dimension of the cube. Loop over the binary representation to figure out which corner you
        are at the slice accordingly.

    >>> arr = np.array([[1,2,3], [4,5,6]])
    >>> pad_arr(arr, [5,3])
    array([[1, 2, 3],
           [0, 0, 0],
           [0, 0, 0],
           [0, 0, 0],
           [4, 5, 6]])
    >>> pad_arr(arr, [2,2])
    array([[1, 3],
           [4, 6]])

    Parameters
    ----------
    arr_in:
        Data to be padded with zeros
    shape:
        Desired shape of the array

    Returns
    -------
    NDArray:
        padded data

    """
    # for _, isize in enumerate(shape):
    #     if isize < arr_in.shape[_]:
    #         raise Warning(
    #             "Some dimension of output array is smaller than the same dimension of input array."
    #         )
    def get_slice(idig, idim, bound_pairs):
        if idig == "0":
            return slice(0, bound_pairs[idim][0])
        elif idig == "1":
            return slice(-bound_pairs[idim][1], None)
        else:
            raise ValueError("Binary digit not 1 or 0")

    dimensions = arr_in.shape
    boundaries = [
        (
            int(np.ceil(min(i_dim, j_dim) + 1) / 2.0),
            int(np.floor(min(i_dim, j_dim)) / 2.0),
        )
        for i_dim, j_dim in zip(dimensions, shape)
    ]
    dim = len(dimensions)
    fmt = f"#0{dim+2}b"
    corners = [format(itr, fmt)[-dim:] for itr in range(2**dim)]
    arr_out = np.zeros(shape, dtype=arr_in.dtype)

    for ic in corners:
        islice = tuple(
            get_slice(idig, idim, boundaries) for idim, idig in enumerate(ic)
        )
        arr_out[islice] = arr_in[islice]
    return arr_out


def interpolate_fourier(arr_in: NDArray, shape: List[int]) -> NDArray:
    """Fourier interpolate an array.

    Interpolate the data to some final shape, keep magnitude the same.
    Will perform best if the input array is periodic in all directions.

    >>> arr = np.array([[5.0, 10.0, 10.0, 7.0],
    ...        [22.0, 12.0, 7.0, 3.0],
    ...        [16.0, 10.0, 3.0, 5.0],
    ...        [16.0, 22.0, 16.0, 6.0],
    ...        [19.0, 20.0, 3.0, 7.0]])
    >>> np.round(interpolate_fourier(arr_in=arr, shape=[10,8]),2)
    array([[ 5.  ,  7.29, 10.  , 10.83, 10.  ,  8.71,  7.  ,  5.17],
           [12.02, 12.22, 11.22, 12.72, 12.72,  8.11,  4.71,  7.62],
           [22.  , 19.49, 12.  ,  8.88,  7.  ,  2.51,  3.  , 13.12],
           [23.2 , 19.82, 10.33,  3.87,  0.64, -1.05,  3.36, 14.9 ],
           [16.  , 14.86, 10.  ,  5.67,  3.  ,  2.14,  5.  , 11.33],
           [11.58, 14.07, 14.8 , 14.24, 11.82,  8.06,  6.05,  7.88],
           [16.  , 20.66, 22.  , 20.66, 16.  ,  9.34,  6.  ,  9.34],
           [21.86, 26.35, 24.64, 18.31, 10.48,  5.16,  6.05, 13.21],
           [19.  , 22.5 , 20.  , 11.19,  3.  ,  2.  ,  7.  , 13.31],
           [ 9.34, 12.33, 13.01,  8.08,  3.33,  4.42,  7.84,  8.67]])

    Parameters
    ----------
    arr_in:
        Input array of data
    shape:
        Desired shape shape of the interpolated data

    Returns
    -------
    NDArray:
        Interpolated data in the desired shape

    """
    fft_res = np.fft.fftn(arr_in)
    fft_res = pad_arr(fft_res, shape)
    results = np.fft.ifftn(fft_res) * np.size(fft_res) / np.size(arr_in)
    # take the real value if the input array is real
    if not np.iscomplexobj(arr_in):
        return np.real(results)
    return results


def roll_array(arr: NDArray, roll_vec: List[int]) -> NDArray:
    """Roll the array along specified axes.

    Shift the index of an ndarray based on roll_vec.

    Parameters
    ----------
    arr:
        array to be rolled
    roll_vec:
        number of indices in each direction to roll

    Returns
    -------
    NDArray:
        The rolled array

    """
    for ii, roll_val in enumerate(roll_vec):
        arr = np.roll(arr, roll_val, ii)
    return arr


def get_sc_interp(
    data_in: NDArray,
    sc_mat: ArrayLike,
    grid_sizes: List[int],
    scipy_interp_method="linear",
    origin: Union[NDArray, List[float], Tuple[float]] = None,
) -> Tuple[NDArray, NDArray]:
    """Get the interpolated data in a supercell.

    Take a data array defined on a regular lattice and a new set of lattice
    vector (in units of the the original lattice), reproduce the data on the new grid.
    We can consider the original cell to be unit cell (UC) and the new cell to be a super cell (SC)
    defined using the basis of the UC -- although the the SC in this case can be any lattice.

    .. note::
        Assuming we have some kind of data stored on a regular grid --- [0,1) in each dimension
        In the frame where UC is orthonormal we can just map all points in the sc_grid to the a
        position in the unit cube.
        Then perform interpolation on the new coordinates that are all within the cube
        We have to include the boundaries of the UC since the mapped grid must lie between the existing points

    .. note::
        This algorithm can be used in real space and fourier space
        For real space we expect the data to be smooth, so we should use "linear" interpolation
        For fourier space we expect the position of the exact fourier component is important so use "nearest"

    Parameters
    ----------
    data_in:
        Data stored on a regular grid
    sc_mat:
        Lattice vectors of new cell in the units of the old cell
    grid_sizes:
        Number of grid points in each direction in the new cell
    scipy_interp_method:
        Interpolation method to be used
    origin:
        Shift applied to the origin in fractional coordinates

    Returns
    -------
    NDArray:
        size ``(ndim x prod(grid_size))`` the cartesian coordinates of each point in the new data
    NDArray:
        size ``(prod(grid_size))`` the regridded data

    """
    # We will need to interpolated near the boundaries so we have to pad the data
    padded_data = get_padded_array(data_in)

    # interpolate the padded data to the sc coordinate in the cube
    uc_vecs = [
        np.linspace(0, 1, isize + 1, endpoint=True) for isize in data_in.shape
    ]  # need to go from ij indexing to xy
    interp_func = RegularGridInterpolator(
        uc_vecs, padded_data, method=scipy_interp_method
    )  # input data from CHGCAR requires transpose
    grid_vec = [np.linspace(0, 1, isize, endpoint=False) for isize in grid_sizes]
    frac_coords = np.meshgrid(
        *grid_vec, indexing="ij"
    )  # indexing to match the labeled array
    frac_coords = np.vstack([icoord.flatten() for icoord in frac_coords])

    sc_coord = np.dot(np.array(sc_mat).T, frac_coords)  # shape (dim, NGRID)

    if origin is not None:
        sc_coord += np.array([[_] for _ in origin])

    mapped_coords = sc_coord - np.floor(sc_coord)

    return sc_coord, interp_func(mapped_coords.T)


def get_padded_array(data_in: NDArray) -> NDArray:
    """Pad the array with zeros.

    Pad an array in each direction with the periodic boundary conditions.

    Parameters
    ----------
    data_in:
        Array to be padded

    Returns
    -------
    NDArray:
        Padded array

    """
    padded_data = data_in.copy()
    slice_arr = [
        [slice(0, 1) if ii == i else slice(0, None) for ii in range(len(data_in.shape))]
        for i in range(len(data_in.shape))
    ]
    for idim, islice in enumerate(slice_arr):
        padded_data = np.concatenate(
            (padded_data, padded_data[tuple(islice)]), axis=idim
        )
    return padded_data


def get_plane_spacing(lattice: NDArray) -> List[float]:
    """Get the cartesian spacing between periodic planes of a unit cell.

    >>> get_plane_spacing([[1,0,0], [1,1,0], [0,0,2]]) # doctest: +ELLIPSIS
    [0.7653..., 1.042..., 2.0]

    Parameters
    ----------
    lattice:
        List of lattice vectors in cartesian coordinates

    Returns
    -------
    List[float]:
        List where the k-th element is is the spacing of planes generated by all
        lattice vectors EXCEPT the k-th one

    """
    # get all pairwise projections i must be smaller than j
    ndim = len(lattice)
    idx_pairs = [*combinations(range(ndim), 2)]
    latt_len = [np.linalg.norm(lattice[i]) for i in range(ndim)]
    pproj = {
        (i, j): np.dot(lattice[i], lattice[j]) / latt_len[i] / latt_len[j]
        for i, j in idx_pairs
    }
    # get the spacing in each direction:
    spacing = []
    for idir in range(ndim):
        idir_proj = [
            np.array(lattice[j]) * pproj[tuple(sorted([idir, j]))]  # type: ignore
            for j in range(ndim)
            if j != idir
        ]
        v_perp_subspace = lattice[idir] - sum(idir_proj)
        spacing.append(np.linalg.norm(v_perp_subspace))
    return spacing


def get_ucell_frac_fit_sphere(lattice: np.ndarray, r: float = 0.2) -> Iterable[float]:
    """Lattice ffractions in each direction that are within r of the origin.

    Get the smallest you can make make the lattice parameter in each direction to fit a
    sphere of radius `r`. For lattice vector `k`, the sphere must be contained within
    the hyperplanes defined by all lattice vectors except `k`.

    >>> get_ucell_frac_fit_sphere([[1,0,0], [1,-1, 0], [0,0,2]], 0.1) # doctest: +ELLIPSIS
    [0.2613..., 0.1919..., 0.1]

    Parameters
    ----------
    lattice:
        List of lattice vectors in cartesian coordinates
    r:
        width of the sphere

    Returns
    -------
    Iterable of floats
        fraction of lattice vector in each direction need to fit the sphere

    """
    rfrac = []
    for ispace in get_plane_spacing(lattice=lattice):
        rfrac.append(2 * r / ispace)
    return rfrac


def gaussian_smear(
    arr: NDArray,
    lattice: NDArray,
    sigma: float = 0.2,
    multiple: float = 4.0,
) -> Tuple[NDArray, NDArray]:
    """Apply an isotropic Gaussian smearing to periodic data.

    Apply a Gaussian smearing of width (standard deviation) `sigma` to
    the periodic field.  The smearing obeys periodic boundary conditions at
    the edges of the cell.

    Parameters
    ----------
    arr:
        input data array to smear, if None: smear self.grid_data
    lattice:
        lattice vectors in cartesian coordinates
    sigma:
        Smearing width in cartesian coordinates, in the same units as the lattice vectors
    multiple:
        ``multiple * sigma`` is the cutoff radius for the smearing

    Returns
    -------
    NDArray:
        The smear data array.

    """
    # Since smearing requires floating point, we need to make sure the input is floating point
    arr = arr.astype(np.float64)

    # if the input is 1 dimensional, we need to make it 2D but and transpose it so that the
    # arr.shape[0] is the size of the first dimension
    is_1d = len(arr.shape) == 1
    if is_1d:
        arr = arr.reshape(len(arr), 1)
        lattice = lattice.reshape(1, 1)

    # get the dimension of the filter needed for to cover multiples of the smearing width
    r_frac = get_ucell_frac_fit_sphere(lattice=lattice, r=sigma * multiple)

    # we want the center of the smearing mask to be the max value so there should an odd number
    # however based on how periodic boundary conditions are implemented, we need to make sure
    # to pad boundary with an extra layer so the masks should be even
    filter_shape = [
        int(
            np.ceil(itr_rf * itr_dim / 2) * 2
        )  # size of the mask should be even in each direction to avoid edge effects
        for itr_rf, itr_dim in zip(r_frac, arr.shape)
    ]

    filter_latt = np.array(
        [
            (filter_shape[_] + 1) / (arr.shape[_] + 1) * lattice[_]
            for _ in range(len(lattice))
        ]
    )

    # Get the fractional positions
    filter_frac_c = [np.linspace(0, 1, ng, endpoint=False) for ng in filter_shape]
    frac_pos = np.meshgrid(*filter_frac_c, indexing="ij")
    frac_pos = [i_dir.flatten() for i_dir in frac_pos]

    # convert to cartesian
    cart_pos = np.matmul(filter_latt.T, np.vstack(frac_pos))

    mid_point = np.sum(filter_latt, axis=0) / 2
    disp2mid2 = [
        (i_coord.reshape(filter_shape) - mp_coord) ** 2
        for mp_coord, i_coord in zip(mid_point, cart_pos)
    ]
    dist2mid = np.sqrt(sum(disp2mid2))
    # make sure the mask is zero outside the sphere
    mm = dist2mid <= sigma * multiple
    gauss = np.exp(-1 / 2 * (dist2mid / sigma) ** 2) * mm
    gauss = gauss / gauss.sum()
    # reduce dimension before convolution if the input is 1D
    if is_1d:
        return convolve(input=arr.flatten(), weights=gauss, mode="wrap"), gauss
    return convolve(input=arr, weights=gauss, mode="wrap"), gauss
