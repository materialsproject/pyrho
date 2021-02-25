from __future__ import annotations
from itertools import combinations
from typing import Iterable, List, Tuple, Union

import numpy as np
from scipy.interpolate import RegularGridInterpolator
import numpy.typing as npt


def pad_arr(arr_in: np.ndarray, shape: List[int]) -> np.ndarray:
    """
    Padding a function on a hypercube.

    Args:
        arr_in: data to be padded with zeros
        shape: desired shape of the array

    Returns:
        ndarray: padded data

    Notes:
    We basically need to move the data at the corners a hypercube to the corners of a bigger/smaller hypercube
    Each corner of the hypercube can be represented as a binary number with length equal to the dimension of the cube.
    Loop over the binary representation to figure out which corner you are at the slice accordingly.

    Examples:
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
        (int(np.ceil(min(i_dim, j_dim) + 1) / 2.0), int(np.floor(min(i_dim, j_dim)) / 2.0),)
        for i_dim, j_dim in zip(dimensions, shape)
    ]
    dim = len(dimensions)
    fmt = f"#0{dim+2}b"
    corners = [format(itr, fmt)[-dim:] for itr in range(2 ** dim)]
    arr_out = np.zeros(shape, dtype=arr_in.dtype)

    for ic in corners:
        islice = tuple(get_slice(idig, idim, boundaries) for idim, idig in enumerate(ic))
        arr_out[islice] = arr_in[islice]
    return arr_out


def interpolate_fourier(arr_in: np.ndarray, shape: List[int]) -> np.ndarray:
    """
    Interpolate the data to some final shape, keep magnitude the same.
    Will perform best if the input array is periodic in all directions.

    Args:
        arr_in: Input array of data
        shape: Desired shape shape of the interpolated data

    Returns:
        interpolated data in the desired shape

    Examples:
>>> arr = np.array([[5.0, 10.0, 10.0, 7.0],
...        [22.0, 12.0, 7.0, 3.0],
...        [16.0, 10.0, 3.0, 5.0],
...        [16.0, 22.0, 16.0, 6.0],
...        [19.0, 20.0, 3.0, 7.0]])
>>> np.round(np.abs(interpolate_fourier(arr_in=arr, shape=[10,8])),2)
array([[ 5.  ,  7.31, 10.  , 10.84, 10.  ,  8.72,  7.  ,  5.2 ],
       [12.02, 12.42, 11.22, 12.91, 12.72,  8.4 ,  4.71,  7.93],
       [22.  , 19.8 , 12.  ,  9.54,  7.  ,  4.31,  3.  , 13.58],
       [23.2 , 19.98, 10.33,  4.63,  0.64,  2.75,  3.36, 15.11],
       [16.  , 14.9 , 10.  ,  5.76,  3.  ,  2.36,  5.  , 11.37],
       [11.58, 14.08, 14.8 , 14.26, 11.82,  8.08,  6.05,  7.9 ],
       [16.  , 20.68, 22.  , 20.68, 16.  ,  9.4 ,  6.  ,  9.4 ],
       [21.86, 26.36, 24.64, 18.31, 10.48,  5.18,  6.05, 13.21],
       [19.  , 22.54, 20.  , 11.26,  3.  ,  2.36,  7.  , 13.37],
       [ 9.34, 12.5 , 13.01,  8.34,  3.33,  4.87,  7.84,  8.91]])
    """
    fft_res = np.fft.fftn(arr_in)
    fft_res = pad_arr(fft_res, shape)
    results = np.fft.ifftn(fft_res) * np.size(fft_res) / np.size(arr_in)
    return results


def roll_array(arr: np.ndarray, roll_vec: List[int]) -> np.ndarray:
    """
    Shift the index of an ndarray based on roll_vec.
    Args:
        arr: array to be rolled
        roll_vec: number of indices in each direction to roll

    Returns:
        The rolled array
    """
    for ii, roll_val in enumerate(roll_vec):
        arr = np.roll(arr, roll_val, ii)
    return arr


def get_sc_interp(
    data_in: np.ndarray,
    sc_mat: npt.ArrayLike,
    grid_sizes: List[int],
    scipy_interp_method="linear",
    origin: Union[np.ndarray, List[float], Tuple[float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Take a data array defined on a regular lattice and a new set of lattice
    vector (in units of the the original lattice), reproduce the data on the new grid.
    We can consider the original cell to be unit cell (UC) and the new cell to be a super cell (SC)
    defined using the basis of the UC -- although the the SC in this case can be any lattice.

    Args:
        data_in: data stored on a regular grid
        sc_mat: lattice vectors of new cell in the units of the old cell
        grid_sizes: number of grid points in each direction in the new cell
        scipy_interp_method: interpolation method to be used
    Returns:
        size (ndim x prod(grid_size)) the cartesian coordinates of each point in the new data
        size (prod(grid_size)) the regridded data

    Note:
    Assuming we have some kind of data stored on a regular grid --- [0,1) in each dimension
    In the frame where UC is orthonormal we can just map all points in the sc_grid to the a position in the unit cube
    Then perform interpolation on the new coordinates that are all within the cube

    Suggestion:
        This algorithm can be used in real space and fourier space
        For real space we expect the data to be smooth, so we should use "real"
        For fourier space we expect the position of the exact fourier component is important so use "nearest"

    We have to include the boundaries of the UC since the mapped grid must lie between the existing points
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
    frac_coords = np.meshgrid(*grid_vec, indexing="ij")  # indexing to match the labeled array
    frac_coords = np.vstack([icoord.flatten() for icoord in frac_coords])

    sc_coord = np.dot(np.array(sc_mat).T, frac_coords)  # shape (dim, NGRID)
    if origin is not None:
        sc_coord += np.array([[_] for _ in origin])

    mapped_coords = sc_coord - np.floor(sc_coord)

    return sc_coord, interp_func(mapped_coords.T)


def get_padded_array(data_in: np.ndarray) -> np.ndarray:
    """
    Pad an array once in each direction with the periodic boundary conditions.
    Args:
        data_in: Array to be padded

    Returns:
        Padded array

    """
    padded_data = data_in.copy()
    slice_arr = [
        [slice(0, 1) if ii == i else slice(0, None) for ii in range(len(data_in.shape))]
        for i in range(len(data_in.shape))
    ]
    for idim, islice in enumerate(slice_arr):
        padded_data = np.concatenate((padded_data, padded_data[tuple(islice)]), axis=idim)
    return padded_data


def get_plane_spacing(lattice: np.ndarray) -> Iterable[float]:
    """
    Get the cartesian spacing between bonding planes of a unit cell
    Args:
        lattice: list of lattice vectors in cartesian coordinates

    Returns:
        List where the k-th element is is the spacing of planes generated by all
        lattice vectors EXCEPT the k-th one

    Examples:
>>> get_plane_spacing([[1,0,0], [1,1,0], [0,0,2]])
[0.7653668647301795, 1.0420107665599743, 2.0]

    """
    # get all pairwise projections i must be smaller than j
    ndim = len(lattice)
    idx_pairs = [*combinations(range(ndim), 2)]
    latt_len = [np.linalg.norm(lattice[i]) for i in range(ndim)]
    pproj = {(i, j): np.dot(lattice[i], lattice[j]) / latt_len[i] / latt_len[j] for i, j in idx_pairs}
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
    """
    The smallest you can make make the lattice parameter in each direction to fit a
    sphere of radius r.  The sphere must be able to fit between the hyperplanes of
    the subspace of all lattice vectors EXCEPT k for all k.
    Args:
        lattice: list of lattice vectors in cartesian coordinates
        r: width of Gaussian
    Returns:
        fraction of lattice vector in each direction need to fit the sphere

    Examples:
>>> get_ucell_frac_fit_sphere([[1,0,0], [1,-1, 0], [0,0,2]], 0.1)
[0.26131259297527537, 0.19193659645213346, 0.1]
    """
    rfrac = []
    for ispace in get_plane_spacing(lattice=lattice):
        rfrac.append(2 * r / ispace)
    return rfrac


if __name__ == "__main__":
    import doctest

    doctest.testmod()
