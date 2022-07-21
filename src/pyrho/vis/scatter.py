"""Helper functions to visualize the data in plotly."""
from __future__ import annotations

import numpy as np
import plotly.graph_objs as go
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

"""Visualizaiton functions do the scatter plots in plotly since it seems to be more efficient."""


def get_scatter_plot(
    data_in: np.ndarray,
    lat_mat: np.ndarray,
    skips: int = 1,
    logcolor: bool = False,
    mask: np.ndarray = None,
    opacity: float = 1.0,
    marker_size: int = 5,
    plotter: str = "matplotlib",
) -> go.Figure | Axes:
    """Return a plotly fig object for plotting.

    Parameters
    ----------
    data_in:
        Structured grid data to be plotted
    lat_mat:
        Lattice vectors of the cell this must be a 2d array
    skips:
        Reduction factor of the grid points for plotting, only show [::skips] in each direction
    logcolor:
        If True, assign the color in log scale
    mask:
        Filter the points to plot
    opacity:
        Opacity of each point being plotted
    marker_size:
        Size of the markers in the 3D scatter plot
    marker_size:
        Marker size for the scatter plot
    plotter:
        Plotter to use, either "matplotlib" or "plotly"


    Returns
    -------
    Figure | Axes:
        `Figure` object or `Axes` object from matplotlib to be rendered in a notebook

    """
    ndim = len(data_in.shape)
    if ndim > 3:
        raise NotImplementedError("Can only render data of 1, 2, or 3 dimensions.")

    ss = slice(0, None, skips)
    trimmed_data = np.real(data_in).copy()
    trimmed_data = trimmed_data[(ss,) * ndim]

    if mask is not None:
        flat_mask = mask[(ss,) * ndim].flatten()
    else:
        flat_mask = np.ones_like(trimmed_data, dtype=bool).flatten()

    vecs = [
        np.linspace(0, 1, trimmed_data.shape[_], endpoint=False) for _ in range(ndim)
    ]
    gridded = np.meshgrid(*vecs, indexing="ij")  # indexing to match the labeled array
    res = np.dot(lat_mat.T, [g_.flatten() for g_ in gridded])

    if logcolor:
        cc = np.log(trimmed_data.flatten())
    else:
        cc = trimmed_data.flatten()

    xx, yy, zz = None, None, None
    if ndim > 0:
        xx = res[0, flat_mask]
    if ndim > 1:
        yy = res[1, flat_mask]
    if ndim > 2:
        zz = res[2, flat_mask]

    cc = cc[flat_mask]
    if plotter == "matplotlib":
        return _scatter_matplotlib(xx, yy, zz, cc, ndim, marker_size, opacity)
    elif plotter == "plotly":
        return _scatter_plotly(xx, yy, zz, cc, ndim, marker_size, opacity)
    else:
        raise ValueError("plotter must be one of 'matplotlib' or 'plotly'")


def _scatter_plotly(xx, yy, zz, cc, ndim, marker_size, opacity):
    """Return the plotly object."""
    if ndim == 1:
        data = go.Scatter(
            x=xx,
            y=cc,
            mode="markers",
            marker=dict(
                size=marker_size,
                color="red",
            ),
        )
    if ndim == 2:
        data = go.Scatter(
            x=xx,
            y=yy,
            mode="markers",
            marker=dict(
                size=marker_size,
                color=cc,  # set color to an array/list of desired values
                colorscale="Viridis",  # choose a colorscale
                opacity=opacity,
            ),
        )
    if ndim == 3:
        data = go.Scatter3d(
            x=xx,
            y=yy,
            z=zz,
            mode="markers",
            marker=dict(
                size=marker_size,
                color=cc,
                colorscale="Viridis",
                opacity=opacity,
            ),
        )
    fig = go.Figure(data=[data])

    fig.update_layout(template="plotly_white")
    if ndim == 2:
        fig.update_layout(
            width=800, height=800, yaxis=dict(scaleanchor="x", scaleratio=1)
        )
    if ndim == 3:
        fig.update_layout(width=800, height=800, scene_aspectmode="data")
    return fig


def _scatter_matplotlib(xx, yy, zz, cc, ndim, marker_size, opacity) -> Axes:
    """Return the matplotlib object."""
    fig = plt.figure(figsize=(8, 8))
    if ndim == 1:
        ax = fig.add_subplot()
        ax.scatter(xx, cc, marker_size, color="red")
    if ndim == 2:
        ax = fig.add_subplot()
        ax.scatter(xx, yy, marker_size, c=cc)
    if ndim == 3:
        ax = fig.add_subplot(projection="3d")
        ax.scatter(xx, yy, zz, marker_size, c=cc)
