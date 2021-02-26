# -*- coding: utf-8 -*-
"""
Helper functions to visualize the data in plotly
"""

import plotly.graph_objs as go
import numpy as np

"""Visualizaiton functions do the scatter plots in plotly since it seems to be more efficient."""


def get_plotly_scatter_plot(
    data_in: np.ndarray,
    lat_mat: np.ndarray,
    skips: int = 5,
    logcolor: bool = False,
    mask: np.ndarray = None,
    opacity: float = 0.5,
    marker_size: int = 5,
) -> go.Figure:
    """
    Returns a plotly fig object for plotting.
    Args:
        data_in: Structured grid data to be plotted
        lat_mat: Lattice vectors of the cell
        skips: reduction factor of the grid points for plotting, only show [::skips] in each direction
        logcolor: If True, assign the color in log scale
        mask: Filter the points to plot
        opacity: opacity of each point being plotted
        marker_size: size of the markers in the 3D scatter plot

    Returns:
        plotly Figure object

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

    vecs = [np.linspace(0, 1, trimmed_data.shape[_], endpoint=False) for _ in range(ndim)]
    gridded = np.meshgrid(*vecs, indexing="ij")  # indexing to match the labeled array
    res = np.dot(lat_mat.T, [g_.flatten() for g_ in gridded])

    if logcolor:
        cc = np.log(trimmed_data.flatten())
    else:
        cc = trimmed_data.flatten()

    if ndim == 1:
        xx = res[flat_mask]
    elif ndim > 1:
        xx = res[0, flat_mask]
        yy = res[1, flat_mask]
    if ndim > 2:
        zz = res[2, flat_mask]

    cc = cc[flat_mask]

    if ndim == 1:
        data = go.Scatter(x=xx, y=cc, mode="markers", marker=dict(size=marker_size, color="red",))
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
            marker=dict(size=marker_size, color=cc, colorscale="Viridis", opacity=opacity,),
        )
    fig = go.Figure(data=[data])

    fig.update_layout(template="plotly_white")
    if ndim == 2:
        fig.update_layout(width=800, height=800, yaxis=dict(scaleanchor="x", scaleratio=1))
    if ndim == 3:
        fig.update_layout(width=800, height=800, scene_aspectmode="data")
    return fig
