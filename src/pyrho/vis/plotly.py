# -*- coding: utf-8 -*-
import plotly.graph_objs as go
import numpy as np

"""Visualizaiton functions do the scatter plots in plotly since it seems to be more efficient."""

# TODO consider dropping large chunks of the data


def get_plotly_scatter_plot_3d(
    data_in: np.ndarray,
    lat_mat: np.ndarray,
    factor: int = 5,
    logcolor: bool = False,
    mask: np.ndarray = None,
    opacity: float = 0.25,
    marker_size: int = 5,
) -> go.Figure:
    """
    Returns a plotly fig object for plotting.
    Args:
        data_in: Structured grid data to be plotted
        lat_mat: Lattice vectors of the cell
        factor: reduction factor of the grid points for plotting, only show [::factor] in each direction
        logcolor: If True, assign the color in log scale
        mask: Filter the points to plot
        opacity: opacity of each point being plotted
        marker_size: size of the markers in the 3D scatter plot

    Returns:
        plotly Figure object

    """

    ss = slice(0, None, factor)
    trimmed_data = np.real(data_in).copy()
    trimmed_data = trimmed_data[ss, ss, ss]
    if mask is not None:
        flat_mask = mask[ss, ss, ss].flatten()
    else:
        flat_mask = np.ones_like(trimmed_data, dtype=bool).flatten()
    av = np.linspace(0, 1, trimmed_data.shape[0], endpoint=False)
    bv = np.linspace(0, 1, trimmed_data.shape[1], endpoint=False)
    cv = np.linspace(0, 1, trimmed_data.shape[2], endpoint=False)
    AA, BB, CC = np.meshgrid(av, bv, cv, indexing="ij")  # indexing to match the labeled array
    res = np.dot(lat_mat.T, [AA.flatten(), BB.flatten(), CC.flatten()])

    if logcolor:
        cc = np.log(trimmed_data.flatten())
    else:
        cc = trimmed_data.flatten()

    xx = res[0, flat_mask]
    yy = res[1, flat_mask]
    zz = res[2, flat_mask]
    cc = cc[flat_mask]
    # df = pandas.DataFrame({'x':xx, 'y':yy, 'z':zz, 'cc': cc})
    # fig = px.scatter_3d(df, x='x', y='y', z='z', color='cc', opacity=opacity, size_max=2)

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=xx,
                y=yy,
                z=zz,
                mode="markers",
                marker=dict(
                    size=marker_size,
                    color=cc,  # set color to an array/list of desired values
                    colorscale="Viridis",  # choose a colorscale
                    opacity=opacity,
                ),
            )
        ]
    )

    # fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    fig.update_layout(width=700, margin={"r": 10, "l": 10, "b": 10, "t": 10})
    # fix the ratio in the top left subplot to be a cube
    fig.update_layout(scene_aspectmode="data")
    return fig
