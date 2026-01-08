import numpy as np

from pyrho.vis.scatter import get_scatter_plot


def test_get_scatter_plot():
    data = [
        [
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
        ],
        [
            [2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2],
        ],
        [
            [3, 3, 3, 3, 3, 3, 3],
            [3, 3, 3, 3, 3, 3, 3],
            [3, 3, 3, 3, 3, 3, 3],
            [3, 3, 3, 3, 3, 3, 3],
            [3, 3, 3, 3, 3, 3, 3],
        ],
        [
            [4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4],
        ],
    ]
    data = np.array(data)
    lat_mat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    fig = get_scatter_plot(
        data_in=data, lat_mat=lat_mat, skips=1, mask=data > 0, plotter="plotly"
    )
    assert fig._data[0]["marker"]["color"].size == data.size
    assert fig._data[0]["x"].size == data.size

    fig = get_scatter_plot(
        data_in=data, lat_mat=lat_mat, skips=1, mask=data > 2, plotter="plotly"
    )
    assert fig._data[0]["marker"]["color"].size == np.sum(data > 2)
