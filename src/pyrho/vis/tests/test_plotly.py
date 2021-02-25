from unittest import TestCase
import numpy as np

from pyrho.vis.plotly import get_plotly_scatter_plot_3d


class Test(TestCase):
    def test_get_plotly_scatter_plot_3d(self):

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
        fig = get_plotly_scatter_plot_3d(data_in=data, lat_mat=lat_mat, factor=1, mask=data > 0)
        assert fig._data[0]["marker"]["color"].size == data.size
        assert fig._data[0]["x"].size == data.size

        fig = get_plotly_scatter_plot_3d(data_in=data, lat_mat=lat_mat, factor=1, mask=data > 2)
        assert fig._data[0]["marker"]["color"].size == np.sum(data > 2)
