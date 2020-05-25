# Jupyter notebook setup

Basic visualizing is handled through Plotly framework.

# Data limit

Since we are visualizing large grid data, it sometimes pushes the limits of the notebook's default data rate.
So we will have to increase the data rate when we call start the notebook.

```
NotebookApp.iopub_data_rate_limit=10000000000
```

# Jupyter-lab integration

Plotly dash's Jupyter lab integration is not always stable, but installing it from `conda` works most of the time.

```
conda install -c conda-forge jupyterlab-plotly-extension
```
