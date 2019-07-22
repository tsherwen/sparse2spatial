"""
Plotting and analysis code for sea-surface example prediction work.
"""
import numpy as np
import xarray as xr
import pandas as pd
import cartopy
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

import sparse2spatial as s2s
import sparse2spatial.utils as utils
import sparse2spatial.plotting as plotting

def main():
    """
    Driver to do plotting/analysis for example
    """
    # - Local variables
    target = 'example'
    # - Explore the predicted concentrations
    # Get the data
    ds = utils.get_predicted_3D_values(target=target)
    # plot up an annual mean
    plotting.plot_up_annual_averages_of_prediction(ds=ds, target=target)
    # Check the comparisons with observations using X vs. Y plots by region
    plotting.plt_X_vs_Y_for_obs_v_params(  )



if __name__ == "__main__":
    main()
