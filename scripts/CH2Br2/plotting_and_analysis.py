"""
Plotting and analysis code for sea-surface CH2Br2 prediction work.
"""
import numpy as np
import xarray as xr
import pandas as pd
import cartopy
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

import sparse2spatial as s2s
import sparse2spatial.utils as utils


def main():
    """
    Driver to do plotting/analysis for CH2Br2
    """
    # - Local variables
    target = 'CH2Br2'
    # - Explore the predicted concentrations
    # Get the data
    ds = get_predicted_3D_values(target=target)
    # plot up an annual mean
    plot_up_annual_averages_of_prediction(ds=ds, target=target)


if __name__ == "__main__":
    main()