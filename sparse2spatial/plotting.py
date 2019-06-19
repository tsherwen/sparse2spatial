"""

Plotting functions for plotting up s2s models/output

Notes
-----
 - Code for direct plotting for RandomForestRegressor output is externally held in the TreeSurgeon package (linked below)
https://github.com/wolfiex/TreeSurgeon

"""
import numpy as np
import xarray as xr
import pandas as pd
import cartopy
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

import sparse2spatial as s2s
import sparse2spatial.utils as utils
import sparse2spatial.RFRanalysis as analysis


def plot_up_annual_averages_of_prediction(ds=None, target=None, version='v0_0_0'):
    """
    Wrapper to plot up the annual averages of the predictions

    Parameters
    -------
    ds (xr.dataset), 3D dataset contraining variable of interest on monthly basis
    target (str), Name of the target variable (e.g. iodide)
    version (str), Version number or string (present in NetCDF names etc)

    Returns
    -------
    (None)
    """
    # get annual average
    var2plot = 'Ensemble_Monthly_mean'
    ds = ds[[var2plot]].mean(dim='time')
    # Set a title
    title = "Annual average ensemble prediction for '{}' (pM)".format(target)
    # Now plot
    plot_spatial_data(ds=ds, var2plot=var2plot, extr_str=version, target=target,
        title=title)


def plot_up_seasonal_averages_of_prediction(ds=None, target=None, version='v0_0_0',
        seperate_plots=True, verbose=False ):
    """
    Wrapper to plot up the annual averages of the predictions

    Parameters
    -------
    ds (xr.dataset), 3D dataset contraining variable of interest on monthly basis
    target (str), Name of the target variable (e.g. iodide)
    version (str), Version number or string (present in NetCDF names etc)

    Returns
    -------
    (None)
    """
    # Get annual average
    var2plot = 'Ensemble_Monthly_mean'
    # Get average by season
    ds = ds.groupby('time.season').mean(dim='time')
    # Plot by season
    if seperate_plots:
        for season in list(ds.season.values):
            # check and name variables
            extr_str = '{}_{}'.format(version, season)
            if verbose:
                print( season, extr_str )
            # Select data for month
            ds2plot = ds[[var2plot]].sel(season=season)
            # Set a title
            title = "Seasonal ({}) average ensemble prediction for '{}' (pM)"
            title = title.format(season, target)
            # Now plot
            plot_spatial_data(ds=ds2plot, var2plot=var2plot, extr_str=extr_str,
                target=target, title=title)
    # Or plot up as a window plot
    else:
        print('TODO: setup to plot window plot by season')


def plot_spatial_data(ds=None, var2plot=None, LatVar='lat', LonVar='lon',
                      extr_str='', fillcontinents=True, target=None, units=None,
                      show_plot=False, save_plot=True, title=None,
                      projection=ccrs.Robinson(), fig=None, ax=None,
                      vmin=None, vmax=None, dpi=320):
    """
    Plot up 2D spatial plot of latitude vs. longitude

    Parameters
    -------
    ds (xr.dataset), 3D dataset contraining variable of interest on monthly basis
    var2plot (str), variable to plot from dataset
    target (str), Name of the target variable (e.g. iodide)
    version (str), Version number or string (present in NetCDF names etc)
    file_and_path (str), folder and filename with location settings as single str
    res (str), horizontal resolution of dataset (e.g. 4x5)

    Returns
    -------
    (None)
    """
    import cartopy.crs as ccrs
    if isinstance(fig, type(None)):
        fig = plt.figure(figsize=(10, 6))
    if isinstance(ax, type(None)):
        ax = fig.add_subplot(111, projection=projection, aspect='auto')
    ds[var2plot].plot.imshow(x='lon', y='lat', ax=ax, vmax=vmax, vmin=vmin,
        transform=ccrs.PlateCarree())
    # Fill the continents
    if fillcontinents:
        ax.add_feature(cartopy.feature.LAND, zorder=100, edgecolor='k')
    # Beautify
    ax.coastlines()
    ax.set_global()
    # Add a title
    if not isinstance(title, type(None)):
        plt.title(title)
    # Save or show plot
    if show_plot:
        plt.show()
    if save_plot:
        filename = 's2s_spatial_{}_{}.png'.format(target, extr_str)
        plt.savefig(filename, dpi=dpi)
