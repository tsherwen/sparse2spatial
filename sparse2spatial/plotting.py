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
import matplotlib
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
    # Get annual average of the variable in the dataset
    var2plot = 'Ensemble_Monthly_mean'
    ds = ds[[var2plot]].mean(dim='time')
    # Set a title for the plot
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
    # Which variable to plot?
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


def plot_spatial_data_TEST(ds=None, var2plot=None, LatVar='lat', LonVar='lon',
                      extr_str='', fillcontinents=True, target=None, units=None,
                      show_plot=False, save_plot=True, title=None,
                      projection=ccrs.Robinson(), fig=None, ax=None, cmap=None,
                      vmin=None, vmax=None, add_meridians_parallels=False,
                      add_borders_coast=True, set_aspect=True, cbar_kwargs=None,
                      xticks=True, yticks=True, dpi=320):
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
    add_meridians_parallels (bool), add the meridians and parallels?

    Returns
    -------
    (None)
    """
    import cartopy.crs as ccrs
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    if isinstance(fig, type(None)):
        fig = plt.figure(figsize=(10, 6))
    if isinstance(ax, type(None)):
        ax = fig.add_subplot(111, projection=projection, aspect='auto')
    plt_object = ds[var2plot].plot.imshow(x='lon', y='lat', ax=ax, vmax=vmax, vmin=vmin,
                             transform=ccrs.PlateCarree(), cmap=cmap,
                             cbar_kwargs=cbar_kwargs)
    # Fill the continents
    if fillcontinents:
        ax.add_feature(cartopy.feature.LAND, zorder=50, facecolor='lightgrey',
                       edgecolor='k')
    # Add the borders and country outlines
    if add_borders_coast:
        ax.add_feature(cartopy.feature.BORDERS, zorder=51, edgecolor='k',
                       linewidth=0.25)
        ax.add_feature(cartopy.feature.COASTLINE, zorder=52, edgecolor='k',
                       linewidth=0.05)
    # Beautify
    ax.coastlines()
    ax.set_global()
    # Add a title
    if not isinstance(title, type(None)):
        plt.title(title)
    # Add meridians and parallels?
    if add_meridians_parallels:
        # setup grdlines object
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                         linewidth=0, color='gray', alpha=0.0, linestyle=None)
        # Setup meridians and parallels
        interval = 1
        parallels = np.arange(-90, 91, 30*interval)
        meridians = np.arange(-180, 181, 60*interval)
        # Now add labels
        gl.xlabels_top = False
        gl.ylabels_right = False
        gl.xlines = False
        gl.ylines = False
        if xticks:
            gl.xticks_bottom = True
            gl.xlocator = matplotlib.ticker.FixedLocator(meridians)
            gl.xformatter = LONGITUDE_FORMATTER
            gl.xlabel_style = {'size': 7.5, 'color': 'gray'}
        else:
            gl.xticks_bottom = False
            gl.xlabels_bottom = False
        if yticks:
            gl.yticks_left = True
            gl.ylocator = matplotlib.ticker.FixedLocator(parallels)
            gl.yformatter = LATITUDE_FORMATTER
            gl.ylabel_style = {'size': 7.5, 'color': 'gray'}
        else:
            gl.yticks_left = False
            gl.ylabel_left = False
    # Save or show plot
    if show_plot:
        plt.show()
    if save_plot:
        filename = 's2s_spatial_{}_{}.png'.format(target, extr_str)
        plt.savefig(filename, dpi=dpi, bbox_inches='tight', pad_inches=0.05)
