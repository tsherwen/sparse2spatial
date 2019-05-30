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

import sparse2spatial as s2s
import sparse2spatial.utils as utils


def plot_up_annual_averages_of_prediction(ds=None, target=None, version='v0_0_0'):
    """
    Plot up the annual averages of the predictions

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


def get_predicted_3D_values(target=None, filename=None, version='v0_0_0',
                            res='0.125x0.125', file_and_path='./sparse2spatial.rc'):
    """
    Get the predicted target values from saved NetCDF

    Parameters
    ------- 
    ds (xr.dataset), 3D dataset contraining variable of interest on monthly basis
    target (str), Name of the target variable (e.g. iodide)
    version (str), Version number or string (present in NetCDF names etc)
    file_and_path (str), folder and filename with location settings as single str
    res (str), horizontal resolution of dataset (e.g. 4x5)

    Returns
    -------                                                                                                                                                     
    (xr.Dataset) 
    """
    # Location of data
    folder = utils.get_file_locations('s2s_root', file_and_path=file_and_path)
    folder += '/{}/outputs/'.format(target)
    # Get file namec
    filename = 'Oi_prj_predicted_{}_{}_{}.nc'.format(target, res, version)
    ds = xr.open_dataset(folder+filename)
    return ds



def plot_spatial_data(ds=None, var2plot=None, LatVar='lat', LonVar='lon',
                      extr_str='', fillcontinents=True, target=None, units=None,
                      show_plot=False, save_plot=True, title=None,
                      dpi=320):
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
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection=ccrs.Robinson(), aspect='auto')
    ds[var2plot].plot.imshow(x='lon', y='lat', ax=ax, transform=ccrs.PlateCarree())
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
