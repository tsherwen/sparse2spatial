"""

Plotting functions for plotting up s2s models/output

Notes
-----
 - Code for direct plotting for RandomForestRegressor output is externally held in the TreeSurgeon package (linked below)
https://github.com/wolfiex/TreeSurgeon / http://doi.org/10.5281/zenodo.2579239

"""
import numpy as np
import xarray as xr
import pandas as pd
import cartopy
import cartopy.crs as ccrs
import matplotlib
import matplotlib.pyplot as plt
#import sparse2spatial as s2s
import sparse2spatial.utils as utils
import sparse2spatial.RFRanalysis as RFRanalysis
import sparse2spatial.analysis as analysis
# import AC_tools (https://github.com/tsherwen/AC_tools.git)
import AC_tools as AC


def plot_up_annual_averages_of_prediction(ds=None, target=None, version='v0_0_0',
                                          var2plot='Ensemble_Monthly_mean'):
    """
    Wrapper to plot up the annual averages of the predictions

    Parameters
    -------
    ds (xr.Dataset): 3D dataset containing variable of interest on monthly basis
    target (str): Name of the target variable (e.g. iodide)
    version (str): Version number or string (present in NetCDF names etc)
    var2plot (str): variable in dataset to be plotted

    Returns
    -------
    (None)
    """
    # Get annual average of the variable in the dataset

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
    ds (xr.Dataset): 3D dataset containing variable of interest on monthly basis
    target (str): Name of the target variable (e.g. iodide)
    version (str): Version number or string (present in NetCDF names etc)
    seperate_plots (bool): plot up output as separate plots
    verbose (bool): print out verbose output?

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


def plot_up_df_data_by_yr(df=None, Datetime_var='datetime', TimeWindow=5,
                          start_from_last_obs=False, drop_bins_without_data=True,
                          target='Iodide', dpi=320):
    """
    Plot up # of obs. data (Y) binned by region against year (X)

    Parameters
    -------
    df (pd.DataFrame): DataFrame of data with and a datetime variable
    target (str): Name of the target variable (e.g. iodide)
    TimeWindow (int): number years to bit observations over
    start_from_last_obs (bool): start from the last observational date
    drop_bins_without_data (bool): exclude bins with no data from plotting
    dpi (int): resolution of figure (dots per sq inch)

    Returns
    -------
    (None)
    """
    # Sort the dataframe by date
    df.sort_values( by=Datetime_var, inplace=True )
    # Get the minimum and maximum dates
    min_date = df[Datetime_var].min()
    max_date = df[Datetime_var].max()
    # How many years of data are there?
    yrs_of_data = (max_date-min_date).total_seconds()/60/60/24/365
    nbins = AC.myround(yrs_of_data/TimeWindow, base=1 )
    # Start from last observation or from last block of time
    sdate_block = AC.myround(max_date.year, 5)
    sdate_block =  datetime.datetime(sdate_block, 1, 1)
    # Make sure the dates used are datetimes
    min_date, max_date = pd.to_datetime( [min_date, max_date] ).values
    min_date, max_date = AC.dt64_2_dt( [min_date, max_date])
    # Calculate the number of points for each bin by region
    dfs = {}
    for nbin in range(nbins+2):
        # Start from last observation or from last block of time?
        days2rm = int(nbin*365*TimeWindow)
        if start_from_last_obs:
            bin_start = AC.add_days( max_date, -int(days2rm+(365*TimeWindow)))
            bin_end = AC.add_days( max_date, -days2rm )
        else:
            bin_start = AC.add_days( sdate_block,-int(days2rm+(365*TimeWindow)))
            bin_end = AC.add_days( sdate_block, -days2rm )
        # Select the data within the observational dates
        bool1 = df[Datetime_var] > bin_start
        bool2 = df[Datetime_var] <= bin_end
        df_tmp = df.loc[bool1 & bool2, :]
        # Print the number of values in regions for bin
        if verbose:
            print(bin_start, bin_end, df_tmp.shape)
        # String to save data with
        if start_from_last_obs:
            bin_start_str = bin_start.strftime( '%Y/%m/%d')
            bin_end_str = bin_end.strftime( '%Y/%m/%d')
        else:
            bin_start_str = bin_start.strftime( '%Y')
            bin_end_str = bin_end.strftime( '%Y')
        str2use = '{}-{}'.format(bin_start_str, bin_end_str)
        # Sum up the number of values by region
        dfs[ str2use] = df_tmp['ocean'].value_counts(dropna=False)
    # Combine to single dataframe and sort by date
    dfA = pd.DataFrame( dfs )
    dfA = dfA[list(sorted(dfA.columns)) ]
    # Drop the years without any data
    if drop_bins_without_data:
        dfA = dfA.T.dropna(how='all').T
    # Update index names
    dfA = dfA.T
    dfA.columns
    rename_cols = {
    np.NaN : 'Other',  'INDIAN OCEAN': 'Indian Ocean', 'SOUTHERN OCEAN' : 'Southern Ocean'
    }
    dfA = dfA.rename(columns=rename_cols)
    dfA = dfA.T
    # Plot up as a stacked bar plot
    import seaborn as sns
    sns.set()
    dfA.T.plot(kind='bar', stacked=True)
    # Add title etc
    plt.ylabel( '# of observations')
    plt.title( '{} obs. data by region'.format(target))
    # Save plotted figure
    savename = 's2s_{}_data_by_year_region'.format(target)
    plt.savefig(savename, dpi=dpi, bbox_inches='tight', pad_inches=0.05)


def plt_X_vs_Y_for_regions(df=None, params2plot=[], LatVar='lat', LonVar='lon',
                           obs_var='Obs.'):
    """
    Plot up the X vs. Y performance by region
    """
    # Add ocean columns to dataframe
    df = add_loc_ocean2df(df=df, LatVar=LatVar, LonVar=LonVar)
    # Split by regions
    regions = set(df['ocean'].dropna())
    dfs = [df.loc[df['ocean']==i,:] for i in regions]
    dfs = dict(zip(regions,dfs))
    # Also get an open ocean dataset
    # TODO ...
    # Use an all data for now
    dfs['all'] = df.copy()
    # Loop and plot by region
    for region in regions:
        print(region)
        df = dfs[region]
        # Now plot
        plt_X_vs_Y_for_obs_v_params(df=df, params2plot=params2plot, obs_var=obs_var,
                                    extr_str=region)


def plt_X_vs_Y_for_obs_v_params(df=None, params2plot=[], obs_var='Obs.',
                                extr_str='', context='paper', dpi=320):
    """
    Plot up comparisons for parameterisations against observations
    """
    import seaborn as sns
    sns.set(color_codes=True)
    sns.set_context(context)
    # Get colours to use
    CB_color_cycle = AC.get_CB_color_cycle()
    color_dict = dict(zip([obs_var]+params2plot, ['k']+CB_color_cycle))
    # Setup the figure and axis for the plot
    fig = plt.figure(dpi=dpi, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(111)
    # Loop by parameter
    for n_param, param in enumerate( params2plot ):
        # Plot a single 1:1 line
        plot_121 = False
        if n_param == 0:
            plot_121 =True
        # Now plot a generic X vs. Y plot
        AC.plt_df_X_vs_Y(df=df, fig=fig, ax=ax, y_var=param, x_var=obs_var,
                         x_label=obs_var, y_label=param, color=color_dict[param],
                         save_plot=False, plot_121=plot_121 )
    # Add a title
    title_str = "Obs. vs. predictions in '{}'".format(extr_str)
    plt.title(title_str)
    # Add a legend
    plt.legend()
    # Save the plot
    png_filename = 's2s_X_vs_Y_{}_vs_{}_{}'.format(obs_var, 'params', extr_str)
    png_filename = AC.rm_spaces_and_chars_from_str(png_filename)
    plt.savefig(png_filename, dpi=dpi)


def plot_spatial_data(ds=None, var2plot=None, LatVar='lat', LonVar='lon',
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
    ds (xr.Dataset): 3D dataset containing variable of interest on monthly basis
    var2plot (str): variable to plot from dataset
    target (str): Name of the target variable (e.g. iodide)
    version (str): Version number or string (present in NetCDF names etc)
    file_and_path (str): folder and filename with location settings as single str
    res (str): horizontal resolution of dataset (e.g. 4x5)
    xticks, yticks (bool): include ticks on y and/or x axis?
    title (str): title to add use for plot
    LatVar, LonVar (str): variables to use for latitude and longitude
    add_meridians_parallels (bool): add the meridians and parallels?
    save_plot (bool): save the plot as png
    show_plot (bool): show the plot on screen
    dpi (int): resolution to use for saved image (dots per square inch)
    projection (cartopy ccrs object), projection to use for spatial plots
    fig (figure instance), figure instance to plot onto
    ax (axis instance), axis to use for plotting

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
