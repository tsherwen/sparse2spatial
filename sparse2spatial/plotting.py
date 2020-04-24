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
import cartopy.crs as ccrs
#from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


def plot_up_annual_averages_of_prediction(ds=None, target=None,
                                          version='v0_0_0',
                                          LatVar='lat', LonVar='lon',
                                          vmin=None, vmax=None, title=None,
                                          var2plot='Ensemble_Monthly_mean',
                                          units=None):
    """
    Wrapper to plot up the annual averages of the predictions

    Parameters
    -------
    ds (xr.Dataset): 3D dataset containing variable of interest on monthly basis
    target (str): Name of the target variable (e.g. iodide)
    version (str): Version number or string (present in NetCDF names etc)
    var2plot (str): variable in dataset to be plotted
    LatVar, LonVar (str): variables to use for latitude and longitude
    vmin, vmax (float): minimum and maximum values to limit colorbar to

    Returns
    -------
    (None)
    """
    # Get annual average of the variable in the dataset
    ds = ds[[var2plot]].mean(dim='time')
    # Set a title for the plot
    if isinstance(title, type(None)):
        title = "Annual average ensemble prediction for '{}' ({})".format(
            target, units)
    # Now plot
    plot_spatial_data(ds=ds, var2plot=var2plot, extr_str=version,
                      target=target,
                      LatVar=LatVar, LonVar=LonVar, vmin=vmin, vmax=vmax,
                      title=title)


def plot_up_seasonal_averages_of_prediction(ds=None, target=None,
                                            version='v0_0_0',
                                            seperate_plots=False, units='pM',
                                            var2plot='Ensemble_Monthly_mean',
                                            vmin=None, vmax=None, dpi=320,
                                            show_plot=False, save_plot=True,
                                            title=None,
                                            var2plot_longname='ensemble prediction',
                                            extension='png', verbose=False):
    """
    Wrapper to plot up the annual averages of the predictions

    Parameters
    -------
    ds (xr.Dataset): 3D dataset containing variable of interest on monthly basis
    var2plot (str): which variable should be plotted?
    target (str): Name of the target variable (e.g. iodide)
    version (str): Version number or string (present in NetCDF names etc)
    LatVar, LonVar (str): variables to use for latitude and longitude
    vmin, vmax (float): minimum and maximum values to limit colorbar to
    title (str): title to use for single seasonal plot (default=None)
    seperate_plots (bool): plot up output as separate plots
    verbose (bool): print out verbose output?

    Returns
    -------
    (None)
    """
    # Get average by season
    ds = ds.groupby('time.season').mean(dim='time')
    # Calculate minimums and maximums over all months to use for all plots
    if isinstance(vmin, type(None)) and isinstance(vmin, type(None)):
        vmin = float(ds[var2plot].min().values)
        vmax = float(ds[var2plot].max().values)
    # Dictionary to convert season acronyms to readable text
    season2text = {
        'DJF': 'Dec-Jan-Feb', 'MAM': 'Mar-Apr-May', 'JJA': 'Jun-Jul-Aug', 'SON': 'Sep-Oct-Nov'
    }
    # Set season ordering to be maintained for all plots
    seasons = ['DJF', 'MAM', 'JJA', 'SON']
    # Plot by season
    if seperate_plots:
        for season in seasons:
            # check and name variables
            extr_str = '{}_{}'.format(version, season2text[season])
            if verbose:
                print(season, extr_str)
            # Select data for month
            ds2plot = ds[[var2plot]].sel(season=season)
            # Set a title
            title = "Seasonal ({}) average {} for '{}' ({})"
            title = title.format(season, target, var2plot_longname, units)
            # Now plot
            plot_spatial_data(ds=ds2plot, var2plot=var2plot, extr_str=extr_str,
                              target=target, title=title, vmin=vmin, vmax=vmax)
    # Or plot up as a window plot
    else:
        fig = plt.figure(figsize=(9, 5), dpi=dpi)
        projection = ccrs.Robinson()
        # Loop by season
        for n_season, season in enumerate(seasons):
            # Select data for month
            ds2plot = ds[[var2plot]].sel(season=season)
            # Setup the axis
            axn = (2, 2, n_season+1)
            ax = fig.add_subplot(*axn, projection=projection, aspect='auto')
            # Now plot
            plot_spatial_data(ds=ds2plot, var2plot=var2plot,
                              ax=ax, fig=fig,
                              target=target, title=season2text[season],
                              vmin=vmin, vmax=vmax,
                              rm_colourbar=True,
                              save_plot=False)
            # Capture the image from the axes
            im = ax.images[0]

        # Add a colorbar using the captured image
        pad = 0.075
        cax = fig.add_axes([0.85, pad*2, 0.035, 1-(pad*4)])
        fig.colorbar(im, cax=cax, orientation='vertical', label=units)
        # Set a title
        if isinstance(title, type(None)):
            title = "Seasonally averaged '{}' ({})"
            title = title.format(var2plot_longname, units)
            fig.suptitle(title)
        # Adjust plot aesthetics
        bottom = pad/4
        top = 1-(pad)
        left = pad/4
        right = 1-(pad*2.5)
        hspace = 0.005
        wspace = pad/3
        fig.subplots_adjust(bottom=bottom, top=top, left=left, right=right,
                            hspace=hspace, wspace=wspace)
        # Save or show plot
        if show_plot:
            plt.show()
        if save_plot:
            filename = 's2s_spatial_by_season_{}_{}'.format(target, version)
            filename = AC.rm_spaces_and_chars_from_str(filename)
            plt.savefig('{}.{}'.format(filename, extension), dpi=dpi)


def plot_up_df_data_by_yr(df=None, Datetime_var='datetime', TimeWindow=5,
                          start_from_last_obs=False,
                          drop_bins_without_data=True,
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
    df.sort_values(by=Datetime_var, inplace=True)
    # Get the minimum and maximum dates
    min_date = df[Datetime_var].min()
    max_date = df[Datetime_var].max()
    # How many years of data are there?
    yrs_of_data = (max_date-min_date).total_seconds()/60/60/24/365
    nbins = AC.myround(yrs_of_data/TimeWindow, base=1)
    # Start from last observation or from last block of time
    sdate_block = AC.myround(max_date.year, 5)
    sdate_block = datetime.datetime(sdate_block, 1, 1)
    # Make sure the dates used are datetimes
    min_date, max_date = pd.to_datetime([min_date, max_date]).values
    min_date, max_date = AC.dt64_2_dt([min_date, max_date])
    # Calculate the number of points for each bin by region
    dfs = {}
    for nbin in range(nbins+2):
        # Start from last observation or from last block of time?
        days2rm = int(nbin*365*TimeWindow)
        if start_from_last_obs:
            bin_start = AC.add_days(max_date, -int(days2rm+(365*TimeWindow)))
            bin_end = AC.add_days(max_date, -days2rm)
        else:
            bin_start = AC.add_days(
                sdate_block, -int(days2rm+(365*TimeWindow)))
            bin_end = AC.add_days(sdate_block, -days2rm)
        # Select the data within the observational dates
        bool1 = df[Datetime_var] > bin_start
        bool2 = df[Datetime_var] <= bin_end
        df_tmp = df.loc[bool1 & bool2, :]
        # Print the number of values in regions for bin
        if verbose:
            print(bin_start, bin_end, df_tmp.shape)
        # String to save data with
        if start_from_last_obs:
            bin_start_str = bin_start.strftime('%Y/%m/%d')
            bin_end_str = bin_end.strftime('%Y/%m/%d')
        else:
            bin_start_str = bin_start.strftime('%Y')
            bin_end_str = bin_end.strftime('%Y')
        str2use = '{}-{}'.format(bin_start_str, bin_end_str)
        # Sum up the number of values by region
        dfs[str2use] = df_tmp['ocean'].value_counts(dropna=False)
    # Combine to single dataframe and sort by date
    dfA = pd.DataFrame(dfs)
    dfA = dfA[list(sorted(dfA.columns))]
    # Drop the years without any data
    if drop_bins_without_data:
        dfA = dfA.T.dropna(how='all').T
    # Update index names
    dfA = dfA.T
    dfA.columns
    rename_cols = {
        np.NaN: 'Other',  'INDIAN OCEAN': 'Indian Ocean', 'SOUTHERN OCEAN': 'Southern Ocean'
    }
    dfA = dfA.rename(columns=rename_cols)
    dfA = dfA.T
    # Plot up as a stacked bar plot
    import seaborn as sns
    sns.set()
    dfA.T.plot(kind='bar', stacked=True)
    # Add title etc
    plt.ylabel('# of observations')
    plt.title('{} obs. data by region'.format(target))
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
    dfs = [df.loc[df['ocean'] == i, :] for i in regions]
    dfs = dict(zip(regions, dfs))
    # Also get an open ocean dataset
    # TODO ...
    # Use an all data for now
    dfs['all'] = df.copy()
    # Loop and plot by region
    for region in regions:
        print(region)
        df = dfs[region]
        # Now plot
        plt_X_vs_Y_for_obs_v_params(df=df, params2plot=params2plot,
                                    obs_var=obs_var,
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
    for n_param, param in enumerate(params2plot):
        # Plot a single 1:1 line
        plot_121 = False
        if n_param == 0:
            plot_121 = True
        # Now plot a generic X vs. Y plot
        AC.plt_df_X_vs_Y(df=df, fig=fig, ax=ax, y_var=param, x_var=obs_var,
                         x_label=obs_var, y_label=param,
                         color=color_dict[param],
                         save_plot=False, plot_121=plot_121)
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
                      extr_str='', fillcontinents=True, target=None,
                      units=None,
                      show_plot=False, save_plot=True, title=None,
                      projection=ccrs.Robinson(), fig=None, ax=None, cmap=None,
                      vmin=None, vmax=None, add_meridians_parallels=False,
                      add_borders_coast=True, set_aspect=True,
                      cbar_kwargs=None,
                      xticks=True, yticks=True, rm_colourbar=False,
                      extension='png',
                      dpi=320):
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
    vmin, vmax (float): minimum and maximum values to limit colorbar to
    add_meridians_parallels (bool): add the meridians and parallels?
    save_plot (bool): save the plot as png
    show_plot (bool): show the plot on screen
    dpi (int): resolution to use for saved image (dots per square inch)
    projection (cartopy ccrs object): projection to use for spatial plots
    rm_colourbar (bool): do not include a colourbar with the plot
    fig (figure instance): figure instance to plot onto
    extension (str): extension to save file with (e.g. .tiff, .eps, .png)
    ax (axis instance): axis to use for plotting

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
    plt_object = ds[var2plot].plot.imshow(x='lon', y='lat', ax=ax, vmax=vmax,
                                          vmin=vmin,
                                          transform=ccrs.PlateCarree(),
                                          cmap=cmap,
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
    # Remove the colour bar
    if rm_colourbar:
        im = ax.images
        cb = im[-1].colorbar
        cb.remove()
    # Save or show plot
    if show_plot:
        plt.show()
    if save_plot:
        filename = 's2s_spatial_{}_{}'.format(target, extr_str)
        filename = '{}.{}'.format(
            AC.rm_spaces_and_chars_from_str(filename), extension)
        plt.savefig(filename, dpi=dpi, bbox_inches='tight', pad_inches=0.05)
    return plt_object


def plot_ODR_window_plot(params=[], show_plot=False, df=None,
                         testset='Test set (strat. 20%)', units='pM',
                         target='Iodide', context="paper", xlim=None,
                         ylim=None,
                         dpi=720, verbose=False):
    """
    Show the correlations between obs. and params. as window plot

    Parameters
    -------
    target (str): Name of the target variable (e.g. iodide)
    testset (str): Testset to use, e.g. stratified sampling over quartiles for 20%:80%
    dpi (int): resolution to use for saved image (dots per square inch)
    RFR_dict (dict): dictionary of core variables and data
    context (str): seaborn context to use for plotting (e.g. paper, poster, talk...)
    show_plot (bool): show the plot on screen
    df (pd.DataFrame): dataframe containing target and feature variables
    units (str): units of the target in the dataframe
    xlim (tuple): limits for plotting x axis
    ylim (tuple): limits for plotting y axis

    Returns
    -------
    (None)
    """
    # Make sure a dataFrame has been provided
    ass_str = "Please provide 'df' of data as a DataFrame type"
    assert type(df) == pd.DataFrame,
    # Setup seabonr plotting environment
    import seaborn as sns
    sns.set(color_codes=True)
    if context == "paper":
        sns.set_context("paper")
    else:
        sns.set_context("talk", font_scale=1.0)
    # Name of PDF to save plots to
    savetitle = 'Oi_prj_point_for_point_comparison_obs_vs_model_ODR_WINDOW_{}'
    savetitle = savetitle.format(target)
    pdff = AC.plot2pdfmulti(title=savetitle, open=True, dpi=dpi)
    # label to use for taget on plots
    target_label = '[{}'.format(target) + '$_{aq}$]'
    # Set location for alt_text
    f_size = 10
    N = int(df.shape[0])
    # Split data into groups
    dfs = {}
    # Entire dataset
    dfs['Entire'] = df.copy()
    # Testdataset
    dfs['Withheld'] = df.loc[df[testset] == True, :].copy()
    dsplits = dfs.keys()
    # Assign colors to splits
    CB_color_cycle = AC.get_CB_color_cycle()
    color_d = dict(zip(dsplits, CB_color_cycle))
    # Intialise figure and axis
    fig, axs = plt.subplots(
        1, 3, sharex=True, sharey=True, dpi=dpi, figsize=(11, 4))
    # Loop by param and compare against whole dataset
    for n_param, param in enumerate(params):
        # set axis to use
        ax = axs[n_param]
        # Use the same asecpt for X and Y
        ax.set_aspect('equal')
        # Add a title the plots
        ax.text(0.5, 1.05, param, horizontalalignment='center',
                verticalalignment='center', transform=ax.transAxes)
        # Add a 1:1 line
        x_121 = np.arange(ylim[0]-(ylim[1]*0.05), ylim[1]*1.05)
        ax.plot(x_121, x_121, alpha=0.5, color='k', ls='--')
        # Plot up data by dataset split
        for nsplit, split in enumerate(dsplits):
            # select the subset of the data
            df = dfs[split].copy()
            # Remove any NaNs
            df = df.dropna()
            # get X
            X = df[target].values
            # get Y
            Y = df[param].values
            # get N
            N = float(df.shape[0])
            # get RMSE
            RMSE = np.sqrt(((Y-X)**2).mean())
            # Plot up just the entire and testset data
            if split in ('Entire', 'Withheld'):
                ax.scatter(X, Y, color=color_d[split], s=3, facecolor='none')
            # add ODR line
            xvalues, Y_ODR = AC.get_linear_ODR(x=X, y=Y, xvalues=x_121,
                                               return_model=False, maxit=10000)

            myoutput = AC.get_linear_ODR(x=X, y=Y, xvalues=x_121,
                                         return_model=True, maxit=10000)
            # print out the parameters from the ODR
            if verbose:
                print(param, split, myoutput.beta)

            ax.plot(xvalues, Y_ODR, color=color_d[split])
            # Add RMSE ( and N value as alt text )
            alt_text_x = 0.01
            alt_text_y = 0.95-(0.05*nsplit)
#            alt_text = 'RMSE={:.1f} ({}, N={:.0f})'.format( RMSE, split, N )
            alt_text = 'RMSE={:.1f} ({})'.format(RMSE, split)
            ax.annotate(alt_text, xy=(alt_text_x, alt_text_y),
                        textcoords='axes fraction', fontsize=f_size,
                        color=color_d[split])
        # Beautify the plot/figure
        plt.xlim(xlim)
        plt.ylim(ylim)
        ax.set_xlabel('Obs. {} ({})'.format(target_label, units))
        if (n_param == 0):
            ax.set_ylabel('Parameterised {} ({})'.format(target_label, units))
    # Adjust the subplots
    if context == "paper":
        top = 0.94
        bottom = 0.1
        left = 0.05
        right = 0.975
        wspace = 0.075
    else:
        top = 0.94
        bottom = 0.14
        left = 0.075
        right = 0.975
        wspace = 0.075
    fig.subplots_adjust(top=top, right=right, left=left, bottom=bottom,
                        wspace=wspace)
    # Save the plot
    AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
    # Save entire pdf
    AC.plot2pdfmulti(pdff, savetitle, close=True, dpi=dpi)
    plt.savefig(savetitle, dpi=dpi)
    if show_plot:
        plt.show()
    plt.close()


def plt_X_vs_Y_for_regions(df=None, params2plot=[], LatVar='lat',
                           LonVar='lon', target='CH3I',
                           obs_var='Obs.', testset='Test set (strat. 20%)',
                           just_plt_testset=False):
    """
    Wrapper to plot up the X vs. Y performance by region
    """
    # Only consider the variables to be plotted
    params2plot = [target, ]
    df = df[params2plot+[LonVar, LatVar, target, testset]]
    # Add ocean columns to dataframe
    df = AC.add_loc_ocean2df(df=df, LatVar=LatVar, LonVar=LonVar)
    # Split by regions
    regions = list(set(df['ocean'].dropna()))
    dfs = [df.loc[df['ocean'] == i, :] for i in regions]
    dfs = dict(zip(regions, dfs))
    # Only consider withheld data
    if just_plt_testset:
        df = df.loc[df[testset] == True, :]
    # Also get an open ocean dataset
    # Use an all data for now
    dfs['all'] = df.copy()
    regions += ['all']
    # loop and plot by region
    for region in regions:
        print(region)
        df = dfs[region]
        # What variable to use in titles?
        if just_plt_testset:
            extr_str = region+' (withheld)'
        else:
            extr_str = region
        # Now plot
        try:
            plt_X_vs_Y_for_obs_v_params(df=df, params2plot=params2plot,
                                        obs_var=target, extr_str=extr_str)
        except ValueError:
            print("WARNING: Not plotting for region ('{}') due to ValueError")

        #
        # TODO ... Update to plot withheld and full dataset on a single plot


def plot_up_PDF_of_obs_and_predictions_WINDOW(show_plot=False, params=[],
                                              testset='Test set (strat. 20%)',
                                              target='Iodide', df=None,
                                              units='pM', xlim=None, dpi=320):
    """
    Plot up CDF and PDF plots to explore point-vs-point data

    Parameters
    -------
    target (str): Name of the target variable (e.g. iodide)
    testset (str): Testset to use, e.g. stratified sampling over quartiles for 20%:80%
    dpi (int): resolution to use for saved image (dots per square inch)
    show_plot (bool): show the plot on screen
    df (pd.DataFrame): DataFrame of data
    units (str): units of the target in the dataframe
    xlim (tuple): limits for plotting x axis
    ylim (tuple): limits for plotting y axis

    Returns
    -------
    (None)
    """
    import seaborn as sns
    sns.set(color_codes=True)
    sns.set_context("paper", font_scale=0.75)
    # Make sure a dataFrame has been provided
    assert type(df) == pd.DataFrame, "Please provide DataFrame ('df') with data"
    # Get a dictionary of different dataset splits
    dfs = {}
    # Entire dataset
    dfs['Entire'] = df.copy()
    # Testdataset
    dfs['All (withheld)'] = df.loc[df[testset] == True, :].copy()
    # Maintain ordering of plotting
    datasets = dfs.keys()
    # Setup color dictionary
    CB_color_cycle = AC.get_CB_color_cycle()
    color_d = dict(zip(params, CB_color_cycle))
    # set a name of file to save data to
    savetitle = 'Oi_prj_point_for_point_comparison_obs_vs_model_PDF_WINDOW_{}'
    savetitle = savetitle.format(target)
    # - Plot up CDF and PDF plots for the dataset and residuals
    fig = plt.figure(dpi=dpi)
    nrows = len(datasets)
    ncols = 2
    for n_dataset, dataset in enumerate(datasets):
        # set Axis for abosulte PDF
        axn = np.arange(1, (nrows*ncols)+1)[::ncols][n_dataset]
        ax1 = fig.add_subplot(nrows, ncols, axn)
        # Get data
        df = dfs[dataset]
        # Drop NaNs
        df = df.dropna()
        # Numer of data points
        N_ = df.shape
        print(dataset, N_)
        # Only add an axis label on to the bottommost plots
        axlabel = None
        if n_dataset in np.arange(1, (nrows*ncols)+1)[::ncols]:
            axlabel = '[{}'.format(target) + '$_{aq}$]'+' ({})'.format(units)
        # - Plot up PDF plots for the dataset
        # Plot observations
        var_ = 'Obs.'
        obs_arr = df[target].values
        ax = sns.distplot(obs_arr, axlabel=axlabel, label=var_,
                          color='k', ax=ax1)
        # Loop and plot model values
        for param in params:
            arr = df[param].values
            ax = sns.distplot(arr, axlabel=axlabel,
                              label=param,
                              color=color_d[param], ax=ax1)
        # Force y axis extent to be correct
        ax1.autoscale()
        # Force x axis to be constant
        ax1.set_xlim(xlim)
        # Beautify the plot/figure
        ylabel = 'Frequency \n ({})'
        ax1.set_ylabel(ylabel.format(dataset))
        # Add legend to first plot
        if (n_dataset == 0):
            plt.legend()
            ax1.set_title('Concentration')
        # Plot up PDF plots for the residual dataset
        # set Axis for abosulte PDF
        axn = np.arange(1, (nrows*ncols)+1)[1::ncols][n_dataset]
        ax2 = fig.add_subplot(nrows, ncols, axn)
        # get observations
        obs_arr = df[target].values
        # Loop and plot model values
        for param in params:
            arr = df[param].values - obs_arr
            ax = sns.distplot(arr, axlabel=axlabel,
                              label=param,
                              color=color_d[param], ax=ax2)
        # Force y axis extent to be correct
        ax2.autoscale()
        # Force x axis to be constant
        ax2.set_xlim(-xlim[1],  xlim[1])
        # Add legend to first plot
        if (n_dataset == 0):
            ax2.set_title('Bias')
    # Save whole figure
    plt.savefig(savetitle)
