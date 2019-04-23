"""

Plotting and analysis code for sea-surface iodide prediction work.



Please see Paper(s) for more details:
Sherwen, T., Chance, R. J., Tinel, L., Ellis, D., Evans, M. J., and Carpenter, L. J.: A machine learning based global sea-surface iodide distribution, Earth Syst. Sci. Data Discuss., https://doi.org/10.5194/essd-2019-40, in review, 2019.


"""
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# ---------- Functions to produce plots/analysis for Oi! paper --------------
# ---------------------------------------------------------------------------
def plot_up_obs_spatially_against_predictions_options(dpi=320,
                                                      RFR_dict=None,
                                                      testset='Test set (strat. 20%)',
                                                      rm_Skagerrak_data=True,
                                                      rm_non_water_boxes=True):
    """ Plot up predicted values overlaid with observations """
    # testset='Test set (strat. 20%)'
    import seaborn as sns
    from matplotlib import colors
    # reset settings as plotting maps
    sns.reset_orig()
    # elephant
    # ---- Get the data
    # - Get the spatial predictions
#    res4param = '4x5'  # use 4x5 for testing
    res4param = '0.125x0.125'  # only 0.125x0.125 should be used for analysis
    if rm_Skagerrak_data:
        extr_str = '_No_Skagerrak'
    else:
        extr_str = ''
    filename = 'Oi_prj_predicted_iodide_{}{}.nc'.format(res4param, extr_str)
#    folder =  './'
    folder = get_file_locations('iodide_data')
    ds = xr.open_dataset(folder + filename)
    # Set the variable to plot underneath observations
    var2plot = 'Ensemble_Monthly_mean'
    # only select boxes where that are fully water (seasonal)
    if rm_non_water_boxes:
        ds = add_LWI2array(ds=ds, res=res4param, var2template=var2plot)
        #
        ds[var2plot] = ds[var2plot].where(ds['IS_WATER'] == True)
#    var2plot = 'RFR(TEMP+DEPTH+SAL)'
    arr = ds[var2plot].mean(dim='time', skipna=True).values
    # - Get the observations
    # select dataframe with observations and predictions in it
    if isinstance(RFR_dict, type(None)):
        RFR_dict = build_or_get_current_models()
    df = RFR_dict['df']
    # get stats on models in RFR_dict
    stats = get_stats_on_current_models(RFR_dict=RFR_dict, verbose=False)
    # only consider that are not outliers.
    df = df.loc[df['Iodide'] <= get_outlier_value(df=df, var2use='Iodide'), :]
#    df.loc[ df['Iodide']<= 400., :]
    # ---- Plot up and save to PDF
    # - setup plotting
    plt.close('all')
    show_plot = False
    # Adjust figure
    top = 0.985
    right = 0.94
    bottom = 0.075
    left = 0.075
    left_cb_pos = 0.96
    # Setup PDF
    res = res4param
    savetitle = 'Oi_prj_spatial_comp_obs_vs_predict_{}'.format(res)
    pdff = AC.plot2pdfmulti(title=savetitle, open=True, dpi=dpi)
    # fix colorbar to be 0 to 240
    fixcb = np.array([0, 240])
    cmap = AC.get_colormap(fixcb)
    norm = colors.Normalize(vmin=fixcb[0], vmax=fixcb[-1])
    nticks = 5
    extend = 'max'
    units = 'nM'
#    figsize = (11, 5)
    figsize = (15, 10)
    #  - plot globally
    plot_options = ((8, 'k'), (8, 'none'), (10, 'k'), (10, 'none'))
    # Initialise plot
    for plot_option_n, plot_option in enumerate(plot_options):
        s, edgecolor = plot_option
        # set plotting options
        fig, ax = plt.subplots(dpi=dpi, figsize=figsize)
        # Plot up background parameterised average.
        if res == '0.125x0.125':
            centre = True
        else:
            centre = False
        AC.plot_spatial_figure(arr, fixcb=fixcb, nticks=nticks, units=units,
                               ax=ax, fig=fig, centre=centre, fillcontinents=True,
                               extend=extend, res=res, left_cb_pos=left_cb_pos,
                               #        cmap=cmap,
                               show=False)
        # Now add point for observations
        x = df[u'Longitude'].values
        y = df[u'Latitude'].values
        z = df['Iodide'].values
        ax.scatter(x, y, c=z, s=s, cmap=cmap, norm=norm, edgecolor=edgecolor)
        # Adjust subplot positions
        fig.subplots_adjust(top=top, right=right, left=left, bottom=bottom)
        # Save to PDF
        AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
        plt.savefig(savetitle+'_{}.png'.format(plot_option_n))
        if show_plot:
            plt.show()
        plt.close()
    # Save entire pdf
    AC.plot2pdfmulti(pdff, savetitle, close=True, dpi=dpi)


def plot_up_obs_spatially_against_predictions(dpi=320,
                                              RFR_dict=None,
                                              testset='Test set (strat. 20%)',
                                              rm_Skagerrak_data=False,
                                              rm_non_water_boxes=True):
    """ Plot up predicted values overlaid with observations """
    # testset='Test set (strat. 20%)'
    import seaborn as sns
    from matplotlib import colors
    # reset settings as plotting maps
    sns.reset_orig()
    # elephant
    # ---- Get the data
    # - Get the spatial predictions
#    res4param = '4x5'  # use 4x5 for testing
    res4param = '0.125x0.125'  # only 0.125x0.125 should be used for analysis
    if rm_Skagerrak_data:
        extr_str = '_No_Skagerrak'
    else:
        extr_str = ''
    filename = 'Oi_prj_predicted_iodide_{}{}.nc'.format(res4param, extr_str)
    folder = get_file_locations('iodide_data')
    ds = xr.open_dataset(folder + filename)
    # Set the variable to plot underneath observations
    var2plot = 'Ensemble_Monthly_mean'
    # only select boxes where the
    if rm_non_water_boxes:
        ds = add_LWI2array(ds=ds, res=res4param, var2template=var2plot)
        #
        ds[var2plot] = ds[var2plot].where(ds['IS_WATER'] == True)
    #    var2plot = 'RTR(TEMP+DEPTH+SAL)'
        arr = ds[var2plot].mean(dim='time', skipna=True).values
    # - Get the observations
    # select dataframe with observations and predictions in it
    if isinstance(RFR_dict, type(None)):
        RFR_dict = build_or_get_current_models()
    df = RFR_dict['df']
    # get stats on models in RFR_dict
    stats = get_stats_on_current_models(RFR_dict=RFR_dict, verbose=False)
    # only consider values below 400
#   df = df.loc[ df['Iodide']<= 400., :]
    # only consider values that are not outliers
    df = df.loc[df['Iodide'] <= get_outlier_value(df=df, var2use='Iodide'), :]
    # ---- Plot up and save to PDF
    # - setup plotting
    plt.close('all')
    show_plot = False
    # Adjust figure
    top = 0.985
    right = 0.94
    bottom = 0.075
    left = 0.075
    left_cb_pos = 0.96
    # Setup PDF
    res = res4param
    savetitle = 'Oi_prj_spatiall_comp_obs_vs_predict_{}_{}'
    savetitle = savetitle.format(res, extr_str)
    pdff = AC.plot2pdfmulti(title=savetitle, open=True, dpi=dpi)
    # fix colorbar to be 0 to 240
    fixcb = np.array([0, 240])
    cmap = AC.get_colormap(fixcb)
    norm = colors.Normalize(vmin=fixcb[0], vmax=fixcb[-1])
    nticks = 5
    extend = 'max'
    edgecolor = 'k'
    s = 10
    units = 'nM'
#    figsize = (11, 5)
    figsize = (15, 10)
    #  - plot globally
    # Initialise plot
    fig, ax = plt.subplots(dpi=dpi, figsize=figsize)
    # Plot up background parameterised average.
    if res == '0.125x0.125':
        centre = True
    else:
        centre = False
    AC.plot_spatial_figure(arr, fixcb=fixcb, nticks=nticks,
                           ax=ax, fig=fig, centre=centre, fillcontinents=True,
                           extend=extend, res=res, left_cb_pos=left_cb_pos,
                           #        cmap=cmap,
                           units=units, show=False)
    # Now add point for observations
    x = df[u'Longitude'].values
    y = df[u'Latitude'].values
    z = df['Iodide'].values
    ax.scatter(x, y, c=z, s=s, cmap=cmap, norm=norm, edgecolor=edgecolor)
    # Adjust subplot positions
    fig.subplots_adjust(top=top, right=right, left=left, bottom=bottom)
    # Save to PDF
    AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
    plt.savefig(savetitle+'.png')
    if show_plot:
        plt.show()
    plt.close()

    # - split world into quarters and plot up
    # get lats and lons for res
    lon, lat, NIU = AC.get_latlonalt4res(res=res)
    #
    latrange = [-90, 0, 90]
    lonrange = np.linspace(-180, 180, 4)
#    latrange = lat[::len( lat ) /2 ]
#    lonrange = lon[::len( lon ) /2 ]
    s = 15
    # loop and plot arrays
    for nlat, latmin in enumerate(latrange[:-1]):
        latmax = latrange[nlat+1]
        for nlon, lonmin in enumerate(lonrange[:-1]):
            lonmax = lonrange[nlon+1]
            # -  plot up obs
            # Initialise plot
            fig, ax = plt.subplots(dpi=dpi, figsize=figsize)
            # Now plot up
            AC.plot_spatial_figure(arr, fixcb=fixcb, nticks=nticks,
                                   ax=ax, fig=fig, centre=centre, fillcontinents=True,
                                   extend=extend, res=res, resolution='h',
                                   left_cb_pos=left_cb_pos,
                                   #        cmap=cmap,
                                   units=units, show=False)
            # -  plot up param
            # Now add point for observations
            x = df[u'Longitude'].values
            y = df[u'Latitude'].values
            z = df['Iodide'].values
            ax.scatter(x, y, c=z, s=s, cmap=cmap, norm=norm,
                       edgecolor=edgecolor)
            # set axis limits
            ax.set_xlim(lonmin, lonmax)
            ax.set_ylim(latmin, latmax)
            # Adjust figure
            fig.subplots_adjust(top=top, right=right, left=left, bottom=bottom)
            # Save to PDF
            AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
            if show_plot:
                plt.show()
            plt.close()
    # Save entire pdf
    AC.plot2pdfmulti(pdff, savetitle, close=True, dpi=dpi)


def plot_up_obs_spatially_against_predictions_at_points(dpi=320,
                                                        RFR_dict=None,
                                                        testset='Test set (strat. 20%)'
                                                        ):
    """ Plot up predicted values against observations at obs. points """
    import seaborn as sns
    from matplotlib import colors
    # reset settings as plotting maps
    sns.reset_orig()
    # elephant
    # ---- Get the data
    # - Get the spatial predictions
    # set models and params to plot
    models2compare = [
        'RFR(TEMP+DEPTH+SAL+NO3+DOC)', 'RFR(TEMP+DEPTH+SAL+NO3)',
        'RFR(TEMP+DEPTH+SAL)', 'RFR(TEMP+SAL+Prod)',
        #    'RFR(TEMP+SAL+NO3)',
        #    'RFR(TEMP+DEPTH+SAL)',
    ]
    params = [u'Chance2014_STTxx2_I', u'MacDonald2014_iodide', ]
    # --- Plot up comparisons spatially just showing obs. points
    savetitle = 'Oi_prj_spatiall_comp_obs_vs_predict_at_points'
    pdff = AC.plot2pdfmulti(title=savetitle, open=True, dpi=dpi)
    # reset settings as plotting maps
    sns.reset_orig()
    # set values to plot all data
    units = 'nM'
    fixcb = np.array([0, 240])
    cmap = AC.get_colormap(fixcb)
    norm = colors.Normalize(vmin=fixcb[0], vmax=fixcb[-1])
    nticks = 5
    extend = 'max'
#    figsize = (11, 5)
    figsize = (15, 10)
    # plot up a white background
    arr = np.zeros(AC.get_dims4res('4x5'))[..., 0]
    # - plot observations
    # Initialise plot
    fig, ax = plt.subplots(dpi=dpi, figsize=figsize)
    # Add a title
    title = 'Entire dataset obs.'
    # Add a blank plot
    AC.plot_spatial_figure(arr, fixcb=fixcb, nticks=nticks,
                           ax=ax, fig=fig, centre=centre, fillcontinents=True,
                           extend=extend, title=title, res='4x5', title_x=0.15,
                           units=units,
                           #        cmap=cmap,
                           show=False)
    # Now add point for observations
    x = df[u'Longitude'].values
    y = df[u'Latitude'].values
    z = df['Iodide'].values
    ax.scatter(x, y, c=z, s=5, cmap=cmap, norm=norm)
    # Save to PDF
    AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
    if show_plot:
        plt.show()
    plt.close()

    # - just plot testset
    df_tmp = df.loc[df[testset] == True, :]
    # Initialise plot
    fig, ax = plt.subplots(dpi=dpi, figsize=figsize)
    # Plot up background parameterised average.
    # Add a title
    title = 'test dataset obs.'
    # Add a blank plot
    AC.plot_spatial_figure(arr, fixcb=fixcb, nticks=nticks,
                           ax=ax, fig=fig, centre=centre, fillcontinents=True,
                           extend=extend, res='4x5', title=title, title_x=0.15,
                           units=units,
                           #        cmap=cmap,
                           show=False)
    # Now add point for observations
    x = df_tmp[u'Longitude'].values
    y = df_tmp[u'Latitude'].values
    z = df_tmp['Iodide'].values
    ax.scatter(x, y, c=z, s=5, cmap=cmap, norm=norm)
    # Save to PDF
    AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
    if show_plot:
        plt.show()
    plt.close()

    # --- plot up bias by param
    units = 'nM'
    # - plot for entire dataset
    fixcb = np.array([-100, 100])
    cmap = AC.get_colormap(fixcb)
    norm = colors.Normalize(vmin=fixcb[0], vmax=fixcb[-1])
    nticks = 5
    extend = 'both'
    for param in params + models2compare:
        # Initialise plot
        fig, ax = plt.subplots(dpi=dpi, figsize=figsize)
        # Add a title
        title = 'Bias in entire dataset ({}-obs)'.format(param)
        # add a blank plot
        AC.plot_spatial_figure(arr, fixcb=fixcb, nticks=nticks,
                               ax=ax, fig=fig, centre=centre, fillcontinents=True,
                               extend=extend, res=res, title=title, title_x=0.15,
                               units=units,
                               #        cmap=cmap,
                               show=False)
        # get the residuals
        x = df[u'Longitude'].values
        y = df[u'Latitude'].values
        z = df[param+'-residual'].values
        ax.scatter(x, y, c=z, s=5, cmap=cmap, norm=norm)
#        print z.max(), z.min()
        # Save to PDF
        AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
        if show_plot:
            plt.show()
        plt.close()

    # - Plot up for just testset
    df_tmp = df.loc[df[testset] == True, :]
    # Loop params
    for param in params + models2compare:
        # Initialise plot
        fig, ax = plt.subplots(dpi=dpi, figsize=figsize)
        # Add a title
        title = 'Bias in testset ({}-obs)'.format(param)
        # add a blank plot
        AC.plot_spatial_figure(arr, fixcb=fixcb, nticks=nticks,
                               ax=ax, fig=fig, centre=centre, fillcontinents=True,
                               extend=extend, res=res, title=title, title_x=0.15,
                               units=units,
                               #        cmap=cmap,
                               show=False)
        # get the residuals
        x = df_tmp[u'Longitude'].values
        y = df_tmp[u'Latitude'].values
        z = df_tmp[param+'-residual'].values
        ax.scatter(x, y, c=z, s=5, cmap=cmap, norm=norm)
#        print z.max(), z.min()
        # Save to PDF
        AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
        if show_plot:
            plt.show()
        plt.close()

    # --- plot up Abs. bias by param
    units = 'nM'
    # - plot for entire dataset
    fixcb = np.array([0, 100])
    cmap = AC.get_colormap(fixcb, cb='Reds')
    norm = colors.Normalize(vmin=fixcb[0], vmax=fixcb[-1])
    nticks = 5
    extend = 'both'
    for param in params + models2compare:
        # Initialise plot
        fig, ax = plt.subplots(dpi=dpi, figsize=figsize)
        # Add a title
        title = 'Abs. bias in entire dataset ({}-obs)'.format(param)
        # add a blank plot
        AC.plot_spatial_figure(arr, fixcb=fixcb, nticks=nticks,
                               ax=ax, fig=fig, centre=centre, fillcontinents=True,
                               extend=extend, res=res, title=title, title_x=0.15,
                               units=units,
                               #        cmap=cmap,
                               show=False)
        # get the residuals
        x = df[u'Longitude'].values
        y = df[u'Latitude'].values
        z = np.abs(df[param+'-residual'].values)
        ax.scatter(x, y, c=z, s=5, cmap=cmap, norm=norm)
#        print z.max(), z.min()
        # Save to PDF
        AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
        if show_plot:
            plt.show()
        plt.close()

    # - Plot up Abs. bias for just testset
    df_tmp = df.loc[df[testset] == True, :]
    # Loop params
    for param in params + models2compare:
        # Initialise plot
        fig, ax = plt.subplots(dpi=dpi, figsize=figsize)
        # Add a title
        title = 'Abs. bias in testset ({}-obs)'.format(param)
        # add a blank plot
        AC.plot_spatial_figure(arr, fixcb=fixcb, nticks=nticks,
                               ax=ax, fig=fig, centre=centre, fillcontinents=True,
                               extend=extend, res=res, title=title, title_x=0.15,
                               units=units,
                               #        cmap=cmap,
                               show=False)
        # get the residuals
        x = df_tmp[u'Longitude'].values
        y = df_tmp[u'Latitude'].values
        z = np.abs(df_tmp[param+'-residual'].values)
        ax.scatter(x, y, c=z, s=5, cmap=cmap, norm=norm)
#        print z.max(), z.min()
        # Save to PDF
        AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
        if show_plot:
            plt.show()
        plt.close()

    # --- plot up % Abs. bias by param
    units = '%'
    # - plot for entire dataset
    fixcb = np.array([-100, 100])
    cmap = AC.get_colormap(fixcb)
    norm = colors.Normalize(vmin=fixcb[0], vmax=fixcb[-1])
    nticks = 5
    extend = 'both'
    for param in params + models2compare:
        # Initialise plot
        fig, ax = plt.subplots(dpi=dpi, figsize=figsize)
        # Add a title
        title = '% Abs. bias in entire dataset ({}-obs)'.format(param)
        # add a blank plot
        AC.plot_spatial_figure(arr, fixcb=fixcb, nticks=nticks,
                               ax=ax, fig=fig, centre=centre, fillcontinents=True,
                               extend=extend, res=res, title=title, title_x=0.15,
                               units=units,
                               #        cmap=cmap,
                               show=False)
        # get the residuals
        x = df[u'Longitude'].values
        y = df[u'Latitude'].values
        z = np.abs(df[param+'-residual'].values) / df['Iodide'].values * 100
        ax.scatter(x, y, c=z, s=5, cmap=cmap, norm=norm)
#        print z.max(), z.min()
        # Save to PDF
        AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
        if show_plot:
            plt.show()
        plt.close()

    # - Plot up % Abs. bias for just testset
    df_tmp = df.loc[df[testset] == True, :]
    # Loop params
    for param in params + models2compare:
        # Initialise plot
        fig, ax = plt.subplots(dpi=dpi, figsize=figsize)
        # Add a title
        title = '% Abs. bias in testset ({}-obs)'.format(param)
        # add a blank plot
        AC.plot_spatial_figure(arr, fixcb=fixcb, nticks=nticks,
                               ax=ax, fig=fig, centre=centre, fillcontinents=True,
                               extend=extend, res=res, title=title, title_x=0.15,
                               units=units,
                               #        cmap=cmap,
                               show=False)
        # get the residuals
        x = df_tmp[u'Longitude'].values
        y = df_tmp[u'Latitude'].values
        z = np.abs(df_tmp[param+'-residual'].values) / df_tmp['Iodide'].values
        z *= 100
        ax.scatter(x, y, c=z, s=5, cmap=cmap, norm=norm)
#        print z.max(), z.min()
        # Save to PDF
        AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
        if show_plot:
            plt.show()
        plt.close()

    # - plot up bias by param ( if abs(bias greater than X)

    # Save entire pdf
    AC.plot2pdfmulti(pdff, savetitle, close=True, dpi=dpi)


def plot_predicted_iodide_vs_lat_figure(dpi=320, plot_avg_as_median=False,
                                        show_plot=False, shade_std=True,
                                        just_plot_existing_params=False,
                                        plot_up_param_iodide=True, context="paper",
                                        ds=None,
                                        rm_Skagerrak_data=False):
    """ Plot a figure of iodide vs laitude """
    import seaborn as sns
    sns.set(color_codes=True)
    if context == "paper":
        sns.set_context("paper", font_scale=0.75)
    else:
        sns.set_context("talk", font_scale=0.9)
    # Get observations
    df_obs = get_processed_df_obs_mod()  # NOTE this df contains values >400nM
    # Get predicted values
#    folder = '/shared/scratch/hpc2/users/ts551/labbook/Python_progs/'
    if isinstance(ds, type(None)):
        folder = get_file_locations('iodide_data')
        filename = 'Oi_prj_predicted_iodide_0.125x0.125{}.nc'
        if rm_Skagerrak_data:
            filename = filename.format('_No_Skagerrak')
        else:
            filename = filename.format('')
        ds = xr.open_dataset(folder + filename)
    # Rename to a more concise name
    try:
        ds.rename(name_dict={'Ensemble_Monthly_mean': 'RFR(Ensemble)'},
                  inplace=True)
    except ValueError:
        # pass if 'Ensemble_Monthly_mean' already is in dataset
        pass
    # Get predicted values binned by latitude
    df = get_spatial_predictions_0125x0125_by_lat(ds=ds)
    # params to pot
    models2compare = [
        #    'RFR(TEMP+DEPTH+SAL+NO3+DOC)', 'RFR(TEMP+SAL+Prod)', 'RFR(TEMP+DEPTH+SAL)'
        'RFR(Ensemble)'
    ]
    params = ['Chance2014_STTxx2_I', u'MacDonald2014_iodide']
    rename_titles = {u'Chance2014_STTxx2_I': 'Chance et al. (2014)',
                     u'MacDonald2014_iodide': 'MacDonald et al. (2014)',
                     'RFR(Ensemble)': 'RFR(Ensemble)',
                     'Iodide': 'Obs.',
                     #                     u'Chance2014_Multivariate': 'Chance et al. (2014) (Multi)',
                     }
    if just_plot_existing_params:
        params2plot = params
    else:
        params2plot = models2compare + params
    # assign colors
    CB_color_cycle = AC.get_CB_color_cycle()
    color_d = dict(zip(params2plot, CB_color_cycle))
    # --- plot up vs. lat
    fig, ax = plt.subplots()
    # if including parameters, loop by param and plot
    if plot_up_param_iodide:
        for param in params2plot:
            # Set color for param
            color = color_d[param]
            # Plot average
            if plot_avg_as_median:
                var2plot = '{} - median'.format(param)
            else:
                var2plot = '{} - mean'.format(param)
            # get X
            X = df[var2plot].index.values
            # plot as line
            plt.plot(X, df[var2plot].values, color=color,
                     label=rename_titles[param])
            # plot up quartiles/std
            if shade_std:
                # Use the std from the ensemble members for the ensemble
                if (param == 'RFR(Ensemble)'):
                    param = 'Ensemble_Monthly_std'
                    if plot_avg_as_median:
                        std_var = '{} - median'.format(param)
                    else:
                        std_var = '{} - mean'.format(param)
                    # Now plot plot this as +/- the average
                    low = df[var2plot].values - df[std_var].values
                    high = df[var2plot].values + df[std_var].values
                    # use the 75th percentile of the monthly average std
                    # of the ensemble members
    #                 std_var = 'Ensemble_Monthly_std - 75%'
    #                 low = df[var2plot].values - df[std_var].values
    #                 high = df[var2plot].values + df[std_var].values
                else:
                    std_var = '{} - std'.format(param)
                    # Now plot plot this as +/- the average
                    low = df[var2plot].values - df[std_var].values
                    high = df[var2plot].values + df[std_var].values
            else:
                low = df['{} - 25%'.format(param)].values
                high = df['{} - 75%'.format(param)].values
            ax.fill_between(X, low, high, alpha=0.2, color=color)

    # Highlight coastal obs
    tmp_df = df_obs.loc[df_obs['Coastal'] == True, :]
    X = tmp_df['Latitude'].values
    Y = tmp_df['Iodide'].values
    plt.scatter(X, Y, color='k', marker='D', facecolor='none', s=3,
                label='Coastal obs.')
    # Non-coastal obs
    tmp_df = df_obs.loc[df_obs['Coastal'] == False, :]
    X = tmp_df['Latitude'].values
    Y = tmp_df['Iodide'].values
    plt.scatter(X, Y, color='k', marker='D', facecolor='k', s=3,
                label='Non-coastal obs.')
    # limit plot y axis
    plt.ylim(-20, 420)
    plt.ylabel('[I$^{-}_{aq}$] (nM)')
    plt.xlabel('Latitude ($^{\\rm o}$N)')
    plt.legend()
    # save or show?
    filename = 'Oi_prj_global_predicted_vals_vs_lat'
    plt.savefig(filename, dpi=dpi)
    if show_plot:
        plt.show()
    plt.close()


def plot_predicted_iodide_vs_lat_figure_with_Skagerrak_too(dpi=320,
                                                           plot_avg_as_median=False,
                                                           show_plot=False,
                                                           shade_std=True,
                                                          just_plot_existing_params=False,
                                                           plot_up_param_iodide=True,
                                                           context="paper", ds=None,
                                                           rm_Skagerrak_data=False):
    """ Plot a figure of iodide vs laitude """
    import seaborn as sns
    sns.set(color_codes=True)
    if context == "paper":
        sns.set_context("paper", font_scale=0.75)
    else:
        sns.set_context("talk", font_scale=0.9)
    # Get observations
    df_obs = get_processed_df_obs_mod()  # NOTE this df contains values >400nM
    # Get predicted values
#    folder = '/shared/scratch/hpc2/users/ts551/labbook/Python_progs/'
    folder = get_file_locations('iodide_data')
    filename = 'Oi_prj_predicted_iodide_0.125x0.125{}.nc'
    ds = xr.open_dataset(folder + filename.format('_No_Skagerrak'))
    # Get data with Skagerrak data too.
    folder = get_file_locations('iodide_data')
    ds2 = xr.open_dataset(folder + filename.format(''))
    # Rename to a more concise name
    try:
        ds.rename(name_dict={'Ensemble_Monthly_mean': 'RFR(Ensemble)'},
                  inplace=True)
    except ValueError:
        # pass if 'Ensemble_Monthly_mean' already is in dataset
        pass
    # Rename to a more concise name
    SkagerrakVarName = 'RFR(Ensemble) - Inc. Skagerrak data'
    try:
        ds2.rename(name_dict={'Ensemble_Monthly_mean': SkagerrakVarName},
                   inplace=True)
    except ValueError:
        # pass if 'Ensemble_Monthly_mean' already is in dataset
        pass
    # Get predicted values binned by latitude
    df = get_spatial_predictions_0125x0125_by_lat(ds=ds)
    df2 = get_spatial_predictions_0125x0125_by_lat(ds=ds2)
    # params to pot
    models2compare = [
        #    'RFR(TEMP+DEPTH+SAL+NO3+DOC)', 'RFR(TEMP+SAL+Prod)', 'RFR(TEMP+DEPTH+SAL)'
        'RFR(Ensemble)'
    ]
    params = ['Chance2014_STTxx2_I', u'MacDonald2014_iodide']
    rename_titles = {u'Chance2014_STTxx2_I': 'Chance et al. (2014)',
                     u'MacDonald2014_iodide': 'MacDonald et al. (2014)',
                     'RFR(Ensemble)': 'RFR(Ensemble)',
                     'Iodide': 'Obs.',
                     #                     u'Chance2014_Multivariate': 'Chance et al. (2014) (Multi)',
                     }
    if just_plot_existing_params:
        params2plot = params
    else:
        params2plot = models2compare + params
    # assign colors
    CB_color_cycle = AC.get_CB_color_cycle()
    color_d = dict(zip(params2plot, CB_color_cycle))
    # - Add a lines for parameterisations
    fig, ax = plt.subplots()
    # if including parameters, loop by param and plot
    if plot_up_param_iodide:
        for param in params2plot:
            # Set color for param
            color = color_d[param]
            # Plot average
            if plot_avg_as_median:
                var2plot = '{} - median'.format(param)
            else:
                var2plot = '{} - mean'.format(param)
            # get X
            X = df[var2plot].index.values
            # plot as line
            plt.plot(X, df[var2plot].values, color=color,
                     label=rename_titles[param])
            # plot up quartiles/std
            if shade_std:
                # Use the std from the ensemble members for the ensemble
                if (param == 'RFR(Ensemble)') or (param == SkagerrakVarName):
                    param = 'Ensemble_Monthly_std'
                    if plot_avg_as_median:
                        std_var = '{} - median'.format(param)
                    else:
                        std_var = '{} - mean'.format(param)
                    # Now plot plot this as +/- the average
                    low = df[var2plot].values - df[std_var].values
                    high = df[var2plot].values + df[std_var].values
                    # use the 75th percentile of the monthly average std
                    # of the ensemble members
    #                 std_var = 'Ensemble_Monthly_std - 75%'
    #                 low = df[var2plot].values - df[std_var].values
    #                 high = df[var2plot].values + df[std_var].values
                else:
                    std_var = '{} - std'.format(param)
                    # Now plot plot this as +/- the average
                    low = df[var2plot].values - df[std_var].values
                    high = df[var2plot].values + df[std_var].values
            else:
                low = df['{} - 25%'.format(param)].values
                high = df['{} - 75%'.format(param)].values
            ax.fill_between(X, low, high, alpha=0.2, color=color)

        # - Add a dashed line for the run including the Skaggerak
        var2plot = 'RFR(Ensemble)'
        param = var2plot
        plt_shading_around_avg = False
    # Set color for param
        color = color_d[param]
        # Plot average
        if plot_avg_as_median:
            var2plot = '{} - median'.format(SkagerrakVarName)
        else:
            var2plot = '{} - mean'.format(SkagerrakVarName)
        # get X
        print(df2.columns)
        X = df2[var2plot].index.values
        # plot as line
        plt.plot(X, df2[var2plot].values, color=color,
                 label=SkagerrakVarName, ls='--')
        # plot up quartiles/std
        if shade_std:
            # Use the std from the ensemble members for the ensemble
            if (param == 'RFR(Ensemble)') or (param == SkagerrakVarName):
                param = 'Ensemble_Monthly_std'
                if plot_avg_as_median:
                    std_var = '{} - median'.format(SkagerrakVarName)
                else:
                    std_var = '{} - mean'.format(SkagerrakVarName)
                # Now plot plot this as +/- the average
                low = df2[var2plot].values - df2[std_var].values
                high = df2[var2plot].values + df2[std_var].values
                # use the 75th percentile of the monthly average std
                # of the ensemble members
#                 std_var = 'Ensemble_Monthly_std - 75%'
#                 low = df[var2plot].values - df[std_var].values
#                 high = df[var2plot].values + df[std_var].values
            else:
                std_var = '{} - std'.format(SkagerrakVarName)
                # Now plot plot this as +/- the average
                low = df2[var2plot].values - df2[std_var].values
                high = df2[var2plot].values + df2[std_var].values
        else:
            low = df2['{} - 25%'.format(SkagerrakVarName)].values
            high = df2['{} - 75%'.format(SkagerrakVarName)].values
        # Include shading around the average?
        if plt_shading_around_avg:
            ax.fill_between(X, low, high, alpha=0.2, color=color)

        # - Plot up the observations and highlight coastal and non-coastal
    # Highlight coastal obs
    tmp_df = df_obs.loc[df_obs['Coastal'] == True, :]
    X = tmp_df['Latitude'].values
    Y = tmp_df['Iodide'].values
    plt.scatter(X, Y, color='k', marker='D', facecolor='none', s=3,
                label='Coastal obs.')
    # Non-coastal obs
    tmp_df = df_obs.loc[df_obs['Coastal'] == False, :]
    X = tmp_df['Latitude'].values
    Y = tmp_df['Iodide'].values
    plt.scatter(X, Y, color='k', marker='D', facecolor='k', s=3,
                label='Non-coastal obs.')
    # limit plot y axis
    plt.ylim(-20, 420)
    plt.ylabel('[I$^{-}_{aq}$] (nM)')
    plt.xlabel('Latitude ($^{\\rm o}$N)')
    plt.legend()
    # save or show?
    filename = 'Oi_prj_global_predicted_vals_vs_lat_with_Skagerrak_too'
    plt.savefig(filename, dpi=dpi)
    if show_plot:
        plt.show()
    plt.close()


def plot_predicted_iodide_vs_lat_figure_ENSEMBLE(dpi=320, extr_str='',
                                                 plot_avg_as_median=False, RFR_dict=None,
                                                 res='0.125x0.125',
                                                 show_plot=False, close_plot=True,
                                                 save_plot=False, shade_std=True,
                                                 folder=None, ds=None, topmodels=None):
    """ Plot a figure of iodide vs laitude - showing all ensemble members """
    from collections import OrderedDict
    import seaborn as sns
    sns.set(color_codes=True)
    sns.set_context("paper", font_scale=0.75)
    # Get observations
    df_obs = get_processed_df_obs_mod()  # NOTE this df contains values >400nM
    # Get predicted values
    if isinstance(folder, type(None)):
        folder = get_file_locations('iodide_data')
    if isinstance(ds, type(None)):
        filename = 'Oi_prj_predicted_iodide_{}{}.nc'.format(res, extr_str)
        ds = xr.open_dataset(folder + filename)
    # Rename to a more concise name
    print(ds.data_vars)
    # Get predicted values binned by latitude
    if res == '0.125x0.125':
        df = get_spatial_predictions_0125x0125_by_lat(ds=ds)
    else:
        df = get_stats_on_spatial_predictions_4x5_2x25_by_lat(res=res, ds=ds)
    # params to pot
    if isinstance(topmodels, type(None)):
        # Get RFR_dict if not provide
        if isinstance(RFR_dict, type(None)):
            RFR_dict = build_or_get_current_models()
        topmodels = get_top_models(RFR_dict=RFR_dict, NO_DERIVED=True, n=10)
    params2plot = topmodels
    # assign colors
    CB_color_cycle = AC.get_CB_color_cycle()
    CB_color_cycle += ['darkgreen']
    color_d = dict(zip(params2plot, CB_color_cycle))
    # --- plot up vs. lat
    fig, ax = plt.subplots()
    # loop by param to plot
    for param in params2plot:
        # Set color for param
        color = color_d[param]
        # Plot average
        if plot_avg_as_median:
            var2plot = '{} - median'.format(param)
        else:
            var2plot = '{} - mean'.format(param)
        # get X
        X = df[var2plot].index.values
        # plot as line
        plt.plot(X, df[var2plot].values, color=color, label=param)
        # plot up quartiles/std as shaded regions too
        if shade_std:
            # Use the std from the ensemble members for the ensemble
            if (param == 'RFR(Ensemble)'):
                param = 'Ensemble_Monthly_std'
                if plot_avg_as_median:
                    std_var = '{} - median'.format(param)
                else:
                    std_var = '{} - mean'.format(param)
                # Now plot plot this as +/- the average
                low = df[var2plot].values - df[std_var].values
                high = df[var2plot].values + df[std_var].values
                # use the 75th percentile of the monthly average std
                # of the ensemble members
#                 std_var = 'Ensemble_Monthly_std - 75%'
#                 low = df[var2plot].values - df[std_var].values
#                 high = df[var2plot].values + df[std_var].values
            else:
                std_var = '{} - std'.format(param)
                # Now plot plot this as +/- the average
                low = df[var2plot].values - df[std_var].values
                high = df[var2plot].values + df[std_var].values
        else:
            low = df['{} - 25%'.format(param)].values
            high = df['{} - 75%'.format(param)].values
        ax.fill_between(X, low, high, alpha=0.2, color=color)

    # highlight coastal obs
    tmp_df = df_obs.loc[df_obs['Coastal'] == True, :]
    X = tmp_df['Latitude'].values
    Y = tmp_df['Iodide'].values
    plt.scatter(X, Y, color='k', marker='D', facecolor='none', s=3,
                label='Coastal obs.')
    # non-coastal obs
    tmp_df = df_obs.loc[df_obs['Coastal'] == False, :]
    X = tmp_df['Latitude'].values
    Y = tmp_df['Iodide'].values
    plt.scatter(X, Y, color='k', marker='D', facecolor='k', s=3,
                label='Non-coastal obs.')
    # limit plot y axis
    plt.ylim(-20, 420)
    plt.ylabel('[I$^{-}_{aq}$] (nM)')
    plt.xlabel('Latitude ($^{\\rm o}$N)')
#    plt.xlim(-80, 80 )
    plt.legend()
    # remove repeats from legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    # save or show?
    filename = 'Oi_prj_global_predicted_vals_vs_lat_ENSEMBLE_{}{}.png'
    if save_plot:
        plt.savefig(filename.format(res, extr_str), dpi=dpi)
    if show_plot:
        plt.show()
    if close_plot:
        plt.close()


# ---------------------------------------------------------------------------
# ---------- Functions to analyse/test models for Oi! paper --------------
# ---------------------------------------------------------------------------
def check_seasonalitity_of_iodide_predcitions(show_plot=False):
    """ Compare the seasonality of obs. and parameterised values """
    # --- Set local variables
    rename_titles = {u'Chance2014_STTxx2_I': 'Chance et al. (2014)',
                     u'MacDonald2014_iodide': 'MacDonald et al. (2014)',
                     'RFR(Ensemble)': 'RFR(Ensemble)',
                     u'Chance2014_Multivariate': 'Chance et al. (2014) (Multi)',
                     }
    CB_color_cycle = AC.get_CB_color_cycle()
    colors_dict = dict(zip(rename_titles.keys(),  CB_color_cycle))

    # --- Get dataset where there is more than a 2 months of data at the same loc
    df = get_processed_df_obs_mod()  # NOTE this df contains values >400nM
    # Exclude data with values above 400 nM
#    df = df.loc[ df['Iodide'].values < 400, : ]
    # Exclude outliers
    df = df.loc[df['Iodide'] <= get_outlier_value(df=df, var2use='Iodide'), :]
    # get metadata
    md_df = get_iodide_obs_metadata()
    datasets = md_df[u'Data_Key']
    # loop datasets and find ones with multiple obs.
    N = []
    ds_seas = []
    MonthVar = 'Month (Orig.)'
    for ds in datasets:
        # get obs for dataset
        ds_tmp = df.loc[df[u'Data_Key'] == ds]
        # save the N value
        N += [ds_tmp.shape[0]]
        #
        ds_months = ds_tmp[MonthVar].value_counts(dropna=True).shape[0]
        print(ds, ds_tmp.shape[0], ds_months)
        # check there is at least more than 1 month
        if ds_months >= 2:
            # check that all the latitudes are within .1 of each other
            ds_lats = set(ds_tmp['Latitude'].round(1))
            ds_lons = set(ds_tmp['Longitude'].round(1))
            if (len(ds_lats) <= 2) and (len(ds_lons) <= 2):
                ds_seas += [ds]

    # --- Loop these datasets and plot the three parameterisation s predictions
    savetitle = 'Oi_prj_seasonality_of_iodide'
    pdff = AC.plot2pdfmulti(title=savetitle, open=True, dpi=dpi)
    # Add a title slide
    fig, ax = plt.subplots()
    plt.text(0.1, 1, 'New parametersation', fontsize=15)
    plt.setp(plt.gca(), frame_on=False, xticks=(), yticks=())
    AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
    plt.close()
    # now loop and plot up vs. new parameterisation
    for ds in ds_seas:
        # get location/data on obs.
        md_df_tmp = md_df.loc[md_df[u'Data_Key'] == ds]
        Source = md_df_tmp['Source'].values[0].strip()
        Loc = md_df_tmp['Location'].values[0].strip()
        Cruise = md_df_tmp['Cruise'].values[0].strip()
        # get obs for dataset
        ds_tmp = df.loc[df[u'Data_Key'] == ds]
        ds_tmp.sort_values(by=MonthVar)
        # plot up against months
        plt.scatter(ds_tmp[MonthVar].values,  ds_tmp['Iodide'].values, label='Obs',
                    color='k')
        # Add values for en
        var2plot = ['RFR(Ensemble)']
        for var in var2plot:
            plt.scatter(ds_tmp[MonthVar].values,  ds_tmp[var].values,
                        label=rename_titles[var], color=colors_dict[var]
                        )
        #
        plt.title('{} ({},{})'.format(Source, Loc, Cruise))
        # Add labelleding for X axis
        plt.xlabel('Month')
        plt.xlim(0.5, 12.5)
        datetime_months = [datetime.datetime(
            2000, i, 1) for i in np.arange(1, 13)]
        labels = [i.strftime("%b") for i in datetime_months]
        ax = plt.gca()
        ax.set_xticks(np.arange(1, 13))
        ax.set_xticklabels(labels, rotation=45)
        # Add a legend
        plt.legend()
        # save plot
        AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
        if show_plot:
            plt.show()
        plt.close()

    # Add a title slide
    fig, ax = plt.subplots()
    plt.text(0.1, 1, 'All parametersations', fontsize=15)
    plt.setp(plt.gca(), frame_on=False, xticks=(), yticks=())
    AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
    plt.close()
    # now loop and plot up vs. all parameterisations parameterisation
    for ds in ds_seas:
        # get location/data on obs.
        md_df_tmp = md_df.loc[md_df[u'Data_Key'] == ds]
        Source = md_df_tmp['Source'].values[0].strip()
        Loc = md_df_tmp['Location'].values[0].strip()
        Cruise = md_df_tmp['Cruise'].values[0].strip()
        # get obs for dataset
        ds_tmp = df.loc[df[u'Data_Key'] == ds]
        ds_tmp.sort_values(by=MonthVar)
        # plot up against months
        plt.scatter(ds_tmp[MonthVar].values,  ds_tmp['Iodide'].values, label='Obs',
                    color='k')
        # Add values for en
        var2plot = [
            'RFR(Ensemble)', u'Chance2014_STTxx2_I', u'MacDonald2014_iodide'
        ]
        for var in var2plot:
            plt.scatter(ds_tmp[MonthVar].values,  ds_tmp[var].values,
                        label=rename_titles[var], color=colors_dict[var]
                        )
        #
        plt.title('{} ({},{})'.format(Source, Loc, Cruise))
        # Add labelleding for X axis
        plt.xlabel('Month')
        plt.xlim(0.5, 12.5)
        datetime_months = [datetime.datetime(
            2000, i, 1) for i in np.arange(1, 13)]
        labels = [i.strftime("%b") for i in datetime_months]
        ax = plt.gca()
        ax.set_xticks(np.arange(1, 13))
        ax.set_xticklabels(labels, rotation=45)
        # Add a legend
        plt.legend()
        # save plot
        AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
        if show_plot:
            plt.show()
        plt.close()

    # Add a title slide
    fig, ax = plt.subplots()
    plt.text(0.1, 1, 'Just existing ones', fontsize=15)
    plt.setp(plt.gca(), frame_on=False, xticks=(), yticks=())
    AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
    plt.close()

    # now loop and plot up vs. all parameterisations parameterisation
    for ds in ds_seas:
        # get location/data on obs.
        md_df_tmp = md_df.loc[md_df[u'Data_Key'] == ds]
        Source = md_df_tmp['Source'].values[0].strip()
        Loc = md_df_tmp['Location'].values[0].strip()
        Cruise = md_df_tmp['Cruise'].values[0].strip()
        # get obs for dataset
        ds_tmp = df.loc[df[u'Data_Key'] == ds]
        ds_tmp.sort_values(by=MonthVar)
        # plot up against months
        plt.scatter(ds_tmp[MonthVar].values,  ds_tmp['Iodide'].values, label='Obs',
                    color='k')
        # Add values for en
        var2plot = [
            u'Chance2014_STTxx2_I', u'MacDonald2014_iodide'
        ]
        for var in var2plot:
            plt.scatter(ds_tmp[MonthVar].values,  ds_tmp[var].values,
                        label=rename_titles[var], color=colors_dict[var]
                        )
        #
        plt.title('{} ({},{})'.format(Source, Loc, Cruise))
        # Add labelleding for X axis
        plt.xlabel('Month')
        plt.xlim(0.5, 12.5)
        datetime_months = [datetime.datetime(
            2000, i, 1) for i in np.arange(1, 13)]
        labels = [i.strftime("%b") for i in datetime_months]
        ax = plt.gca()
        ax.set_xticks(np.arange(1, 13))
        ax.set_xticklabels(labels, rotation=45)
        # Add a legend
        plt.legend()
        # save plot
        AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
        if show_plot:
            plt.show()
        plt.close()

    # Add a title slide
    fig, ax = plt.subplots()
    plt.text(0.1, 1, 'Chance Mulit variant', fontsize=15)
    plt.setp(plt.gca(), frame_on=False, xticks=(), yticks=())
    AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
    plt.close()

    # now loop and plot up vs. all parameterisations parameterisation
    for ds in ds_seas:
        # get location/data on obs.
        md_df_tmp = md_df.loc[md_df[u'Data_Key'] == ds]
        Source = md_df_tmp['Source'].values[0].strip()
        Loc = md_df_tmp['Location'].values[0].strip()
        Cruise = md_df_tmp['Cruise'].values[0].strip()
        # get obs for dataset
        ds_tmp = df.loc[df[u'Data_Key'] == ds]
        ds_tmp.sort_values(by=MonthVar)
        # plot up against months
        plt.scatter(ds_tmp[MonthVar].values,  ds_tmp['Iodide'].values, label='Obs',
                    color='k')
        # Add values for en
        var2plot = [
            u'Chance2014_Multivariate'
        ]
        for var in var2plot:
            plt.scatter(ds_tmp[MonthVar].values,  ds_tmp[var].values,
                        label=rename_titles[var], color=colors_dict[var]
                        )
        #
        plt.title('{} ({},{})'.format(Source, Loc, Cruise))
        # Add labelleding for X axis
        plt.xlabel('Month')
        plt.xlim(0.5, 12.5)
        datetime_months = [datetime.datetime(
            2000, i, 1) for i in np.arange(1, 13)]
        labels = [i.strftime("%b") for i in datetime_months]
        ax = plt.gca()
        ax.set_xticks(np.arange(1, 13))
        ax.set_xticklabels(labels, rotation=45)
        # Add a legend
        plt.legend()
        # save plot
        AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
        if show_plot:
            plt.show()
        plt.close()

    # Save entire pdf
    AC.plot2pdfmulti(pdff, savetitle, close=True, dpi=dpi)
    plt.savefig(savetitle, dpi=dpi)
    if show_plot:
        plt.show()
    plt.close()

    # elephant


def test_model_sensitiivty2training_test_split(models2compare=None,
                                               models_dict=None):
    """ Driver to test/training set sensitivity for a set of models """
    # list of models to test?
    if isinstance(models2compare, type(None)):
        models2compare = ['RFR(TEMP+DEPTH+SAL)']
    # Get the unprocessed obs and variables as a DataFrame
    df = get_processed_df_obs_mod()  # NOTE this df contains values >400nM
    # Get variables to use for each model
    model_feature_dict = get_model_testing_features_dict(rtn_dict=True)
    # setup a DataFrame to store statistics
    dfs = {}
    # Now models and assess the sensitivity to the training/test set
    for model_name in models2compare:
        # Get the model to test
        model = models_dict[model_name]
        # Get the training features for the model
        training_features = model_feature_dict[model_name]
        # Run the analysis
        df = run_tests_on_testing_dataset_split(model_name=model_name,
                                                model=model, df=df,
                                                training_features=training_features)

    # Compare different bias for the difference test sets with existing params.

    # Plot these  as a line graph...


def analyse_model_selection_error_in_ensemble_members(RFR_dict=None,
                                                      rm_Skagerrak_data=False):
    """ Calculation of model selection bias """
    # --- Set local variables
    if rm_Skagerrak_data:
        extr_str = '_nSkagerrak'
    else:
        extr_str = ''
    # Get key data as a dictionary
    if isinstance(RFR_dict, type(None)):
        RFR_dict = build_or_get_current_models(
            rm_Skagerrak_data=rm_Skagerrak_data,
        )
    # select dataframe with observations and predictions in it
    df = RFR_dict['df']
    # Also make a dictionary
    testing_features_dict = RFR_dict['testing_features_dict']
    # Get the names of the ensemble members (topten models )
    if isinstance(topmodels, type(None)):
        topmodels = get_top_models(RFR_dict=RFR_dict, NO_DERIVED=True, n=10)

    # ---- Get data at observation points
    # add lines extract data from files...
#	filename = 'Oi_prj_models_built_stats_on_models_at_obs_points.csv'
#	folder ='../'
#	dfP = pd.read_csv( folder + filename )
#	dfP.index = dfP['Unnamed: 0'].values
 # Just select the
    df = RFR_dict['df']
    df = df.loc[df[testset] == True, :]
    # Get stats on model tuns runs
    dfP = get_stats_on_current_models(RFR_dict=RFR_dict, df=df,
                                      verbose=False)
    # only consider topmodels
    dfP = dfP.T[topmodels].T

    # ---- Get data at spatial points
    # add lines extract data from files...

    # just use outputted file for now.
    filename = 'Oi_prj_annual_stats_global_ocean_0.125x0.125.csv'
    folder = '../'
    dfG = pd.read_csv(folder + filename)
    dfG.index = dfG['Unnamed: 0'].values
    # only consider topmodels
    dfG = dfG.T[topmodels].T

    # - Set summary stats and print to a txt file
    file2save = 'Oi_prj_Error_calcs_model_selction{}{}.txt'
    a = open(file2save.format(res, extr_str), 'w')
    # ----  Calculate model selection error spatially
    print('---- Model choice affect spatially /n', file=a)
    # get stats
    var2use = u'mean (weighted)'
    min_ = dfG[var2use].min()
    max_ = dfG[var2use].max()
    mean_ = dfG[var2use].mean()
    range_ = max_ - min_
    # print out the range
    ptr_str = 'Model choice stats: min={:.5g}, max={:.5g} (range={:.5g})'
    print(ptr_str.format(min_, max_, range_), file=a)
    # print out the error
    ptr_str = 'Model choice affect on mean: {:.5g}-{:.5g} %'
    print(ptr_str.format(range_/max_*100, range_/min_*100), file=a)

    # ----  Calculate model selection error at observational points
    print('---- Model choice affect at point locations (mean) \n', file=a)
    # get stats
    var2use = u'mean'
    min_ = dfP[var2use].min()
    max_ = dfP[var2use].max()
    mean_ = dfP[var2use].mean()
    range_ = max_ - min_
    # print out the range
    ptr_str = 'Model choice stats: min={:.5g}, max={:.5g} (range={:.5g})'
    print(ptr_str.format(min_, max_, range_), file=a)
    # print out the error
    ptr_str = 'Model choice affect on mean: {:.5g}-{:.5g} %'
    print(ptr_str.format(range_/max_*100, range_/min_*100), file=a)
    # -  Now calculate for RMSE
    print('---- Model choice affect at point locations (mean) \n', file=a)
    # get stats
    var2use = 'RMSE (Test set (strat. 20%))'
    min_ = dfP[var2use].min()
    max_ = dfP[var2use].max()
    mean_ = dfP[var2use].mean()
    range_ = max_ - min_
    # print out the range
    ptr_str = 'Model choice stats: min={:.5g}, max={:.5g} (range={:.5g})'
    print(ptr_str.format(min_, max_, range_), file=a)
    # print out the error
    ptr_str = 'Model choice affect on mean: {:.5g}-{:.5g} %'
    print(ptr_str.format(range_/max_*100, range_/min_*100), file=a)
    # close the file
    a.close()


def analyse_dataset_error_in_ensemble_members(RFR_dict=None,
                                              rebuild_models=False, remake_NetCDFs=False,
                                              res='0.125x0.125',
                                              rm_Skagerrak_data=False, topmodels=None):
    """ Analyse the variation in spatial prediction on a per model basis """
    from multiprocessing import Pool
    from functools import partial
    # --- Set local variables
    if rm_Skagerrak_data:
        extr_str = '_nSkagerrak'
    else:
        extr_str = ''
    # Get key data as a dictionary
    if isinstance(RFR_dict, type(None)):
        RFR_dict = build_or_get_current_models(
            rm_Skagerrak_data=rm_Skagerrak_data
        )
    # select dataframe with observations and predictions in it
    df = RFR_dict['df']
    # Also make a dictionary
    testing_features_dict = RFR_dict['testing_features_dict']
    # Get the names of the ensemble members (topten models )
    if isinstance(topmodels, type(None)):
        topmodels = get_top_models(RFR_dict=RFR_dict, NO_DERIVED=True, n=10)
    # --- build 20 variable models for the ensemble memberse
    if rebuild_models:
        for model_name in topmodels:
            # get the training features for a given model
            testing_features = testing_features_dict[model_name]
            testing_features = testing_features.split('+')
            # Now build 20 separate initiations of the model
            build_the_same_model_mulitple_times(model_name=model_name,
                                                testing_features=testing_features, df=df,
                                                rm_Skagerrak_data=rm_Skagerrak_data
                                                )

    # --- Predict the surface concentrations for each of members' repeat builds
    if remake_NetCDFs:
        # Initialise pool to parrellise over
        p = Pool(len(topmodels))
        # Use Pool to concurrently process all predictions
        p.map(partial(mk_predictions_NetCDF_4_many_builds, res=res,
                      RFR_dict=RFR_dict,
                      rm_Skagerrak_data=rm_Skagerrak_data
                      ), topmodels)
        # close the pool
        p.close()

        # --------- Get data
    # Get the data for the variance in the prediction of members (spatially)
    # - Get stats on the model builds
    dfs = {}
    for model_name in topmodels:
        dfs[model_name] = get_stats_on_multiple_global_predictions(
            model_name=model_name, RFR_dict=RFR_dict, res=res,
            rm_Skagerrak_data=rm_Skagerrak_data,
        )
    # combine all the models
    dfG = pd.concat([dfs[i].T for i in dfs.keys()])
    save_str = 'Oi_prj_RAW_stats_on_ENSEMBLE_predictions_globally{}.csv'
    dfG.T.to_csv(save_str.format(extr_str))

    # - Get the data for variance in the prediction of members (point-for-point)
    # get predictions for repeat builds of ensemble members.
    dfs = {}
    for model_name in topmodels:
        # get the training features for a given model
        testing_features = testing_features_dict[model_name]
        testing_features = testing_features.split('+')
        # Now build 20 separate initiations of the model
        dfs[model_name] = get_stats4mulitple_model_builds(
            model_name=model_name,
            testing_features=testing_features, df=df,
            RFR_dict=RFR_dict
        )
    # concatenate into a single dataframe
    dfP = pd.concat([dfs[i] for i in dfs.keys()], axis=0)
    save_str = 'Oi_prj_RAW_stats_on_ENSEMBLE_predictions_at_obs_locs{}.csv'
    dfP.to_csv(save_str.format(extr_str))

    # --------- Do analysis
    # --- analyse the variance in the prediction of members (spatially)
    # - Get summary stats and print to a txt file
    file2save = 'Oi_prj_Error_calcs_dataset_selction{}{}.txt'
    a = open(file2save.format(res, extr_str), 'w')
    # Set a header
    print('This file contains analysis of the ensemble members', file=a)
    print('\n', file=a)
    # which files are being analysed?
    ptr_str = '----- Detail on variation within ensemble members for (20) builds'
    print(ptr_str, file=a)
    ptr_str = ' ----- ( 1st analysis is spatial - global stats )'
    print(ptr_str, file=a)
    # - what are the stats on individual model builds
    ranges4models, highest_means, lowest_means, means4models = {}, {}, {}, {}
    for model_ in topmodels:
        print('------- Analysis for: {}'.format(model_), file=a)
        # Select runs just for model
        df_tmp = dfG.T[[i for i in dfG.T.columns if model_ in i]] .T
        min_ = min(df_tmp['min'])
        max_ = max(df_tmp['max'])
        range_ = max_ - min_
        # print range for test_
        ptr_str = "range : {:.5g} ({:.5g}-{:.5g})"
        print(ptr_str.format(range_, min_, max_), file=a)
        # average value and range of this
        var2use = 'mean (weighted)'
        stats_on_var = df_tmp[var2use].describe()
        mean_ = stats_on_var['mean']
        min_mean_ = stats_on_var['min']
        max_mean_ = stats_on_var['max']
        range_ = max_mean_ - min_mean_
        lowest_means[model_] = min_mean_
        highest_means[model_] = max_mean_
        ranges4models[model_] = range_
        means4models[model_] = mean_
        ptr_str = "Avg. value of mean: {:.5g} ({:.5g} - {:.5g} )"
        print(ptr_str.format(mean_, min_mean_, max_mean_), file=a)
    # what is the total variance for the whole ensemble?
    print('\n', file=a)
    print('------- Analysis for all ensemble members ', file=a)
    dfA = pd.DataFrame({'mean range': [ranges4models[i] for i in topmodels]})
    dfA['mean4model'] = [means4models[i] for i in topmodels]
    dfA['min_mean4model'] = [lowest_means[i] for i in topmodels]
    dfA['max_mean4model'] = [highest_means[i] for i in topmodels]
    # add percent errors
    MerrPCmax = 'max % err. (mean)'
    dfA[MerrPCmax] = dfA['mean range'] / dfA['min_mean4model'] * 100
    MerrPCmin = 'min % err. (mean)'
    dfA[MerrPCmin] = dfA['mean range'] / dfA['max_mean4model'] * 100
    # use names for the index
    dfA.index = topmodels
    # save processed data
    save_str = 'Oi_prj_RAW_stats_on_ENSEMBLE_predictions_globally{}{}'
    dfA.to_csv(save_str.format(extr_str, '_PROCESSED', '.csv'))
    # print output to screen
    ptr_str = 'The avg. range in ensemble members is {:.5g} ({:.5g} - {:.5g} )'
    mean_ = dfA['mean range'].mean()
    min_ = dfA['mean range'].min()
    max_ = dfA['mean range'].max()
    # print the maximum range
    print(ptr_str.format(mean_, min_, max_), file=a)
    ptr_str = 'The max. range in ensemble is: {:.5g} '
    print(ptr_str.format(max_-min_), file=a)
    # print range in all ensemble members rebuilds (20*10)
    mean_ = dfG[var2use].mean()
    min_ = dfG[var2use].min()
    max_ = dfG[var2use].max()
    range_ = max_-min_
    ptr_str = 'The max. range in all ensemble member re-builds is: {:.5g} '
    print(ptr_str.format(range_), file=a)
    ptr_str = 'The avg. of all ensemble member re-builds is: '
    ptr_str += '{:.5g} ({:.5g} - {:.5g} )'
    print(ptr_str.format(mean_, min_, max_), file=a)
    # In percentage terms
    ptr_str = '- Percentage numbers: '
    print(ptr_str, file=a)
    ptr_str = 'Dataset split effect on mean: {:.5g}-{:.5g} %'
    Emax = max(highest_means.values())
    Emin = min(lowest_means.values())
    # for "dataset error"
    ptr_str = 'Dataset split effect on mean: {:.5g}-{:.5g} %'
    print(ptr_str.format(min_/Emax*100, max_/Emin*100), file=a)
#    Emin = dfA[ MerrPCmin ].min()
#    Emax = dfA[ MerrPCmax ].max()
#    print( ptr_str.format( Emin, Emax ),  file=a )

    # --- analyse the variance in the prediction of members (point-for-point)
    # - Get summary stats and print to a txt file
    # which files are being analysed?
    print('\n', file=a)
    ptr_str = '----- Detail on variation at observational points in testset'
    print(ptr_str, file=a)
    # - what are the stats on individual model builds
    ranges4models, highest_means, lowest_means, RMSE4models = {}, {}, {}, {}
    mean4models, RMSEranges4models, lowest_RMSE_, highest_RMSE_ = {}, {}, {}, {}
    for model_ in topmodels:
        print('------- Analysis for: {}'.format(model_), file=a)
        # Select runs just for model
        df_tmp = dfP.T[[i for i in dfP.T.columns if model_ in i]] .T
        min_ = min(df_tmp['min'])
        max_ = max(df_tmp['max'])
        range_ = max_ - min_
        # print range for test_
        ptr_str = "range : {:.5g} ({:.5g}-{:.5g})"
        print(ptr_str.format(range_, min_, max_), file=a)
        # average value and range of this
        mean_ = df_tmp['mean'].mean()
        min_mean_ = min(df_tmp['mean'])
        max_mean_ = max(df_tmp['mean'])
        range_ = max_mean_ - min_mean_
        ranges4models[model_] = range_
        ptr_str = "Avg. value of mean: {:.5g} ({:.5g} - {:.5g}, range={:.5g})"
        print(ptr_str.format(mean_, min_mean_, max_mean_, range_), file=a)
        # now print the RMSE and range of
        RMSE_ = df_tmp['RMSE'].mean()
        min_RMSE_ = df_tmp['RMSE'].min()
        max_RMSE_ = df_tmp['RMSE'].max()
        range_ = max_RMSE_ - min_RMSE_
        lowest_means[model_] = min_mean_
        highest_means[model_] = max_mean_
        lowest_RMSE_[model_] = min_RMSE_
        highest_RMSE_[model_] = max_RMSE_
        RMSEranges4models[model_] = range_
        RMSE4models[model_] = RMSE_
        mean4models[model_] = mean_
        ptr_str = "Avg. value of RMSE: {:.5g} ({:.5g} - {:.5g}, range={:.5g})"
        print(ptr_str.format(RMSE_, min_RMSE_, max_RMSE_, range_), file=a)
    # - Now consider the ensemble members en mass
    print('\n', file=a)
    ptr_str = '----- Summary overview'
    print(ptr_str, file=a)
    ptr_str = '- Actual numbers: '
    print(ptr_str, file=a)
    # setup a dataframe with stats for all models
    dfPbP = pd.DataFrame({'mean range': [ranges4models[i] for i in topmodels]})
    dfPbP['mean4model'] = [mean4models[i] for i in topmodels]
    dfPbP['min_mean4model'] = [lowest_means[i] for i in topmodels]
    dfPbP['max_mean4model'] = [highest_means[i] for i in topmodels]
    dfPbP['RMSE'] = [RMSE4models[i] for i in topmodels]
    dfPbP['RMSE range'] = [RMSEranges4models[i] for i in topmodels]
    dfPbP['min_RMSE4model'] = [lowest_RMSE_[i] for i in topmodels]
    dfPbP['max_RMSE4model'] = [highest_RMSE_[i] for i in topmodels]
    # add percent errors
    MerrPCmax = 'max % err. (mean)'
    dfPbP[MerrPCmax] = dfPbP['mean range'] / dfPbP['min_mean4model'] * 100
    MerrPCmin = 'min % err. (mean)'
    dfPbP[MerrPCmin] = dfPbP['mean range'] / dfPbP['max_mean4model'] * 100
    # add percent errors
    RerrPCmax = 'max % err. (RMSE)'
    dfPbP[RerrPCmax] = dfPbP['RMSE range'] / dfPbP['min_RMSE4model'] * 100
    RerrPCmin = 'min % err. (RMSE)'
    dfPbP[RerrPCmin] = dfPbP['RMSE range'] / dfPbP['max_RMSE4model'] * 100
    # update the index
    dfPbP.index = topmodels
    # save processed data
    save_str = 'Oi_prj_RAW_stats_on_ENSEMBLE_predictions_at_obs_locs{}{}'
    dfPbP.to_csv(save_str.format(extr_str, '_PROCESSED', '.csv'))
    # -- RMSE
    # effect of model choice on RMSE
    # don't calculate this here - just use the core model output in a tables
#     ptr_str = "Model choice effect on RMSE: "
#     ptr_str += "{:.5g} (mean={:.5g},range={:.5g} - {:.5g})"
#     mean_M = dfPbP['RMSE'].mean()
#     min_M = dfPbP['RMSE'].min()
#     max_M = dfPbP['RMSE'].max()
#     range_M = max_M - min_M
#     print( ptr_str.format( range_M, mean_M, min_M, max_M ), file=a )

    # effect of dataset split choice on RMSE
    ptr_str = "Dataset split choice effect on RMSE: "
    ptr_str += "{:.5g} (range={:.5g} - {:.5g}), "
#    ptr_str += "(factor better than model choice={:.5g} - {:.5g})"
    mean_ = dfPbP['RMSE range'].mean()
    min_ = dfPbP['RMSE range'].min()
    max_ = dfPbP['RMSE range'].max()
    range_ = max_ - min_
    print(ptr_str.format(mean_, min_, max_, ), file=a)
# min_/range_M, max_/range_M ),

    # Now print % numbers
    Emax = max(highest_RMSE_.values())
    Emin = min(lowest_RMSE_.values())
    ptr_str = '- Percentage numbers: '
    print(ptr_str, file=a)
    # for "dataset error" on RMSE
    ptr_str = 'Dataset split effect on RMSE: {:.5g}-{:.5g} %'
    print(ptr_str.format(min_/Emax*100, max_/Emin*100), file=a)
    print(ptr_str.format(Emin, Emax), file=a)
    # for "dataset error" on mean
    Emin = dfPbP[MerrPCmin].min()
    Emax = dfPbP[MerrPCmax].max()
    ptr_str = 'Dataset split on mean: {:.5g}-{:.5g} %'
    print(ptr_str.format(Emin, Emax), file=a)

    # -- Mean
    # effect of model choice on mean
    # don't calculate this here - just use the core model output in a tables
#     print('\n', file=a )
#     ptr_str = "Model choice effect on mean: "
#     ptr_str += "{:.5g} (mean={:.5g},range={:.5g} - {:.5g})"
#     mean_M = dfPbP['mean'].mean()
#     min_M = dfPbP['mean'].min()
#     max_M = dfPbP['mean'].max()
#     range_M = max_M - min_M
#     print( ptr_str.format( range_M, mean_M, min_M, max_M ), file=a )

    # effect of dataset split choice on mean
    ptr_str = "Dataset split choice effect on mean: "
    ptr_str += "{:.5g} (range={:.5g} - {:.5g}), "
#    ptr_str += "(factor better than model choice={:.5g} - {:.5g})"
    mean_ = dfPbP['mean range'].mean()
    min_ = dfPbP['mean range'].min()
    max_ = dfPbP['mean range'].max()
    range_ = max_ - min_
#    print( ptr_str.format( mean_, min_, max_,  min_/range_M, max_/range_M ),
#        file=a )
    print(ptr_str.format(mean_, min_, max_,), file=a)

    # Now print % numbers
    ptr_str = '- Percentage numbers: '
    print(ptr_str, file=a)
    Emax = max(highest_means.values())
    Emin = min(lowest_means.values())
    # for "dataset error"
    ptr_str = 'Dataset split effect on mean: {:.5g}-{:.5g} %'
    print(ptr_str.format(min_/Emax*100, max_/Emin*100),  file=a)
    # for "model error"

    # close the file
    a.close()

def plot_ODR_window_plot(RFR_dict=None, show_plot=False, df=None,
                         testset='Test set (strat. 20%)', target_name='Iodide',
                         target='Iodide', context="paper", dpi=720):
    """ Show the correlations between obs. and params. as window plot """
    if isinstance(RFR_dict, type(None)):
        RFR_dict = build_or_get_current_models()
    # select dataframe with observations and predictions in it
    if isinstance(RFR_dict, type(None)):
        testing_features_dict = RFR_dict['testing_features_dict']
        models_dict = RFR_dict['models_dict']
    if isinstance(df, type(None)):
        df = RFR_dict['df']

    # --- Evaluate model using various approaches
    import seaborn as sns
    sns.set(color_codes=True)
    if context == "paper":
        sns.set_context("paper")
    else:
        sns.set_context("talk", font_scale=1.0)

    # - Evaluate point for point
    savetitle = 'Oi_prj_point_for_point_comparison_obs_vs_model_ODR_WINDOW'
    pdff = AC.plot2pdfmulti(title=savetitle, open=True, dpi=dpi)
    # models to compare
    models2compare = [
        # Earlier well performing models
        #    'RFR(TEMP+SAL+Prod)',
        #    'RFR(TEMP)',
        # Later well performing models
        #    'RFR(TEMP+DEPTH+SAL+NO3+DOC)', 'RFR(TEMP+SAL+NO3)',
        #    'RFR(TEMP+DOC+Phos)', 'RFR(TEMP+SWrad+NO3+MLD+SAL)',
        #    'RFR(TEMP+SAL+NO3)',
        #    'RFR(TEMP+DEPTH+SAL)',
        'RFR(Ensemble)',
    ]
    # rename titles
    rename_titles = {u'Chance2014_STTxx2_I': 'Chance et al. (2014)',
                     u'MacDonald2014_iodide': 'MacDonald et al. (2014)',
                     'RFR(Ensemble)': 'RFR(Ensemble)', }
    # units
    units = 'nM'
    # iodide in aq
    Iaq = '[I$^{-}_{aq}$]'
    # also compare existing parameters
    params = [u'Chance2014_STTxx2_I', u'MacDonald2014_iodide', ]
    # set location for alt_text
    f_size = 10
    N = int(df.shape[0])
    # split data into groups
    dfs = {}
    # Entire dataset
    dfs['Entire'] = df.copy()
    # Testdataset
    dfs['Withheld'] = df.loc[df[testset] == True, :].copy()
    # Coastal testdataset
    df_tmp = df.loc[df['Coastal'] == 1, :]
    dfs['Withheld coastal'] = df_tmp.loc[df_tmp[testset] == True, :].copy()
    # Non-coastal testdataset
    df_tmp = df.loc[df['Coastal'] == 0, :]
    dfs['Withheld non-coastal'] = df_tmp.loc[df_tmp[testset] == True, :].copy()
    # keep order by setting it here
    dsplits = [
        'Entire', 'Withheld', 'Withheld coastal', 'Withheld non-coastal'
    ]
    # assign colors
    CB_color_cycle = AC.get_CB_color_cycle()
    color_d = dict(zip(dsplits, CB_color_cycle))
    # params to plot
    params2plot = ['RFR(Ensemble)'] + params
    # Intialise figure and axis
    fig, axs = plt.subplots(1, 3, sharex=True, sharey=True, dpi=dpi, \
                            #        figsize=(12, 5)
                            figsize=(11, 4)
                            )
    # Loop by param and compare against whole dataset
    for n_param, param in enumerate(params2plot):
        #        fig = plt.figure()
        # set axis to use
        ax = axs[n_param]
        # Use the same asecpt for X and Y
        ax.set_aspect('equal')
        # title the plots
#        plt.title( rename_titles[param] )
        ax.text(0.5, 1.05, rename_titles[param], horizontalalignment='center',
                verticalalignment='center', transform=ax.transAxes)
#        ax = fig.add_subplot(1, 3, n_param+1 )
        # add a 1:1 line
        x_121 = np.arange(-50, 500)
        ax.plot(x_121, x_121, alpha=0.5, color='k', ls='--')
        # Plot up data by dataset split
        for nsplit, split in enumerate(dsplits):
            #
            df = dfs[split].copy()
            # get X
            X = df['Iodide'].values
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
        # Beautify
        plt.xlim(-10, 410)
        plt.ylim(-10, 410)
        ax.set_xlabel('Obs. {} ({})'.format(Iaq, units))
        if (n_param == 0):
            ax.set_ylabel('Parameterised {} ({})'.format(Iaq, units))
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
    # Save plot
    AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
    # Save entire pdf
    AC.plot2pdfmulti(pdff, savetitle, close=True, dpi=dpi)
    plt.savefig(savetitle, dpi=dpi)
    if show_plot:
        plt.show()
    plt.close()


def analyse_X_Y_correlations_ODR(RFR_dict=None, show_plot=False,
                                 testset='Test set (strat. 20%)', target_name='Iodide',
                                 target='Iodide', context="paper", dpi=320):
    """ Analyse the correlations between obs. and params. """
    # --- Get data
    if isinstance(RFR_dict, type(None)):
        RFR_dict = build_or_get_current_models()
    # select dataframe with observations and predictions in it
    df = RFR_dict['df']
    testing_features_dict = RFR_dict['testing_features_dict']
    models_dict = RFR_dict['models_dict']
    # get stats on models in RFR_dict
#    stats = get_stats_on_current_models( RFR_dict=RFR_dict, verbose=False )

    # --- Evaluate model using various approaches
    import seaborn as sns
    sns.set(color_codes=True)
    sns.set_context(context)
    # - Evaluate point for point
    savetitle = 'Oi_prj_point_for_point_comparison_obs_vs_model_ODR'
    pdff = AC.plot2pdfmulti(title=savetitle, open=True, dpi=dpi)
    # models to compare
    models2compare = [
        # Earlier well performing models
        #    'RFR(TEMP+SAL+Prod)',
        #    'RFR(TEMP)',
        # Later well performing models
        #    'RFR(TEMP+DEPTH+SAL+NO3+DOC)', 'RFR(TEMP+SAL+NO3)',
        #    'RFR(TEMP+DOC+Phos)', 'RFR(TEMP+SWrad+NO3+MLD+SAL)',
        #    'RFR(TEMP+SAL+NO3)',
        #    'RFR(TEMP+DEPTH+SAL)',
        'RFR(Ensemble)',
    ]
    # units
    units = 'nM'
    # iodide in aq
    Iaq = '[I$^{-}_{aq}$]'
    # also compare existing parameters
    params = [u'Chance2014_STTxx2_I', u'MacDonald2014_iodide', ]
    # set location for alt_text
    f_size = 10
    N = int(df.shape[0])
    # split data into groups
    dfs = {}
    # Entire dataset
    dfs['Entire'] = df.copy()
    # Testdataset
    dfs['Withheld'] = df.loc[df[testset] == True, :].copy()
    # Coastal testdataset
    df_tmp = df.loc[df['Coastal'] == 1, :]
    dfs['Coastal (withheld)'] = df_tmp.loc[df_tmp[testset] == True, :].copy()
    # Non-coastal testdataset
    df_tmp = df.loc[df['Coastal'] == 0, :]
    dfs['Non-coastal (withheld)'] = df_tmp.loc[df_tmp[testset]
                                               == True, :].copy()
    # keep order by setting it here
    dsplits = [
        'Entire', 'Withheld', 'Coastal (withheld)', 'Non-coastal (withheld)'
    ]
    # assign colors
    CB_color_cycle = AC.get_CB_color_cycle()
    color_d = dict(zip(dsplits, CB_color_cycle))
    # Loop by param and compare against whole dataset
    for param in models2compare + params:
        # Intialise figure and axis
        fig, ax = plt.subplots()
        # add a 1:1 line
        x_121 = np.arange(-50, 500)
        plt.plot(x_121, x_121, alpha=0.5, color='k', ls='--')
        #
        # Plot up data by dataset split
        for nsplit, split in enumerate(dsplits):
            df = dfs[split].copy()
            # get X
            X = df['Iodide'].values
            # get Y
            Y = df[param].values
            # get N
            N = float('{:.3g}'.format(df.shape[0]))
            # get RMSE
            RMSE = np.sqrt(((Y-X)**2).mean())
            # Plot up just the entire and testset data
            if split in ('Entire', 'Withheld'):
                plt.scatter(X, Y, color=color_d[split], s=3, facecolor='none')
            # add ODR line
            xvalues, Y_ODR = AC.get_linear_ODR(x=X, y=Y, xvalues=x_121,
                                               return_model=False,
                                               maxit=10000)
            plt.plot(xvalues, Y_ODR, color=color_d[split])
            # Add RMSE ( and N value as alt text )
            alt_text_x = 0.05
            alt_text_y = 0.95-(0.05*nsplit)
            alt_text = 'RMSE={:.1f} ({}, N={:.0f})'.format(RMSE, split, N)
            ax.annotate(alt_text, xy=(alt_text_x, alt_text_y),
                        textcoords='axes fraction', fontsize=f_size,
                        color=color_d[split])
        # Beautify
        plt.xlim(-10, 410)
        plt.ylim(-10, 410)
        #
        plt.ylabel('{} {} ({})'.format(param, Iaq, units))
        plt.xlabel('Obs. {} ({})'.format(Iaq, units))
#        plt.title('Obs. vs param. for entire dataset')
        # save plot
        AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
        if show_plot:
            plt.show()
        plt.close()
    # Save entire pdf
    AC.plot2pdfmulti(pdff, savetitle, close=True, dpi=dpi)


def analyse_X_Y_correlations(RFR_dict=None, show_plot=False,
                             testset='Test set (strat. 20%)', target_name='Iodide',
                             target='Iodide', dpi=320):
    """ Analyse the correlations between obs. and params. """
    # --- Get data
    if isinstance(RFR_dict, type(None)):
        RFR_dict = build_or_get_current_models()
    # select dataframe with observations and predictions in it
    df = RFR_dict['df']
    testing_features_dict = RFR_dict['testing_features_dict']
    models_dict = RFR_dict['models_dict']
    # get stats on models in RFR_dict
    stats = get_stats_on_current_models(RFR_dict=RFR_dict, verbose=False)

    # --- Evaluate model using various approaches
    import seaborn as sns
    sns.set(color_codes=True)
    sns.set_context("paper")
    # - Evaluate point for point
    savetitle = 'Oi_prj_point_for_point_comparison_obs_vs_model'
    pdff = AC.plot2pdfmulti(title=savetitle, open=True, dpi=dpi)
    # plot up X vs. Y for all points

    # models to compare
    models2compare = [
        'RFR(TEMP+DEPTH+SAL+NO3+DOC)', 'RFR(TEMP+DEPTH+SAL+NO3)',
        'RFR(TEMP+DEPTH+SAL)', 'RFR(TEMP+SAL+Prod)',
        #    'RFR(TEMP+SAL+NO3)',
        #    'RFR(TEMP+DEPTH+SAL)',
    ]
    # also compare existing parameters
    params = [u'Chance2014_STTxx2_I', u'MacDonald2014_iodide', ]
    # set location for alt_text
    alt_text_x = 0.1
    alt_text_y = 0.875
    alt_text_str = 'RMSE={:.2f} (N={})'
    f_size = 10
    N = df.shape[0]
    # Loop by param and compare against whole dataset
    for param in models2compare + params:
        # Intialise figure and axis
        fig, ax = plt.subplots()
        # add a 1:1 line
        x_121 = np.arange(-50, 500)
        plt.plot(x_121, x_121, alpha=0.5, color='k', ls='--')
        # Plot up data
        sns.regplot(x='Iodide', y=param, data=df)
        # beautify
        plt.xlim(-10, 410)
        plt.ylim(-10, 410)
        plt.title('Obs. vs param. for entire dataset')
        # TODO (inc. RMSE on plot)
        stats_tmp = stats.loc[stats.index == param]
        stat_var = 'RMSE (all)'
        alt_text = alt_text_str.format(stats_tmp[stat_var][0], N)
        ax.annotate(alt_text, xy=(alt_text_x, alt_text_y),
                    textcoords='axes fraction', fontsize=f_size*1.5)
        # save plot
        AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
        if show_plot:
            plt.show()
        plt.close()

    # Loop by param and compare against test dataset
    df_tmp = df.loc[df[testset] == True, :]
    N = df_tmp.shape[0]
    for param in models2compare + params:
        # Intialise figure and axis
        fig, ax = plt.subplots()
        # add a 1:1 line
        x_121 = np.arange(-50, 500)
        plt.plot(x_121, x_121, alpha=0.5, color='k', ls='--')
        # Plot up data
        sns.regplot(x='Iodide', y=param, data=df_tmp)
        # beautify
        plt.xlim(-10, 410)
        plt.ylim(-10, 410)
        plt.title('Obs. vs param. for test dataset')
        # TODO (inc. RMSE on plot)
        stats_tmp = stats.loc[stats.index == param]
        stat_var = 'RMSE ({})'.format(testset)
        alt_text = alt_text_str.format(stats_tmp[stat_var][0], N)
        ax.annotate(alt_text, xy=(alt_text_x, alt_text_y),
                    textcoords='axes fraction', fontsize=f_size*1.5)
        # save plot
        AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
        if show_plot:
            plt.show()
        plt.close()

    # Loop by param and compare against test dataset that is coastal
    df_tmp = df.loc[df[testset] == True, :]
    df_tmp = df_tmp.loc[df_tmp['Coastal'] == 1, :]
    N = df_tmp.shape[0]
    for param in models2compare + params:
        # Intialise figure and axis
        fig, ax = plt.subplots()
        # add a 1:1 line
        x_121 = np.arange(-50, 500)
        plt.plot(x_121, x_121, alpha=0.5, color='k', ls='--')
        # Plot up data
        sns.regplot(x='Iodide', y=param, data=df_tmp)
        # beautify
        plt.xlim(-10, 410)
        plt.ylim(-10, 410)
        plt.title('Obs. vs param. for coastal test dataset')
        # TODO (inc. RMSE on plot)
        stats_tmp = stats.loc[stats.index == param]
        stat_var = 'RMSE (Coastal ({}))'.format(testset)
        alt_text = alt_text_str.format(stats_tmp[stat_var][0], N)
        ax.annotate(alt_text, xy=(alt_text_x, alt_text_y),
                    textcoords='axes fraction', fontsize=f_size*1.5)
        # save plot
        AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
        if show_plot:
            plt.show()
        plt.close()

    # Loop by param and compare against test dataset that is coastal
    df_tmp = df.loc[df[testset] == True, :]
    df_tmp = df_tmp.loc[df_tmp['Coastal'] == 0, :]
    N = df_tmp.shape[0]
    for param in models2compare + params:
        # Intialise figure and axis
        fig, ax = plt.subplots()
        # add a 1:1 line
        x_121 = np.arange(-50, 500)
        plt.plot(x_121, x_121, alpha=0.5, color='k', ls='--')
        # Plot up data
        sns.regplot(x='Iodide', y=param, data=df_tmp)
        # beautify
        plt.xlim(-10, 410)
        plt.ylim(-10, 410)
        plt.title('Obs. vs param. for non-coastal test dataset')
        # TODO (inc. RMSE on plot)
        stats_tmp = stats.loc[stats.index == param]
        stat_var = 'RMSE (Non coastal ({}))'.format(testset)
        alt_text = alt_text_str.format(stats_tmp[stat_var][0], N)
        ax.annotate(alt_text, xy=(alt_text_x, alt_text_y),
                    textcoords='axes fraction', fontsize=f_size*1.5)
        # save plot
        AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
        if show_plot:
            plt.show()
        plt.close()

    # - Now look at residuals
    # get residual
    for param in models2compare + params:
        df[param+'-residual'] = df[param] - df['Iodide']

    # Loop by param and compare against whole dataset ( as residuals )
    for param in models2compare + params:
        # Intialise figure and axis
        fig, ax = plt.subplots()
        # add a 1:1 line
        x_121 = np.arange(-50, 500)
        plt.plot(x_121, [0]*len(x_121), alpha=0.5, color='k', ls='--')
        # Plot up data
        sns.regplot(x='Iodide', y=param+'-residual', data=df)
        # beautify
        plt.xlim(-10, 410)
        plt.ylim(-150, 150)
        plt.title('Residual (param.-Obs.) for entire dataset')
        # TODO (inc. RMSE on plot)
        stats_tmp = stats.loc[stats.index == param]
        stat_var = 'RMSE (all)'
        alt_text = alt_text_str.format(stats_tmp[stat_var][0], N)
        ax.annotate(alt_text, xy=(alt_text_x, alt_text_y),
                    textcoords='axes fraction', fontsize=f_size*1.5)
        # save plot
        AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
        if show_plot:
            plt.show()
        plt.close()

    # Loop by param and compare against test dataset ( as residuals )
    df_tmp = df.loc[df[testset] == True, :]
    N = df_tmp.shape[0]
    for param in models2compare + params:
        # Intialise figure and axis
        fig, ax = plt.subplots()
        # add a 1:1 line
        x_121 = np.arange(-50, 500)
        plt.plot(x_121, [0]*len(x_121), alpha=0.5, color='k', ls='--')
        # Plot up data
        sns.regplot(x='Iodide', y=param+'-residual', data=df_tmp)
        # beautify
        plt.xlim(-10, 410)
        plt.ylim(-150, 150)
        plt.title('Residual (param.-Obs.) for test dataset')
        # TODO (inc. RMSE on plot)
        stats_tmp = stats.loc[stats.index == param]
        stat_var = 'RMSE ({})'.format(testset)
        alt_text = alt_text_str.format(stats_tmp[stat_var][0], N)
        ax.annotate(alt_text, xy=(alt_text_x, alt_text_y),
                    textcoords='axes fraction', fontsize=f_size*1.5)
        # save plot
        AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
        if show_plot:
            plt.show()
        plt.close()

    # Loop by param and compare against test dataset that is coastal
    df_tmp = df.loc[df[testset] == True, :]
    df_tmp = df_tmp.loc[df_tmp['Coastal'] == 1, :]
    N = df_tmp.shape[0]
    for param in models2compare + params:
        # Intialise figure and axis
        fig, ax = plt.subplots()
        # add a 1:1 line
        x_121 = np.arange(-50, 500)
        plt.plot(x_121, [0]*len(x_121), alpha=0.5, color='k', ls='--')
        # Plot up data
        sns.regplot(x='Iodide', y=param+'-residual', data=df_tmp)
        # beautify
        plt.xlim(-10, 410)
        plt.ylim(-150, 150)
        plt.title('Residual (param.-Obs.) for coastal test dataset')
        # TODO (inc. RMSE on plot)
        stats_tmp = stats.loc[stats.index == param]
        stat_var = 'RMSE (Coastal ({}))'.format(testset)
        alt_text = alt_text_str.format(stats_tmp[stat_var][0], N)
        ax.annotate(alt_text, xy=(alt_text_x, alt_text_y),
                    textcoords='axes fraction', fontsize=f_size*1.5)
        # save plot
        AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
        if show_plot:
            plt.show()
        plt.close()

    # Loop by param and compare against test dataset that is coastal
    df_tmp = df.loc[df[testset] == True, :]
    df_tmp = df_tmp.loc[df_tmp['Coastal'] == 0, :]
    N = df_tmp.shape[0]
    for param in models2compare + params:
        # Intialise figure and axis
        fig, ax = plt.subplots()
        # add a 1:1 line
        x_121 = np.arange(-50, 500)
        plt.plot(x_121, [0]*len(x_121), alpha=0.5, color='k', ls='--')
        # Plot up data
        sns.regplot(x='Iodide', y=param+'-residual', data=df_tmp)
        # beautify
        plt.xlim(-10, 410)
        plt.ylim(-150, 150)
        plt.title('Residual (param.-Obs.) for non-coastal test dataset')
        # TODO (inc. RMSE on plot)
        stats_tmp = stats.loc[stats.index == param]
        stat_var = 'RMSE (Non coastal ({}))'.format(testset)
        alt_text = alt_text_str.format(stats_tmp[stat_var][0], N)
        ax.annotate(alt_text, xy=(alt_text_x, alt_text_y),
                    textcoords='axes fraction', fontsize=f_size*1.5)
        # save plot
        AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
        if show_plot:
            plt.show()
        plt.close()

    # Save entire pdf
    AC.plot2pdfmulti(pdff, savetitle, close=True, dpi=dpi)


def calculate_biases_in_predictions(testset='Test set (strat. 20%)',
                                    target_name='Iodide'):
    """ """
    # Get data
    if isinstance(df, type(None)):
        RFR_dict = build_or_get_current_models()
        df = RFR_dict['df']
    # Select parameterisations
    models2compare = ['RFR(Ensemble)']
    params = [u'Chance2014_STTxx2_I', u'MacDonald2014_iodide', ]
    params2plot = models2compare + params
    # --- Calculate bias for all params
    dfs = {}
    for param in params2plot:
        dfs[param] = df[param]-df[target_name]
    # Make new dataframe with params as columns
    dfNEW = pd.DataFrame([dfs[i] for i in params2plot]).T
    dfNEW.columns = params2plot
    dfNEW[testset] = df[testset]
    dfNEW['Coastal'] = df['Coastal']
    df = dfNEW

    # --- Split dataset by testset, training, all, etc...
    dfs = {}
    # Entire dataset
    dfs['Entire'] = df.copy()
    # Testdataset
    dfs['Withheld'] = df.loc[df[testset] == True, :].copy()
    # Coastal testdataset
    df_tmp = df.loc[df['Coastal'] == 1, :]
    dfs['Coastal (withheld)'] = df_tmp.loc[df_tmp[testset] == True, :].copy()
    # Non-coastal testdataset
    df_tmp = df.loc[df['Coastal'] == 0, :]
    dfs['Non-coastal (withheld)'] = df_tmp.loc[df_tmp[testset]
                                               == True, :].copy()
    # maintain ordering of plotting
    datasets = dfs.keys()

    # --- Loop by data and write out stats
    print(" -----  -----  -----  -----  ----- ")
    # write out by dataset and by param
    dataset_stats = {}
    for dataset in datasets:
        # print header
        ptr_str = " ----- For dataset split '{}' ----- "
        print(ptr_str.format(dataset))
        # get data
        df = dfs[dataset]
        # Loop by param, print values and save overalls
        param_stats = {}
        for param in params2plot:
            # get stats on dataset biases
            stats = df[param].describe().to_dict()
            #  Now print these
            pstr2 = '{:<25} - mean={mean:.2f}, median={50%:.2f} '
            pstr2 += '(min={min:.2f}, max={max:.2f}), std={std:.2f}'
            print(pstr2.format(param, **stats))
            #
            param_stats[param] = stats
        # Print summary
        stats = pd.DataFrame(param_stats).T
        dataset_stats[dataset] = stats
    # Write out summary of dataset
    print(" -----  -----  -----  -----  ----- ")
    ptr_str = " ----- Summary ----- "
    print(ptr_str)
    for dataset in datasets:
        # Get stats
        stats = dataset_stats[dataset]
        # header for dataset
        pstr3 = ' --- for dataset: {:<25}'
        print(pstr3.format(dataset))

        # find out which param has the largest min bias
        min_vals = stats['min'].sort_values(ascending=True)
        min_var = min_vals.head(1).index[0]
        min_var2 = min_vals.head(2).index[1]
        min_val = min_vals.head(1).values[0]
        min_val2 = min_vals.head(2).values[1]
        pstr4 = 'largest min bias: {:<22} ({:.2f}) - 2nd min: {:<20} ({:.2f})'
        print(pstr4.format(min_var, min_val, min_var2, min_val2))

        # find out which param has the largest max bias
        max_vals = stats['max'].sort_values(ascending=False)
        max_var = max_vals.head(1).index[0]
        max_var2 = max_vals.head(2).index[1]
        max_val = max_vals.head(1).values[0]
        max_val2 = max_vals.head(2).values[1]
        print(pstr4.format(max_var, max_val, max_var2, max_val2))


def plot_up_CDF_and_PDF_of_obs_and_predictions(show_plot=False,
                                               testset='Test set (strat. 20%)',
                                               target_name='Iodide',
                                               target='Iodide', df=None,
                                               plot_up_CDF=False, dpi=320):
    """ Plot up CDF and PDF plots to explore point-vs-point data """
    import seaborn as sns
    sns.set(color_codes=True)
    sns.set_context("paper", font_scale=0.75)
    # show_plot=False; testset='Test set (strat. 20%)'; target_name='Iodide'; target='Iodide'
    # Get data
    if isinstance(df, type(None)):
        RFR_dict = build_or_get_current_models()
        df = RFR_dict['df']
    # Get a dictionary of different dataset splits
    dfs = {}
    #
    # Entire dataset
    dfs['Entire'] = df.copy()
    # Testdataset
    dfs['Withheld'] = df.loc[df[testset] == True, :].copy()
    # Coastal testdataset
    df_tmp = df.loc[df['Coastal'] == 1, :]
    dfs['Coastal (withheld)'] = df_tmp.loc[df_tmp[testset] == True, :].copy()
    # Non-coastal testdataset
    df_tmp = df.loc[df['Coastal'] == 0, :]
    dfs['Non-coastal (withheld)'] = df_tmp.loc[df_tmp[testset]
                                               == True, :].copy()
    # maintain ordering of plotting
    datasets = dfs.keys()

    # models2compare
    models2compare = [
        'RFR(TEMP+DEPTH+SAL+NO3+DOC)', 'RFR(TEMP+SAL+NO3)',
        'RFR(TEMP+DOC+Phos)', 'RFR(TEMP+SWrad+NO3+MLD+SAL)',
        #    'RFR(TEMP+SAL+Prod)',
        #    'RFR(TEMP)',
        'RFR(TEMP+SAL+NO3)',
        'RFR(TEMP+DEPTH+SAL)',
    ]
    # params
    params = [u'Chance2014_STTxx2_I', u'MacDonald2014_iodide', ]
    params2plot = models2compare + params
    # setup color dictionary
    CB_color_cycle = AC.get_CB_color_cycle()
    color_d = dict(zip(params2plot, CB_color_cycle))
    # plotting variables
    axlabel = '[I$^{-}_{aq}$] (nM)'
    # set a PDF to save data to
    savetitle = 'Oi_prj_point_for_point_comparison_PDF'
    if plot_up_CDF:
        savetitle += '_CDF'
    pdff = AC.plot2pdfmulti(title=savetitle, open=True, dpi=dpi)
    # --- Plot up CDF and PDF plots for the dataset and residuals
    for dataset in datasets:
        # get data
        df = dfs[dataset]

        # - Plot up PDF plots for the dataset
        # plot observations
        var_ = 'Obs.'
        obs_arr = df[target_name].values
        ax = sns.distplot(obs_arr, axlabel=axlabel, label=var_,
                          color='k',)
        # loop and plot model values
        for param in params2plot:
            arr = df[param].values
            ax = sns.distplot(arr, axlabel=axlabel, label=param,
                              color=color_d[param], ax=ax)
        # force y axis extend to be correct
        ax.autoscale()
        # Beautify
        title = 'PDF of {} data ({}) at obs. locations'
        plt.title(title.format(dataset, axlabel))
        plt.legend()
        # Save to PDF and close plot
        AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
        plt.close()

        # - Plot up CDF plots for the dataset
        # plot observations
        if plot_up_CDF:
            var_ = 'Obs.'
            obs_arr = df[target_name].values
            ax = sns.distplot(arr, axlabel=axlabel, label=var_, color='k',
                              hist_kws=dict(cumulative=True),
                              kde_kws=dict(cumulative=True))
            # loop and plot model values
            for param in params2plot:
                arr = df[param].values
                ax = sns.distplot(arr, axlabel=axlabel, label=param,
                                  color=color_d[param], ax=ax,
                                  hist_kws=dict(cumulative=True),
                                  kde_kws=dict(cumulative=True))
            # force y axis extend to be correct
            ax.autoscale()
            # Beautify
            title = 'CDF of {} data ({}) at obs. locations'
            plt.title(title.format(dataset, axlabel))
            plt.legend()
            # Save to PDF and close plot
            AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
            plt.close()

        # - Plot up PDF plots for the residual dataset
        # get observations
        obs_arr = df[target_name].values
        fig, ax = plt.subplots()
        # loop and plot model values
        for param in params2plot:
            arr = df[param].values - obs_arr
            ax = sns.distplot(arr, axlabel=axlabel, label=param,
                              color=color_d[param], ax=ax)
        # force y axis extend to be correct
        ax.autoscale()
        # Beautify
        title = 'PDF of residual in {} data ({}) at obs. locations'
        plt.title(title.format(dataset, axlabel))
        plt.legend()
        # Save to PDF and close plot
        AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
        plt.close()

        # - Plot up CDF plots for the residual  dataset
        if plot_up_CDF:
            # plot observations
            obs_arr = df[target_name].values
            fig, ax = plt.subplots()
            # loop and plot model values
            for param in params2plot:
                arr = df[param].values - obs_arr
                ax = sns.distplot(arr, axlabel=axlabel, label=param,
                                  color=color_d[param], ax=ax,
                                  hist_kws=dict(cumulative=True),
                                  kde_kws=dict(cumulative=True))
            # force y axis extend to be correct
            ax.autoscale()
            # Beautify
            title = 'CDF of residual in {} data ({}) at obs. locations'
            plt.title(title.format(dataset, axlabel))
            plt.legend()
            # Save to PDF and close plot
            AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
            plt.close()

    # Save entire pdf
    AC.plot2pdfmulti(pdff, savetitle, close=True, dpi=dpi)


def plot_up_PDF_of_obs_and_predictions_WINDOW(show_plot=False,
                                              testset='Test set (strat. 20%)',
                                              target_name='Iodide',
                                              target='Iodide', df=None, plot_up_CDF=False,
                                              dpi=320):
    """ Plot up CDF and PDF plots to explore point-vs-point data """
    import seaborn as sns
    sns.set(color_codes=True)
    sns.set_context("paper", font_scale=0.75)
    # show_plot=False; testset='Test set (strat. 20%)'; target_name='Iodide'; target='Iodide'
    # Get data
    if isinstance(df, type(None)):
        RFR_dict = build_or_get_current_models()
        df = RFR_dict['df']
    # Get a dictionary of different dataset splits
    dfs = {}
    #
    # Entire dataset
#    dfs['Entire'] = df.copy()
    # Testdataset
    dfs['All (withheld)'] = df.loc[df[testset] == True, :].copy()
    # Coastal testdataset
    df_tmp = df.loc[df['Coastal'] == 1, :]
    dfs['Coastal (withheld)'] = df_tmp.loc[df_tmp[testset] == True, :].copy()
    # Non-coastal testdataset
    df_tmp = df.loc[df['Coastal'] == 0, :]
    dfs['Non-coastal (withheld)'] = df_tmp.loc[df_tmp[testset]
                                               == True, :].copy()
    # maintain ordering of plotting
    datasets = dfs.keys()
    # models2compare
    models2compare = ['RFR(Ensemble)']
    # params
    params = [u'Chance2014_STTxx2_I', u'MacDonald2014_iodide', ]
    params2plot = models2compare + params
    # titles to rename plots with
    rename_titles = {u'Chance2014_STTxx2_I': 'Chance et al. (2014)',
                     u'MacDonald2014_iodide': 'MacDonald et al. (2014)',
                     'RFR(Ensemble)': 'RFR(Ensemble)', }
    # setup color dictionary
    CB_color_cycle = AC.get_CB_color_cycle()
    color_d = dict(zip(params2plot, CB_color_cycle))
    # plotting variables
    axlabel = '[I$^{-}_{aq}$] (nM)'
    # limit of PDF plots of acutal data?
    xlim_iodide = -50
    xlim_iodide = 0
    # set a PDF to save data to
    savetitle = 'Oi_prj_point_for_point_comparison_obs_vs_model_PDF_WINDOW'
    # --- Plot up CDF and PDF plots for the dataset and residuals
    fig = plt.figure(dpi=dpi)
    for n_dataset, dataset in enumerate(datasets):
        # set Axis for abosulte PDF
        ax1 = fig.add_subplot(3, 2, [1, 3, 5][n_dataset])
        # get data
        df = dfs[dataset]
        N_ = df.shape
        print(dataset, N_)
        # - Plot up PDF plots for the dataset
        # plot observations
        var_ = 'Obs.'
        obs_arr = df[target_name].values
        ax = sns.distplot(obs_arr, axlabel=axlabel, label=var_,
                          color='k', ax=ax1)
        # loop and plot model values
        for param in params2plot:
            arr = df[param].values
            ax = sns.distplot(arr, axlabel=axlabel,
                              label=rename_titles[param],
                              color=color_d[param], ax=ax1)
        # force y axis extent to be correct
        ax1.autoscale()
        # force x axis to be constant
        ax1.set_xlim(xlim_iodide, 420)
        # Beautify
        ylabel = 'Frequency \n ({})'
        ax1.set_ylabel(ylabel.format(dataset))
        # Add legend to first plot
        if (n_dataset == 0):
            plt.legend()
            ax1.set_title('Concentration')
        # - Plot up PDF plots for the residual dataset
        # set Axis for abosulte PDF
        ax2 = fig.add_subplot(3, 2, [2, 4, 6][n_dataset])
        # get observations
        obs_arr = df[target_name].values
        # loop and plot model values
        for param in params2plot:
            arr = df[param].values - obs_arr
            ax = sns.distplot(arr, axlabel=axlabel,
                              label=rename_titles[param],
                              color=color_d[param], ax=ax2)
        # force y axis extent to be correct
        ax2.autoscale()
        # force x axis to be constant
        ax2.set_xlim(-320, 220)
        # Add legend to first plot
        if (n_dataset == 0):
            ax2.set_title('Bias')
    # save whole figure
    plt.savefig(savetitle)


def plot_monthly_predicted_iodide_diff(res='0.125x0.125', dpi=640,
                                       stats=None, show_plot=False, save2png=True,
                                       skipna=True, fillcontinents=True,
                                       rm_non_water_boxes=True):
    """ Plot up a window plot of predicted iodide """
    import seaborn as sns
    sns.reset_orig()
    # get data
#    filename = 'Oi_prj_predicted_iodide_{}.nc'.format( res )
#    folder = '/shared/earth_home/ts551/labbook/Python_progs/'
#    filename= 'Oi_prj_predicted_iodide_4x5_UPDATED_Depth_GEBCO.nc'
    filename = 'Oi_prj_predicted_iodide_{}.nc'.format(res)
    folder = get_file_locations('iodide_data')
    ds = xr.open_dataset(folder + filename)
    # use center points if plotting 0.125x0.125
    if res == '0.125x0.125':
        centre = True
    else:
        centre = False
    # --- Now plot up by month
    var2plot = 'Ensemble_Monthly_mean'
    if rm_non_water_boxes:
        ds = add_LWI2array(ds=ds, res=res, var2template=var2plot)
        # set non water boxes to np.NaN
        ds[var2plot] = ds[var2plot].where(ds['IS_WATER'] == True)
    #
    avg_arr = ds[var2plot].mean(dim='time', skipna=skipna)
#    var2plot = 'RFR(TEMP+SWrad+NO3+MLD+SAL)'
    units = '%'
    cb_max = 240
    fig = plt.figure(dpi=dpi)
    # loop by month and plot
    extend = 'neither'
    range = []
    for n_month, month in enumerate(ds.time):
        # Select data
        #        ds_tmp = ds.sel( time=month ) / avg_arr
        ds_tmp = (ds.sel(time=month) - avg_arr) / avg_arr * 100
        arr = ds_tmp[var2plot].values
        title = AC.dt64_2_dt([month.values])[0].strftime('%b')
        title_x, title_y = 0.425, 1.075
        # Set colorbar range
        fixcb, nticks = np.array([-100, 100]), 5
        range += [arr.min(), arr.max()]
        if np.array(range).max() > 100:
            if extend == 'neither':
                extend = 'max'
            if extend == 'min':
                extend == 'both'
        if np.array(range).min() < -100:
            if extend == 'neither':
                extend = 'min'
            if extend == 'max':
                extend == 'both'
        f_size = 12
        # Set axis labelling
        ylabel = False
        if n_month in np.arange(12)[::3]:
            ylabel = True
        xlabel = False
        if n_month in np.arange(12)[-3:]:
            xlabel = True
        # Now plot
        ax = fig.add_subplot(4, 3, n_month+1)
        AC.plot_spatial_figure(arr, fixcb=fixcb, nticks=nticks,
                               extend=extend, res=res, show=False, title=title,
                               fillcontinents=fillcontinents, centre=centre, units=units,
                               fig=fig, ax=ax, xlabel=xlabel, ylabel=ylabel,
                               title_x=title_x, window=True, f_size=f_size)
#        AC.map_plot( arr, res=res )
    # Adjust figure
    # Adjust plot ascetics
    bottom = 0.05
    left = 0.05
    hspace = 0.075
    wspace = 0.05
    fig.subplots_adjust(bottom=bottom, left=left, hspace=hspace, wspace=wspace)
    # Save to png
    savetitle = 'Oi_prj_seasonal_predicted_iodide_{}_pcent'.format(res)
    savetitle = AC.rm_spaces_and_chars_from_str(savetitle)
    if save2png:
        plt.savefig(savetitle+'.png', dpi=dpi)
    plt.close()


def plot_monthly_predicted_iodide(res='0.125x0.125', dpi=640,
                                  stats=None, show_plot=False, save2png=True,
                                  fillcontinents=True, rm_Skagerrak_data=False,
                                  rm_non_water_boxes=True, debug=False):
    """ Plot up a window plot of predicted iodide """
    import seaborn as sns
    sns.reset_orig()
    # get data
#    filename = 'Oi_prj_predicted_iodide_{}.nc'.format( res )
#    folder = '/shared/earth_home/ts551/labbook/Python_progs/'
#    filename= 'Oi_prj_predicted_iodide_4x5_UPDATED_Depth_GEBCO.nc'
    if rm_Skagerrak_data:
        extr_str = '_No_Skagerrak'
    else:
        extr_str = ''
    filename = 'Oi_prj_predicted_iodide_{}{}.nc'.format(res, extr_str)
    folder = get_file_locations('iodide_data')
    ds = xr.open_dataset(folder + filename)
    # use center points if plotting 0.125x0.125
    if res == '0.125x0.125':
        centre = True
    else:
        centre = False

    # --- Now plot up by month
    var2plot = 'Ensemble_Monthly_mean'
#    var2plot = 'RFR(TEMP+SWrad+NO3+MLD+SAL)'
    # only select boxes where that are fully water (seasonal)
    if rm_non_water_boxes:
        ds = add_LWI2array(ds=ds, res=res, var2template=var2plot)
        # set non water boxes to np.NaN
        ds[var2plot] = ds[var2plot].where(ds['IS_WATER'] == True)
    units = 'nM'
    cb_max = 240
    fig = plt.figure(dpi=dpi)
    # loop by month and plot
    for n_month, month in enumerate(ds.time):
        # Select data
        ds_tmp = ds.sel(time=month)
        arr = ds_tmp[var2plot].values
        title = AC.dt64_2_dt([month.values])[0].strftime('%b')
        title_x, title_y = 0.425, 1.075
        # Set colorbar range
        fixcb, nticks = np.array([0., cb_max]), 5
        extend = 'max'
        f_size = 12
        # Set axis labelling
        ylabel = False
        if n_month in range(12)[::3]:
            ylabel = True
        xlabel = False
        if n_month in range(12)[-3:]:
            xlabel = True
        # Now plot
        ax = fig.add_subplot(4, 3, n_month+1)
        AC.plot_spatial_figure(arr, fixcb=fixcb, nticks=nticks,
                               extend=extend, res=res, show=False, title=title,
                               fillcontinents=fillcontinents, centre=centre, units=units,
                               fig=fig, ax=ax, xlabel=xlabel, ylabel=ylabel,
                               title_x=title_x, window=True, f_size=f_size)
#        AC.map_plot( arr, res=res )
    # Adjust figure
    # Adjust plot ascetics
    bottom = 0.05
    left = 0.05
    hspace = 0.075
    wspace = 0.05
    fig.subplots_adjust(bottom=bottom, left=left, hspace=hspace, wspace=wspace)
    # Save to png
    savetitle = 'Oi_prj_seasonal_predicted_iodide_{}_{}'.format(res, extr_str)
    savetitle = AC.rm_spaces_and_chars_from_str(savetitle)
    if save2png:
        plt.savefig(savetitle+'.png', dpi=dpi)
    plt.close()




def plot_update_existing_params_spatially_window(res='0.125x0.125', dpi=320,
                                                 stats=None, show_plot=False,
                                                 save2png=True, fillcontinents=True):
    """ Plot up predictions from existing parameters spatially """
    import seaborn as sns
    sns.reset_orig()
    # get data
    filename = 'Oi_prj_predicted_iodide_{}.nc'.format(res)
#    filename = 'Oi_prj_predicted_iodide_4x5_UPDATED_Depth_GEBCO.nc'
#    folder = '/shared/earth_home/ts551/labbook/Python_progs/'
    folder = get_file_locations('iodide_data')
    ds = xr.open_dataset(folder + filename)
    # use center points if plotting 0.125x0.125
    if res == '0.125x0.125':
        centre = True
    else:
        centre = False
    units = 'nM'
    # titles / vars2plot
    titles = {
        u'Chance2014_STTxx2_I': 'Chance et al. (2014)',
        u'MacDonald2014_iodide': 'MacDonald et al. (2014)',
    }
    vars2plot = titles.keys()
    # Setup plot
    fig = plt.figure(dpi=dpi, figsize=(6.0, 5.5))
    f_size = 15
    # as a mean
    for n_var, var_ in enumerate(vars2plot):
        # Get the axis
        ax = fig.add_subplot(2, 1, n_var+1)
        # get the annual average
        arr = ds[var_].mean(dim='time').values
        # Plot up
        title = titles[var_]
        fixcb, nticks = np.array([0., 240.]), 5
        extend = 'max'
        # Only label x axis if bottom plot
        xlabel = False
        if (n_var == 1):
            xlabel = True
        # Now plot
        AC.plot_spatial_figure(arr, fixcb=fixcb, nticks=nticks,
                               extend=extend, res=res, show=False, title=title, fig=fig,
                               ax=ax,
                               fillcontinents=fillcontinents, centre=centre, units=units,
                               title_x=.2, title_y=1.02, window=True, xlabel=xlabel,
                               f_size=f_size)
        # add A/B label
        ax.annotate(['(A)', '(B)'][n_var], xy=(0.025, 1.02),
                    textcoords='axes fraction', fontsize=f_size)
    # Adjust plot ascetics
    bottom = 0.025
    left = 0.075
    hspace = 0.075
    top = 0.955
    fig.subplots_adjust(bottom=bottom, left=left, hspace=hspace, top=top)
    # save as png
    savetitle = 'Oi_prj_annual_avg_existing_parameters_{}'.format(res)
    if save2png:
        plt.savefig(savetitle+'.png', dpi=dpi)
    plt.close()


def plot_up_ensemble_avg_and_std_spatially(res='0.125x0.125', dpi=320,
                                           stats=None, show_plot=False, save2png=True,
                                           fillcontinents=True,
                                           rm_Skagerrak_data=False,
                                           rm_non_water_boxes=True, skipna=True,
                                           verbose=True, debug=False):
    """ Plot up the ensemble average and uncertainty (std. dev.) spatially  """
    import seaborn as sns
    sns.reset_orig()
    # Use the predicted values with or without the Skagerrak data?
    if rm_Skagerrak_data:
        extr_str = '_No_Skagerrak'
    else:
        extr_str = ''
    # Get spatial data from saved NetCDF
    filename = 'Oi_prj_predicted_iodide_{}{}.nc'.format(res, extr_str)
    folder = get_file_locations('iodide_data')
    ds = xr.open_dataset(folder + filename)
    # setup a PDF
    savetitle = 'Oi_prj_spatial_avg_and_std_ensemble_models_{}_{}'
    savetitle = savetitle.format(res, extr_str)
    pdff = AC.plot2pdfmulti(title=savetitle, open=True, dpi=dpi)
    # use center points if plotting 0.125x0.125
    if res == '0.125x0.125':
        centre = True
    else:
        centre = False
    # --- Now plot up montly mean of ensemble spatially
    vars2plot = ['Ensemble_Monthly_mean']
    #
    if rm_non_water_boxes:
        ds = add_LWI2array(ds=ds, res=res, var2template=vars2plot[0])
        for var2mask in ['Ensemble_Monthly_mean', 'Ensemble_Monthly_std']:
            # set non water boxes to np.NaN
            ds[var2mask] = ds[var2mask].where(ds['IS_WATER'] == True)
    # set plotting vars
    units = '[I$^{-}_{(aq)}$], (nM)'
    cb_max = 240
    f_size = 20
    # as a mean
    for var in vars2plot:
        arr = ds[var].mean(dim='time', skipna=skipna).values
        # Plot up
        title = 'Annual average I ({})'.format(var)
        fixcb, nticks = np.array([0., cb_max]), 5
        extend = 'max'
        AC.plot_spatial_figure(arr, fixcb=fixcb, nticks=nticks,
                               extend=extend, res=res, show=False, title=title,
                               fillcontinents=fillcontinents, centre=centre, units=units,
                               f_size=f_size)
#        AC.map_plot( arr, res=res )
        # beautify
        # save plot
        AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
        if show_plot:
            plt.show()
        png_filename = AC.rm_spaces_and_chars_from_str(savetitle + '_'+var)
        png_filename += '_{}_{}_{}.png'.format(cb_max, units, extr_str)
        if save2png:
            plt.savefig(png_filename, dpi=dpi)
        plt.close()

    # --- Now plot up montly mean of ensemble spatially
    vars2plot = ['Ensemble_Monthly_std']
    units = 'nM'
    cb_max = 50.
    # as a mean
    for var in vars2plot:
        arr = ds[var].mean(dim='time', skipna=skipna).values
        # Plot up
        title = "Spatial I uncertainity (from '{}')".format(var)
        fixcb, nticks = np.array([0., cb_max]), 6
        extend = 'max'
        AC.plot_spatial_figure(arr, fixcb=fixcb, nticks=nticks,
                               extend=extend, res=res, show=False, title=title,
                               fillcontinents=fillcontinents, centre=centre, units=units,
                               f_size=f_size)
#        AC.map_plot( arr, res=res )
        # beautify
        # save plot
        AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
        if show_plot:
            plt.show()
        png_filename = AC.rm_spaces_and_chars_from_str(savetitle + '_'+var)
        png_filename += '_{}_{}_{}.png'.format(cb_max, units, extr_str)
        if save2png:
            plt.savefig(png_filename, dpi=dpi)
        plt.close()

    # --- Now plot up montly mean of ensemble spatially
    vars2plot = ['Ensemble_Monthly_std']
    units = 'nM'
    cb_max = 30.
    # as a mean
    for var in vars2plot:
        arr = ds[var].mean(dim='time', skipna=skipna).values
        # Plot up
        title = "Spatial I uncertainity (from '{}')".format(var)
        fixcb, nticks = np.array([0., cb_max]), 6
        extend = 'max'
        AC.plot_spatial_figure(arr, fixcb=fixcb, nticks=nticks,
                               extend=extend, res=res, show=False, title=title,
                               fillcontinents=fillcontinents, centre=centre, units=units,
                               f_size=f_size)
#        AC.map_plot( arr, res=res )
        # beautify
        # save plot
        AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
        if show_plot:
            plt.show()
        png_filename = AC.rm_spaces_and_chars_from_str(savetitle + '_'+var)
        png_filename += '_{}_{}_{}.png'.format(cb_max, units, extr_str)
        if save2png:
            plt.savefig(png_filename, dpi=dpi)
        plt.close()

    # --- Now plot up montly mean of ensemble spatially
    vars2plot = ['Ensemble_Monthly_std']
    units = 'nM'
    cb_max = 25.
    # as a mean
    for var in vars2plot:
        arr = ds[var].mean(dim='time', skipna=skipna).values
        # Plot up
        title = "Spatial I uncertainity (from '{}')".format(var)
        fixcb, nticks = np.array([0., cb_max]), 6
        extend = 'max'
        AC.plot_spatial_figure(arr, fixcb=fixcb, nticks=nticks,
                               extend=extend, res=res, show=False, title=title,
                               fillcontinents=fillcontinents, centre=centre, units=units,
                               f_size=f_size)
#        AC.map_plot( arr, res=res )
        # beautify
        # save plot
        AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
        if show_plot:
            plt.show()
        png_filename = AC.rm_spaces_and_chars_from_str(savetitle + '_'+var)
        png_filename += '_{}_{}_{}.png'.format(cb_max, units, extr_str)
        if save2png:
            plt.savefig(png_filename, dpi=dpi)
        plt.close()

    # --- Now plot up monthly (relative) std of ensemble spatially
    vars2plot = ['Ensemble_Monthly_std']
    REFvar = 'Ensemble_Monthly_mean'
    units = '%'
    cb_max = 50.
    # as a mean
    for var in vars2plot:
        arr = ds[var].mean(dim='time', skipna=skipna).values
        arr2 = ds[REFvar].mean(dim='time', skipna=skipna).values
        arr = arr / arr2 * 100
        # Plot up
        title = "% Spatial I uncertainity (from '{}')".format(var)
        fixcb, nticks = np.array([0., cb_max]), 6
        extend = 'max'
        AC.plot_spatial_figure(arr, fixcb=fixcb, nticks=nticks,
                               extend=extend, res=res, show=False, title=title,
                               fillcontinents=fillcontinents, centre=centre, units=units,
                               f_size=f_size)
#        AC.map_plot( arr, res=res )
        # beautify
        # save plot
        AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
        if show_plot:
            plt.show()
        png_filename = AC.rm_spaces_and_chars_from_str(savetitle + '_'+var)
        png_filename += '_{}_{}_{}.png'.format(cb_max, units, extr_str)
        if save2png:
            plt.savefig(png_filename, dpi=dpi)
        plt.close()

    # --- Now plot up monthly (relative) std of ensemble spatially
    vars2plot = ['Ensemble_Monthly_std']
    REFvar = 'Ensemble_Monthly_mean'
    units = '%'
    cb_max = 30.
    # as a mean
    for var in vars2plot:
        arr = ds[var].mean(dim='time', skipna=skipna).values
        arr2 = ds[REFvar].mean(dim='time', skipna=skipna).values
        arr = arr / arr2 * 100
        # Plot up
        title = "% Spatial I uncertainity (from '{}')".format(var)
        fixcb, nticks = np.array([0., cb_max]), 6
        extend = 'max'
        AC.plot_spatial_figure(arr, fixcb=fixcb, nticks=nticks,
                               extend=extend, res=res, show=False, title=title,
                               fillcontinents=fillcontinents, centre=centre, units=units,
                               f_size=f_size)
#        AC.map_plot( arr, res=res )
        # beautify
        # save plot
        AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
        if show_plot:
            plt.show()
        png_filename = AC.rm_spaces_and_chars_from_str(savetitle + '_'+var)
        png_filename += '_{}_{}_{}.png'.format(cb_max, units, extr_str)
        if save2png:
            plt.savefig(png_filename, dpi=dpi)
        plt.close()

    # --- Now plot up monthly (relative) std of ensemble spatially
    vars2plot = ['Ensemble_Monthly_std']
    REFvar = 'Ensemble_Monthly_mean'
    cb_max = 25.
    # as a mean
    for var in vars2plot:
        arr = ds[var].mean(dim='time', skipna=skipna).values
        arr2 = ds[REFvar].mean(dim='time', skipna=skipna).values
        arr = arr / arr2 * 100
        # Plot up
        title = "% Spatial I uncertainity (from '{}')".format(var)
        fixcb, nticks = np.array([0., cb_max]), 6
        extend = 'max'
        AC.plot_spatial_figure(arr, fixcb=fixcb, nticks=nticks,
                               extend=extend, res=res, show=False, title=title,
                               fillcontinents=fillcontinents, centre=centre, units=units,
                               f_size=f_size)
#        AC.map_plot( arr, res=res )
        # beautify
        # save plot
        AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
        if show_plot:
            plt.show()
        png_filename = AC.rm_spaces_and_chars_from_str(savetitle + '_'+var)
        png_filename += '_{}_{}_{}.png'.format(cb_max, units, extr_str)
        if save2png:
            plt.savefig(png_filename, dpi=dpi)
        plt.close()

    # Save entire pdf
    AC.plot2pdfmulti(pdff, savetitle, close=True, dpi=dpi)

def plot_up_input_ancillaries_spatially(res='4x5', dpi=320,
                                        show_plot=False, save2png=True,
                                        fillcontinents=True,
                                        window=False, f_size=20, RFR_dict=None):
    """ Plot up the spatial changes between models  """
    import seaborn as sns
    sns.reset_orig()
    # Get dictionary of shared data if not provided
    if isinstance(RFR_dict, type(None)):
        RFR_dict = build_or_get_current_models()
    # get XR Dataset of data
    filename = 'Oi_prj_feature_variables_{}.nc'.format(res)
    folder = get_file_locations('iodide_data')
    ds = xr.open_dataset(folder + filename)
    # setup a PDF
    savetitle = 'Oi_prj_input_ancillaries_spatailly_{}'.format(res)
    pdff = AC.plot2pdfmulti(title=savetitle, open=True, dpi=dpi)
    # Use appropriate plotting settings for resolution
    if res == '0.125x0.125':
        centre = True
    else:
        centre = False
    # variables to plot?
    vars2plot = get_features_used_by_model_list(RFR_dict=RFR_dict)
    # --- Now plot up concentrations spatially
    # as a mean
    for var in vars2plot:
        if ('time' in ds[var].coords):
            arr = ds[var].mean(dim='time').values
        else:
            arr = ds[var].values
        # adjust temperature to celcuis
        if (var == u'WOA_TEMP_K'):
            arr = arr - 273.15
        # special treatment for depth
        if (var == 'Depth_GEBCO'):
            arr[arr > -2] = -2
            upperlimit = 0
            lowerlimit = AC.myround(np.percentile(arr, 25), base=5)
            nticks = 10
            extend = 'min'
        else:
            base = 5
            upperlimit = AC.myround(np.percentile(arr, 95), base=base)
            lowerlimit = AC.myround(np.percentile(arr, 5), base=base)
            if (lowerlimit == upperlimit):
                base = 1
                upperlimit = AC.myround(np.percentile(arr, 95), base=base)
            nticks = ((upperlimit-lowerlimit) / base) + 1
            if nticks <= 3:
                nticks = nticks*2
            extend = 'both'

        print(var, lowerlimit, upperlimit)
        # Plot up
        maxval = arr.max()
        title = 'Annual average {} (max={:.2f})'.format(var, maxval)
        fixcb = np.array([lowerlimit, upperlimit])
        AC.plot_spatial_figure(arr, fixcb=fixcb, nticks=nticks,
                               extend=extend, res=res, show=False, title=title,
                               fillcontinents=fillcontinents, centre=centre,
                               f_size=f_size, units=None, window=window)
#        AC.map_plot( arr, res=res )
        # beautify
        # save plot
        AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
        if show_plot:
            plt.show()
        png_filename = AC.rm_spaces_and_chars_from_str(savetitle + '_'+var)
        png_filename = png_filename+'_annual_avg'
        if save2png:
            plt.savefig(png_filename, dpi=dpi, bbox_inches='tight')
        plt.close()

    # as a maximum
#     for var in vars2plot:
#         arr = ds[var].max(dim='time').values
#         # Plot up
#         title = 'Annual maximum I ({})'.format( var )
#         fixcb, nticks = np.array( [0., 240.] ), 5
#         extend='max'
#         AC.plot_spatial_figure( arr, fixcb=fixcb, nticks=nticks, \
#             extend=extend, res=res, show=False, title=title, \
#             fillcontinents=fillcontinents, centre=centre )
# #        AC.map_plot( arr, res=res )
#         # beautify
#         # save plot
#         AC.plot2pdfmulti( pdff, savetitle, dpi=dpi )
#         if show_plot: plt.show()
#         plt.close()

        # as fraction of mean
    for var in vars2plot:

        if ('time' in ds[var].coords):
            arr = ds[var].mean(dim='time').values
        else:
            arr = ds[var].values
            # adjust temperature to celcuis
        if (var == u'WOA_TEMP_K'):
            arr = arr - 273.15
            # special treatment for depth
        if (var == 'Depth_GEBCO'):
            arr[arr > -2] = -2
            #
        base = 2
        arr = arr / arr.mean()
        upperlimit = AC.myround(arr.max(), base=base)
        lowerlimit = AC.myround(arr.min(), base=base)
        if lowerlimit < -5:
            lowerlimit = -5
        if upperlimit > 5:
            upperlimit = 5

        nticks = 10
        extend = 'both'

        print(var, lowerlimit, upperlimit)
        # Plot up
        maxval = arr.max()
        title = 'Annual average {} (max={:.2f})'.format(var, maxval)
        fixcb = np.array([lowerlimit, upperlimit])
        AC.plot_spatial_figure(arr, fixcb=fixcb, nticks=nticks,
                               extend=extend, res=res, show=False, title=title,
                               fillcontinents=fillcontinents, centre=centre,
                               f_size=f_size, units=None, window=window)
#        AC.map_plot( arr, res=res )
        # beautify
        # save plot
        AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
        if show_plot:
            plt.show()
        png_filename = AC.rm_spaces_and_chars_from_str(savetitle + '_'+var)
        png_filename = png_filename+'_frac_from_mean'
        if save2png:
            plt.savefig(png_filename, dpi=dpi, bbox_inches='tight')
        plt.close()

    # Save entire pdf
    AC.plot2pdfmulti(pdff, savetitle, close=True, dpi=dpi)


def plot_up_spatial_changes_in_predicted_values(res='4x5', dpi=320,
                                                show_plot=False, save2png=True,
                                                fillcontinents=True,
                                                window=False, f_size=20):
    """ Plot up the spatial changes between models  """
    import seaborn as sns
    sns.reset_orig()
    #
#     models2compare = [
# #    'RFR(TEMP+DEPTH+SAL+NO3+DOC)', 'RFR(TEMP+DEPTH+SAL+NO3)',
# #    'RFR(TEMP+DEPTH+SAL)'
#     # Ones using all variable options
#     'RFR(TEMP+DEPTH+SAL+NO3+DOC)', 'RFR(TEMP+DOC+Phos)',
#     # ones just using variable options
#     'RFR(TEMP+SAL+NO3)', 'RFR(TEMP+SAL+Prod)', 'RFR(TEMP+DEPTH+SAL+Phos)',
#     'RFR(TEMP+SWrad+NO3+MLD+SAL)','RFR(TEMP+DEPTH+SAL)',
#     # Temperature for zeroth order
#     'RFR(TEMP)',
#     ]
    # get data
    filename = 'Oi_prj_predicted_iodide_{}.nc'.format(res)
#    folder = '/shared/earth_home/ts551/labbook/Python_progs/'
    folder = get_file_locations('iodide_data')
    ds = xr.open_dataset(folder + filename)
    # setup a PDF
    savetitle = 'Oi_prj_spatial_comparison_models_{}'.format(res)
    pdff = AC.plot2pdfmulti(title=savetitle, open=True, dpi=dpi)
    #
    if res == '0.125x0.125':
        centre = True
    else:
        centre = False
    # plot settings
    units = '[I$^{-}_{(aq)}$], (nM)'
    # variables to plot?
#    vars2plot = ds.data_vars
    vars2plot = [
        'Chance2014_STTxx2_I', 'MacDonald2014_iodide', 'Ensemble_Monthly_mean'
    ]
    # --- Now plot up concentrations spatially
    # as a mean
    for var in vars2plot:
        arr = ds[var].mean(dim='time').values
        # Plot up
        title = 'Annual average I ({})'.format(var)
        fixcb, nticks = np.array([0., 240.]), 5
        extend = 'max'
        AC.plot_spatial_figure(arr, fixcb=fixcb, nticks=nticks,
                               extend=extend, res=res, show=False, title=title,
                               fillcontinents=fillcontinents, centre=centre,
                               f_size=f_size, units=units, window=window)
#        AC.map_plot( arr, res=res )
        # beautify
        # save plot
        AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
        if show_plot:
            plt.show()
        png_filename = AC.rm_spaces_and_chars_from_str(savetitle + '_'+var)
        png_filename = png_filename+'_II'
        if save2png:
            plt.savefig(png_filename, dpi=dpi, bbox_inches='tight')
        plt.close()

    # as a maximum
#     for var in vars2plot:
#         arr = ds[var].max(dim='time').values
#         # Plot up
#         title = 'Annual maximum I ({})'.format( var )
#         fixcb, nticks = np.array( [0., 240.] ), 5
#         extend='max'
#         AC.plot_spatial_figure( arr, fixcb=fixcb, nticks=nticks, \
#             extend=extend, res=res, show=False, title=title, \
#             fillcontinents=fillcontinents, centre=centre )
# #        AC.map_plot( arr, res=res )
#         # beautify
#         # save plot
#         AC.plot2pdfmulti( pdff, savetitle, dpi=dpi )
#         if show_plot: plt.show()
#         plt.close()

    # --- Difference from Chance et al 2013
#     REF = 'Chance2014_STTxx2_I'
#     vars2plot = [i for i in ds.data_vars if i != REF ]
#     # as a mean
#     for var in vars2plot:
#         arr = ds[var].mean(dim='time').values
#         REF_arr = ds[REF].mean(dim='time').values
#         arr = ( arr - REF_arr ) / REF_arr *100
#         # Plot up
#         title = '% $\Delta$ in annual I ({} vs {})'.format( var, REF )
#         fixcb, nticks = np.array( [-100, 100.] ), 11
#         extend='max'
#         AC.plot_spatial_figure( arr, fixcb=fixcb, nticks=nticks, \
#             extend=extend, res=res, show=False, title=title, \
#             fillcontinents=fillcontinents, centre=centre )
# #        AC.map_plot( arr, res=res )
#         # beautify
#         # save plot
#         AC.plot2pdfmulti( pdff, savetitle, dpi=dpi )
#         if show_plot: plt.show()
#         plt.close()
    # Save entire pdf
    AC.plot2pdfmulti(pdff, savetitle, close=True, dpi=dpi)


# def plot_up_spatial_uncertainty_predicted_values( res='4x5', dpi=320,
#         show_plot=False, save2png=True, fillcontinents=True,
#         rm_Skagerrak_data=False ):
#     """ Plot up the spatial uncertainty between models  """
#     import seaborn as sns
#     sns.reset_orig()
#     # --- set variables
#     #
#     models2compare = [
# #    'RFR(TEMP+DEPTH+SAL+NO3+DOC)', 'RFR(TEMP+DEPTH+SAL+NO3)',
# #    'RFR(TEMP+DEPTH+SAL)'
#     'RFR(TEMP+DEPTH+SAL+NO3+DOC)',
# #    'RFR(TEMP+SAL+NO3)',
#     'RFR(TEMP+SAL+Prod)',
# #    'RFR(TEMP)',
#     'RFR(TEMP+DEPTH+SAL)'
#     ]
# 	#
#     if rm_Skagerrak_data:
#         extr_str = '_No_Skagerrak'
#     else:
#         extr_str = ''
#     # Get data
#     filename = 'Oi_prj_predicted_iodide_{}{}.nc'.format( res, extr_str )
#     folder = get_file_locations( 'iodide_data'  )
# #    folder = '/shared/earth_home/ts551/labbook/Python_progs/'
#     ds = xr.open_dataset( folder + filename )
#     # setup a PDF
#     savetitle = 'Oi_prj_spatial_uncertainity_models_{}'.format( res )
#     pdff = AC.plot2pdfmulti( title=savetitle, open=True, dpi=dpi )
#     #
#     if res == '0.125x0.125':
#         centre=True
#     else:
#         centre=False
#     # iodide in aq
#     Iaq = '[I$^{-}_{aq}$]'
#     # --- Calculate the uncertainty on annual basis
#     print( ds.data_vars )
#     # get all the predicted values
#     ars = [ ds[i].mean(dim='time') for i in models2compare ]
#     units = 'nM'
#     # calculate an average 2D field
#     mean_values = np.array( ars ).mean(axis=0)
#     # subtract this from the averages of the other values
#     UNars = [i-mean_values for i in ars ]
#     # get the standard deviation of this
#     arr = np.std( np.stack( UNars ), axis=0 )
#     # calculate value as % of mean
# #    arr = arr / mean_values *100
# #    units = '%'
#     # Plot up this uncertainty
#     title = 'Std. dev. ({}) of annual average predicted {}'.format( units, Iaq )
#     if units == 'nM':
#         fixcb, nticks = np.array( [0., 20.] ), 5
#     else:
#         fixcb, nticks = np.array( [0., 50.] ), 6
#     extend='max'
#     AC.plot_spatial_figure( arr, fixcb=fixcb, nticks=nticks, \
#         extend=extend, res=res, show=False, title=title, \
#         fillcontinents=True, centre=centre )
# #        AC.map_plot( arr, res=res )
#     # beautify
#     # save plot
#     AC.plot2pdfmulti( pdff, savetitle, dpi=dpi )
#     if show_plot: plt.show()
#     png_filename = AC.rm_spaces_and_chars_from_str( savetitle +'_'+'Annual' )
#     if save2png: plt.savefig( png_filename, dpi=dpi )
#     plt.close()
#
#     # --- Calculate the uncertainty considering temporal variation
#     # get all the predicted values
#     ars = [ ds[i]for i in models2compare ]
#     units = 'nM'
#     # calculate an average 2D field
#     mean_values = np.concatenate( ars, axis=0 ).mean(axis=0)
#     # subtract this from the averages of the other values
#     UNars = [ i-mean_values for i in ars ]
#     # get the standard deviation of this
#     arr = np.std( np.concatenate( UNars,  axis=0 ), axis=0 )
#     # calculate value as % of mean
# #    arr = arr / mean_values *100
# #    units = '%'
#     # Plot up this uncertainty
#     title = 'Std. dev. ({}) of monthly predicted {}'.format( units, Iaq )
#     if units == 'nM':
#         fixcb, nticks = np.array( [0., 30.] ), 7
#     else:
#         fixcb, nticks = np.array( [0., 50.] ), 6
#     extend='max'
#     AC.plot_spatial_figure( arr, fixcb=fixcb, nticks=nticks, \
#         extend=extend, res=res, show=False, title=title, \
#         fillcontinents=True, centre=centre )
# #        AC.map_plot( arr, res=res )
#     # beautify
#     # save plot
#     AC.plot2pdfmulti( pdff, savetitle, dpi=dpi )
#     if show_plot: plt.show()
#     png_filename = AC.rm_spaces_and_chars_from_str( savetitle +'_'+'monthly' )
#     if save2png: plt.savefig( png_filename, dpi=dpi )
#     plt.close()
#
#     # --- Save entire pdf
#     AC.plot2pdfmulti( pdff, savetitle, close=True, dpi=dpi )


def calculate_average_predicted_surface_conc():
    """ Calculate the average predicted surface concentration """
    # directory
    dir_ = '/Users/tomassherwen/Google_Drive/data/iodide_Oi_project/'
    # files
    file_dict = {
        'Macdonald2014': 'Oi_prj_Iodide_monthly_param_4x5_Macdonald2014.nc',
        'Chance2014': 'Oi_prj_Iodide_monthly_param_4x5_Chance2014.nc',
        'NEW_PARAM': 'Oi_prj_Iodide_monthly_param_4x5_NEW_PARAM.nc',
    }
    s_area = AC.get_surface_area(res=res)[..., 0]  # m2 land map
    #
    for param in file_dict.keys():
        filename = file_dict[param]
#        print( param, filename )
        ds = xr.open_dataset(dir_+filename)
        ds = ds['iodide'].mean(dim='time')
        # mask for ocean
        MASK = AC.ocean_unmasked()
        arr = np.ma.array(ds.values, mask=MASK[..., 0])
        print(param, arr.mean())
        # Area weight (masked) array by surface area
        value = AC.get_2D_arr_weighted_by_X(arr.T,  s_area=s_area)
        print(param, value)


# def check_partial_dependence( model=None,
#         features=None, feature_names=None
#         ):
#     """ N/A only model setup is 'BaseGradientBoosting' """
# #    features = [0, 5, 1, 2, (5, 1)]
#     fig, axs = plot_partial_dependence( model, X_train, features=feature_names,
#                                        feature_names=feature_names,
#                                        n_jobs=3, grid_resolution=50)
#     fig.suptitle('Partial dependence of house value on nonlocation features\n'
#                  'for the California housing dataset')
#     plt.subplots_adjust(top=0.9)  # tight_layout causes overlap with suptitle
#
#     print('Custom 3d plot via ``partial_dependence``')
#     fig = plt.figure()
#
#     target_feature = (1, 5)
#     pdp, axes = partial_dependence(clf, target_feature,
#                                    X=X_train, grid_resolution=50)
#     XX, YY = np.meshgrid(axes[0], axes[1])
#     Z = pdp[0].reshape(list(map(np.size, axes))).T
#     ax = Axes3D(fig)
#     surf = ax.plot_surface(XX, YY, Z, rstride=1, cstride=1,
#                            cmap=plt.cm.BuPu, edgecolor='k')
#     ax.set_xlabel(names[target_feature[0]])
#     ax.set_ylabel(names[target_feature[1]])
#     ax.set_zlabel('Partial dependence')
#     #  pretty init view
#     ax.view_init(elev=22, azim=122)
#     plt.colorbar(surf)
#     plt.suptitle('Partial dependence of house value on median\n'
#                  'age and average occupancy')
#     plt.subplots_adjust(top=0.9)
#     plt.show()


def get_diagnostic_plots_analysis4model(res='4x5', extr_str=''):
    """ Plot up a selection of diagnostic plots the model (& exsiting param) """
    # res='4x5'; extr_str='tree_X_STRAT_JUST_TEMP_K_GEBCO_SALINTY'
    # Get the model
    model = get_current_model(extr_str=extr_str)
    testing_features = ['WOA_TEMP_K', 'WOA_Salinity', 'Depth_GEBCO']
    target_name = ['Iodide']
    # Initialise a dictionary to store data
    ars_dict = {}

    # get array of predictor for lats and lons (at res... )
    df_predictors = get_predict_lat_lon_array(res=res)
    # now make predictions for target ("y") from loaded predictors
    target_predictions = model.predict(df_predictors[testing_features])
    # Convert output vector to 2D lon/lat array
    model_name = "RandomForestRegressor '{}'"
    model_name = model_name.format('+'.join(testing_features))
    ars_dict[model_name] = mk_uniform_2D_array(df_predictors=df_predictors,
                                               target_name=target_name, res=res,
                                               target_predictions=target_predictions)

    # - Also get arrays of data for Chance et al and MacDonald et al...
    param_name = 'Chance et al (2014)'
    ars_dict[param_name] = get_equivlient_Chance_arr(res=res,
                                                    target_predictions=target_predictions,
                                                     df=df_predictors,
                                                     testing_features=testing_features)
    param_name = 'MacDonald et al (2014)'
    ars_dict[param_name] = get_equivlient_MacDonald_arr(res=res,
                                                    target_predictions=target_predictions,
                                                        df=df_predictors,
                                                        testing_features=testing_features)
    # -- Also get the working output from processed file for obs.
    pro_df = pd.read_csv(get_file_locations(
        'iodide_data')+'Iodine_obs_WOA.csv')
    # Exclude v. high values (N=4 -  in intial dataset)
    # Exclude v. high values (N=7 -  in final dataset)
    pro_df = pro_df.loc[pro_df['Iodide'] < 400.]

    # sort a fixed order of param names
    param_names = sorted(ars_dict.keys())

    # - Also build 2D arrays for input testing variables
    feature_dict = {}
    extras = [u'SeaWIFs_ChlrA', u'WOA_Nitrate']
    for feature in testing_features + extras:
        feature_dict[feature] = mk_uniform_2D_array(
            df_predictors=df_predictors, target_name=target_name, res=res,
            target_predictions=df_predictors[feature])

    # - Also build 2D arrays for input testing point data
#     NOTE: This needs to be updates to consider overlapping data points.
#     point_dict_ars ={}
#     for feature in testing_features + extras + target_name:
#
#
#         point_dict_ars[feature] = mk_uniform_2D_array(
#         df_predictors=pro_df, target_name=target_name, res=res,
#         target_predictions=pro_df[feature] )

    # ---- Create diagnostics and save as a PDF
    # misc. shared variables
    axlabel = '[I$^{-}_{aq}$] (nM)'
    # setup PDf
    savetitle = 'Oi_prj_param_plots_{}_{}'.format(res, extr_str)
    dpi = 320
    pdff = AC.plot2pdfmulti(title=savetitle, open=True, dpi=dpi)
    # colours to use?
    import seaborn as sns
    sns.reset_orig()
#    current_palette = sns.color_palette()
    current_palette = sns.color_palette("colorblind")
    colour_dict = dict(zip(param_names, current_palette[:len(param_names)]))
    colour_dict['Obs.'] = 'K'
    #  --- Plot up locations of old and new data
    plot_up_data_locations_OLD_and_new(save_plot=False, show_plot=False)
    # Save to PDF and close plot
    AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
    plt.close()

    # --- Plot up parameterisations as spatial plot
    # reset Seaborn settings
    import seaborn as sns
    sns.reset_orig()

#    plot_current_parameterisations()
    for param_name in param_names:
        plot_up_surface_iodide(arr=ars_dict[param_name], title=param_name,
                               res=res)
        # Save to PDF and close plot
        AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
        plt.close()

    # ---- plot up input fields for Temp, salinity, GEBCO
    # reset Seaborn settings
    import seaborn as sns
    sns.reset_orig()

#    plot_current_parameterisations()
    for feature in feature_dict.keys():
        arr = feature_dict[feature]
        fixcb = np.array([arr.min(), arr.max()])
        extend = 'neither'
        nticks = 10
#        plot_up_surface_iodide( arr=, title=feature,\
#            res=res, fixcb, extend=extend )
        title = feature + '(inputed feature)'

        #
        AC.plot_spatial_figure(arr, fixcb=fixcb, nticks=nticks, extend=extend,
                               res=res, units=None, title=title, show=False)
        # Save to PDF and close plot
        AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
        plt.close()

    # ---- plot up ***normalised*** input fields for Temp, salinity, GEBCO
    # ( as before, but- normalised to the mean )
    # reset Seaborn settings
    import seaborn as sns
    sns.reset_orig()

#    plot_current_parameterisations()
    for feature in feature_dict.keys():
        arr = feature_dict[feature].copy()
        arr = arr / arr.mean()
        arr = np.ma.log(arr)
        fixcb = np.array([arr.min(), arr.max()])
        extend = 'neither'
        nticks = 10
        if fixcb[0] == 0:
            fixcb[0] = 0.01
            extend = 'min'
#        plot_up_surface_iodide( arr=, title=feature,\
#            res=res, fixcb, extend=extend )
        title = feature + '( log fraction of mean )'
        #
        AC.plot_spatial_figure(arr, fixcb=fixcb, nticks=nticks, extend=extend,
                               res=res, units=axlabel, title=title, show=False, )
        # Save to PDF and close plot
        AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
        plt.close()

    # ---- plot up ***normalised*** input fields for Temp, salinity, GEBCO
    # ( as before, but- normalised to the median )
    # reset Seaborn settings
    import seaborn as sns
    sns.reset_orig()

#    plot_current_parameterisations()
    for feature in feature_dict.keys():
        print(feature)
        arr = feature_dict[feature].copy()
        arr = arr / np.median(arr)
        arr = np.ma.log(arr)
        fixcb = np.array([arr.min(), arr.max()])
        extend = 'neither'
        nticks = 10
        if fixcb[0] == 0:
            fixcb[0] = 0.01
            extend = 'min'
#        plot_up_surface_iodide( arr=, title=feature,\
#            res=res, fixcb, extend=extend )
        title = feature + '( log fraction of median )'
        #
        if arr.mask.all():
            #            np.min(arr[~arr.mask])
            # check if array is completely masked
            # ValueError: zero-size array to reduction operation minimum which
            # has no identity
            ax, fig = plt.subplots()
            msg = 'WARNING: entire array masked for {}'.format(feature)
            plt.text(0.1, 0.5, msg)
            plt.title(title)
            print(msg)
        else:
            AC.plot_spatial_figure(arr, fixcb=fixcb, nticks=nticks,
                                   extend=extend,
                                   res=res, units=axlabel, title=title, show=False, )
        # Save to PDF and close plot
        AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
        plt.close()

    # --- Plot up parameterisations as a PDF
    import seaborn as sns
    sns.set(color_codes=True)
    sns.set_context("paper")
    # plot 1st model...
    param_name = param_names[0]
    arr = ars_dict[param_name].flatten()
    ax = sns.distplot(arr, axlabel=axlabel, label=param_name,
                      color=colour_dict[param_name])
    # Then loop rest of params
    for param_name in param_names[1:]:
        arr = ars_dict[param_name].flatten()
        ax = sns.distplot(arr, axlabel=axlabel,
                          label=param_name,
                          color=colour_dict[param_name], ax=ax)
    # force y axis extend to be correct
    max_yval = max([h.get_height() for h in ax.patches])
    plt.ylim(0, max_yval+max_yval*0.025)
    # Beautify
    plt.title('PDF of predicted ocean surface '+axlabel)
    plt.legend()
    # Save to PDF and close plot
    AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
    plt.close()

    # --- Plot up parameterisations as a CDF
    # plot 1st model...
    param_name = param_names[0]
    arr = ars_dict[param_name].flatten()
    ax = sns.distplot(arr, axlabel=axlabel, label=param_name,
                      color=colour_dict[param_name],
                      hist_kws=dict(cumulative=True), kde_kws=dict(cumulative=True))
    # Then loop rest of params
    for param_name in param_names[1:]:
        arr = ars_dict[param_name].flatten()
        ax = sns.distplot(arr, axlabel=axlabel,
                          label=param_name,
                          color=colour_dict[param_name], ax=ax,
                          hist_kws=dict(cumulative=True), kde_kws=dict(cumulative=True))
    # force y axis extend to be correct
    max_yval = max([h.get_height() for h in ax.patches])
    plt.ylim(0, max_yval+max_yval*0.025)
    # Beautify
    plt.title('CDF of predicted ocean surface '+axlabel)
    plt.legend()
    # Save to PDF and close plot
    AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
    plt.close()

    # --- --- --- --- --- --- --- ---
    # --- Calculate differs between point observations and params

    # --- All of observational dataset

    # --- Just the test set of observations (20% of original data )

    # ---- plot up input fields for Temp, salinity, GEBCO
    # ---- plot up ***normalised*** input fields for Temp, salinity, GEBCO
    # reset Seaborn settings
#     import seaborn as sns
#     sns.reset_orig()
#
# #    plot_current_parameterisations()
#     for feature in feature_dict.keys():
#         arr = pro_df[feature].copy()
#         #
#
#         fixcb = np.array([ arr.min(), arr.max() ] )
#         extend = 'neither'
#         nticks =10
#         if fixcb[0] == 0:
#             fixcb[0] = 0.01
#             extend = 'min'
# #        plot_up_surface_iodide( arr=, title=feature,\
# #            res=res, fixcb, extend=extend )
#         title = feature + '( log fraction of median )'
#         #
#         AC.plot_spatial_figure(arr, fixcb=fixcb, nticks=nticks, extend=extend,
#             res=res, units=axlabel, title=title, show=False, )
#         # Save to PDF and close plot
#         AC.plot2pdfmulti( pdff, savetitle, dpi=dpi )
#         plt.close()
#

    # -- Save entire pdf
    AC.plot2pdfmulti(pdff, savetitle, close=True, dpi=dpi)




def get_equivlient_Chance_arr(df=None, target_predictions=None,
                              testing_features=None, target_name=['Iodide'], res='4x5'):
    """ Calculate Iodide from Chance parametistaion and input data"""
    # calculate dependency of iodide from Chance et al 2014
    C = (df['WOA_TEMP'].values)**2
    arr = (0.225*C) + 19.
    # Transpose to be a 2D array, that can then be plotted
    arr = mk_uniform_2D_array(df_predictors=df, target_predictions=arr,
                              target_name=target_name, res=res)
    return arr


def get_equivlient_MacDonald_arr(df=None, target_predictions=None,
                                 testing_features=None, target_name=['Iodide'],
                                 res='4x5'):
    """ Calculate Iodide from Chance parametistaion and input data"""
    # calculate dependency of iodide from MacDonald et al 2014
    # NOTE: conversion of M to nM of I-
    C = (df['WOA_TEMP'].values+273.15)
#    C = df['WOA_TEMP_K']
    # NOTE: conversion of M to nM of I-
    arr = 1.45E6 * (np.exp((-9134. / C))) * 1E9
    # Transpose to be a 2D array, that can then be plotted
    arr = mk_uniform_2D_array(df_predictors=df, target_predictions=arr,
                              target_name=target_name, res=res)
    return arr



def plot_up_surface_iodide(arr=None, res='4x5', title=None, plot4poster=False,
                           fixcb=np.array([0., 240.]), nticks=5, extend='max',
                           show_plot=False,
                           f_size=15, window=False, axlabel='[I$^{-}_{aq}$] (nM)'):
    """ Plot up surface concentrations of Iodide """
    # local params
    if plot4poster:
        title = None
        f_size = 30
        window = True
    # Use AC_tools...
    AC.plot_spatial_figure(arr, fixcb=fixcb, nticks=nticks, extend=extend,
                           res=res, units=axlabel, title=title, show=False,
                           f_size=f_size, window=window)
#    if save_plot:
    if show_plot:
        AC.show_plot()


def get_hexbin_plot(x=None, y=None, xlabel=None, ylabel=None, log=False,
                    title=None, add_ODR_trendline2plot=True):
    """ Plot up a hexbin comparison with marginals """
    # detail: http://seaborn.pydata.org/examples/hexbin_marginals.html
    import numpy as np
    import seaborn as sns
    # Setup regression plot
    sns.set(style="ticks")
    g = sns.jointplot(x, y, kind="hex",
                      # set bins to log to increase contrast.
                      #            bins='log' , \
                      #        stat_func=kendalltau,
                      color="#4CB391")
    # Set X and Y ranges to be equal?
    x0, x1 = g.ax_joint.get_xlim()
    y0, y1 = g.ax_joint.get_ylim()
    print('ranges of data:', x0, x1, y0, y1)
    # Or just add a 1:1 line?
    lims = [max(x0, y0), min(x1, y1)]
    g.ax_joint.plot(lims, lims, ':k')
    if all((not isinstance(xlabel, type(None)),
            (not isinstance(ylabel, type(None))))):
        g.set_axis_labels(xlabel, ylabel)
    # Add ODR best fit line
    if add_ODR_trendline2plot:
        plot_ODR_linear(X=x, Y=y, ax=g.ax_joint)
    if not isinstance(title, type(None)):
        g.fig.suptitle(title)
    return g




def get_literature_predicted_iodide(df=None,
                                    verbose=True, debug=False):
    """ Get predicted iodide from literature parametersations """
    # --- Set local variables
    TEMPvar = 'WOA_TEMP'  # temperature
    # Add temperature in Kelvin to array
    TEMPvar_K = TEMPvar+'_K'
    try:
        df[TEMPvar_K]
    except KeyError:
        print('Adding temperature in Kelvin')
        df[TEMPvar+'_K'] = df[TEMPvar].values+273.15
    # Add Modulus to Dataframe, if not present
    MOD_LAT_var = "Latitude (Modulus)"
    try:
        df[MOD_LAT_var]
    except KeyError:
        print('Adding modulus of Latitude')
        df[MOD_LAT_var] = np.sqrt(df["Latitude"].copy()**2)
    # Other variables used in module
    NO3_var = u'WOA_Nitrate'
    sumMLDpt_var = 'WOA_MLDpt_sum'
    salinity_var = u'WOA_Salinity'
    # --- Function to calculate Chance et al correlation
    # --- Relationships from Rosie's paper.
    # functions to calculate (main) Chance et al correlation
    # In order of table 2

    # --- Add two main parameterisations to dataframe
    # Chance et al. (2014)
    var2use = 'Chance2014_STTxx2_I'
    try:
        df[var2use]
    except KeyError:
        df[var2use] = df[TEMPvar].map(calc_iodide_chance2014_STTxx2_I)
    # MacDonald et al. (2014)
    var2use = 'MacDonald2014_iodide'
    try:
        df[var2use]
    except KeyError:
        df[var2use] = df[TEMPvar].map(calc_iodide_MacDonald2014)
    # --- Add all parameterisations from Chance et al (2014) to dataframe
    df = add_all_Chance2014_correlations(df=df, debug=debug)
#    print df.shape
    # --- Add multivariate parameterisation too (Chance et al. (2014))

    def calc_iodide_chance2014_Multivariate(
        TEMP=None, MOD_LAT=None, NO3=None, sumMLDpt=None, salinity=None
    ):
        """ Take variable and returns (multivariate) parameterised [iodide] """
        iodide = (0.28*TEMP**2) + (1.7*MOD_LAT) + (0.9*NO3) -  \
            (0.020*sumMLDpt) + (7.0*salinity) - 309
        return iodide
    # Chance et al. (2014). multivariate
    var2use = 'Chance2014_Multivariate'
    try:
        df[var2use]
    except KeyError:
        df[var2use] = df.apply(lambda x:
                               calc_iodide_chance2014_Multivariate(NO3=x[NO3_var],
                                                                 sumMLDpt=x[sumMLDpt_var],
                                                                   MOD_LAT=x[MOD_LAT_var],
                                                                   TEMP=x[TEMPvar],
                                                                salinity=x[salinity_var]),
                                                                   axis=1)

    # --- Add the ensemble to the dataframe
# 	use_vals_from_NetCDF =  False # Use the values from the spatial prediction
# 	avg_values_
#     if add_ensemble:
#         var2use = 'RFR(Ensemble)'
#         try:
#             df[var2use]
#         except KeyError:
# 			if use_vals_from_NetCDF:
# 				month_var = 'Month'
# 				# Save the original values
# 				df['Month (Orig.)'] = df[month_var].values
# 				# Make sure month is numeric (if not given)
# 				NaN_months_bool = ~np.isfinite(df[month_var].values)
# 				NaN_months_df = df.loc[NaN_months_bool, :]
# 				N_NaN_months =  NaN_months_df.shape[0]
# 				if N_NaN_months >1:
# 					print_str = 'DataFrame contains NaNs for {} months - '
# 					print_str += 'Replacing these with month # 3 months '
# 					print_str += 'before (hemispheric) summer solstice'
# 					if verbose: print( print_str.format( N_NaN_months ) )
# 					NaN_months_df.loc[:,month_var] = NaN_months_df.apply(
# 						lambda x:
# 						set_backup_month_if_unkonwn(
# 						lat=x['Latitude'],
# 			#            main_var=var2use, var2use=var2use,
# 			#            Data_key_ID_=Data_key_ID_,
# 						debug=False), axis=1 ).values
# 					# Add back into DataFrame
# 					tmp_vals =  NaN_months_df[month_var].values
# 					df.loc[NaN_months_bool, month_var] = tmp_vals
# 				# Now calculate the month
# 				df[var2use] = extract_4_nearest_points_in_NetCDF(\
# 					lats=df['Latitude'].values, lons=df[u'Longitude'].values,\
# 					months=df['Month'].values, \
# 					rm_Skagerrak_data=rm_Skagerrak_data,\
# 				)
# 			else: # average the topmodels output.
# 				pass
    return df


def get_ensemble_predicted_iodide(df=None,
                                  RFR_dict=None, topmodels=None, stats=None,
                                  rm_Skagerrak_data=False, use_vals_from_NetCDF=False,
                                  var2use='RFR(Ensemble)', verbose=True, debug=False):
    """ Get predicted iodide from literature parametersations """
    # Just use top 10 models are included
    # ( with derivative variables )
    if isinstance(topmodels, type(None)):
        # extract the models...
        if isinstance(RFR_dict, type(None)):
            RFR_dict = build_or_get_current_models(
                rm_Skagerrak_data=rm_Skagerrak_data
            )
        # get stats on models in RFR_dict
        if isinstance(stats, type(None)):
            stats = get_stats_on_current_models(RFR_dict=RFR_dict,
                                                verbose=False)
        # get list of
        topmodels = get_top_models(RFR_dict=RFR_dict, NO_DERIVED=True)

    # --- Add the ensemble to the dataframe
    use_vals_from_NetCDF = False  # Use the values from the spatial prediction
    # If the variable is not already there then do not add
    try:
        df[var2use]
        print("Ensemble data not added - '{}' already in df".format(var2use))
    except KeyError:
        if use_vals_from_NetCDF:
            month_var = 'Month'
            # Save the original values
            df['Month (Orig.)'] = df[month_var].values
            # Make sure month is numeric (if not given)
            NaN_months_bool = ~np.isfinite(df[month_var].values)
            NaN_months_df = df.loc[NaN_months_bool, :]
            N_NaN_months = NaN_months_df.shape[0]
            if N_NaN_months > 1:
                print_str = 'DataFrame contains NaNs for {} months - '
                print_str += 'Replacing these with month # 3 months '
                print_str += 'before (hemispheric) summer solstice'
                if verbose:
                    print(print_str.format(N_NaN_months))
                NaN_months_df.loc[:, month_var] = NaN_months_df.apply(
                    lambda x:
                    set_backup_month_if_unkonwn(
                        lat=x['Latitude'],
                        #            main_var=var2use, var2use=var2use,
                        #            Data_key_ID_=Data_key_ID_,
                        debug=False), axis=1).values
                # Add back into DataFrame
                tmp_vals = NaN_months_df[month_var].values
                df.loc[NaN_months_bool, month_var] = tmp_vals
                # Now calculate the month
            df[var2use] = extract_4_nearest_points_in_NetCDF(
                lats=df['Latitude'].values, lons=df[u'Longitude'].values,
                months=df['Month'].values,
                rm_Skagerrak_data=rm_Skagerrak_data,
            )
        else:  #  average the topmodels output.
                        # Get all the model predictions from the RFR_dict
            df_tmp = RFR_dict['df']
            df_tmp.index = df_tmp['Data_Key_ID']
            #
            df_tmp[var2use] = df_tmp[topmodels].mean(axis=1)
            # Add a column for the
            df[var2use] = np.NaN
            # Now add along the  index
            Data_Key_IDs = df_tmp['Data_Key_ID'].values
            for nDK_ID, DK_ID in enumerate(Data_Key_IDs):
                # Get predicted value
                val = df_tmp.loc[df_tmp['Data_Key_ID'] == DK_ID, var2use][0]
                # print out diagnostic
                pstr = "Adding {} prediction for {:<20} of:  {:.2f} ({:.2f})"
                pcent = float(nDK_ID) / len(Data_Key_IDs) * 100.
                if debug:
                    print(pstr.format(var2use, DK_ID, val, pcent))
                    # fill in value to input DataFrame
                df.loc[df['Data_Key_ID'] == DK_ID, var2use] = val
    return df




def plot_difference2_input_PDF_on_update_of_var(res='4x5'):
    """
    Set coordinates to use for plotting data spatially
    """
    # Use appropriate plotting settings for resolution
    if res == '0.125x0.125':
        centre = True
    else:
        centre = False

    pass



def mk_PDFs_to_show_the_sensitivty_input_vars_65N_and_up(
        RFR_dict=None, stats=None, res='4x5', dpi=320,
        perturb_by_mutiple=False, save_str='', show_plot=False):
    """ Graphically plot the sensitivity of iodide in input variables """
    import matplotlib
#    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.style.use('ggplot')
    import seaborn as sns
    sns.set()
    # Get the dictionary of shared data
    if isinstance(RFR_dict, type(None)):
        RFR_dict = build_or_get_current_models()
    # Get the stats on the models built
    if isinstance(stats, type(None)):
        stats = get_stats_on_current_models(RFR_dict=RFR_dict, verbose=False)
    # Get the core input variables
    iodide_dir = get_file_locations('iodide_data')
    filename = 'Oi_prj_feature_variables_{}.nc'.format(res)
    ds = xr.open_dataset(iodide_dir + filename)
    # set up a dictionary for different dataset splits
    dss = {}
    # Keep the base case to use as a reference
    dss['BASE'] = ds.copy()
    # Which variables should be plotted?
    topmodels = get_top_models(RFR_dict=RFR_dict, NO_DERIVED=True)
    var2test = get_features_used_by_model_list(RFR_dict=RFR_dict,
                                               models_list=topmodels)
#	var2test =  ['WOA_Nitrate']  # for testing just change nitrate
    # perturb vars (e.g. by  by -/+ 10, 20, 30 % )
#	perturb_by_mutiple = False
    perturb_by_mutiple = True
#	perturb2use = [ 0.7, 0.8, 0.9, 1.1, 1.2, 1.3 ]
#	perturb2use = [ 0.1, 0.5, 0.6, 0.8, 1.2, 1.4, 1.5, 10 ]
#	perturb2use = [ 0.1, 10 ]
    perturb2use = [0.25, 0.5, 0.75, 1.25, 1.5, 1.75, ]
#	perturb2use = [ 1, 2, 4, 6,8, 9, 10 ]
#	perturb2use = [1.5,  2.5, 4 , 5, 6, 7.5, 10 ]
#	perturb2use = [ 1, 2, 3, 4 ]
    print(var2test)
    for var in var2test:
        # add perturbations
        for perturb in perturb2use:
            ds_tmp = ds.copy()
            if perturb_by_mutiple:
                var_str = '{} (x{})'.format(var, perturb)
                ds_tmp[var] = ds_tmp[var].copy() * perturb
            else:
                var_str = '{} (+{})'.format(var, perturb)
                ds_tmp[var] = ds_tmp[var].copy() + perturb
            dss[var_str] = ds_tmp
            del ds_tmp

    # --- Make predictions for dataset splits
    dssI = {}
#    keys2add = [i for i in dss.keys() if i not in dssI.keys()]
    keys2add = dss.keys()
    for key_ in keys2add:
        # predict the values for the locations
        ds_tmp = mk_iodide_predictions_from_ancillaries(None,
                                                        dsA=dss[key_], RFR_dict=RFR_dict,
                                                       use_updated_predictor_NetCDF=False,
                                                        save2NetCDF=False,
                                                        topmodels=topmodels)
        # add ensemble to ds
        ds_tmp = add_ensemble_avg_std_to_dataset(ds=ds_tmp,
                                                 RFR_dict=RFR_dict, topmodels=topmodels,
                                                 res=res,
                                                 save2NetCDF=False)
        # add LWI and surface area to array
        ds_tmp = add_LWI2array(ds=ds_tmp, res=res,
                               var2template='Chance2014_STTxx2_I')
        # save to dict
        dssI[key_] = ds_tmp

    # --- plot these up
    # setup a PDF
    savetitle = 'Oi_prj_sensitivity_to_perturbation_{}{}'
    savetitle = savetitle.format(res, save_str)
    pdff = AC.plot2pdfmulti(title=savetitle, open=True, dpi=dpi)
    # make sure BASE is first
    var_ = 'BASE'
#    vars2plot = dssI.keys()
#    del vars2plot[vars2plot.index(var_) ]
#    vars2plot = [var_] + list( sorted(vars2plot) )
    # loop and plot
    for key_ in var2test:
        print('plotting: {}'.format(key_))
        # set a single axis to use.
        fig, ax = plt.subplots()
        # plot the base values as a shadow
        vars2plot = ['BASE']
        var2plot = vars2plot[0]
        ds = dssI[var2plot]
        model2plot = 'Ensemble_Monthly_mean'
        # select iodide to plot and only above 65N
        ds = ds.sel(lat=(ds['lat'] >= 65))
        arr = ds[model2plot].values
        # select only the water locations
        arr[(ds['IS_WATER'] == False).values] = np.NaN
        df = pd.DataFrame(arr.flatten()).dropna()
        sns.distplot(df, ax=ax, label=var2plot, color='k',
                     kde_kws={"linestyle": "--"})
        # Add title to plot
        plt.title("Perturbations to '{}' > 65N".format(key_))
        # Make sure the values are correctly scaled
        ax.autoscale()
        # Loop the perturbations
        vars2plot = [i for i in dssI.keys() if key_ in i]
        # get perturbations
        if perturb_by_mutiple:
            p2plot = [i.split('(x')[-1][:-1] for i in vars2plot]
        else:
            p2plot = [i.split('(+')[-1][:-1] for i in vars2plot]
        perturb2var = dict(zip(p2plot, vars2plot))
        for var_ in sorted(p2plot):
            # set colour
            if (float(var_) <= 1) and perturb_by_mutiple:
                cmap = plt.cm.Blues_r
                norm = matplotlib.colors.Normalize(
                    vmin=float(min(p2plot)), vmax=1
                )
            else:
                cmap = plt.cm.Reds
                norm = matplotlib.colors.Normalize(
                    vmin=1, vmax=float(max(p2plot))
                )
            color = cmap(norm(float(var_)))
            ds = dssI[perturb2var[var_]]
            # select only the water locations
            arr = ds[model2plot].values
            arr[(ds['IS_WATER'] == False).values] = np.NaN
            # select iodide to plot and only above 65N
            arr[:, (ds['lat'] <= 65), :] = np.NaN
            df = pd.DataFrame(arr.flatten()).dropna()
            # now plot non-NaNs
            sns.distplot(df, ax=ax, label=var_, color=color, hist=False)
        # Make sure the values are correctly scaled
        ax.autoscale()
        plt.legend()
        # Save to PDF and close plot
        print('saving: {}'.format(key_))
        AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
        if show_plot:
            plt.show()
        plt.close()
    # Save entire pdf
    AC.plot2pdfmulti(pdff, savetitle, close=True, dpi=dpi)



def plot_spatial_area4core_decisions(res='4x5'):
    """ Plot various spatial extents of input vairables  """

    # --- Get core decision points for variables

    d = {
        'WOA_TEMP_K':  {'value': 17.4+273.15, 'std': 2.0},
    }
    # ---
    for var in d.keys():
        # Get value and std for variable
        value = d[var]['value']
        std = d[var]['std']
        # plot up threshold
        plot_threshold_plus_SD_spatially(var=var, value=value, std=std,
                                         res=res)


def explore_sensitivity_of_65N2data_denial(res='4x5', RFR_dict=None, dpi=320,
                                           verbose=True, debug=False):
    """ Explore the sensitivity of the prediction to data denial """
    import gc
    # res='4x5'; dpi=320
    # --- Local variables
    Iaq = '[I$^{-}_{aq}$]'
    # Get the core input variables
    iodide_dir = get_file_locations('iodide_data')
    filename = 'Oi_prj_feature_variables_{}.nc'.format(res)
    ds = xr.open_dataset(iodide_dir + filename)
    # Get the models
    if isinstance(RFR_dict, type(None)):
        RFR_dict = build_or_get_current_models()
    topmodels = get_top_models(RFR_dict=RFR_dict, NO_DERIVED=True, n=10)
    topmodels = list(set(topmodels))
    # other local settings?
    plt_option_tired_but_didnt_help = False
    plt_option_tired_but_only_slightly_helped = False
    plt_excluded_obs_locations = False

    # --- Now build the models without certain values
    dfA = get_dataset_processed4ML(restrict_data_max=False,
                                   rm_Skagerrak_data=False, rm_iodide_outliers=False,)
    RFR_dict_d = {}
    Nvals = {}
    # Add Base
    VarName = 'BASE'
    RFR_dict_d[VarName] = RFR_dict  # base currently inc. all iodide values
    NA = float(dfA.shape[0])
    Nvals[VarName] = NA

    # Clean memory
    gc.collect()

    # - no where obs where low temperature and coastal (NH)
#     VarName = 'No Skagerrak'
#     bool1 = dfA['Data_Key'].values == 'Truesdale_2003_I'
#     index2drop = dfA.loc[ bool1, : ].index
#     df = dfA.drop( index2drop )
#     # reset index of updated DataFrame (and save out the rm'd data prior)
#     df2plot = dfA.drop( df.index ).copy()
#     df.index = np.arange(df.shape[0])
#     # Reset the training/withhel data split
#     returned_vars = mk_iodide_ML_testing_and_training_set(df=df.copy(),
#         random_20_80_split=False, random_strat_split=True,
#         testing_features=df.columns.tolist(),
#         test_plots_of_iodide_dist=False,
#         )
#     train_set, test_set, test_set_targets = returned_vars
#     key_varname = 'Test set ({})'.format( 'strat. 20%' )
#     df[key_varname] =  False
#     df.loc[ test_set.index,key_varname ] = True
#     df.loc[ train_set.index, key_varname ] = False
#     # print the size of the input set
#     N = float(df.shape[0])
#     Nvals[ VarName ] = N
#     prt_str =  "N={:.0f} ({:.2f} % of total) for '{}'"
#     if verbose: print( prt_str.format( N, N/NA*100,VarName ) )
#     # Test the locations?
#     if plt_excluded_obs_locations:
#         import seaborn as sns
#         sns.reset_orig()
#         lats = df2plot['Latitude'].values
#         lons = df2plot['Longitude'].values
#         title4plt = "Points excluded (N={}) for \n '{}'".format( int(NA-N), VarName )
#         AC.plot_lons_lats_spatial_on_map( lats=lats, lons=lons, title=title4plt )
#         savestr = 'Oi_prj_locations4data_split_{}'.format( VarName )
#         savestr = AC.rm_spaces_and_chars_from_str( savestr )
#         plt.savefig( savestr, dpi=320 )
#         plt.close()
#     # rebuild (just the top models)
#     RFR_dict_d[VarName] = build_or_get_current_models( df=df,
#         model_names = topmodels,
#         save_model_to_disk=False,
#         read_model_from_disk=False,
#         delete_existing_model_files=False
#     )

    # - no where obs where low temperature and coastal (NH)
    VarName = 'No outliers'
    bool1 = dfA['Iodide'] > get_outlier_value(df=dfA, var2use='Iodide')
    index2drop = dfA.loc[bool1, :].index
    df = dfA.drop(index2drop)
    # reset index of updated DataFrame (and save out the rm'd data prior)
    df2plot = dfA.drop(df.index).copy()
    df.index = np.arange(df.shape[0])
    # Reset the training/withhel data split
    returned_vars = mk_iodide_ML_testing_and_training_set(df=df.copy(),
                                                          random_20_80_split=False,
                                                          random_strat_split=True,
                                                     testing_features=df.columns.tolist(),
                                                          test_plots_of_iodide_dist=False,
                                                          )
    train_set, test_set, test_set_targets = returned_vars
    key_varname = 'Test set ({})'.format('strat. 20%')
    df[key_varname] = False
    df.loc[test_set.index, key_varname] = True
    df.loc[train_set.index, key_varname] = False
    # print the size of the input set
    N = float(df.shape[0])
    Nvals[VarName] = N
    prt_str = "N={:.0f} ({:.2f} % of total) for '{}'"
    if verbose:
        print(prt_str.format(N, N/NA*100, VarName))
    # Test the locations?
    if plt_excluded_obs_locations:
        import seaborn as sns
        sns.reset_orig()
        lats = df2plot['Latitude'].values
        lons = df2plot['Longitude'].values
        title4plt = "Points excluded (N={}) for \n '{}'".format(
            int(NA-N), VarName)
        AC.plot_lons_lats_spatial_on_map(lats=lats, lons=lons, title=title4plt)
        savestr = 'Oi_prj_locations4data_split_{}'.format(VarName)
        savestr = AC.rm_spaces_and_chars_from_str(savestr)
        plt.savefig(savestr, dpi=320)
        plt.close()
    # rebuild (just the top models)
    RFR_dict_d[VarName] = build_or_get_current_models(df=df,
                                                      model_names=topmodels,
                                                      save_model_to_disk=False,
                                                      read_model_from_disk=False,
                                                      delete_existing_model_files=False
                                                      )

    # - No outliers or skaggerak
    VarName = 'No outliers \or Skagerrak'
    bool1 = dfA['Iodide'] > get_outlier_value(df=dfA, var2use='Iodide')
    bool2 = dfA['Data_Key'].values == 'Truesdale_2003_I'
    index2drop = dfA.loc[bool1 | bool2, :].index
    df = dfA.drop(index2drop)
    # reset index of updated DataFrame (and save out the rm'd data prior)
    df2plot = dfA.drop(df.index).copy()
    df.index = np.arange(df.shape[0])
    # Reset the training/withhel data split
    returned_vars = mk_iodide_ML_testing_and_training_set(df=df.copy(),
                                                          random_20_80_split=False,
                                                          random_strat_split=True,
                                                    testing_features=df.columns.tolist(),
                                                          test_plots_of_iodide_dist=False,
                                                          )
    train_set, test_set, test_set_targets = returned_vars
    key_varname = 'Test set ({})'.format('strat. 20%')
    df[key_varname] = False
    df.loc[test_set.index, key_varname] = True
    df.loc[train_set.index, key_varname] = False
    # print the size of the input set
    N = float(df.shape[0])
    Nvals[VarName] = N
    prt_str = "N={:.0f} ({:.2f} % of total) for '{}'"
    if verbose:
        print(prt_str.format(N, N/NA*100, VarName))
    # Test the locations?
    if plt_excluded_obs_locations:
        import seaborn as sns
        sns.reset_orig()
        lats = df2plot['Latitude'].values
        lons = df2plot['Longitude'].values
        title4plt = "Points excluded (N={}) for \n '{}'".format(
            int(NA-N), VarName)
        AC.plot_lons_lats_spatial_on_map(lats=lats, lons=lons, title=title4plt)
        savestr = 'Oi_prj_locations4data_split_{}'.format(VarName)
        savestr = AC.rm_spaces_and_chars_from_str(savestr)
        plt.savefig(savestr, dpi=320)
        plt.close()
    # rebuild (just the top models)
    RFR_dict_d[VarName] = build_or_get_current_models(df=df,
                                                      model_names=topmodels,
                                                      save_model_to_disk=False,
                                                      read_model_from_disk=False,
                                                      delete_existing_model_files=False
                                                      )

    # --- Include options that didn't improve things in PDF
    if plt_option_tired_but_only_slightly_helped:
        # 		- no where obs where low temperature and coastal (NH)
        # 		VarName = '{} '.format( Iaq ) +'<98$^{th}$'
        # 		bool1 = dfA['Iodide'] > np.percentile( df['Iodide'].values, 98 )
        # 		index2drop = dfA.loc[ bool1, : ].index
        # 		df = dfA.drop( index2drop )
        # 		reset index of updated DataFrame (and save out the rm'd data prior)
        # 		df2plot = dfA.drop( df.index ).copy()
        # 		df.index = np.arange(df.shape[0])
        # 		Reset the training/withhel data split
        # 		returned_vars = mk_iodide_ML_testing_and_training_set(df=df.copy(),
        # 			random_20_80_split=False, random_strat_split=True,
        # 			testing_features=df.columns.tolist(),
        # 			test_plots_of_iodide_dist=False,
        # 			)
        # 		train_set, test_set, test_set_targets = returned_vars
        # 		key_varname = 'Test set ({})'.format( 'strat. 20%' )
        # 		df[key_varname] =  False
        # 		df.loc[ test_set.index,key_varname ] = True
        # 		df.loc[ train_set.index, key_varname ] = False
        # 		print the size of the input set
        # 		N = float(df.shape[0])
        # 		Nvals[ VarName ] = N
        # 		prt_str =  "N={:.0f} ({:.2f} % of total) for '{}'"
        # 		if verbose: print( prt_str.format( N, N/NA*100,VarName ) )
        # 		Test the locations?
        # 		if plt_excluded_obs_locations:
        # 			import seaborn as sns
        # 			sns.reset_orig()
        # 			lats = df2plot['Latitude'].values
        # 			lons = df2plot['Longitude'].values
        # 			title4plt = "Points excluded (N={}) for \n '{}'".format( int(NA-N), VarName )
        # 			AC.plot_lons_lats_spatial_on_map( lats=lats, lons=lons, title=title4plt )
        # 			savestr = 'Oi_prj_locations4data_split_{}'.format( VarName )
        # 			savestr = AC.rm_spaces_and_chars_from_str( savestr )
        # 			plt.savefig( savestr, dpi=320 )
        # 			plt.close()
        # 		rebuild (just the top models)
        # 		RFR_dict_d[VarName] = build_or_get_current_models( df=df,
        # 			model_names = topmodels,
        # 			save_model_to_disk=False,
        # 			read_model_from_disk=False,
        # 			delete_existing_model_files=False
        # 		)
        #
        # 		- no where obs where low temperature and coastal (NH)
        # 		VarName = '{}  + \n No Skaggerak'.format( Iaq ) +'<98$^{th}$'
        # 		bool1 = dfA['Iodide'] > np.percentile( df['Iodide'].values, 98 )
        # 		bool2 = dfA['Data_Key'].values == 'Truesdale_2003_I'
        # 		index2drop = dfA.loc[ bool1 | bool2, : ].index
        # 		df = dfA.drop( index2drop )
        # 		reset index of updated DataFrame (and save out the rm'd data prior)
        # 		df2plot = dfA.drop( df.index ).copy()
        # 		df.index = np.arange(df.shape[0])
        # 		Reset the training/withhel data split
        # 		returned_vars = mk_iodide_ML_testing_and_training_set(df=df.copy(),
        # 			random_20_80_split=False, random_strat_split=True,
        # 			testing_features=df.columns.tolist(),
        # 			test_plots_of_iodide_dist=False,
        # 			)
        # 		train_set, test_set, test_set_targets = returned_vars
        # 		key_varname = 'Test set ({})'.format( 'strat. 20%' )
        # 		df[key_varname] =  False
        # 		df.loc[ test_set.index,key_varname ] = True
        # 		df.loc[ train_set.index, key_varname ] = False
        # 		print the size of the input set
        # 		N = float(df.shape[0])
        # 		Nvals[ VarName ] = N
        # 		prt_str =  "N={:.0f} ({:.2f} % of total) for '{}'"
        # 		if verbose: print( prt_str.format( N, N/NA*100,VarName ) )
        # 		Test the locations?
        # 		if plt_excluded_obs_locations:
        # 			import seaborn as sns
        # 			sns.reset_orig()
        # 			lats = df2plot['Latitude'].values
        # 			lons = df2plot['Longitude'].values
        # 			title4plt = "Points excluded (N={}) for \n '{}'".format( int(NA-N), VarName )
        # 			AC.plot_lons_lats_spatial_on_map( lats=lats, lons=lons, title=title4plt )
        # 			savestr = 'Oi_prj_locations4data_split_{}'.format( VarName )
        # 			savestr = AC.rm_spaces_and_chars_from_str( savestr )
        # 			plt.savefig( savestr, dpi=320 )
        # 			plt.close()
        # 		rebuild (just the top models)
        # 		RFR_dict_d[VarName] = build_or_get_current_models( df=df,
        # 			model_names = topmodels,
        # 			save_model_to_disk=False,
        # 			read_model_from_disk=False,
        # 			delete_existing_model_files=False
        # 		)
        #
        # 		Clean memory
        # 		gc.collect()

        # - no where obs where low temperature and coastal (SH)
        VarName = 'No SH coastal <280K'
        bool1 = dfA['WOA_TEMP_K'].values < 280
        bool2 = dfA['Coastal'].values == 1.0
    #    datasets2inc = [
    #    u'Truesdale_U_2003', u'Truesdale_2003_I', u'Luther_1988', u'Wong_C_1998'
    #    ]
    #    bool3 = dfA['Data_Key'].values in datasets2inc
        bool3 = dfA['Latitude'].values < 0
        index2drop = dfA.loc[bool1 & bool2 & bool3, :].index
        df = dfA.drop(index2drop)
        # reset index of updated DataFrame (and save out the rm'd data prior)
        df2plot = dfA.drop(df.index)
        df.index = np.arange(df.shape[0])
        # Reset the training/withhel data split
        returned_vars = mk_iodide_ML_testing_and_training_set(df=df.copy(),
                                                              random_20_80_split=False,
                                                              random_strat_split=True,
                                                     testing_features=df.columns.tolist(),
                                                         test_plots_of_iodide_dist=False,
                                                              )
        train_set, test_set, test_set_targets = returned_vars
        key_varname = 'Test set ({})'.format('strat. 20%')
        df[key_varname] = False
        df.loc[test_set.index, key_varname] = True
        df.loc[train_set.index, key_varname] = False
        # print the size of the input set
        N = float(df.shape[0])
        Nvals[VarName] = N
        prt_str = "N={:.0f} ({:.2f} % of total) for '{}'"
        if verbose:
            print(prt_str.format(N, N/NA*100, VarName))
        # Test the locations?
        if plt_excluded_obs_locations:
            import seaborn as sns
            sns.reset_orig()
            lats = df2plot['Latitude'].values
            lons = df2plot['Longitude'].values
            title4plt = "Points excluded (N={}) for \n '{}'".format(
                int(NA-N), VarName)
            AC.plot_lons_lats_spatial_on_map(
                lats=lats, lons=lons, title=title4plt)
            savestr = 'Oi_prj_locations4data_split_{}'.format(VarName)
            savestr = AC.rm_spaces_and_chars_from_str(savestr)
            plt.savefig(savestr, dpi=320)
            plt.close()
        # rebuild (just the top models)
        RFR_dict_d[VarName] = build_or_get_current_models(df=df,
                                                          model_names=topmodels,
                                                          save_model_to_disk=False,
                                                          read_model_from_disk=False,
                                                        delete_existing_model_files=False
                                                          )

        # - where obs are n coastal
        VarName = 'No Coastal'
        df = dfA.loc[dfA['Coastal'] != 1.0, :]
        # reset index of updated DataFrame
        df.index = np.arange(df.shape[0])
        # Reset the training/withhel data split
        returned_vars = mk_iodide_ML_testing_and_training_set(df=df.copy(),
                                                              random_20_80_split=False,
                                                              random_strat_split=True,
                                                     testing_features=df.columns.tolist(),
                                                           test_plots_of_iodide_dist=False
                                                              )
        train_set, test_set, test_set_targets = returned_vars
        key_varname = 'Test set ({})'.format('strat. 20%')
        df[key_varname] = False
        df.loc[test_set.index, key_varname] = True
        df.loc[train_set.index, key_varname] = False
        # print the size of the input set
        N = float(df.shape[0])
        Nvals[VarName] = N
        prt_str = "N={:.0f} ({:.2f} % of total) for '{}'"
        if verbose:
            print(prt_str.format(N, N/NA*100, VarName))
        # rebuild (just the top models)
        RFR_dict_d[VarName] = build_or_get_current_models(df=df,
                                                          model_names=topmodels,
                                                          save_model_to_disk=False,
                                                          read_model_from_disk=False,
                                                         delete_existing_model_files=False
                                                          )

        # - no where obs where low temperature.
        VarName = 'No <280K'
        df = dfA.loc[dfA['WOA_TEMP_K'].values >= 280, :]
        # reset index of updated DataFrame
        df.index = np.arange(df.shape[0])
        # Reset the training/withhel data split
        returned_vars = mk_iodide_ML_testing_and_training_set(df=df.copy(),
                                                              random_20_80_split=False,
                                                              random_strat_split=True,
                                                     testing_features=df.columns.tolist(),
                                                          test_plots_of_iodide_dist=False,
                                                              )
        train_set, test_set, test_set_targets = returned_vars
        key_varname = 'Test set ({})'.format('strat. 20%')
        df[key_varname] = False
        df.loc[test_set.index, key_varname] = True
        df.loc[train_set.index, key_varname] = False
        # print the size of the input set
        N = float(df.shape[0])
        Nvals[VarName] = N
        prt_str = "N={:.0f} ({:.2f} % of total) for '{}'"
        if verbose:
            print(prt_str.format(N, N/NA*100, VarName))
        # rebuild (just the top models)
        RFR_dict_d[VarName] = build_or_get_current_models(df=df,
                                                          model_names=topmodels,
                                                          save_model_to_disk=False,
                                                          read_model_from_disk=False,
                                                         delete_existing_model_files=False
                                                          )

        # - no where obs where low temperature.
        VarName = 'No coastal <280K'
        bool1 = dfA['WOA_TEMP_K'].values < 280
        bool2 = dfA['Coastal'].values == 1.0
        index2drop = dfA.loc[bool1 & bool2, :].index
        df = dfA.drop(index2drop)
        # reset index of updated DataFrame (and save out the rm'd data prior)
        df2plot = dfA.drop(df.index)
        df.index = np.arange(df.shape[0])
        # Reset the training/withhel data split
        returned_vars = mk_iodide_ML_testing_and_training_set(df=df.copy(),
                                                              random_20_80_split=False,
                                                              random_strat_split=True,
                                                    testing_features=df.columns.tolist(),
                                                         test_plots_of_iodide_dist=False,
                                                              )
        train_set, test_set, test_set_targets = returned_vars
        key_varname = 'Test set ({})'.format('strat. 20%')
        df[key_varname] = False
        df.loc[test_set.index, key_varname] = True
        df.loc[train_set.index, key_varname] = False
        # print the size of the input set
        N = float(df.shape[0])
        Nvals[VarName] = N
        prt_str = "N={:.0f} ({:.2f} % of total) for '{}'"
        if verbose:
            print(prt_str.format(N, N/NA*100, VarName))
        # Test the locations?
        if plt_excluded_obs_locations:
            import seaborn as sns
            sns.reset_orig()
            lats = df2plot['Latitude'].values
            lons = df2plot['Longitude'].values
            title4plt = "Points excluded (N={}) for \n '{}'".format(
                int(NA-N), VarName)
            AC.plot_lons_lats_spatial_on_map(
                lats=lats, lons=lons, title=title4plt)
            savestr = 'Oi_prj_locations4data_split_{}'.format(VarName)
            savestr = AC.rm_spaces_and_chars_from_str(savestr)
            plt.savefig(savestr, dpi=320)
            plt.close()
        # rebuild (just the top models)
        RFR_dict_d[VarName] = build_or_get_current_models(df=df,
                                                          model_names=topmodels,
                                                          save_model_to_disk=False,
                                                          read_model_from_disk=False,
                                                      delete_existing_model_files=False
                                                          )

        # - no where obs where low temperature and coastal (NH)
        VarName = 'No NH coastal <280K'
        bool1 = dfA['WOA_TEMP_K'].values < 280
        bool2 = dfA['Coastal'].values == 1.0
    #    datasets2inc = [
    #    u'Truesdale_U_2003', u'Truesdale_2003_I', u'Luther_1988', u'Wong_C_1998'
    #    ]
    #    bool3 = dfA['Data_Key'].values in datasets2inc
        bool3 = dfA['Latitude'].values > 0
        index2drop = dfA.loc[bool1 & bool2 & bool3, :].index
        df = dfA.drop(index2drop)
        # reset index of updated DataFrame (and save out the rm'd data prior)
        df2plot = dfA.drop(df.index)
        df.index = np.arange(df.shape[0])
        # Reset the training/withhel data split
        returned_vars = mk_iodide_ML_testing_and_training_set(df=df.copy(),
                                                              random_20_80_split=False,
                                                              random_strat_split=True,
                                                     testing_features=df.columns.tolist(),
                                                          test_plots_of_iodide_dist=False,
                                                              )
        train_set, test_set, test_set_targets = returned_vars
        key_varname = 'Test set ({})'.format('strat. 20%')
        df[key_varname] = False
        df.loc[test_set.index, key_varname] = True
        df.loc[train_set.index, key_varname] = False
        # print the size of the input set
        N = float(df.shape[0])
        Nvals[VarName] = N
        prt_str = "N={:.0f} ({:.2f} % of total) for '{}'"
        if verbose:
            print(prt_str.format(N, N/NA*100, VarName))
        # Test the locations?
        if plt_excluded_obs_locations:
            import seaborn as sns
            sns.reset_orig()
            lats = df2plot['Latitude'].values
            lons = df2plot['Longitude'].values
            title4plt = "Points excluded (N={}) for \n '{}'".format(
                int(NA-N), VarName)
            AC.plot_lons_lats_spatial_on_map(
                lats=lats, lons=lons, title=title4plt)
            savestr = 'Oi_prj_locations4data_split_{}'.format(VarName)
            savestr = AC.rm_spaces_and_chars_from_str(savestr)
            plt.savefig(savestr, dpi=320)
            plt.close()
        # rebuild (just the top models)
        RFR_dict_d[VarName] = build_or_get_current_models(df=df,
                                                          model_names=topmodels,
                                                          save_model_to_disk=False,
                                                          read_model_from_disk=False,
                                                        delete_existing_model_files=False
                                                          )

        # - no where obs where low temperature and coastal (NH)
        VarName = 'No NH coastal <280K, Salinity <30'
        bool1 = dfA['WOA_TEMP_K'].values < 280
        bool2 = dfA['Coastal'].values == 1.0
    #    datasets2inc = [
    #    u'Truesdale_U_2003', u'Truesdale_2003_I', u'Luther_1988', u'Wong_C_1998'
    #    ]
    #    bool3 = dfA['Data_Key'].values in datasets2inc
        bool3 = dfA['Latitude'].values > 0
        bool4 = dfA['WOA_Salinity'].values < 30
        index2drop = dfA.loc[bool1 & bool2 & bool3 & bool4, :].index
        df = dfA.drop(index2drop)
        # reset index of updated DataFrame (and save out the rm'd data prior)
        df2plot = dfA.drop(df.index)
        df.index = np.arange(df.shape[0])
        # Reset the training/withhel data split
        returned_vars = mk_iodide_ML_testing_and_training_set(df=df.copy(),
                                                              random_20_80_split=False,
                                                              random_strat_split=True,
                                                     testing_features=df.columns.tolist(),
                                                        test_plots_of_iodide_dist=False,
                                                              )
        train_set, test_set, test_set_targets = returned_vars
        key_varname = 'Test set ({})'.format('strat. 20%')
        df[key_varname] = False
        df.loc[test_set.index, key_varname] = True
        df.loc[train_set.index, key_varname] = False
        # print the size of the input set
        N = float(df.shape[0])
        Nvals[VarName] = N
        prt_str = "N={:.0f} ({:.2f} % of total) for '{}'"
        if verbose:
            print(prt_str.format(N, N/NA*100, VarName))
        # Test the locations?
        if plt_excluded_obs_locations:
            import seaborn as sns
            sns.reset_orig()
            lats = df2plot['Latitude'].values
            lons = df2plot['Longitude'].values
            title4plt = "Points excluded (N={}) for \n '{}'".format(
                int(NA-N), VarName)
            AC.plot_lons_lats_spatial_on_map(
                lats=lats, lons=lons, title=title4plt)
            savestr = 'Oi_prj_locations4data_split_{}'.format(VarName)
            savestr = AC.rm_spaces_and_chars_from_str(savestr)
            plt.savefig(savestr, dpi=320)
            plt.close()
        # rebuild (just the top models)
        RFR_dict_d[VarName] = build_or_get_current_models(df=df,
                                                          model_names=topmodels,
                                                          save_model_to_disk=False,
                                                          read_model_from_disk=False,
                                                        delete_existing_model_files=False
                                                          )

        # - Get just the obs above 65S
        VarName = 'No <65S'
        df = dfA.loc[dfA['Latitude'].values >= -65, :]
        # reset index of updated DataFrame
        df.index = np.arange(df.shape[0])
        # Reset the training/withhel data split
        returned_vars = mk_iodide_ML_testing_and_training_set(df=df.copy(),
                                                              random_20_80_split=False,
                                                              random_strat_split=True,
                                                     testing_features=df.columns.tolist(),
                                                        test_plots_of_iodide_dist=False,
                                                              )
        train_set, test_set, test_set_targets = returned_vars
        key_varname = 'Test set ({})'.format('strat. 20%')
        df[key_varname] = False
        df.loc[test_set.index, key_varname] = True
        df.loc[train_set.index, key_varname] = False
        # print the size of the input set
        N = float(df.shape[0])
        Nvals[VarName] = N
        prt_str = "N={:.0f} ({:.2f} % of total) for '{}'"
        if verbose:
            print(prt_str.format(N, N/NA*100, VarName))
        # rebuild (just the top models)
        RFR_dict_d[VarName] = build_or_get_current_models(df=df,
                                                          model_names=topmodels,
                                                          save_model_to_disk=False,
                                                          read_model_from_disk=False,
                                                        delete_existing_model_files=False
                                                          )

        # - Get all except the obs below 65S which are coastal
        VarName = 'No coastal <65S'
        bool1 = dfA['Latitude'].values <= -65
        bool2 = dfA['Coastal'].values == 1.0
        index2drop = dfA.loc[bool1 & bool2, :].index
        df = dfA.drop(index2drop)
        # reset index of updated DataFrame
        df.index = np.arange(df.shape[0])
        # Reset the training/withhel data split
        returned_vars = mk_iodide_ML_testing_and_training_set(df=df.copy(),
                                                              random_20_80_split=False,
                                                              random_strat_split=True,
                                                     testing_features=df.columns.tolist(),
                                                          test_plots_of_iodide_dist=False,
                                                              )
        train_set, test_set, test_set_targets = returned_vars
        key_varname = 'Test set ({})'.format('strat. 20%')
        df[key_varname] = False
        df.loc[test_set.index, key_varname] = True
        df.loc[train_set.index, key_varname] = False
        # print the size of the input set
        N = float(df.shape[0])
        Nvals[VarName] = N
        prt_str = "N={:.0f} ({:.2f} % of total) for '{}'"
        if verbose:
            print(prt_str.format(N, N/NA*100, VarName))
        # rebuild (just the top models)
        RFR_dict_d[VarName] = build_or_get_current_models(df=df,
                                                          model_names=topmodels,
                                                          save_model_to_disk=False,
                                                          read_model_from_disk=False,
                                                        delete_existing_model_files=False
                                                          )

        # - Get just the obs above 55S
        VarName = 'No <55S'
        df = dfA.loc[dfA['Latitude'].values >= -55, :]
        # reset index of updated DataFrame
        df.index = np.arange(df.shape[0])
        # Reset the training/withhel data split
        returned_vars = mk_iodide_ML_testing_and_training_set(df=df.copy(),
                                                              random_20_80_split=False,
                                                              random_strat_split=True,
                                                    testing_features=df.columns.tolist(),
                                                        test_plots_of_iodide_dist=False,
                                                              )
        train_set, test_set, test_set_targets = returned_vars
        key_varname = 'Test set ({})'.format('strat. 20%')
        df[key_varname] = False
        df.loc[test_set.index, key_varname] = True
        df.loc[train_set.index, key_varname] = False
        # print the size of the input set
        N = float(df.shape[0])
        Nvals[VarName] = N
        prt_str = "N={:.0f} ({:.2f} % of total) for '{}'"
        if verbose:
            print(prt_str.format(N, N/NA*100, VarName))
        # rebuild (just the top models)
        RFR_dict_d[VarName] = build_or_get_current_models(df=df,
                                                          model_names=topmodels,
                                                          save_model_to_disk=False,
                                                          read_model_from_disk=False,
                                                         delete_existing_model_files=False
                                                          )
        # - Get just the obs above 45S
        VarName = 'No <45S'
        df = dfA.loc[dfA['Latitude'].values >= -45, :]
        # reset index of updated DataFrame
        df.index = np.arange(df.shape[0])
        # Reset the training/withhel data split
        returned_vars = mk_iodide_ML_testing_and_training_set(df=df.copy(),
                                                              random_20_80_split=False,
                                                              random_strat_split=True,
                                                     testing_features=df.columns.tolist(),
                                                    test_plots_of_iodide_dist=False,
                                                              )
        train_set, test_set, test_set_targets = returned_vars
        key_varname = 'Test set ({})'.format('strat. 20%')
        df[key_varname] = False
        df.loc[test_set.index, key_varname] = True
        df.loc[train_set.index, key_varname] = False
        # print the size of the input set
        N = float(df.shape[0])
        Nvals[VarName] = N
        prt_str = "N={:.0f} ({:.2f} % of total) for '{}'"
        if verbose:
            print(prt_str.format(N, N/NA*100, VarName))
        # rebuild (just the top models)
        RFR_dict_d[VarName] = build_or_get_current_models(df=df,
                                                          model_names=topmodels,
                                                          save_model_to_disk=False,
                                                          read_model_from_disk=False,
                                                    delete_existing_model_files=False
                                                          )

        # - Get just the obs above 45S
        VarName = 'No coastal <45S'
        bool1 = dfA['Latitude'].values <= -45
        bool2 = dfA['Coastal'].values == 1.0
        index2drop = dfA.loc[bool1 & bool2, :].index
        df = dfA.drop(index2drop)
        # reset index of updated DataFrame
        df.index = np.arange(df.shape[0])
        # Reset the training/withhel data split
        returned_vars = mk_iodide_ML_testing_and_training_set(df=df.copy(),
                                                              random_20_80_split=False,
                                                              random_strat_split=True,
                                                     testing_features=df.columns.tolist(),
                                                        test_plots_of_iodide_dist=False,
                                                              )
        train_set, test_set, test_set_targets = returned_vars
        key_varname = 'Test set ({})'.format('strat. 20%')
        df[key_varname] = False
        df.loc[test_set.index, key_varname] = True
        df.loc[train_set.index, key_varname] = False
        # print the size of the input set
        N = float(df.shape[0])
        Nvals[VarName] = N
        prt_str = "N={:.0f} ({:.2f} % of total) for '{}'"
        if verbose:
            print(prt_str.format(N, N/NA*100, VarName))
        # rebuild (just the top models)
        RFR_dict_d[VarName] = build_or_get_current_models(df=df,
                                                          model_names=topmodels,
                                                          save_model_to_disk=False,
                                                          read_model_from_disk=False,
                                                    delete_existing_model_files=False
                                                          )

    # --- Include options that didn't improve things in PDF
    if plt_option_tired_but_didnt_help:
        # - where obs are Just coastal
        VarName = 'Just Coastal'
        df = dfA.loc[dfA['Coastal'] == 1.0, :]
        # reset index of updated DataFrame
        df.index = np.arange(df.shape[0])
        # Reset the training/withhel data split
        returned_vars = mk_iodide_ML_testing_and_training_set(df=df.copy(),
                                                              random_20_80_split=False,
                                                              random_strat_split=True,
                                                     testing_features=df.columns.tolist(),
                                                        test_plots_of_iodide_dist=False
                                                              )
        train_set, test_set, test_set_targets = returned_vars
        key_varname = 'Test set ({})'.format('strat. 20%')
        df[key_varname] = False
        df.loc[test_set.index, key_varname] = True
        df.loc[train_set.index, key_varname] = False
        # print the size of the input set
        N = float(df.shape[0])
        Nvals[VarName] = N
        prt_str = "N={:.0f} ({:.2f} % of total) for '{}'"
        if verbose:
            print(prt_str.format(N, N/NA*100, VarName))
        # rebuild (just the top models)
        RFR_dict_d[VarName] = build_or_get_current_models(df=df,
                                                          model_names=topmodels,
                                                          save_model_to_disk=False,
                                                          read_model_from_disk=False,
                                                    delete_existing_model_files=False
                                                          )

        # - no where obs where low temperature.
        VarName = 'No <17.4C'
        df = dfA.loc[dfA['WOA_TEMP_K'].values >= (273.15+17.4), :]
        # reset index of updated DataFrame
        df.index = np.arange(df.shape[0])
        # Reset the training/withhel data split
        returned_vars = mk_iodide_ML_testing_and_training_set(df=df.copy(),
                                                              random_20_80_split=False,
                                                              random_strat_split=True,
                                                     testing_features=df.columns.tolist(),
                                                        test_plots_of_iodide_dist=False,
                                                              )
        train_set, test_set, test_set_targets = returned_vars
        key_varname = 'Test set ({})'.format('strat. 20%')
        df[key_varname] = False
        df.loc[test_set.index, key_varname] = True
        df.loc[train_set.index, key_varname] = False
        # print the size of the input set
        N = float(df.shape[0])
        Nvals[VarName] = N
        prt_str = "N={:.0f} ({:.2f} % of total) for '{}'"
        if verbose:
            print(prt_str.format(N, N/NA*100, VarName))
        # rebuild (just the top models)
        RFR_dict_d[VarName] = build_or_get_current_models(df=df,
                                                          model_names=topmodels,
                                                          save_model_to_disk=False,
                                                          read_model_from_disk=False,
                                                        delete_existing_model_files=False
                                                          )

        # where obs are low nitrate and low temperature.
        VarName = 'No <17.4C & no<0.15 Nitrate'
        bool1 = dfA['WOA_TEMP_K'].values >= (273.15+17.4)
        bool2 = dfA['WOA_Nitrate'].values >= 0.15
        df = dfA.loc[bool1 & bool2, :]
        # reset index of updated DataFrame
        df.index = np.arange(df.shape[0])
        # Reset the training/withhel data split
        returned_vars = mk_iodide_ML_testing_and_training_set(df=df.copy(),
                                                              random_20_80_split=False,
                                                              random_strat_split=True,
                                                    testing_features=df.columns.tolist(),
                                                    test_plots_of_iodide_dist=False
                                                              )
        train_set, test_set, test_set_targets = returned_vars
        key_varname = 'Test set ({})'.format('strat. 20%')
        df[key_varname] = False
        df.loc[test_set.index, key_varname] = True
        df.loc[train_set.index, key_varname] = False
        # print the size of the input set
        N = float(df.shape[0])
        Nvals[VarName] = N
        prt_str = "N={:.0f} ({:.2f} % of total) for '{}'"
        if verbose:
            print(prt_str.format(N, N/NA*100, VarName))
        # rebuild (just the top models)
        RFR_dict_d[VarName] = build_or_get_current_models(df=df,
                                                          model_names=topmodels,
                                                          save_model_to_disk=False,
                                                          read_model_from_disk=False,
                                                        delete_existing_model_files=False
                                                          )

        # - No cold of coastal
        VarName = 'No <17.4C & no coastal'
        bool1 = dfA['WOA_TEMP_K'].values >= (273.15+17.4)
        bool2 = dfA['Coastal'].values != 1.0
        df = dfA.loc[bool1 | bool2, :]
        # reset index of updated DataFrame
        df.index = np.arange(df.shape[0])
        # Reset the training/withhel data split
        returned_vars = mk_iodide_ML_testing_and_training_set(df=df.copy(),
                                                              random_20_80_split=False,
                                                              random_strat_split=True,
                                                     testing_features=df.columns.tolist(),
                                                        test_plots_of_iodide_dist=False
                                                              )
        train_set, test_set, test_set_targets = returned_vars
        key_varname = 'Test set ({})'.format('strat. 20%')
        df[key_varname] = False
        df.loc[test_set.index, key_varname] = True
        df.loc[train_set.index, key_varname] = False
        # print the size of the input set
        N = float(df.shape[0])
        Nvals[VarName] = N
        prt_str = "N={:.0f} ({:.2f} % of total) for '{}'"
        if verbose:
            print(prt_str.format(N, N/NA*100, VarName))
        # rebuild (just the top models)
        RFR_dict_d[VarName] = build_or_get_current_models(df=df,
                                                          model_names=topmodels,
                                                          save_model_to_disk=False,
                                                          read_model_from_disk=False,
                                                        delete_existing_model_files=False
                                                          )

        # - where obs are low nitrate
        VarName = 'No <0.5 Nitrate'
        df = dfA.loc[dfA['WOA_Nitrate'] >= 0.5, :]
        # reset index of updated DataFrame
        df.index = np.arange(df.shape[0])
        # Reset the training/withhel data split
        returned_vars = mk_iodide_ML_testing_and_training_set(df=df.copy(),
                                                              random_20_80_split=False,
                                                              random_strat_split=True,
                                                     testing_features=df.columns.tolist(),
                                                        test_plots_of_iodide_dist=False
                                                              )
        train_set, test_set, test_set_targets = returned_vars
        key_varname = 'Test set ({})'.format('strat. 20%')
        df[key_varname] = False
        df.loc[test_set.index, key_varname] = True
        df.loc[train_set.index, key_varname] = False
        # print the size of the input set
        N = float(df.shape[0])
        Nvals[VarName] = N
        prt_str = "N={:.0f} ({:.2f} % of total) for '{}'"
        if verbose:
            print(prt_str.format(N, N/NA*100, VarName))
        # rebuild (just the top models)
        RFR_dict_d[VarName] = build_or_get_current_models(df=df,
                                                          model_names=topmodels,
                                                          save_model_to_disk=False,
                                                          read_model_from_disk=False,
                                                         delete_existing_model_files=False
                                                          )

        # Clean memory
        gc.collect()

        # - where obs are low nitrate
        VarName = 'No <1 Nitrate'
        df = dfA.loc[dfA['WOA_Nitrate'] >= 1.0, :]
        # reset index of updated DataFrame
        df.index = np.arange(df.shape[0])
        # Reset the training/withhel data split
        returned_vars = mk_iodide_ML_testing_and_training_set(df=df.copy(),
                                                              random_20_80_split=False,
                                                              random_strat_split=True,
                                                     testing_features=df.columns.tolist(),
                                                           test_plots_of_iodide_dist=False
                                                              )
        train_set, test_set, test_set_targets = returned_vars
        key_varname = 'Test set ({})'.format('strat. 20%')
        df[key_varname] = False
        df.loc[test_set.index, key_varname] = True
        df.loc[train_set.index, key_varname] = False
        # print the size of the input set
        N = float(df.shape[0])
        Nvals[VarName] = N
        prt_str = "N={:.0f} ({:.2f} % of total) for '{}'"
        if verbose:
            print(prt_str.format(N, N/NA*100, VarName))
        # rebuild (just the top models)
        RFR_dict_d[VarName] = build_or_get_current_models(df=df,
                                                          model_names=topmodels,
                                                          save_model_to_disk=False,
                                                          read_model_from_disk=False,
                                                        delete_existing_model_files=False
                                                          )

    # Clean memory
    gc.collect()

    # --- Make predictions for dataset splits
    dssI = {}
    # -  first BASE
    n_keys = float(len(RFR_dict_d.keys()))
    for n_key_, key_ in enumerate(RFR_dict_d.keys()):
        # Print status to screen
        prt_str = "Predicting for '{}' @ {} and mk'ing Dataset object ({:.2f}%) - {}"
        Tnow = strftime("%c", gmtime())
        if verbose:
            print(prt_str.format(key_, res, n_key_/n_keys*100, Tnow))
        # Predict the values for the locations
        ds_tmp = mk_iodide_predictions_from_ancillaries(None,
                                                        dsA=ds, RFR_dict=RFR_dict_d[key_],
                                                       use_updated_predictor_NetCDF=False,
                                                        save2NetCDF=False,
                                                        topmodels=topmodels,
                                                        models2compare=topmodels)
        # Add ensemble to ds
        ds_tmp = add_ensemble_avg_std_to_dataset(ds=ds_tmp,
                                                 RFR_dict=RFR_dict_d[key_],
                                                 topmodels=topmodels,
                                                 res=res,
                                                 save2NetCDF=False)
        # Save to dict
        dssI[key_] = ds_tmp
        del ds_tmp
        # Clean memory
        gc.collect()

    # Clean memory
    gc.collect()
    # --- plot these up
    # setup a PDF
    savetitle = 'Oi_prj_test_impact_changing_input_features_Arctic_DATA_DENIAL_{}'
    savetitle += '_JUST_SKAGERAK_earth0'
    savetitle = savetitle.format(res)
    pdff = AC.plot2pdfmulti(title=savetitle, open=True, dpi=dpi)
    # make sure BASE is first
    var_ = 'BASE'
    vars2plot = dssI.keys()
    del vars2plot[vars2plot.index(var_)]
    vars2plot = [var_] + list(sorted(vars2plot))
    # loop and plot
    for n_key_, key_ in enumerate(vars2plot):
        # Print status to screen
        prt_str = "Plotting for {} @ {} and mk'ing Dataset object ({:.2f}%) - {}"
        Tnow = strftime("%c", gmtime())
        Pcent = n_key_/float(len(vars2plot))*100
        if verbose:
            print(prt_str.format(key_, res, Pcent, Tnow))
        # Plot up as a latitudeinal plot
        plot_predicted_iodide_vs_lat_figure_ENSEMBLE(ds=dssI[key_].copy(),
                                                     RFR_dict=RFR_dict, res=res,
                                                     show_plot=False, close_plot=False,
                                                     save_plot=False, topmodels=topmodels,
                                                     )
        plt_str = "Obs.+Params. vs Lat for '{}' (N={}, {:.2f}% of dataset)"
        plt.title(plt_str.format(key_, Nvals[key_], Nvals[key_]/NA*100))
        # Save to PDF and close plot
        AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
        plt.close()
        # Clean memory
        gc.collect()
    # Save entire pdf
    AC.plot2pdfmulti(pdff, savetitle, close=True, dpi=dpi)


def explore_sensitivity_of_65N(res='4x5'):
    """ Explore sensitivty of iodide parameterisations to  input vairables """
    # --- Local variables
    # Get the core input variables
    iodide_dir = get_file_locations('iodide_data')
    filename = 'Oi_prj_feature_variables_{}.nc'.format(res)
    ds = xr.open_dataset(iodide_dir + filename)
    # Get the models
    if isinstance(RFR_dict, type(None)):
        RFR_dict = build_or_get_current_models()
    topmodels = get_top_models(RFR_dict=RFR_dict, NO_DERIVED=True, n=10)

    # set up a dictionary for different dataset splits
    dss = {}
    # include the starting output for reference
    dss['BASE'] = ds.copy()
    # Set the depth to a generic value (-4100 )
    lat_above2set = 65.
    var2set = 'Depth_GEBCO'
    fixed_value2use = -4500.
    var_ = 'Depth set to {} >={}N'.format(fixed_value2use, lat_above2set)
    dss[var_] = set_values_at_of_var_above_X_lat_2_avg(var2set=var2set,
                                                       fixed_value2use=fixed_value2use,
                                                       ds=ds.copy(), res=res,
                                                       save2NetCDF=False,
                                                       lat_above2set=lat_above2set)
    # Set the depth to a generic value (-4100 )
    var2set = 'Depth_GEBCO'
    fixed_value2use = -10000.
    var_ = 'Depth set to {} >={}N'.format(fixed_value2use, lat_above2set)
    dss[var_] = set_values_at_of_var_above_X_lat_2_avg(var2set=var2set,
                                                       fixed_value2use=fixed_value2use,
                                                       ds=ds.copy(), res=res,
                                                       save2NetCDF=False,
                                                       lat_above2set=lat_above2set)
    # set temperature to global mean above x
    var2set = 'WOA_Phosphate'
    fixed_value2use = 0.5
    var_ = 'Phos set to {} >={}N'.format(fixed_value2use, lat_above2set)
    dss[var_] = set_values_at_of_var_above_X_lat_2_avg(var2set=var2set,
                                                       fixed_value2use=fixed_value2use,
                                                       ds=ds.copy(), res=res,
                                                       save2NetCDF=False,
                                                       lat_above2set=lat_above2set)
    # set temperature to global mean above x
    var2set = 'WOA_Nitrate'
    fixed_value2use = 1.
    var_ = 'Nitrate set to {} >={}N'.format(fixed_value2use, lat_above2set)
    dss[var_] = set_values_at_of_var_above_X_lat_2_avg(var2set=var2set,
                                                       fixed_value2use=fixed_value2use,
                                                       ds=ds.copy(), res=res,
                                                       save2NetCDF=False,
                                                       lat_above2set=lat_above2set)
    # set temperature to global mean above x
    var2set = 'WOA_Nitrate'
    fixed_value2use = 1.5
    var_ = 'Nitrate set to {} >={}N'.format(fixed_value2use, lat_above2set)
    dss[var_] = set_values_at_of_var_above_X_lat_2_avg(var2set=var2set,
                                                       fixed_value2use=fixed_value2use,
                                                       ds=ds.copy(), res=res,
                                                       save2NetCDF=False,
                                                       lat_above2set=lat_above2set)
    # set temperature to global mean above x
    var2set = 'WOA_Nitrate'
    fixed_value2use = 2.
    var_ = 'Nitrate set to {} >={}N'.format(fixed_value2use, lat_above2set)
    dss[var_] = set_values_at_of_var_above_X_lat_2_avg(var2set=var2set,
                                                       fixed_value2use=fixed_value2use,
                                                       ds=ds.copy(), res=res,
                                                       save2NetCDF=False,
                                                       lat_above2set=lat_above2set)
    # set temperature to global mean above x
    var2set = 'WOA_Nitrate'
    fixed_value2use = 3.
    var_ = 'Nitrate set to {} >={}N'.format(fixed_value2use, lat_above2set)
    dss[var_] = set_values_at_of_var_above_X_lat_2_avg(var2set=var2set,
                                                       fixed_value2use=fixed_value2use,
                                                       ds=ds.copy(), res=res,
                                                       save2NetCDF=False,
                                                       lat_above2set=lat_above2set)
    # set temperature to global mean above x
    var2set = 'WOA_Nitrate'
    fixed_value2use = 4.
    var_ = 'Nitrate set to {} >={}N'.format(fixed_value2use, lat_above2set)
    dss[var_] = set_values_at_of_var_above_X_lat_2_avg(var2set=var2set,
                                                       fixed_value2use=fixed_value2use,
                                                       ds=ds.copy(), res=res,
                                                       save2NetCDF=False,
                                                       lat_above2set=lat_above2set)
    # set temperature to global mean above x
    var2set = 'WOA_Nitrate'
    fixed_value2use = 5.
    var_ = 'Nitrate set to {} >={}N'.format(fixed_value2use, lat_above2set)
    dss[var_] = set_values_at_of_var_above_X_lat_2_avg(var2set=var2set,
                                                       fixed_value2use=fixed_value2use,
                                                       ds=ds.copy(), res=res,
                                                       save2NetCDF=False,
                                                       lat_above2set=lat_above2set)
    # set temperature to global mean above x
    var2set = 'WOA_Nitrate'
    fixed_value2use = 6.
    var_ = 'Nitrate set to {} >={}N'.format(fixed_value2use, lat_above2set)
    dss[var_] = set_values_at_of_var_above_X_lat_2_avg(var2set=var2set,
                                                       fixed_value2use=fixed_value2use,
                                                       ds=ds.copy(), res=res,
                                                       save2NetCDF=False,
                                                       lat_above2set=lat_above2set)
    # set temperature to global mean above x
    var2set = 'WOA_Nitrate'
    fixed_value2use = 7.
    var_ = 'Nitrate set to {} >={}N'.format(fixed_value2use, lat_above2set)
    dss[var_] = set_values_at_of_var_above_X_lat_2_avg(var2set=var2set,
                                                       fixed_value2use=fixed_value2use,
                                                       ds=ds.copy(), res=res,
                                                       save2NetCDF=False,
                                                       lat_above2set=lat_above2set)
    # set temperature to global mean above x
    var2set = 'WOA_Nitrate'
    fixed_value2use = 8.
    var_ = 'Nitrate set to {} >={}N'.format(fixed_value2use, lat_above2set)
    dss[var_] = set_values_at_of_var_above_X_lat_2_avg(var2set=var2set,
                                                       fixed_value2use=fixed_value2use,
                                                       ds=ds.copy(), res=res,
                                                       save2NetCDF=False,
                                                       lat_above2set=lat_above2set)
    # set temperature to global mean above x
    var2set = 'WOA_Nitrate'
    fixed_value2use = 9.
    var_ = 'Nitrate set to {} >={}N'.format(fixed_value2use, lat_above2set)
    dss[var_] = set_values_at_of_var_above_X_lat_2_avg(var2set=var2set,
                                                       fixed_value2use=fixed_value2use,
                                                       ds=ds.copy(), res=res,
                                                       save2NetCDF=False,
                                                       lat_above2set=lat_above2set)
    # set temperature to global mean above x
    var2set = 'WOA_Nitrate'
    fixed_value2use = 10.
    var_ = 'Nitrate set to {} >={}N'.format(fixed_value2use, lat_above2set)
    dss[var_] = set_values_at_of_var_above_X_lat_2_avg(var2set=var2set,
                                                       fixed_value2use=fixed_value2use,
                                                       ds=ds.copy(), res=res,
                                                       save2NetCDF=False,
                                                       lat_above2set=lat_above2set)
    # set temperature to global mean above x
    var2set = 'WOA_Nitrate'
    fixed_value2use = 15.
    var_ = 'Nitrate set to {} >={}N'.format(fixed_value2use, lat_above2set)
    dss[var_] = set_values_at_of_var_above_X_lat_2_avg(var2set=var2set,
                                                       fixed_value2use=fixed_value2use,
                                                       ds=ds.copy(), res=res,
                                                       save2NetCDF=False,
                                                       lat_above2set=lat_above2set)
    # set temperature to global mean above x
    var2set = 'WOA_Nitrate'
    fixed_value2use = 20.
    var_ = 'Nitrate set to {} >={}N'.format(fixed_value2use, lat_above2set)
    dss[var_] = set_values_at_of_var_above_X_lat_2_avg(var2set=var2set,
                                                       fixed_value2use=fixed_value2use,
                                                       ds=ds.copy(), res=res,
                                                       save2NetCDF=False,
                                                       lat_above2set=lat_above2set)
    # set temperature to global mean above x
    var2set = 'WOA_Salinity'
    fixed_value2use = 30.
    var_ = 'Salinity set to {} >={}N'.format(fixed_value2use, lat_above2set)
    dss[var_] = set_values_at_of_var_above_X_lat_2_avg(var2set=var2set,
                                                       fixed_value2use=fixed_value2use,
                                                       ds=ds.copy(), res=res,
                                                       save2NetCDF=False,
                                                       lat_above2set=lat_above2set)
    # set temperature to global mean above x
    var2set = 'WOA_Salinity'
    fixed_value2use = 34.
    var_ = 'Salinity set to {} >={}N'.format(fixed_value2use, lat_above2set)
    dss[var_] = set_values_at_of_var_above_X_lat_2_avg(var2set=var2set,
                                                       fixed_value2use=fixed_value2use,
                                                       ds=ds.copy(), res=res,
                                                       save2NetCDF=False,
                                                       lat_above2set=lat_above2set)
    # set temperature to global mean above x
    var2set = 'WOA_Salinity'
    fixed_value2use = 35.
    var_ = 'Salinity set to {} >={}N'.format(fixed_value2use, lat_above2set)
    dss[var_] = set_values_at_of_var_above_X_lat_2_avg(var2set=var2set,
                                                       fixed_value2use=fixed_value2use,
                                                       ds=ds.copy(), res=res,
                                                       save2NetCDF=False,
                                                       lat_above2set=lat_above2set)
    # set temperature to global mean above x
    var2set = 'WOA_TEMP_K'
    fixed_value2use = 273.
    var_ = 'Temp. set to {} >={}N'.format(fixed_value2use, lat_above2set)
    dss[var_] = set_values_at_of_var_above_X_lat_2_avg(var2set=var2set,
                                                       fixed_value2use=fixed_value2use,
                                                       ds=ds.copy(), res=res,
                                                       save2NetCDF=False,
                                                       lat_above2set=lat_above2set)
    # set temperature to global mean above x
    var2set = 'WOA_TEMP_K'
    fixed_value2use = 275.
    var_ = 'Temp. set to {} >={}N'.format(fixed_value2use, lat_above2set)
    dss[var_] = set_values_at_of_var_above_X_lat_2_avg(var2set=var2set,
                                                       fixed_value2use=fixed_value2use,
                                                       ds=ds.copy(), res=res,
                                                       save2NetCDF=False,
                                                       lat_above2set=lat_above2set)
    # set temperature to global mean above x
    var2set = 'WOA_TEMP_K'
    fixed_value2use = 276.
    var_ = 'Temp. set to {} >={}N'.format(fixed_value2use, lat_above2set)
    dss[var_] = set_values_at_of_var_above_X_lat_2_avg(var2set=var2set,
                                                       fixed_value2use=fixed_value2use,
                                                       ds=ds.copy(), res=res,
                                                       save2NetCDF=False,
                                                       lat_above2set=lat_above2set)
    # set temperature to global mean above x
    var2set = 'WOA_TEMP_K'
    fixed_value2use = 278.
    var_ = 'Temp. set to {} >={}N'.format(fixed_value2use, lat_above2set)
    ds_tmp = set_values_at_of_var_above_X_lat_2_avg(var2set=var2set,
                                                    fixed_value2use=fixed_value2use,
                                                    ds=ds.copy(), res=res,
                                                    save2NetCDF=False,
                                                    lat_above2set=lat_above2set)
    dss[var_] = ds_tmp
    # set temperature to global mean above x
    var2set = 'WOA_TEMP_K'
    fixed_value2use = 279.
    var_ = 'Temp. set to {} >={}N'.format(fixed_value2use, lat_above2set)
    ds_tmp = set_values_at_of_var_above_X_lat_2_avg(var2set=var2set,
                                                    fixed_value2use=fixed_value2use,
                                                    ds=ds.copy(), res=res,
                                                    save2NetCDF=False,
                                                    lat_above2set=lat_above2set)
    dss[var_] = ds_tmp
    # set temperature to global mean above x
    var2set = 'WOA_TEMP_K'
    fixed_value2use = 281.
    var_ = 'Temp. set to {} >={}N'.format(fixed_value2use, lat_above2set)
    ds_tmp = set_values_at_of_var_above_X_lat_2_avg(var2set=var2set,
                                                    fixed_value2use=fixed_value2use,
                                                    ds=ds.copy(), res=res,
                                                    save2NetCDF=False,
                                                    lat_above2set=lat_above2set)
    dss[var_] = ds_tmp
    # set temperature to global mean above x
    var2set = 'WOA_TEMP_K'
    fixed_value2use = 282.
    var_ = 'Temp. set to {} >={}N'.format(fixed_value2use, lat_above2set)
    ds_tmp = set_values_at_of_var_above_X_lat_2_avg(var2set=var2set,
                                                    fixed_value2use=fixed_value2use,
                                                    ds=ds.copy(), res=res,
                                                    save2NetCDF=False,
                                                    lat_above2set=lat_above2set)
    dss[var_] = ds_tmp
    # set temperature to global mean above x
    var2set = 'WOA_TEMP_K'
    fixed_value2use = 285.
    var_ = 'Temp. set to {} >={}N'.format(fixed_value2use, lat_above2set)
    ds_tmp = set_values_at_of_var_above_X_lat_2_avg(var2set=var2set,
                                                    fixed_value2use=fixed_value2use,
                                                    ds=ds.copy(), res=res,
                                                    save2NetCDF=False,
                                                    lat_above2set=lat_above2set)
    # set temperature to global mean above x
    var2set = 'WOA_TEMP_K'
    fixed_value2use = 280.
    var_ = 'Temp. set to {} >={}N'.format(fixed_value2use, lat_above2set)
    ds_tmp = set_values_at_of_var_above_X_lat_2_avg(var2set=var2set,
                                                    fixed_value2use=fixed_value2use,
                                                    ds=ds.copy(), res=res,
                                                    save2NetCDF=False,
                                                    lat_above2set=lat_above2set)
    dss[var_] = ds_tmp
    # set temperature to global mean above and depth of value
    lat_above2set = 65.
    var2set = 'Depth_GEBCO'
    fixed_value2use = -4500.
    var_ = 'Depth set to {} >={}N'.format(fixed_value2use, lat_above2set)
    var_ += '\n (and Temp to 280 K)'
    dss[var_] = set_values_at_of_var_above_X_lat_2_avg(var2set=var2set,
                                                       fixed_value2use=fixed_value2use,
                                                       ds=ds_tmp, res=res,
                                                       save2NetCDF=False,
                                                       lat_above2set=lat_above2set)

    # --- Make predictions for dataset splits
    dssI = {}
    keys2add = [i for i in dss.keys() if i not in dssI.keys()]
    for key_ in keys2add:
        # predict the values for the locations
        ds_tmp = mk_iodide_predictions_from_ancillaries(None,
                                                        dsA=dss[key_], RFR_dict=RFR_dict,
                                                       use_updated_predictor_NetCDF=False,
                                                        save2NetCDF=False,
                                                        topmodels=topmodels)
        # add ensemble to ds
        ds_tmp = add_ensemble_avg_std_to_dataset(ds=ds_tmp,
                                                 RFR_dict=RFR_dict, topmodels=topmodels,
                                                 res=res,
                                                 save2NetCDF=False)
        # save to dict
        dssI[key_] = ds_tmp

    # --- plot these up
    # setup a PDF
    savetitle = 'Oi_prj_test_impact_changing_input_features_Arctic_{}'
    savetitle = savetitle.format(res)
    pdff = AC.plot2pdfmulti(title=savetitle, open=True, dpi=dpi)
    # make sure BASE is first
    var_ = 'BASE'
    vars2plot = dssI.keys()
    del vars2plot[vars2plot.index(var_)]
    vars2plot = [var_] + list(sorted(vars2plot))
    # loop and plot
    for key_ in vars2plot:
        # plot up as a latitudeinal plot
        plot_predicted_iodide_vs_lat_figure_ENSEMBLE(ds=dssI[key_].copy(),
                                                     RFR_dict=RFR_dict, res=res,
                                                     show_plot=False, close_plot=False,
                                                     save_plot=False, topmodels=topmodels,
                                                     )
        plt.title("Obs.+Params. vs Lat for '{}'".format(key_))
        # Save to PDF and close plot
        AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
        plt.close()
    # Save entire pdf
    AC.plot2pdfmulti(pdff, savetitle, close=True, dpi=dpi)

    # plot up up each splits prediction
    # (also plot the reference case for comparison)


def plot_predicted_iodide_PDF4region(dpi=320, extr_str='',
                                     plot_avg_as_median=False, RFR_dict=None,
                                     res='0.125x0.125',
                                     show_plot=False, close_plot=True, save_plot=False,
                                     folder=None, ds=None, topmodels=None):
    """ Plot a figure of iodide vs laitude - showing all ensemble members """
    import seaborn as sns
    sns.set(color_codes=True)
    sns.set_context("paper", font_scale=0.75)
    # Get RFR_dict if not provide
    if isinstance(RFR_dict, type(None)):
        RFR_dict = build_or_get_current_models()
    # Get predicted values
    if isinstance(folder, type(None)):
        folder = get_file_locations('iodide_data')
    if isinstance(ds, type(None)):
        filename = 'Oi_prj_predicted_iodide_{}{}.nc'.format(res, extr_str)
        ds = xr.open_dataset(folder + filename)
    # Rename to a more concise name
    print(ds.data_vars)
    # Get predicted values binned by latitude
    if res == '0.125x0.125':
        df = get_spatial_predictions_0125x0125_by_lat(ds=ds)
    else:
        df = get_stats_on_spatial_predictions_4x5_2x25_by_lat(res=res, ds=ds)
    # params to pot
    if isinstance(topmodels, type(None)):
        topmodels = get_top_models(RFR_dict=RFR_dict, NO_DERIVED=True, n=10)
    params2plot = topmodels
    # assign colors
    CB_color_cycle = AC.get_CB_color_cycle()
    CB_color_cycle += ['darkgreen']
    color_d = dict(zip(params2plot, CB_color_cycle))
    # --- plot up vs. lat
    fig, ax = plt.subplots()
    #
    for param in params2plot:
        # Set color for param
        color = color_d[param]
        # Plot average
        if plot_avg_as_median:
            var2plot = '{} - median'.format(param)
        else:
            var2plot = '{} - mean'.format(param)
        # get X
        X = df[var2plot].index.values
        # plot as line
        plt.plot(X, df[var2plot].values, color=color, label=param)
        # plot up quartiles tooo
        low = df['{} - 25%'.format(param)].values
        high = df['{} - 75%'.format(param)].values
        ax.fill_between(X, low, high, alpha=0.2, color=color)

    # highlight coastal obs
    tmp_df = df_obs.loc[df_obs['Coastal'] == True, :]
    X = tmp_df['Latitude'].values
    Y = tmp_df['Iodide'].values
    plt.scatter(X, Y, color='k', marker='D', facecolor='none', s=3,
                label='Coastal obs.')
    # non-coastal obs
    tmp_df = df_obs.loc[df_obs['Coastal'] == False, :]
    X = tmp_df['Latitude'].values
    Y = tmp_df['Iodide'].values
    plt.scatter(X, Y, color='k', marker='D', facecolor='k', s=3,
                label='Non-coastal obs.')
    # limit plot y axis
    plt.ylim(-20, 420)
    plt.ylabel('[I$^{-}_{aq}$] (nM)')
    plt.xlabel('Latitude ($^{\\rm o}$N)')
#    plt.xlim(-80, 80 )
    plt.legend()
    # save or show?
    filename = 'Oi_prj_global_predicted_vals_vs_lat_ENSEMBLE_{}{}'
    if save_plot:
        plt.savefig(filename.format(res, extr_str), dpi=dpi)
    if show_plot:
        plt.show()
    if close_plot:
        plt.close()



def set_values_at_of_var_above_X_lat_2_avg(lat_above2set=65, ds=None,
                                           use_avg_at_lat=True, res='0.125x0.125',
                                           var2set=None,
                                           only_consider_water_boxes=True,
                                           fixed_value2use=None,
                                           save2NetCDF=True):
    """ Set values above a latitude to the monthly lon average """
    print(var2set)
    # local variables
    iodide_dir = get_file_locations('iodide_data')
    # Get existing file
    if isinstance(ds, type(None)):
        filename = 'Oi_prj_feature_variables_{}.nc'.format(res)
        ds = xr.open_dataset(iodide_dir + filename)
    # get the average value at lat
    avg = ds[var2set].sel(lat=lat_above2set, method='nearest')
    # get index of lat to set values from
    idx = AC.find_nearest(avg['lat'].values, ds['lat'].values)
    # Setup a bool for values above or equal to lat
    bool_ = avg['lat'].values <= ds['lat'].values
    # Just use a fixed value?
    if not use_avg_at_lat:
        assert type(fixed_value2use) != int, 'Fixed value must be a float!'
    if isinstance(fixed_value2use, float):
        print('Set all values above lat to: {}'.format(fixed_value2use))
        avg[:] = fixed_value2use
    # Make sure there is one value per month
    if len(avg.shape) == 1:
        try:
            avg = np.ma.array([avg.values]*12)
        except AttributeError:
            avg = np.ma.array([avg]*12)
    # Only consider the ware boxes
    if only_consider_water_boxes:
        # add LWI to array
        if res == '0.125x0.125':
            #            folderLWI = '/shared/earthfs//NASA/nature_run/LWI/monthly/'
            #            filenameLWI = 'nature_run_lev_72_res_0.125_spec_LWI_monthly_ctm.nc'
            folderLWI = get_file_locations('AC_tools')
            folderLWI += '/data/LM/TEMP_NASA_Nature_run/'
            filenameLWI = 'ctm.nc'
            LWI = xr.open_dataset(folderLWI+filenameLWI)
            bool_water = LWI.to_array().values[0, :, idx, :] == 0.0
        else:
            LWI = AC.get_land_map(res=res)[..., 0]
            bool_water = (LWI[:, idx] == 0.0)
            # Use the annual value for ewvery month
            bool_water = np.ma.array([bool_water]*12)
        # Set land/ice values to NaN
        for n_month in range(12):
            avg[n_month, ~bool_water[n_month]] = np.NaN
    # get the average over lon
    avg = np.nanmean(avg, axis=-1)
    pstr = '>={}N = monthly avg. @ lat (avg={:.2f},min={:.2f},max={:.2f})'
    print(pstr.format(lat_above2set, avg.mean(), avg.min(), avg.max()))
    # Get the data
    values = ds[var2set].values
    # Update the values above the specific lat
    # Do this on a monthly basis if data is monthly.
    if len(values.shape) == 3:
        for month in np.arange(values.shape[0]):
            # Updated array of values
            arr = np.zeros(values[month, bool_, :].shape)
            arr[:] = avg[month]
            # now replace values
            values[month, bool_, :] = arr
            del arr
    else:
        # Updated array of values
        arr = np.zeros(values[bool_, :].shape)
        arr[:] = np.nanmean(avg)
        # now replace values
        values[bool_, :] = arr
    ds[var2set].values = values
    # Update the history attribute to record the update.
    attrs = ds.attrs
    #
    try:
        History = attrs['History']
    except KeyError:
        attrs['History'] = ''
    hist_str = "; '{}' above lat ({}N) set to monthly lon average at that lat.;"
    hist_str = hist_str.format(var2set, lat_above2set)
    attrs['History'] = attrs['History'] + hist_str
    # Save updated file
    if save2NetCDF:
        ext_str = '_INTERP_NEAREST_DERIVED_UPDATED_{}'.format(var2set)
        filename = 'Oi_prj_feature_variables_{}{}.nc'.format(res, ext_str)
        ds.to_netcdf(filename)
    else:
        return ds


def set_SAL_and_NIT_above_65N_to_avg(res='0.125x0.125'):
    """ Driver to build NetCDF files with updates """
    # Local variables
    vars2set = [
        'WOA_Nitrate', 'WOA_Salinity', 'WOA_Phosphate',
        'Depth_GEBCO',
        'WOA_TEMP_K',
    ]
    fixed_value2use = None
#    fixed_value2use = -4500.
#    fixed_value2use = 280.  # temp
    # Loop vars and set
    for var2set in vars2set:
        set_values_at_of_var_above_X_lat_2_avg(var2set=var2set, res=res,
                                               fixed_value2use=fixed_value2use)

