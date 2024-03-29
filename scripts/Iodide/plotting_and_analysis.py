#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Plotting and analysis code for sea-surface iodide prediction work.

Please see Paper(s) for more details:

Sherwen, T., Chance, R. J., Tinel, L., Ellis, D., Evans, M. J., and Carpenter, L. J.: A machine learning based global sea-surface iodide distribution, Earth Syst. Sci. Data Discuss., https://doi.org/10.5194/essd-2019-40, in review, 2019.

Chance, R.J., Tinel, L., Sherwen, T., Baker, A.R., Bell, T., Brindle, J., Campos, M.L.A., Croot, P., Ducklow, H., Peng, H. and Hopkins, F., 2019. Global sea-surface iodide observations, 1967–2018. Scientific data, 6(1), pp.1-8.

"""
import numpy as np
import pandas as pd
# import AC_tools (https://github.com/tsherwen/AC_tools.git)
import AC_tools as AC
import matplotlib
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
# s2s imports
import sparse2spatial.RFRanalysis as RFRanalysis
import sparse2spatial.analysis as analysis
import sparse2spatial.RFRbuild as build
import sparse2spatial.utils as utils
from sparse2spatial.RFRbuild import build_or_get_models
from sparse2spatial.RFRbuild import get_top_models
#from sparse2spatial.RFRanalysis import get_stats_on_models
from sparse2spatial.RFRanalysis import get_stats_on_multiple_global_predictions
from sparse2spatial.RFRanalysis import get_spatial_predictions_0125x0125_by_lat
# Local modules specific to iodide work
from sea_surface_iodide import *
import project_misc as misc


def plt_obs_spatially_vs_predictions_options(dpi=320, target='Iodide',
                                             RFR_dict=None,
                                             testset='Test set (strat. 20%)',
                                             rm_Skagerrak_data=True,
                                             rm_non_water_boxes=True):
    """
    Plot up predicted values overlaid with observations

    Parameters
    -------
    target (str): Name of the target variable (e.g. iodide)
    testset (str): Testset to use, e.g. stratified sampling over quartiles for 20%:80%
    rm_Skagerrak_data (bool): remove the data from the Skagerrak region
    rm_non_water_boxes (bool): fill all non-water grid boxes with NaNs
    dpi (int): resolution to use for saved image (dots per square inch)
    RFR_dict (dict): dictionary of core variables and data

    Returns
    -------
    (dict)
    """
    # testset='Test set (strat. 20%)'
    import seaborn as sns
    from matplotlib import colors
    # reset settings as plotting maps
    sns.reset_orig()
    # - Get the data
    # - Get the spatial predictions
#    res4param = '4x5'  # use 4x5 for testing
    res4param = '0.125x0.125'  # only 0.125x0.125 should be used for analysis
    if rm_Skagerrak_data:
        extr_str = '_No_Skagerrak'
    else:
        extr_str = ''
    filename = 'Oi_prj_predicted_{}_{}{}.nc'.format(
        target, res4param, extr_str)
    data_root = utils.get_file_locations('data_root')
    folder = '/{}/{}/outputs/'.format(data_root, target)
    ds = xr.open_dataset(folder + filename)
    # Set the variable to plot underneath observations
    var2plot = 'Ensemble_Monthly_mean'
    # Only select boxes where that are fully water (seasonal)
    if rm_non_water_boxes:
        ds = utils.add_LWI2array(ds=ds, res=res4param, var2template=var2plot)
        #
        ds[var2plot] = ds[var2plot].where(ds['IS_WATER'] == True)
#    var2plot = 'RFR(TEMP+DEPTH+SAL)'
    arr = ds[var2plot].mean(dim='time', skipna=True).values
    # - Get the observations
    # select dataframe with observations and predictions in it
    if isinstance(RFR_dict, type(None)):
        RFR_dict = build_or_get_models_iodide()
    df = RFR_dict['df']
    # Get stats on models in RFR_dict
    stats = get_stats_on_models(RFR_dict=RFR_dict, verbose=False)
    # Only consider that are not outliers.
    df = df.loc[df[target] <= utils.get_outlier_value(df=df, var2use=target),:]
#    df.loc[ df[target]<= 400., :]
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
    #  - Plot globally
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
                               ax=ax, fig=fig, centre=centre,
                               fillcontinents=True,
                               extend=extend, res=res, left_cb_pos=left_cb_pos,
                               #        cmap=cmap,
                               show=False)
        # Now add point for observations
        x = df[u'Longitude'].values
        y = df[u'Latitude'].values
        z = df[target].values
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


def plt_obs_spatially_vs_predictions(dpi=320, target='Iodide',
                                     RFR_dict=None,
                                     testset='Test set (strat. 20%)',
                                     rm_Skagerrak_data=False,
                                     rm_non_water_boxes=True):
    """
    Plot up predicted values overlaid with observations

    Parameters
    -------
    target (str): Name of the target variable (e.g. iodide)
    testset (str): Testset to use, e.g. stratified sampling over quartiles for 20%:80%
    rm_Skagerrak_data (bool): remove the data from the Skagerrak region
    RFR_dict (dict): dictionary of core variables and data
    dpi (int): resolution to use for saved image (dots per square inch)
    rm_non_water_boxes (bool): fill all non-water grid boxes with NaNs

    Returns
    -------
    (None)
    """
    import seaborn as sns
    from matplotlib import colors
    # reset settings as plotting maps
    sns.reset_orig()
    # - Get the data
    # - Get the spatial predictions
#    res4param = '4x5'  # use 4x5 for testing
    res4param = '0.125x0.125'  # only 0.125x0.125 should be used for analysis
    if rm_Skagerrak_data:
        extr_str = '_No_Skagerrak'
    else:
        extr_str = ''
    filename = 'Oi_prj_predicted_{}_{}{}.nc'.format(
        target, res4param, extr_str)
    folder = utils.get_file_locations('data_root') + '/Iodide/outputs/'
    ds = xr.open_dataset(folder + filename)
    # Set the variable to plot underneath observations
    var2plot = 'Ensemble_Monthly_mean'
    # Only select boxes where the
    if rm_non_water_boxes:
        ds = utils.add_LWI2array(ds=ds, res=res4param, var2template=var2plot)
        #
        ds[var2plot] = ds[var2plot].where(ds['IS_WATER'] == True)
    #    var2plot = 'RFR(TEMP+DEPTH+SAL)'
        arr = ds[var2plot].mean(dim='time', skipna=True).values
    # - Get the observations
    # select dataframe with observations and predictions in it
    if isinstance(RFR_dict, type(None)):
        RFR_dict = build_or_get_models_iodide()
    df = RFR_dict['df']
    # Get stats on models in RFR_dict
    stats = get_stats_on_models(RFR_dict=RFR_dict, verbose=False)
    # Only consider values below 400
#   df = df.loc[ df[target]<= 400., :]
    # Only consider values that are not outliers
    df = df.loc[df[target] <= utils.get_outlier_value(df=df, var2use=target), :]
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
    #  - Plot globally
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
    z = df[target].values
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
    # Get lats and lons for res
    lon, lat, NIU = AC.get_latlonalt4res(res=res)
    #
    latrange = [-90, 0, 90]
    lonrange = np.linspace(-180, 180, 4)
#    latrange = lat[::len( lat ) /2 ]
#    lonrange = lon[::len( lon ) /2 ]
    s = 15
    # Loop and plot arrays
    for nlat, latmin in enumerate(latrange[:-1]):
        latmax = latrange[nlat+1]
        for nlon, lonmin in enumerate(lonrange[:-1]):
            lonmax = lonrange[nlon+1]
            # - Plot up obs
            # Initialise plot
            fig, ax = plt.subplots(dpi=dpi, figsize=figsize)
            # Now plot up
            AC.plot_spatial_figure(arr, fixcb=fixcb, nticks=nticks,
                                   ax=ax, fig=fig, centre=centre,
                                   fillcontinents=True,
                                   extend=extend, res=res, resolution='h',
                                   left_cb_pos=left_cb_pos,
                                   #        cmap=cmap,
                                   units=units, show=False)
            # - Plot up param
            # Now add point for observations
            x = df[u'Longitude'].values
            y = df[u'Latitude'].values
            z = df[target].values
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


def plt_obs_spatially_vs_predictions_at_points(dpi=320,
                                                        target='Iodide',
                                                        RFR_dict=None,
                                                        testset='Test set (strat. 20%)'
                                                        ):
    """
    Plot up predicted values against observations at obs. points

    Parameters
    -------
    testset (str): Testset to use, e.g. stratified sampling over quartiles for 20%:80%
    RFR_dict (dict): dictionary of core variables and data
    dpi (int): resolution to use for saved image (dots per square inch)

    Returns
    -------
    (None)
    """
    import seaborn as sns
    from matplotlib import colors
    # reset settings as plotting maps
    sns.reset_orig()
    # - Get the spatial predictions
    # set models and params to plot
    models2compare = [
        'RFR(TEMP+DEPTH+SAL+NO3+DOC)', 'RFR(TEMP+DEPTH+SAL+NO3)',
        'RFR(TEMP+DEPTH+SAL)', 'RFR(TEMP+SAL+Prod)',
        #    'RFR(TEMP+SAL+NO3)',
        #    'RFR(TEMP+DEPTH+SAL)',
    ]
    params = [u'Chance2014_STTxx2_I', u'MacDonald2014_iodide', ]
    # - Plot up comparisons spatially just showing obs. points
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
    # Plot up a white background
    arr = np.zeros(AC.get_dims4res('4x5'))[..., 0]
    # - Plot observations
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
    z = df[target].values
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
    z = df_tmp[target].values
    ax.scatter(x, y, c=z, s=5, cmap=cmap, norm=norm)
    # Save to PDF
    AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
    if show_plot:
        plt.show()
    plt.close()

    # - Plot up bias by param
    units = 'nM'
    # - Plot for entire dataset
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
        # Add a blank plot
        AC.plot_spatial_figure(arr, fixcb=fixcb, nticks=nticks,
                               ax=ax, fig=fig, centre=centre,
                               fillcontinents=True,
                               extend=extend, res=res, title=title,
                               title_x=0.15,
                               units=units,
                               #        cmap=cmap,
                               show=False)
        # Get the residuals
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
        # Add a blank plot
        AC.plot_spatial_figure(arr, fixcb=fixcb, nticks=nticks,
                               ax=ax, fig=fig, centre=centre,
                               fillcontinents=True,
                               extend=extend, res=res, title=title,
                               title_x=0.15,
                               units=units,
                               #        cmap=cmap,
                               show=False)
        # Get the residuals
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

    # --- Plot up Abs. bias by param
    units = 'nM'
    # - Plot for entire dataset
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
        # Add a blank plot
        AC.plot_spatial_figure(arr, fixcb=fixcb, nticks=nticks,
                               ax=ax, fig=fig, centre=centre,
                               fillcontinents=True,
                               extend=extend, res=res, title=title,
                               title_x=0.15,
                               units=units,
                               #        cmap=cmap,
                               show=False)
        # Get the residuals
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
        # Add a blank plot
        AC.plot_spatial_figure(arr, fixcb=fixcb, nticks=nticks,
                               ax=ax, fig=fig, centre=centre,
                               fillcontinents=True,
                               extend=extend, res=res, title=title,
                               title_x=0.15,
                               units=units,
                               #        cmap=cmap,
                               show=False)
        # Get the residuals
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

    # --- Plot up % Abs. bias by param
    units = '%'
    # - Plot for entire dataset
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
        # Add a blank plot
        AC.plot_spatial_figure(arr, fixcb=fixcb, nticks=nticks,
                               ax=ax, fig=fig, centre=centre,
                               fillcontinents=True,
                               extend=extend, res=res, title=title,
                               title_x=0.15,
                               units=units,
                               #        cmap=cmap,
                               show=False)
        # Get the residuals
        x = df[u'Longitude'].values
        y = df[u'Latitude'].values
        z = np.abs(df[param+'-residual'].values) / df[target].values * 100
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
        # Add a blank plot
        AC.plot_spatial_figure(arr, fixcb=fixcb, nticks=nticks,
                               ax=ax, fig=fig, centre=centre,
                               fillcontinents=True,
                               extend=extend, res=res, title=title,
                               title_x=0.15,
                               units=units,
                               #        cmap=cmap,
                               show=False)
        # Get the residuals
        x = df_tmp[u'Longitude'].values
        y = df_tmp[u'Latitude'].values
        z = np.abs(df_tmp[param+'-residual'].values) / df_tmp[target].values
        z *= 100
        ax.scatter(x, y, c=z, s=5, cmap=cmap, norm=norm)
        # Save to PDF
        AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
        if show_plot:
            plt.show()
        plt.close()
    # Save entire pdf
    AC.plot2pdfmulti(pdff, savetitle, close=True, dpi=dpi)


def plot_predicted_iodide_vs_lat_figure(dpi=320, plot_avg_as_median=False,
                                        show_plot=False, shade_std=True,
                                        just_plot_existing_params=False,
                                        plot_up_param_iodide=True,
                                        context="paper",
                                        ds=None, target='Iodide',
                                        rm_Skagerrak_data=False):
    """
    Plot a figure of iodide vs laitude

    Parameters
    -------
    ds (xr.Dataset): 3D dataset containing variable of interest on monthly basis
    rm_Skagerrak_data (bool): remove the data from the Skagerrak region
    dpi (int): resolution to use for saved image (dots per square inch)
    plot_avg_as_median (bool): use median as the average in plots
    just_plot_existing_params (bool): just plot up the existing parameterisations
    plot_up_param_iodide (bool): plot parameterised iodide
    shade_std (bool): shae in a standard deviation around the predictions
    context (str): seaborn context to use for plotting (e.g. paper, poster, talk...)
    target (str): Name of the target variable (e.g. iodide)
    show_plot (bool): show the plot on screen

    Returns
    -------
    (None)
    """
    import seaborn as sns
    sns.set(color_codes=True)
    if context == "paper":
        sns.set_context("paper", font_scale=0.75)
    else:
        sns.set_context("talk", font_scale=0.9)
    # Get observations
    df_obs = get_processed_df_obs_mod()  # NOTE this df contains values >400nM
    # Get predicted values
    if isinstance(ds, type(None)):
        folder = utils.get_file_locations('data_root') + '/Iodide/outputs/'
        filename = 'Oi_prj_predicted_{}_0.125x0.125{}.nc'
        if rm_Skagerrak_data:
            filename = filename.format(target, '_No_Skagerrak')
        else:
            filename = filename.format(target, '')
        ds = xr.open_dataset(folder + filename)
    # Rename to a more concise name
    try:
        ds.rename(name_dict={'Ensemble_Monthly_mean': 'RFR(Ensemble)'},
                  inplace=True)
    except ValueError:
        # Pass if 'Ensemble_Monthly_mean' already is in dataset
        pass
    # Get predicted values binned by latitude
    df = get_spatial_predictions_0125x0125_by_lat(ds=ds)
    # Params to pot
    models2compare = [
        'RFR(Ensemble)'
    ]
    params = ['Chance2014_STTxx2_I', u'MacDonald2014_iodide',  'Wadley2020']
    rename_titles = {u'Chance2014_STTxx2_I': 'Chance et al. (2014)',
                     u'MacDonald2014_iodide': 'MacDonald et al. (2014)',
                     'RFR(Ensemble)': 'RFR(Ensemble)',
                     'Wadley2020': 'Wadley et al. (2020)',
                     'Iodide': 'Obs.',
                     }
    if just_plot_existing_params:
        params2plot = params
    else:
        params2plot = models2compare + params
    # Assign colors
    CB_color_cycle = AC.get_CB_color_cycle()
    color_d = dict(zip(params2plot, CB_color_cycle))
    # --- Plot up vs. lat
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
            # Get X
            X = df[var2plot].index.values
            # Plot as line
            plt.plot(X, df[var2plot].values, color=color,
                     label=rename_titles[param])
            # Plot up quartiles/std
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
                    # Use the 75th percentile of the monthly average std
                    # Of the ensemble members
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
    Y = tmp_df[target].values
    plt.scatter(X, Y, color='k', marker='D', facecolor='none', s=3,
                label='Coastal obs.')
    # Non-coastal obs
    tmp_df = df_obs.loc[df_obs['Coastal'] == False, :]
    X = tmp_df['Latitude'].values
    Y = tmp_df[target].values
    plt.scatter(X, Y, color='k', marker='D', facecolor='k', s=3,
                label='Non-coastal obs.')
    # Limit plot y axis
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


def plt_predicted_I_vs_lat_fig_with_Skagerrak_too(dpi=320, target='Iodide',
                                                  plot_avg_as_median=False,
                                                  show_plot=False,
                                                  shade_std=True,
                                               just_plot_existing_params=False,
                                                  plot_up_param_iodide=True,
                                                  context="paper", ds=None,
                                                  rm_Skagerrak_data=False):
    """
    Plot a figure of iodide vs laitude

    Parameters
    -------
    rm_Skagerrak_data (bool): remove the data from the Skagerrak region
    target (str): Name of the target variable (e.g. iodide)
    dpi (int): resolution to use for saved image (dots per square inch)
    context (str): seaborn context to use for plotting (e.g. paper, poster, talk...)
    just_plot_existing_params (bool): just plot up the existing parameterisations
    plot_up_param_iodide (bool): plot parameterised iodide
    ds (xr.Dataset): 3D dataset containing variable of interest on monthly basis
    plot_avg_as_median (bool): use median as the average in plots
    show_plot (bool): show the plot on screen

    Returns
    -------
    (None)
    """
    import seaborn as sns
    sns.set(color_codes=True)
    if context == "paper":
        sns.set_context("paper", font_scale=0.75)
    else:
        sns.set_context("talk", font_scale=0.9)
    # Get observations
    df_obs = get_processed_df_obs_mod()  # NOTE this df contains values >400nM
    # Get predicted values
    folder = utils.get_file_locations('data_root') + '/Iodide/outputs/'
    filename = 'Oi_prj_predicted_{}_0.125x0.125{}.nc'
    ds = xr.open_dataset(folder + filename.format(target, '_No_Skagerrak'))
    # Get data with Skagerrak data too.
    folder = utils.get_file_locations('data_root') + '/Iodide/outputs/'
    ds2 = xr.open_dataset(folder + filename.format(''))
    # Rename to a more concise name
    try:
        ds.rename(name_dict={'Ensemble_Monthly_mean': 'RFR(Ensemble)'},
                  inplace=True)
    except ValueError:
        # Pass if 'Ensemble_Monthly_mean' already is in dataset
        pass
    # Rename to a more concise name
    SkagerrakVarName = 'RFR(Ensemble) - Inc. Skagerrak data'
    try:
        ds2.rename(name_dict={'Ensemble_Monthly_mean': SkagerrakVarName},
                   inplace=True)
    except ValueError:
        # Pass if 'Ensemble_Monthly_mean' already is in dataset
        pass
    # Get predicted values binned by latitude
    df = get_spatial_predictions_0125x0125_by_lat(ds=ds)
    df2 = get_spatial_predictions_0125x0125_by_lat(ds=ds2)
    # Params to pot
    models2compare = [
        'RFR(Ensemble)'
    ]
    params = ['Chance2014_STTxx2_I', u'MacDonald2014_iodide']
    rename_titles = {u'Chance2014_STTxx2_I': 'Chance et al. (2014)',
                     u'MacDonald2014_iodide': 'MacDonald et al. (2014)',
                     'RFR(Ensemble)': 'RFR(Ensemble)',
                     'Iodide': 'Obs.',
                     }
    if just_plot_existing_params:
        params2plot = params
    else:
        params2plot = models2compare + params
    # Assign colors
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
            # Get X
            X = df[var2plot].index.values
            # Plot as line
            plt.plot(X, df[var2plot].values, color=color,
                     label=rename_titles[param])
            # Plot up quartiles/std
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
                    # Use the 75th percentile of the monthly average std
                    # Of the ensemble members
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
        # Get X
        print(df2.columns)
        X = df2[var2plot].index.values
        # Plot as line
        plt.plot(X, df2[var2plot].values, color=color,
                 label=SkagerrakVarName, ls='--')
        # Plot up quartiles/std
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
                # Use the 75th percentile of the monthly average std
                # Of the ensemble members
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
    Y = tmp_df[target].values
    plt.scatter(X, Y, color='k', marker='D', facecolor='none', s=3,
                label='Coastal obs.')
    # Non-coastal obs
    tmp_df = df_obs.loc[df_obs['Coastal'] == False, :]
    X = tmp_df['Latitude'].values
    Y = tmp_df[target].values
    plt.scatter(X, Y, color='k', marker='D', facecolor='k', s=3,
                label='Non-coastal obs.')
    # Limit plot y axis
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
                                                 plot_avg_as_median=False,
                                                 RFR_dict=None,
                                                 res='0.125x0.125',
                                                 target='Iodide',
                                                 show_plot=False,
                                                 close_plot=True,
                                                 save2png=False,
                                                 shade_std=True,
                                                 folder=None, ds=None,
                                                 topmodels=None):
    """
    Plot a figure of iodide vs laitude - showing all ensemble members

    Parameters
    -------
    RFR_dict (dict): dictionary of core variables and data
    target (str): Name of the target variable (e.g. iodide)
    dpi (int): resolution to use for saved image (dots per square inch)
    ds (xr.Dataset): 3D dataset containing variable of interest on monthly basis
    plot_avg_as_median (bool): use median as the average in plots
    topmodels (list): list of models to make spatial predictions for
    show_plot (bool): show the plot on screen
    save2png (bool): save the plot as png

    Returns
    -------
    (None)
    """
    from collections import OrderedDict
    import seaborn as sns
    sns.set(color_codes=True)
    sns.set_context("paper", font_scale=0.75)
    # Get observations
    df_obs = get_processed_df_obs_mod()  # NOTE this df contains values >400nM
    # Get predicted values
    if isinstance(folder, type(None)):
        folder = utils.get_file_locations('data_root') + '/Iodide/outputs/'
    if isinstance(ds, type(None)):
        filename = 'Oi_prj_predicted_{}_{}{}.nc'.format(target, res, extr_str)
        ds = xr.open_dataset(folder + filename)
    # Rename to a more concise name
    print(ds.data_vars)
    # Get predicted values binned by latitude
    if res == '0.125x0.125':
        df = get_spatial_predictions_0125x0125_by_lat(ds=ds)
    else:
        df = get_stats_on_spatial_predictions_4x5_2x25_by_lat(res=res, ds=ds)
    # Params to pot
    if isinstance(topmodels, type(None)):
        # Get RFR_dict if not provide
        if isinstance(RFR_dict, type(None)):
            RFR_dict = build_or_get_models_iodide()
        topmodels = get_top_models(RFR_dict=RFR_dict, vars2exclude=['DOC', 'Prod'], n=10)
    params2plot = topmodels
    # Assign colors
    CB_color_cycle = AC.get_CB_color_cycle()
    CB_color_cycle += ['darkgreen']
    color_d = dict(zip(params2plot, CB_color_cycle))
    # --- Plot up vs. lat
    fig, ax = plt.subplots()
    # Loop by param to plot
    for param in params2plot:
        # Set color for param
        color = color_d[param]
        # Plot average
        if plot_avg_as_median:
            var2plot = '{} - median'.format(param)
        else:
            var2plot = '{} - mean'.format(param)
        # Get X
        X = df[var2plot].index.values
        # Plot as line
        plt.plot(X, df[var2plot].values, color=color, label=param)
        # Plot up quartiles/std as shaded regions too
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
                # Use the 75th percentile of the monthly average std
                # Of the ensemble members
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
    Y = tmp_df[target].values
    plt.scatter(X, Y, color='k', marker='D', facecolor='none', s=3,
                label='Coastal obs.')
    # Non-coastal obs
    tmp_df = df_obs.loc[df_obs['Coastal'] == False, :]
    X = tmp_df['Latitude'].values
    Y = tmp_df[target].values
    plt.scatter(X, Y, color='k', marker='D', facecolor='k', s=3,
                label='Non-coastal obs.')
    # Limit plot y axis
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
    if save2png:
        plt.savefig(filename.format(res, extr_str), dpi=dpi)
    if show_plot:
        plt.show()
    if close_plot:
        plt.close()


# ---------------------------------------------------------------------------
# ---------- Functions to analyse/test models for Oi! paper --------------
# ---------------------------------------------------------------------------
def check_seasonality_of_iodide_predictions(show_plot=False,
                                            target='Iodide'):
    """
    Compare the seasonality of obs. and parameterised values
    """
    # --- Set local variables
    rename_titles = {
    u'Chance2014_STTxx2_I': 'Chance et al. (2014)',
    u'MacDonald2014_iodide': 'MacDonald et al. (2014)',
    'RFR(Ensemble)': 'RFR(Ensemble)',
    u'Chance2014_Multivariate': 'Chance et al. (2014) (Multi)',
    }
    CB_color_cycle = AC.get_CB_color_cycle()
    colors_dict = dict(zip(rename_titles.keys(),  CB_color_cycle))

    # - Get dataset where there is more than a 2 months of data at the same loc
    df = get_processed_df_obs_mod()  # NOTE this df contains values >400nM
    # Exclude data with values above 400 nM
#    df = df.loc[ df[target].values < 400, : ]
    # Exclude outliers
    df = df.loc[df[target] <= utils.get_outlier_value(df=df, var2use=target),:]
    # Get metadata
    md_df = get_iodide_obs_metadata()
    datasets = md_df[u'Data_Key']
    # Loop datasets and find ones with multiple obs.
    N = []
    ds_seas = []
    MonthVar = 'Month (Orig.)'
    for ds in datasets:
        # Get obs for dataset
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

    # - Loop these datasets and plot the three parameterisation s predictions
    savetitle = 'Oi_prj_seasonality_of_iodide'
    pdff = AC.plot2pdfmulti(title=savetitle, open=True, dpi=dpi)
    # Add a title slide
    fig, ax = plt.subplots()
    plt.text(0.1, 1, 'New parametersation', fontsize=15)
    plt.setp(plt.gca(), frame_on=False, xticks=(), yticks=())
    AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
    plt.close()
    # Now loop and plot up vs. new parameterisation
    for ds in ds_seas:
        # Get location/data on obs.
        md_df_tmp = md_df.loc[md_df[u'Data_Key'] == ds]
        Source = md_df_tmp['Source'].values[0].strip()
        Loc = md_df_tmp['Location'].values[0].strip()
        Cruise = md_df_tmp['Cruise'].values[0].strip()
        # Get obs for dataset
        ds_tmp = df.loc[df[u'Data_Key'] == ds]
        ds_tmp.sort_values(by=MonthVar)
        # Plot up against months
        plt.scatter(ds_tmp[MonthVar].values, ds_tmp[target].values,
                    label='Obs', color='k')
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
        # Save the plot
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
    # Now loop and plot up vs. all parameterisations parameterisation
    for ds in ds_seas:
        # Get location/data on obs.
        md_df_tmp = md_df.loc[md_df[u'Data_Key'] == ds]
        Source = md_df_tmp['Source'].values[0].strip()
        Loc = md_df_tmp['Location'].values[0].strip()
        Cruise = md_df_tmp['Cruise'].values[0].strip()
        # Get obs for dataset
        ds_tmp = df.loc[df[u'Data_Key'] == ds]
        ds_tmp.sort_values(by=MonthVar)
        # Plot up against months
        plt.scatter(ds_tmp[MonthVar].values,  ds_tmp[target].values,
                    label='Obs',
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
        # Save the plot
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

    # Now loop and plot up vs. all parameterisations parameterisation
    for ds in ds_seas:
        # Get location/data on obs.
        md_df_tmp = md_df.loc[md_df[u'Data_Key'] == ds]
        Source = md_df_tmp['Source'].values[0].strip()
        Loc = md_df_tmp['Location'].values[0].strip()
        Cruise = md_df_tmp['Cruise'].values[0].strip()
        # Get obs for dataset
        ds_tmp = df.loc[df[u'Data_Key'] == ds]
        ds_tmp.sort_values(by=MonthVar)
        # Plot up against months
        plt.scatter(ds_tmp[MonthVar].values,  ds_tmp[target].values,
                    label='Obs', color='k')
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
        # Save the plot
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

    # Now loop and plot up vs. all parameterisations parameterisation
    for ds in ds_seas:
        # Get location/data on obs.
        md_df_tmp = md_df.loc[md_df[u'Data_Key'] == ds]
        Source = md_df_tmp['Source'].values[0].strip()
        Loc = md_df_tmp['Location'].values[0].strip()
        Cruise = md_df_tmp['Cruise'].values[0].strip()
        # Get obs for dataset
        ds_tmp = df.loc[df[u'Data_Key'] == ds]
        ds_tmp.sort_values(by=MonthVar)
        # Plot up against months
        plt.scatter(ds_tmp[MonthVar].values,  ds_tmp[target].values,
                    label='Obs', color='k')
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
        # Save the plot
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


def test_model_sensitiivty2training_test_split(models2compare=None,
                                               models_dict=None):
    """
    Driver to test/training set sensitivity for a set of models
    """
    # List of models to test?
    if isinstance(models2compare, type(None)):
        models2compare = ['RFR(TEMP+DEPTH+SAL)']
    # Get the unprocessed obs and variables as a DataFrame
    df = get_processed_df_obs_mod()  # NOTE this df contains values >400nM
    # Get variables to use for each model
    model_feature_dict = utils.get_model_features_used_dict(rtn_dict=True)
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


def analyse_model_selection_error_in_ensemble_members(RFR_dict=None,
                                                      rm_Skagerrak_data=False):
    """
    Calculate the error caused by model selection

    Parameters
    -------
    RFR_dict (dict): dictionary of core variables and data
    rm_Skagerrak_data (bool): Remove specific data

    Returns
    -------
    (None)
    """
    # - Set local variables
    if rm_Skagerrak_data:
        extr_str = '_nSkagerrak'
    else:
        extr_str = ''
    # Get key data as a dictionary
    if isinstance(RFR_dict, type(None)):
        RFR_dict = build_or_get_models_iodide(
            rm_Skagerrak_data=rm_Skagerrak_data,
        )
    # select dataframe with observations and predictions in it
    df = RFR_dict['df']
    # Also make a dictionary
    features_used_dict = RFR_dict['features_used_dict']
    # Get the names of the ensemble members (topten models )
    if isinstance(topmodels, type(None)):
        topmodels = get_top_models(RFR_dict=RFR_dict, vars2exclude=['DOC', 'Prod'], n=10)

    # - Get data at observation points
    # Add lines extract data from files...
#	filename = 'Oi_prj_models_built_stats_on_models_at_obs_points.csv'
#	folder ='../'
#	dfP = pd.read_csv( folder + filename )
#	dfP.index = dfP['Unnamed: 0'].values
 # Just select the
    df = RFR_dict['df']
    testset='Test set (strat. 20%)'
    df = df.loc[df[testset] == True, :]
    # Get stats on model tuns runs
    dfP = get_stats_on_models(RFR_dict=RFR_dict, df=df,
                              verbose=False)
    # Only consider topmodels
    dfP = dfP.T[topmodels].T

    # -  Get data at spatial points
    # Add lines extract data from files...

    # just use outputted file for now.
    filename = 'Oi_prj_annual_stats_global_ocean_0.125x0.125.csv'
    folder = '../'
    dfG = pd.read_csv(folder + filename)
    dfG.index = dfG['Unnamed: 0'].values
    # Only consider topmodels
    dfG = dfG.T[topmodels].T

    # - Set summary stats and print to a txt file
    file2save = 'Oi_prj_Error_calcs_model_selction{}{}.txt'
    a = open(file2save.format(res, extr_str), 'w')
    # -  Calculate model selection error spatially
    print('---- Model choice affect spatially /n', file=a)
    # Get stats
    var2use = u'mean (weighted)'
    min_ = dfG[var2use].min()
    max_ = dfG[var2use].max()
    mean_ = dfG[var2use].mean()
    range_ = max_ - min_
    # Print out the range
    ptr_str = 'Model choice stats: min={:.5g}, max={:.5g} (range={:.5g})'
    print(ptr_str.format(min_, max_, range_), file=a)
    # Print out the error
    ptr_str = 'Model choice affect on mean: {:.5g}-{:.5g} %'
    print(ptr_str.format(range_/max_*100, range_/min_*100), file=a)

    # - Calculate model selection error at observational points
    print('---- Model choice affect at point locations (mean) \n', file=a)
    # Get stats
    var2use = u'mean'
    min_ = dfP[var2use].min()
    max_ = dfP[var2use].max()
    mean_ = dfP[var2use].mean()
    range_ = max_ - min_
    # Print out the range
    ptr_str = 'Model choice stats: min={:.5g}, max={:.5g} (range={:.5g})'
    print(ptr_str.format(min_, max_, range_), file=a)
    # Print out the error
    ptr_str = 'Model choice affect on mean: {:.5g}-{:.5g} %'
    print(ptr_str.format(range_/max_*100, range_/min_*100), file=a)
    # -  Now calculate for RMSE
    print('---- Model choice affect at point locations (mean) \n', file=a)
    # Get stats
    var2use = 'RMSE (Test set (strat. 20%))'
    min_ = dfP[var2use].min()
    max_ = dfP[var2use].max()
    mean_ = dfP[var2use].mean()
    range_ = max_ - min_
    # Print out the range
    ptr_str = 'Model choice stats: min={:.5g}, max={:.5g} (range={:.5g})'
    print(ptr_str.format(min_, max_, range_), file=a)
    # Print out the error
    ptr_str = 'Model choice affect on mean: {:.5g}-{:.5g} %'
    print(ptr_str.format(range_/max_*100, range_/min_*100), file=a)
    # close the file
    a.close()


def analyse_dataset_error_in_ensemble_members(RFR_dict=None,
                                              rebuild_models=False,
                                              remake_NetCDFs=False,
                                              res='0.125x0.125',
                                              rm_Skagerrak_data=False,
                                              topmodels=None):
    """
    Analyse the variation in spatial prediction on a per model basis

    Parameters
    -------
    rm_Skagerrak_data (bool): remove the data from the Skagerrak region
    RFR_dict (dict): dictionary of core variables and data
    remake_NetCDFs (bool): remake the NetCDF files of pseudo-random repeated builds
    rebuild_models (bool): rebuild all of the models?
    res (str): horizontal resolution of dataset (e.g. 4x5)
    topmodels (list): list of models to make spatial predictions for

    Returns
    -------
    (None)
    """
    from multiprocessing import Pool
    from functools import partial
    # --- Set local variables
    if rm_Skagerrak_data:
        extr_str = '_nSkagerrak'
    else:
        extr_str = ''
    # Get key data as a dictionary
    if isinstance(RFR_dict, type(None)):
        RFR_dict = build_or_get_models_iodide(
            rm_Skagerrak_data=rm_Skagerrak_data
        )
    # select dataframe with observations and predictions in it
    df = RFR_dict['df']
    # Also make a dictionary
    features_used_dict = RFR_dict['features_used_dict']
    # Get the names of the ensemble members (topten models )
    if isinstance(topmodels, type(None)):
        topmodels = get_top_models(RFR_dict=RFR_dict,
                                   vars2exclude=['DOC', 'Prod'], n=10)
    # --- build 20 variable models for the ensemble memberse
    if rebuild_models:
        for model_name in topmodels:
            # Get the training features for a given model
            features_used = features_used_dict[model_name]
            features_used = features_used.split('+')
            # Now build 20 separate initiations of the model
            build_the_same_model_mulitple_times(model_name=model_name,
                                                features_used=features_used,
                                                df=df,
                                            rm_Skagerrak_data=rm_Skagerrak_data
                                                )

    # - Predict the surface concentrations for each of members' repeat builds
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

    # - Get data
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
    # Get predictions for repeat builds of ensemble members.
    dfs = {}
    for model_name in topmodels:
        # Get the training features for a given model
        features_used = features_used_dict[model_name]
        features_used = features_used.split('+')
        # Now build 20 separate initiations of the model
        dfs[model_name] = RFRanalysis.get_stats4mulitple_model_builds(
            model_name=model_name,
            features_used=features_used, df=df,
            RFR_dict=RFR_dict
        )
    # concatenate into a single dataframe
    dfP = pd.concat([dfs[i] for i in dfs.keys()], axis=0)
    save_str = 'Oi_prj_RAW_stats_on_ENSEMBLE_predictions_at_obs_locs{}.csv'
    dfP.to_csv(save_str.format(extr_str))

    # --- Do analysis
    # - analyse the variance in the prediction of members (spatially)
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
        # Print range for test_
        ptr_str = "range : {:.5g} ({:.5g}-{:.5g})"
        print(ptr_str.format(range_, min_, max_), file=a)
        # Average value and range of this
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
    # Add percent errors
    MerrPCmax = 'max % err. (mean)'
    dfA[MerrPCmax] = dfA['mean range'] / dfA['min_mean4model'] * 100
    MerrPCmin = 'min % err. (mean)'
    dfA[MerrPCmin] = dfA['mean range'] / dfA['max_mean4model'] * 100
    # Use names for the index
    dfA.index = topmodels
    # save processed data
    save_str = 'Oi_prj_RAW_stats_on_ENSEMBLE_predictions_globally{}{}'
    dfA.to_csv(save_str.format(extr_str, '_PROCESSED', '.csv'))
    # Print output to screen
    ptr_str = 'The avg. range in ensemble members is {:.5g} ({:.5g} - {:.5g} )'
    mean_ = dfA['mean range'].mean()
    min_ = dfA['mean range'].min()
    max_ = dfA['mean range'].max()
    # Print the maximum range
    print(ptr_str.format(mean_, min_, max_), file=a)
    ptr_str = 'The max. range in ensemble is: {:.5g} '
    print(ptr_str.format(max_-min_), file=a)
    # Print range in all ensemble members rebuilds (20*10)
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
        # Print range for test_
        ptr_str = "range : {:.5g} ({:.5g}-{:.5g})"
        print(ptr_str.format(range_, min_, max_), file=a)
        # Average value and range of this
        mean_ = df_tmp['mean'].mean()
        min_mean_ = min(df_tmp['mean'])
        max_mean_ = max(df_tmp['mean'])
        range_ = max_mean_ - min_mean_
        ranges4models[model_] = range_
        ptr_str = "Avg. value of mean: {:.5g} ({:.5g} - {:.5g}, range={:.5g})"
        print(ptr_str.format(mean_, min_mean_, max_mean_, range_), file=a)
        # Now print the RMSE and range of
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
    # Add percent errors
    MerrPCmax = 'max % err. (mean)'
    dfPbP[MerrPCmax] = dfPbP['mean range'] / dfPbP['min_mean4model'] * 100
    MerrPCmin = 'min % err. (mean)'
    dfPbP[MerrPCmin] = dfPbP['mean range'] / dfPbP['max_mean4model'] * 100
    # Add percent errors
    RerrPCmax = 'max % err. (RMSE)'
    dfPbP[RerrPCmax] = dfPbP['RMSE range'] / dfPbP['min_RMSE4model'] * 100
    RerrPCmin = 'min % err. (RMSE)'
    dfPbP[RerrPCmin] = dfPbP['RMSE range'] / dfPbP['max_RMSE4model'] * 100
    # Update the index
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
#    print(ptr_str.format(Emin, Emax), file=a)
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
                         testset='Test set (strat. 20%)',
                         target='Iodide', context="paper", dpi=720):
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

    Returns
    -------
    (None)

    """
    if isinstance(RFR_dict, type(None)):
        RFR_dict = build_or_get_models_iodide()
    # select dataframe with observations and predictions in it
    if isinstance(RFR_dict, type(None)):
        features_used_dict = RFR_dict['features_used_dict']
        models_dict = RFR_dict['models_dict']
    if isinstance(df, type(None)):
        df = RFR_dict['df']

    # - Evaluate model using various approaches
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
    models2compare = [ 'RFR(Ensemble)',]
    # rename titles
    rename_titles = {u'Chance2014_STTxx2_I': 'Chance et al. (2014)',
                     u'MacDonald2014_iodide': 'MacDonald et al. (2014)',
                     'RFR(Ensemble)': 'RFR(Ensemble)', }
    # Units
    units = 'nM'
    # iodide in aq
    Iaq = '[I$^{-}_{aq}$]'
    # Also compare existing parameters
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
    # Assign colors
    CB_color_cycle = AC.get_CB_color_cycle()
    color_d = dict(zip(dsplits, CB_color_cycle))
    # Params to plot
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
        # Add a 1:1 line
        x_121 = np.arange(-50, 500)
        ax.plot(x_121, x_121, alpha=0.5, color='k', ls='--')
        # Plot up data by dataset split
        for nsplit, split in enumerate(dsplits):
            #
            df = dfs[split].copy()
            # Get X
            X = df[target].values
            # Get Y
            Y = df[param].values
            # Get N
            N = float(df.shape[0])
            # Get RMSE
            RMSE = np.sqrt(((Y-X)**2).mean())
            # Plot up just the entire and testset data
            if split in ('Entire', 'Withheld'):
                ax.scatter(X, Y, color=color_d[split], s=3, facecolor='none')
            # Add ODR line
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
        # Beautify the plot/figure
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
    # Save the plot
    AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
    # Save entire pdf
    AC.plot2pdfmulti(pdff, savetitle, close=True, dpi=dpi)
    plt.savefig(savetitle, dpi=dpi)
    if show_plot:
        plt.show()
    plt.close()


def analyse_X_Y_correlations_ODR(RFR_dict=None, show_plot=False,
                                 testset='Test set (strat. 20%)',
                                 target='Iodide', context="paper", dpi=320):
    """
    Analyse the correlations between obs. and params. using ODR

    Parameters
    -------
    target (str): Name of the target variable (e.g. iodide)
    testset (str): Testset to use, e.g. stratified sampling over quartiles for 20%:80%
    dpi (int): resolution to use for saved image (dots per square inch)
    RFR_dict (dict): dictionary of core variables and data
    context (str): seaborn context to use for plotting (e.g. paper, poster, talk...)
    show_plot (bool): show the plot on screen

    Returns
    -------
    (None)
    """
    # - Get data
    if isinstance(RFR_dict, type(None)):
        RFR_dict = build_or_get_models_iodide()
    # select dataframe with observations and predictions in it
    df = RFR_dict['df']
    features_used_dict = RFR_dict['features_used_dict']
    models_dict = RFR_dict['models_dict']
    # Get stats on models in RFR_dict
#    stats = get_stats_on_models( RFR_dict=RFR_dict, verbose=False )

    # - Evaluate model using various approaches
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
    # Units
    units = 'nM'
    # iodide in aq
    Iaq = '[I$^{-}_{aq}$]'
    # Also compare existing parameters
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
    # Assign colors
    CB_color_cycle = AC.get_CB_color_cycle()
    color_d = dict(zip(dsplits, CB_color_cycle))
    # Loop by param and compare against whole dataset
    for param in models2compare + params:
        # Intialise figure and axis
        fig, ax = plt.subplots()
        # Add a 1:1 line
        x_121 = np.arange(-50, 500)
        plt.plot(x_121, x_121, alpha=0.5, color='k', ls='--')
        #
        # Plot up data by dataset split
        for nsplit, split in enumerate(dsplits):
            df = dfs[split].copy()
            # Get X
            X = df[target].values
            # Get Y
            Y = df[param].values
            # Get N
            N = float('{:.3g}'.format(df.shape[0]))
            # Get RMSE
            RMSE = np.sqrt(((Y-X)**2).mean())
            # Plot up just the entire and testset data
            if split in ('Entire', 'Withheld'):
                plt.scatter(X, Y, color=color_d[split], s=3, facecolor='none')
            # Add ODR line
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
        # Beautify the plot/figure
        plt.xlim(-10, 410)
        plt.ylim(-10, 410)
        #
        plt.ylabel('{} {} ({})'.format(param, Iaq, units))
        plt.xlabel('Obs. {} ({})'.format(Iaq, units))
#        plt.title('Obs. vs param. for entire dataset')
        # Save the plot
        AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
        if show_plot:
            plt.show()
        plt.close()
    # Save entire pdf
    AC.plot2pdfmulti(pdff, savetitle, close=True, dpi=dpi)


def analyse_X_Y_correlations(RFR_dict=None, show_plot=False,
                             testset='Test set (strat. 20%)',
                             target='Iodide', dpi=320):
    """
    Analyse the correlations between obs. and params.

    Parameters
    -------
    target (str): Name of the target variable (e.g. iodide)
    testset (str): Testset to use, e.g. stratified sampling over quartiles for 20%:80%
    dpi (int): resolution to use for saved image (dots per square inch)
    RFR_dict (dict): dictionary of core variables and data
    show_plot (bool): show the plot on screen

    Returns
    -------
    (None)
    """
    # - Get data
    if isinstance(RFR_dict, type(None)):
        RFR_dict = build_or_get_models_iodide()
    # select dataframe with observations and predictions in it
    df = RFR_dict['df']
    features_used_dict = RFR_dict['features_used_dict']
    models_dict = RFR_dict['models_dict']
    # Get stats on models in RFR_dict
    stats = get_stats_on_models(RFR_dict=RFR_dict, verbose=False)

    # - Evaluate model using various approaches
    import seaborn as sns
    sns.set(color_codes=True)
    sns.set_context("paper")
    # - Evaluate point for point
    savetitle = 'Oi_prj_point_for_point_comparison_obs_vs_model'
    pdff = AC.plot2pdfmulti(title=savetitle, open=True, dpi=dpi)
    # Plot up X vs. Y for all points

    # models to compare
    models2compare = [
        'RFR(TEMP+DEPTH+SAL+NO3+DOC)', 'RFR(TEMP+DEPTH+SAL+NO3)',
        'RFR(TEMP+DEPTH+SAL)', 'RFR(TEMP+SAL+Prod)',
        #    'RFR(TEMP+SAL+NO3)',
        #    'RFR(TEMP+DEPTH+SAL)',
    ]
    # Also compare existing parameters
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
        # Add a 1:1 line
        x_121 = np.arange(-50, 500)
        plt.plot(x_121, x_121, alpha=0.5, color='k', ls='--')
        # Plot up data
        sns.regplot(x='Iodide', y=param, data=df)
        # Beautify the plot/figure
        plt.xlim(-10, 410)
        plt.ylim(-10, 410)
        plt.title('Obs. vs param. for entire dataset')
        # TODO (inc. RMSE on plot)
        stats_tmp = stats.loc[stats.index == param]
        stat_var = 'RMSE (all)'
        alt_text = alt_text_str.format(stats_tmp[stat_var][0], N)
        ax.annotate(alt_text, xy=(alt_text_x, alt_text_y),
                    textcoords='axes fraction', fontsize=f_size*1.5)
        # Save the plot
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
        # Add a 1:1 line
        x_121 = np.arange(-50, 500)
        plt.plot(x_121, x_121, alpha=0.5, color='k', ls='--')
        # Plot up data
        sns.regplot(x='Iodide', y=param, data=df_tmp)
        # Beautify the plot/figure
        plt.xlim(-10, 410)
        plt.ylim(-10, 410)
        plt.title('Obs. vs param. for test dataset')
        # TODO (inc. RMSE on plot)
        stats_tmp = stats.loc[stats.index == param]
        stat_var = 'RMSE ({})'.format(testset)
        alt_text = alt_text_str.format(stats_tmp[stat_var][0], N)
        ax.annotate(alt_text, xy=(alt_text_x, alt_text_y),
                    textcoords='axes fraction', fontsize=f_size*1.5)
        # Save the plot
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
        # Add a 1:1 line
        x_121 = np.arange(-50, 500)
        plt.plot(x_121, x_121, alpha=0.5, color='k', ls='--')
        # Plot up data
        sns.regplot(x='Iodide', y=param, data=df_tmp)
        # Beautify the plot/figure
        plt.xlim(-10, 410)
        plt.ylim(-10, 410)
        plt.title('Obs. vs param. for coastal test dataset')
        # TODO (inc. RMSE on plot)
        stats_tmp = stats.loc[stats.index == param]
        stat_var = 'RMSE (Coastal ({}))'.format(testset)
        alt_text = alt_text_str.format(stats_tmp[stat_var][0], N)
        ax.annotate(alt_text, xy=(alt_text_x, alt_text_y),
                    textcoords='axes fraction', fontsize=f_size*1.5)
        # Save the plot
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
        # Add a 1:1 line
        x_121 = np.arange(-50, 500)
        plt.plot(x_121, x_121, alpha=0.5, color='k', ls='--')
        # Plot up data
        sns.regplot(x='Iodide', y=param, data=df_tmp)
        # Beautify the plot/figure
        plt.xlim(-10, 410)
        plt.ylim(-10, 410)
        plt.title('Obs. vs param. for non-coastal test dataset')
        # TODO (inc. RMSE on plot)
        stats_tmp = stats.loc[stats.index == param]
        stat_var = 'RMSE (Non coastal ({}))'.format(testset)
        alt_text = alt_text_str.format(stats_tmp[stat_var][0], N)
        ax.annotate(alt_text, xy=(alt_text_x, alt_text_y),
                    textcoords='axes fraction', fontsize=f_size*1.5)
        # Save the plot
        AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
        if show_plot:
            plt.show()
        plt.close()

    # - Now look at residuals
    # Get residual
    for param in models2compare + params:
        df[param+'-residual'] = df[param] - df[target]

    # Loop by param and compare against whole dataset ( as residuals )
    for param in models2compare + params:
        # Intialise figure and axis
        fig, ax = plt.subplots()
        # Add a 1:1 line
        x_121 = np.arange(-50, 500)
        plt.plot(x_121, [0]*len(x_121), alpha=0.5, color='k', ls='--')
        # Plot up data
        sns.regplot(x='Iodide', y=param+'-residual', data=df)
        # Beautify the plot/figure
        plt.xlim(-10, 410)
        plt.ylim(-150, 150)
        plt.title('Residual (param.-Obs.) for entire dataset')
        # TODO (inc. RMSE on plot)
        stats_tmp = stats.loc[stats.index == param]
        stat_var = 'RMSE (all)'
        alt_text = alt_text_str.format(stats_tmp[stat_var][0], N)
        ax.annotate(alt_text, xy=(alt_text_x, alt_text_y),
                    textcoords='axes fraction', fontsize=f_size*1.5)
        # Save the plot
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
        # Add a 1:1 line
        x_121 = np.arange(-50, 500)
        plt.plot(x_121, [0]*len(x_121), alpha=0.5, color='k', ls='--')
        # Plot up data
        sns.regplot(x='Iodide', y=param+'-residual', data=df_tmp)
        # Beautify the plot/figure
        plt.xlim(-10, 410)
        plt.ylim(-150, 150)
        plt.title('Residual (param.-Obs.) for test dataset')
        # TODO (inc. RMSE on plot)
        stats_tmp = stats.loc[stats.index == param]
        stat_var = 'RMSE ({})'.format(testset)
        alt_text = alt_text_str.format(stats_tmp[stat_var][0], N)
        ax.annotate(alt_text, xy=(alt_text_x, alt_text_y),
                    textcoords='axes fraction', fontsize=f_size*1.5)
        # Save the plot
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
        # Add a 1:1 line
        x_121 = np.arange(-50, 500)
        plt.plot(x_121, [0]*len(x_121), alpha=0.5, color='k', ls='--')
        # Plot up data
        sns.regplot(x='Iodide', y=param+'-residual', data=df_tmp)
        # Beautify the plot/figure
        plt.xlim(-10, 410)
        plt.ylim(-150, 150)
        plt.title('Residual (param.-Obs.) for coastal test dataset')
        # TODO (inc. RMSE on plot)
        stats_tmp = stats.loc[stats.index == param]
        stat_var = 'RMSE (Coastal ({}))'.format(testset)
        alt_text = alt_text_str.format(stats_tmp[stat_var][0], N)
        ax.annotate(alt_text, xy=(alt_text_x, alt_text_y),
                    textcoords='axes fraction', fontsize=f_size*1.5)
        # Save the plot
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
        # Add a 1:1 line
        x_121 = np.arange(-50, 500)
        plt.plot(x_121, [0]*len(x_121), alpha=0.5, color='k', ls='--')
        # Plot up data
        sns.regplot(x='Iodide', y=param+'-residual', data=df_tmp)
        # Beautify the plot/figure
        plt.xlim(-10, 410)
        plt.ylim(-150, 150)
        plt.title('Residual (param.-Obs.) for non-coastal test dataset')
        # TODO (inc. RMSE on plot)
        stats_tmp = stats.loc[stats.index == param]
        stat_var = 'RMSE (Non coastal ({}))'.format(testset)
        alt_text = alt_text_str.format(stats_tmp[stat_var][0], N)
        ax.annotate(alt_text, xy=(alt_text_x, alt_text_y),
                    textcoords='axes fraction', fontsize=f_size*1.5)
        # Save the plot
        AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
        if show_plot:
            plt.show()
        plt.close()

    # Save entire pdf
    AC.plot2pdfmulti(pdff, savetitle, close=True, dpi=dpi)


def plt_X_vs_Y_for_regions(df=None, params2plot=[], LatVar='lat', LonVar='lon',
                           obs_var='Obs.'):
    """
    Plot up the X vs. Y performance by region - using core s2s functions

    Parameters
    -------
    df (pd.DataFrame): DataFrame of data
    LatVar (str): variable name in DataFrame for latitude
    LonVar (str): variable name in DataFrame for longitude
    params2plot (list): parameterisations to plot

    Returns
    -------
    (None)
    """
    # Local hardwired settings
    rm_Skagerrak_data = True
    rebuild = False
    # Use top models from full dataset  ( now: nOutliers + nSkagerak
    RFR_dict = build_or_get_models_iodide(
        rebuild=rebuild,
        rm_Skagerrak_data=rm_Skagerrak_data)
    # Get the dataframe of observations and predictions
    df = RFR_dict['df']
    # Add ensemble to the df
    LatVar = 'Latitude'
    LonVar = 'Longitude'
    ds = utils.get_predicted_values_as_ds(rm_Skagerrak_data=rm_Skagerrak_data)
    vals = utils.extract4nearest_points_in_ds(ds=ds, lons=df[LonVar].values,
                                              lats=df[LatVar].values,
                                              months=df['Month'].values,
                                              var2extract='Ensemble Monthly mean',)
    var = 'RFR(Ensemble)'
    df[var] = vals
    # Just withheld data?
    testset = 'Test set (strat. 20%)'
    df = df.loc[df[testset] == True, :]
    # Only consider the variables to be plotted
    obs_var = 'Iodide'
    params2plot = [var,  'Chance2014_STTxx2_I', 'MacDonald2014_iodide',]
    df = df[params2plot+[LonVar, LatVar, obs_var]]
    # Add ocean columns to dataframe
    df = AC.add_loc_ocean2df(df=df, LatVar=LatVar, LonVar=LonVar)
    # Split by regions
    regions = list(set(df['ocean'].dropna()))
    dfs = [df.loc[df['ocean']==i,:] for i in regions]
    dfs = dict(zip(regions,dfs))
    # Also get an open ocean dataset
    # TODO ...
    # Use an all data for now
    dfs['all'] = df.copy()
    regions += ['all']
    # Loop and plot by region
    for region in regions:
        print(region)
        df = dfs[region]
        extr_str=region+' (withheld)'
        # Now plot
        plt_X_vs_Y_for_obs_v_params(df=df, params2plot=params2plot,
                                             obs_var=obs_var,
                                             extr_str=extr_str)


def calculate_biases_in_predictions(testset='Test set (strat. 20%)',
                                    target='Iodide'):
    """
    Calculate the bias within the predictions

    Parameters
    -------
    target (str): Name of the target variable (e.g. iodide)
    testset (str): Testset to use, e.g. stratified sampling over quartiles for 20%:80%

    Returns
    -------
    (None)
    """
    # Get data
    if isinstance(df, type(None)):
        RFR_dict = build_or_get_models_iodide()
        df = RFR_dict['df']
    # Select parameterisations
    models2compare = ['RFR(Ensemble)']
    params = [u'Chance2014_STTxx2_I', u'MacDonald2014_iodide', ]
    params2plot = models2compare + params
    # --- Calculate bias for all params
    dfs = {}
    for param in params2plot:
        dfs[param] = df[param]-df[target]
    # Make new dataframe with params as columns
    dfNEW = pd.DataFrame([dfs[i] for i in params2plot]).T
    dfNEW.columns = params2plot
    dfNEW[testset] = df[testset]
    dfNEW['Coastal'] = df['Coastal']
    df = dfNEW

    # - Split dataset by testset, training, all, etc...
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

    # - Loop by data and write out stats
    print(" -----  -----  -----  -----  ----- ")
    # write out by dataset and by param
    dataset_stats = {}
    for dataset in datasets:
        # Print header
        ptr_str = " ----- For dataset split '{}' ----- "
        print(ptr_str.format(dataset))
        # Get data
        df = dfs[dataset]
        # Loop by param, print values and save overalls
        param_stats = {}
        for param in params2plot:
            # Get stats on dataset biases
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
        # Header for dataset
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
                                               target='Iodide', df=None,
                                               plot_up_CDF=False, dpi=320):
    """
    Plot up CDF and PDF plots to explore point-vs-point data

    Parameters
    -------
    target (str): Name of the target variable (e.g. iodide)
    testset (str): Testset to use, e.g. stratified sampling over quartiles for 20%:80%
    dpi (int): resolution to use for saved image (dots per square inch)
    show_plot (bool): show the plot on screen
    plot_up_CDF (bool): plot up as a cumulative distribution function

    Returns
    -------
    (None)
    """
    import seaborn as sns
    sns.set(color_codes=True)
    sns.set_context("paper", font_scale=0.75)
    # Get data
    if isinstance(df, type(None)):
        RFR_dict = build_or_get_models_iodide()
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
    # Params
    params = [u'Chance2014_STTxx2_I', u'MacDonald2014_iodide', ]
    params2plot = models2compare + params
    # setup color dictionary
    CB_color_cycle = AC.get_CB_color_cycle()
    color_d = dict(zip(params2plot, CB_color_cycle))
    # Plotting variables
    axlabel = '[I$^{-}_{aq}$] (nM)'
    # set a PDF to save data to
    savetitle = 'Oi_prj_point_for_point_comparison_PDF'
    if plot_up_CDF:
        savetitle += '_CDF'
    pdff = AC.plot2pdfmulti(title=savetitle, open=True, dpi=dpi)
    # - Plot up CDF and PDF plots for the dataset and residuals
    for dataset in datasets:
        # Get data
        df = dfs[dataset]

        # - Plot up PDF plots for the dataset
        # Plot observations
        var_ = 'Obs.'
        obs_arr = df[target].values
        ax = sns.distplot(obs_arr, axlabel=axlabel, label=var_,
                          color='k',)
        # Loop and plot model values
        for param in params2plot:
            arr = df[param].values
            ax = sns.distplot(arr, axlabel=axlabel, label=param,
                              color=color_d[param], ax=ax)
        # Force y axis extent to be correct
        ax.autoscale()
        # Beautify the plot/figure
        title = 'PDF of {} data ({}) at obs. locations'
        plt.title(title.format(dataset, axlabel))
        plt.legend()
        # Save to PDF and close plot
        AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
        plt.close()

        # - Plot up CDF plots for the dataset
        # Plot observations
        if plot_up_CDF:
            var_ = 'Obs.'
            obs_arr = df[target].values
            ax = sns.distplot(arr, axlabel=axlabel, label=var_, color='k',
                              hist_kws=dict(cumulative=True),
                              kde_kws=dict(cumulative=True))
            # Loop and plot model values
            for param in params2plot:
                arr = df[param].values
                ax = sns.distplot(arr, axlabel=axlabel, label=param,
                                  color=color_d[param], ax=ax,
                                  hist_kws=dict(cumulative=True),
                                  kde_kws=dict(cumulative=True))
            # Force y axis extent to be correct
            ax.autoscale()
            # Beautify the plot/figure
            title = 'CDF of {} data ({}) at obs. locations'
            plt.title(title.format(dataset, axlabel))
            plt.legend()
            # Save to PDF and close plot
            AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
            plt.close()

        # - Plot up PDF plots for the residual dataset
        # Get observations
        obs_arr = df[target].values
        fig, ax = plt.subplots()
        # Loop and plot model values
        for param in params2plot:
            arr = df[param].values - obs_arr
            ax = sns.distplot(arr, axlabel=axlabel, label=param,
                              color=color_d[param], ax=ax)
        # Force y axis extent to be correct
        ax.autoscale()
        # Beautify the plot/figure
        title = 'PDF of residual in {} data ({}) at obs. locations'
        plt.title(title.format(dataset, axlabel))
        plt.legend()
        # Save to PDF and close plot
        AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
        plt.close()

        # - Plot up CDF plots for the residual  dataset
        if plot_up_CDF:
            # Plot observations
            obs_arr = df[target].values
            fig, ax = plt.subplots()
            # Loop and plot model values
            for param in params2plot:
                arr = df[param].values - obs_arr
                ax = sns.distplot(arr, axlabel=axlabel, label=param,
                                  color=color_d[param], ax=ax,
                                  hist_kws=dict(cumulative=True),
                                  kde_kws=dict(cumulative=True))
            # Force y axis extent to be correct
            ax.autoscale()
            # Beautify the plot/figure
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
                                              target='Iodide', df=None,
                                              plot_up_CDF=False,
                                              dpi=320):
    """
    Plot up CDF and PDF plots to explore point-vs-point data

    Parameters
    -------
    target (str): Name of the target variable (e.g. iodide)
    testset (str): Testset to use, e.g. stratified sampling over quartiles for 20%:80%
    dpi (int): resolution to use for saved image (dots per square inch)
    show_plot (bool): show the plot on screen
    df (pd.DataFrame): DataFrame of data
    plot_up_CDF (bool): plot up as a cumulative distribution function

    Returns
    -------
    (None)
    """
    import seaborn as sns
    sns.set(color_codes=True)
    sns.set_context("paper", font_scale=0.75)
    # Get data
    if isinstance(df, type(None)):
        RFR_dict = build_or_get_models_iodide()
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
    # Params
    params = [u'Chance2014_STTxx2_I', u'MacDonald2014_iodide', ]
    params2plot = models2compare + params
    # titles to rename plots with
    rename_titles = {u'Chance2014_STTxx2_I': 'Chance et al. (2014)',
                     u'MacDonald2014_iodide': 'MacDonald et al. (2014)',
                     'RFR(Ensemble)': 'RFR(Ensemble)', }
    # setup color dictionary
    CB_color_cycle = AC.get_CB_color_cycle()
    color_d = dict(zip(params2plot, CB_color_cycle))
    # Plotting variables
    axlabel = '[I$^{-}_{aq}$] (nM)'
    # Limit of PDF plots of acutal data?
    xlim_iodide = -50
    xlim_iodide = 0
    # set a PDF to save data to
    savetitle = 'Oi_prj_point_for_point_comparison_obs_vs_model_PDF_WINDOW'
    # --- Plot up CDF and PDF plots for the dataset and residuals
    fig = plt.figure(dpi=dpi)
    for n_dataset, dataset in enumerate(datasets):
        # set Axis for abosulte PDF
        ax1 = fig.add_subplot(3, 2, [1, 3, 5][n_dataset])
        # Get data
        df = dfs[dataset]
        N_ = df.shape
        print(dataset, N_)
        # - Plot up PDF plots for the dataset
        # Plot observations
        var_ = 'Obs.'
        obs_arr = df[target].values
        ax = sns.distplot(obs_arr, axlabel=axlabel, label=var_,
                          color='k', ax=ax1)
        # Loop and plot model values
        for param in params2plot:
            arr = df[param].values
            ax = sns.distplot(arr, axlabel=axlabel,
                              label=rename_titles[param],
                              color=color_d[param], ax=ax1)
        # Force y axis extent to be correct
        ax1.autoscale()
        # Force x axis to be constant
        ax1.set_xlim(xlim_iodide, 420)
        # Beautify the plot/figure
        ylabel = 'Frequency \n ({})'
        ax1.set_ylabel(ylabel.format(dataset))
        # Add legend to first plot
        if (n_dataset == 0):
            plt.legend()
            ax1.set_title('Concentration')
        # - Plot up PDF plots for the residual dataset
        # set Axis for abosulte PDF
        ax2 = fig.add_subplot(3, 2, [2, 4, 6][n_dataset])
        # Get observations
        obs_arr = df[target].values
        # Loop and plot model values
        for param in params2plot:
            arr = df[param].values - obs_arr
            ax = sns.distplot(arr, axlabel=axlabel,
                              label=rename_titles[param],
                              color=color_d[param], ax=ax2)
        # Force y axis extent to be correct
        ax2.autoscale()
        # Force x axis to be constant
        ax2.set_xlim(-320, 220)
        # Add legend to first plot
        if (n_dataset == 0):
            ax2.set_title('Bias')
    # Save whole figure
    plt.savefig(savetitle)


def plot_monthly_predicted_iodide_diff(ds=None,
                                       res='0.125x0.125', dpi=640,
                                       target='Iodide',
                                       stats=None, show_plot=False,
                                       save2png=True,
                                       skipna=True, fillcontinents=True,
                                       var2plot = 'Ensemble_Monthly_mean',
                                       rm_non_water_boxes=True):
    """
    Plot up a window plot of predicted iodide

    Parameters
    -------
    dpi (int): resolution to use for saved image (dots per square inch)
    show_plot (bool): show the plot on screen
    res (str): horizontal resolution of dataset (e.g. 4x5)
    target (str): Name of the target variable (e.g. iodide)
    stats (pd.DataFrame): dataframe of statistics on model/obs.
    fillcontinents (bool): plot up data with continents greyed out
    save2png (bool): save the plot as png
    skipna (bool): exclude NaNs from analysis
    rm_non_water_boxes (bool): fill all non-water grid boxes with NaNs

    Returns
    -------
    (None)
    """
    import seaborn as sns
    sns.reset_orig()
    # Get predicted target data
    if isinstance(ds, type(None)):
        filename = 'Oi_prj_predicted_{}_{}.nc'.format(target, res)
        folder = utils.get_file_locations('data_root') + '/Iodide/outputs/'
        ds = xr.open_dataset(folder + filename)
    # Use center points if plotting 0.125x0.125
    if res == '0.125x0.125':
        centre = True
    else:
        centre = False
    # - Now plot up by month
    if rm_non_water_boxes:
        ds = utils.add_LWI2array(ds=ds, res=res, var2template=var2plot)
        # set non water boxes to np.NaN
        ds[var2plot] = ds[var2plot].where(ds['IS_WATER'] == True)
    # Average over time
    avg_arr = ds[var2plot].mean(dim='time', skipna=skipna)
    units = '%'
    cb_max = 240
    fig = plt.figure(dpi=dpi)
    # Loop by month and plot
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
                               fillcontinents=fillcontinents, centre=centre,
                               units=units,
                               fig=fig, ax=ax, xlabel=xlabel, ylabel=ylabel,
                               title_x=title_x, window=True, f_size=f_size)
    # Adjust figure plot aesthetics
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


def plot_monthly_predicted_iodide(ds=None,
                                  res='0.125x0.125', dpi=640, target='Iodide',
                                  stats=None, show_plot=False, save2png=True,
                                  fillcontinents=True, rm_Skagerrak_data=False,
                                  var2plot = 'Ensemble_Monthly_mean',
                                  rm_non_water_boxes=True, debug=False):
    """
    Plot up a window plot of predicted iodide

    Parameters
    -------
    rm_Skagerrak_data (bool): remove the data from the Skagerrak region
    dpi (int): resolution to use for saved image (dots per square inch)
    show_plot (bool): show the plot on screen
    save2png (bool): save the plot as png
    rm_non_water_boxes (bool): fill all non-water grid boxes with NaNs
    res (str): horizontal resolution of dataset (e.g. 4x5)
    target (str): name of the target variable being predicted by the feature variables
    fillcontinents (bool): plot up data with continents greyed out
    stats (pd.DataFrame): dataframe of statistics on model/obs.

    Returns
    -------
    (None)
    """
    import seaborn as sns
    sns.reset_orig()
    # Get data predicted target data
    if rm_Skagerrak_data:
        extr_str = '_No_Skagerrak'
    else:
        extr_str = ''
    if isinstance(ds, type(None)):
        filename = 'Oi_prj_predicted_{}_{}{}.nc'.format(target, res, extr_str)
        data_root = utils.get_file_locations('data_root')
        folder = '{}/{}/outputs/'.format(data_root, target)
        ds = xr.open_dataset(folder + filename)
    # Use center points if plotting 0.125x0.125
    if res == '0.125x0.125':
        centre = True
    else:
        centre = False

    # Now plot up by month
    # Only select boxes where that are fully water (seasonal)
    if rm_non_water_boxes:
        ds = utils.add_LWI2array(ds=ds, res=res, var2template=var2plot)
        # set non water boxes to np.NaN
        ds[var2plot] = ds[var2plot].where(ds['IS_WATER'] == True)
    units = 'nM'
    cb_max = 240
    fig = plt.figure(dpi=dpi)
    # Loop by month and plot
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
                               fillcontinents=fillcontinents, centre=centre,
                               units=units,
                               fig=fig, ax=ax, xlabel=xlabel, ylabel=ylabel,
                               title_x=title_x, window=True, f_size=f_size)
    # Adjust plot aesthetics
    bottom = 0.05
    left = 0.05
    hspace = 0.075
    wspace = 0.05
    fig.subplots_adjust(bottom=bottom, left=left, hspace=hspace, wspace=wspace)
    # Save to png
    savetitle_str = 'Oi_prj_seasonal_predicted_{}_{}_{}'
    savetitle = savetitle_str.format(target, res, extr_str)
    savetitle = AC.rm_spaces_and_chars_from_str(savetitle)
    if save2png:
        plt.savefig(savetitle+'.png', dpi=dpi)
    plt.close()


def plot_update_existing_params_spatially_window(res='0.125x0.125', dpi=320,
                                                 target='Iodide',
                                                 stats=None, show_plot=False,
                                                 save2png=True,
                                                 fillcontinents=True):
    """
    Plot up predictions from existing parameters spatially

    Parameters
    -------
    target (str): name of the target variable being predicted by the feature variables
    dpi (int): resolution to use for saved image (dots per square inch)
    show_plot (bool): show the plot on screen
    save2png (bool): save the plot as png
    fillcontinents (bool): plot up data with continents greyed out
    res (str): horizontal resolution of dataset (e.g. 4x5)
    stats (pd.DataFrame): dataframe of statistics on model/obs.

    Returns
    -------
    (None)
    """
    import seaborn as sns
    sns.reset_orig()
    # Get data predicted target data
    filename = 'Oi_prj_predicted_{}_{}.nc'.format(target, res)
    folder = utils.get_file_locations('data_root') + '/Iodide/outputs/'
    ds = xr.open_dataset(folder + filename)
    # Use center points if plotting 0.125x0.125
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
    # As a mean
    for n_var, var_ in enumerate(vars2plot):
        # Get the axis
        ax = fig.add_subplot(2, 1, n_var+1)
        # Get the annual average
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
                               extend=extend, res=res, show=False, title=title,
                               fig=fig,
                               ax=ax,
                               fillcontinents=fillcontinents, centre=centre,
                               units=units,
                               title_x=.2, title_y=1.02, window=True,
                               xlabel=xlabel,
                               f_size=f_size)
        # Add A/B label
        ax.annotate(['(A)', '(B)'][n_var], xy=(0.025, 1.02),
                    textcoords='axes fraction', fontsize=f_size)
    # Adjust plot aesthetics
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
                                           stats=None, show_plot=False,
                                           save2png=True,
                                           fillcontinents=True,
                                           target='Iodide',
                                           rm_Skagerrak_data=False,
                                           rm_non_water_boxes=True,
                                           skipna=True,
                                           verbose=True, debug=False):
    """
    Plot up the ensemble average and uncertainty (std. dev.) spatially

    Parameters
    -------
    rm_Skagerrak_data (bool): remove the data from the Skagerrak region
    dpi (int): resolution to use for saved image (dots per square inch)
    show_plot (bool): show the plot on screen
    save2png (bool): save the plot as png
    rm_non_water_boxes (bool): fill all non-water grid boxes with NaNs
    fillcontinents (bool): plot up data with continents greyed out
    skipna (bool): exclude NaNs from analysis
    stats (pd.DataFrame): dataframe of statistics on model/obs.
    verbose (bool): print out verbose output?
    debug (bool): print out debugging output?

    Returns
    -------
    (None)
    """
    import seaborn as sns
    sns.reset_orig()
    # Use the predicted values with or without the Skagerrak data?
    if rm_Skagerrak_data:
        extr_str = '_No_Skagerrak'
    else:
        extr_str = ''
    # Get spatial data from saved NetCDF
    filename = 'Oi_prj_predicted_{}_{}{}.nc'.format(target, res, extr_str)
    folder = utils.get_file_locations('data_root') + '/Iodide/outputs/'
    ds = xr.open_dataset(folder + filename)
    # setup a PDF
    savetitle = 'Oi_prj_spatial_avg_and_std_ensemble_models_{}_{}'
    savetitle = savetitle.format(res, extr_str)
    pdff = AC.plot2pdfmulti(title=savetitle, open=True, dpi=dpi)
    # Use center points if plotting 0.125x0.125
    if res == '0.125x0.125':
        centre = True
    else:
        centre = False
    # --- Now plot up montly mean of ensemble spatially
    vars2plot = ['Ensemble_Monthly_mean']
    #
    if rm_non_water_boxes:
        ds = utils.add_LWI2array(ds=ds, res=res, var2template=vars2plot[0])
        for var2mask in ['Ensemble_Monthly_mean', 'Ensemble_Monthly_std']:
            # set non water boxes to np.NaN
            ds[var2mask] = ds[var2mask].where(ds['IS_WATER'] == True)
    # set plotting vars
    units = '[I$^{-}_{(aq)}$], (nM)'
    cb_max = 240
    f_size = 20
    # As a mean
    for var in vars2plot:
        arr = ds[var].mean(dim='time', skipna=skipna).values
        # Plot up
        title = 'Annual average I ({})'.format(var)
        fixcb, nticks = np.array([0., cb_max]), 5
        extend = 'max'
        AC.plot_spatial_figure(arr, fixcb=fixcb, nticks=nticks,
                               extend=extend, res=res, show=False, title=title,
                               fillcontinents=fillcontinents, centre=centre,
                               units=units,
                               f_size=f_size)
#        AC.map_plot( arr, res=res )
        # Beautify the plot/figure
        # Save the plot
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
    # As a mean
    for var in vars2plot:
        arr = ds[var].mean(dim='time', skipna=skipna).values
        # Plot up
        title = "Spatial I uncertainity (from '{}')".format(var)
        fixcb, nticks = np.array([0., cb_max]), 6
        extend = 'max'
        AC.plot_spatial_figure(arr, fixcb=fixcb, nticks=nticks,
                               extend=extend, res=res, show=False, title=title,
                               fillcontinents=fillcontinents, centre=centre,
                               units=units,
                               f_size=f_size)
#        AC.map_plot( arr, res=res )
        # Beautify the plot/figure
        # Save the plot
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
    # As a mean
    for var in vars2plot:
        arr = ds[var].mean(dim='time', skipna=skipna).values
        # Plot up
        title = "Spatial I uncertainity (from '{}')".format(var)
        fixcb, nticks = np.array([0., cb_max]), 6
        extend = 'max'
        AC.plot_spatial_figure(arr, fixcb=fixcb, nticks=nticks,
                               extend=extend, res=res, show=False, title=title,
                               fillcontinents=fillcontinents, centre=centre,
                               units=units,
                               f_size=f_size)
#        AC.map_plot( arr, res=res )
        # Beautify the plot/figure
        # Save the plot
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
    # As a mean
    for var in vars2plot:
        arr = ds[var].mean(dim='time', skipna=skipna).values
        # Plot up
        title = "Spatial I uncertainity (from '{}')".format(var)
        fixcb, nticks = np.array([0., cb_max]), 6
        extend = 'max'
        AC.plot_spatial_figure(arr, fixcb=fixcb, nticks=nticks,
                               extend=extend, res=res, show=False, title=title,
                               fillcontinents=fillcontinents, centre=centre,
                               units=units,
                               f_size=f_size)
#        AC.map_plot( arr, res=res )
        # Beautify the plot/figure
        # Save the plot
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
    # As a mean
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
                               fillcontinents=fillcontinents, centre=centre,
                               units=units,
                               f_size=f_size)
#        AC.map_plot( arr, res=res )
        # Beautify the plot/figure
        # Save the plot
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
    # As a mean
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
                               fillcontinents=fillcontinents, centre=centre,
                               units=units,
                               f_size=f_size)
#        AC.map_plot( arr, res=res )
        # Beautify the plot/figure
        # Save the plot
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
    # As a mean
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
                               fillcontinents=fillcontinents, centre=centre,
                               units=units,
                               f_size=f_size)
#        AC.map_plot( arr, res=res )
        # Beautify the plot/figure
        # Save the plot
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
                                        window=False, f_size=20,
                                        RFR_dict=None):
    """
    Plot up the spatial changes between models

    Parameters
    -------
    dpi (int): resolution to use for saved image (dots per square inch)
    RFR_dict (dict): dictionary of core variables and data
    show_plot (bool): show the plot on screen
    save2png (bool): save the plot as png
    fillcontinents (bool): plot up data with continents greyed out
    window (bool): plot up with larger plot text as part of a window plot

    Returns
    -------
    (None)
    """
    import seaborn as sns
    sns.reset_orig()
    # Get dictionary of shared data if not provided
    if isinstance(RFR_dict, type(None)):
        RFR_dict = build_or_get_models_iodide()
    # Get XR Dataset of data
    filename = 'Oi_prj_feature_variables_{}.nc'.format(res)
    folder = utils.get_file_locations('data_root')+'/data/'
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
    vars2plot = get_features_used_by_model(RFR_dict=RFR_dict)
    # --- Now plot up concentrations spatially
    # As a mean
    for var in vars2plot:
        if ('time' in ds[var].coords):
            arr = ds[var].mean(dim='time').values
        else:
            arr = ds[var].values
        # Adjust temperature to celcuis
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
        # Beautify the plot/figure
        # Save the plot
        AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
        if show_plot:
            plt.show()
        png_filename = AC.rm_spaces_and_chars_from_str(savetitle + '_'+var)
        png_filename = png_filename+'_annual_avg'
        if save2png:
            plt.savefig(png_filename, dpi=dpi, bbox_inches='tight')
        plt.close()

        # As fraction of mean
    for var in vars2plot:

        if ('time' in ds[var].coords):
            arr = ds[var].mean(dim='time').values
        else:
            arr = ds[var].values
            # Adjust temperature to celcuis
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
        # Beautify the plot/figure
        # Save the plot
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
                                                target='Iodide',
                                                show_plot=False, save2png=True,
                                                fillcontinents=True,
                                                window=False, f_size=20):
    """
    Plot up the spatial changes between models

    Parameters
    -------
    dpi (int): resolution to use for saved image (dots per square inch)
    show_plot (bool): show the plot on screen
    save2png (bool): save the plot as png
    fillcontinents (bool): plot up data with continents greyed out
    f_size (float): fontsize to use for plotting

    Returns
    -------
    (None)
    """
    import seaborn as sns
    sns.reset_orig()
    # Get data
    filename = 'Oi_prj_predicted_{}_{}.nc'.format(target, res)
    folder = utils.get_file_locations('data_root') + '/Iodide/outputs/'
    ds = xr.open_dataset(folder + filename)
    # setup a PDF
    savetitle = 'Oi_prj_spatial_comparison_models_{}'.format(res)
    pdff = AC.plot2pdfmulti(title=savetitle, open=True, dpi=dpi)
    #
    if res == '0.125x0.125':
        centre = True
    else:
        centre = False
    # Plot settings
    units = '[I$^{-}_{(aq)}$], (nM)'
    # variables to plot?
#    vars2plot = ds.data_vars
    vars2plot = [
        'Chance2014_STTxx2_I', 'MacDonald2014_iodide', 'Ensemble_Monthly_mean'
    ]
    # --- Now plot up concentrations spatially
    # As a mean
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
        # Beautify the plot/figure
        # Save the plot
        AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
        if show_plot:
            plt.show()
        png_filename = AC.rm_spaces_and_chars_from_str(savetitle + '_'+var)
        png_filename = png_filename+'_II'
        if save2png:
            plt.savefig(png_filename, dpi=dpi, bbox_inches='tight')
        plt.close()

    # Save entire pdf
    AC.plot2pdfmulti(pdff, savetitle, close=True, dpi=dpi)


def calculate_average_predicted_surface_conc(target='Iodide'):
    """
    Calculate the average predicted surface concentration
    """
    # directory
    folder = '/Users/tomassherwen/Google_Drive/data/iodide_Oi_project/'
    # files
    file_dict = {
        'Macdonald2014': 'Oi_prj_{}_monthly_param_4x5_Macdonald2014.nc'.format(target),
        'Chance2014': 'Oi_prj_{}_monthly_param_4x5_Chance2014.nc'.format(target),
        'NEW_PARAM': 'Oi_prj_{}_monthly_param_4x5_NEW_PARAM.nc'.format(target),
    }
    s_area = AC.get_surface_area(res=res)[..., 0]  # m2 land map
    #
    for param in file_dict.keys():
        filename = file_dict[param]
#        print( param, filename )
        ds = xr.open_dataset(folder+filename)
        ds = ds[target].mean(dim='time')
        # mask for ocean
        MASK = AC.ocean_unmasked()
        arr = np.ma.array(ds.values, mask=MASK[..., 0])
        print(param, arr.mean())
        # Area weight (masked) array by surface area
        value = AC.get_2D_arr_weighted_by_X(arr.T,  s_area=s_area)
        print(param, value)


def get_ensemble_predicted_iodide(df=None,
                                  RFR_dict=None, topmodels=None, stats=None,
                                  rm_Skagerrak_data=False,
                                  use_vals_from_NetCDF=False,
                                  var2use='RFR(Ensemble)', verbose=True,
                                  debug=False):
    """
    Get predicted iodide from literature parametersations

    Parameters
    -------
    rm_Skagerrak_data (bool): remove the data from the Skagerrak region
    use_vals_from_NetCDF (bool): use the values for the prediction from offline NetCDF
    RFR_dict (dict): dictionary of core variables and data
    stats (pd.DataFrame): dataframe of statistics on model/obs.
    verbose (bool): print out verbose output?
    debug (bool): print out debugging output?
    topmodels (list): list of models to make spatial predictions for

    Returns
    -------
    (pd.DataFrame)
    """
    # Just use top 10 models are included
    # ( with derivative variables )
    if isinstance(topmodels, type(None)):
        # extract the models...
        if isinstance(RFR_dict, type(None)):
            RFR_dict = build_or_get_models_iodide(
                rm_Skagerrak_data=rm_Skagerrak_data
            )
        # Get stats on models in RFR_dict
        if isinstance(stats, type(None)):
            stats = get_stats_on_models(RFR_dict=RFR_dict,
                                        verbose=False)
        # Get list of the top models to use
        topmodels = get_top_models(RFR_dict=RFR_dict,
                                   vars2exclude=['DOC', 'Prod'])

    # Add the ensemble to the dataframe
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
                    set_backup_month_if_unknown(
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
            df_tmp = RFR_dict['df'].copy()
            df_tmp.index = df_tmp['Data_Key_ID']
            # Set the ensemble as the arithmetic mean
            df_tmp[var2use] = df_tmp[topmodels].mean(axis=1)
            # Add a column for the
            df[var2use] = np.NaN
            # Now add along the  index
            Data_Key_IDs = df_tmp['Data_Key_ID'].values
            for nDK_ID, DK_ID in enumerate(Data_Key_IDs):
                # Get predicted value
                val = df_tmp.loc[df_tmp['Data_Key_ID'] == DK_ID, var2use][0]
                # Print out diagnostic
                pstr = "Adding {} prediction for {:<20} of:  {:.2f} ({:.2f})"
                pcent = float(nDK_ID) / len(Data_Key_IDs) * 100.
                if debug:
                    print(pstr.format(var2use, DK_ID, val, pcent))
                    # fill in value to input DataFrame
                df.loc[df['Data_Key_ID'] == DK_ID, var2use] = val
    return df


# def plot_difference2_input_PDF_on_update_of_var(res='4x5'):
#     """
#     Set coordinates to use for plotting data spatially
#     """
#     # Use appropriate plotting settings for resolution
#     if res == '0.125x0.125':
#         centre = True
#     else:
#         centre = False
#     pass


def mk_PDFs_to_show_the_sensitivty_input_vars_65N_and_up(RFR_dict=None,
                                                         stats=None,
                                                         res='4x5', dpi=320,
                                                      perturb_by_mutiple=False,
                                                         save_str='',
                                                         show_plot=False):
    """
    Graphically plot the sensitivity of iodide in input variables

    Parameters
    -------
    dpi (int): resolution to use for saved image (dots per square inch)
    RFR_dict (dict): dictionary of core variables and data
    show_plot (bool): show the plot on screen
    stats (pd.DataFrame): dataframe of statistics on model/obs.

    Returns
    -------
    (None)
    """
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.style.use('ggplot')
    import seaborn as sns
    sns.set()
    # Get the dictionary of shared data
    if isinstance(RFR_dict, type(None)):
        RFR_dict = build_or_get_models_iodide()
    # Get the stats on the models built
    if isinstance(stats, type(None)):
        stats = get_stats_on_models(RFR_dict=RFR_dict, verbose=False)
    # Get the core input variables
    folder = utils.get_file_locations('data_root')+'/data/'
    filename = 'Oi_prj_feature_variables_{}.nc'.format(res)
    ds = xr.open_dataset(folder + filename)
    # set up a dictionary for different dataset splits
    dss = {}
    # Keep the base case to use as a reference
    dss['BASE'] = ds.copy()
    # Which variables should be plotted?
    topmodels = get_top_models(RFR_dict=RFR_dict, vars2exclude=['DOC', 'Prod'])
    var2test = get_features_used_by_model(RFR_dict=RFR_dict,
                                          models_list=topmodels)
#	var2test =  ['WOA_Nitrate']  # for testing just change nitrate
    # Perturb vars (e.g. by  by -/+ 10, 20, 30 % )
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
        # Add perturbations
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
        # Predict the values for the locations
        ds_tmp = mk_iodide_predictions_from_ancillaries(None,
                                                        dsA=dss[key_],
                                                        RFR_dict=RFR_dict,
                                                       use_updated_predictor_NetCDF=False,
                                                        save2NetCDF=False,
                                                        topmodels=topmodels)
        # Add ensemble to ds
        ds_tmp = add_ensemble_avg_std_to_dataset(ds=ds_tmp,
                                                 RFR_dict=RFR_dict,
                                                 topmodels=topmodels,
                                                 res=res,
                                                 save2NetCDF=False)
        # Add LWI and surface area to array
        ds_tmp = utils.add_LWI2array(ds=ds_tmp, res=res,
                               var2template='Chance2014_STTxx2_I')
        # save to dict
        dssI[key_] = ds_tmp

    # --- Plot these up
    # setup a PDF
    savetitle = 'Oi_prj_sensitivity_to_perturbation_{}{}'
    savetitle = savetitle.format(res, save_str)
    pdff = AC.plot2pdfmulti(title=savetitle, open=True, dpi=dpi)
    # make sure BASE is first
    var_ = 'BASE'
#    vars2plot = dssI.keys()
#    del vars2plot[vars2plot.index(var_) ]
#    vars2plot = [var_] + list( sorted(vars2plot) )
    # Loop and plot
    for key_ in var2test:
        print('plotting: {}'.format(key_))
        # set a single axis to use.
        fig, ax = plt.subplots()
        # Plot the base values as a shadow
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
        # Get perturbations
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
            # Now plot non-NaNs
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
    """
    Plot various spatial extents of input vairables
    """
    # Get core decision points for variables (for data=v6?)
    d = {
        'WOA_TEMP_K':  {'value': 17.4+273.15, 'std': 2.0},
    }
    # Loop and plot these threshold
    for var in d.keys():
        # Get value and std for variable
        value = d[var]['value']
        std = d[var]['std']
        # Plot up threshold
        misc.plot_threshold_plus_SD_spatially(var=var, value=value, std=std,
                                         res=res)


def explore_sensitivity_of_65N2data_denial(res='4x5', RFR_dict=None, dpi=320,
                                           target='Iodide', verbose=True,
                                           debug=False):
    """
    Explore the sensitivity of the prediction to data denial

    Parameters
    -------
    dpi (int): resolution to use for saved image (dots per square inch)
    RFR_dict (dict): dictionary of core variables and data
    verbose (bool): print out verbose output?
    debug (bool): print out debugging output?

    Returns
    -------
    (None)
    """
    import gc
    # res='4x5'; dpi=320
    # --- Local variables
    Iaq = '[I$^{-}_{aq}$]'
    # Get the core input variables
    folder = utils.get_file_locations('data_root')+'/data/'
    filename = 'Oi_prj_feature_variables_{}.nc'.format(res)
    ds = xr.open_dataset(folder + filename)
    # Get the models
    if isinstance(RFR_dict, type(None)):
        RFR_dict = build_or_get_models_iodide()
    topmodels = get_top_models(RFR_dict=RFR_dict,
                               vars2exclude=['DOC', 'Prod'], n=10)
    topmodels = list(set(topmodels))
    # Other local settings?
    plt_option_tired_but_didnt_help = False
    plt_option_tried_but_only_slightly_helped = False
    plt_excluded_obs_locations = False

    # --- Now build the models without certain values
    dfA = get_dataset_processed4ML(restrict_data_max=False,
                                   rm_Skagerrak_data=False, rm_outliers=False,)
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
#     returned_vars = mk_iodide_test_train_sets(df=df.copy(),
#         rand_20_80=False, rand_strat=True,
#         features_used=df.columns.tolist(),
#         )
#     train_set, test_set, test_set_targets = returned_vars
#     key_varname = 'Test set ({})'.format( 'strat. 20%' )
#     df[key_varname] =  False
#     df.loc[ test_set.index,key_varname ] = True
#     df.loc[ train_set.index, key_varname ] = False
#     # Print the size of the input set
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
#     RFR_dict_d[VarName] = build_or_get_models_iodide( df=df,
#         model_names = topmodels,
#         save_model_to_disk=False,
#         read_model_from_disk=False,
#         delete_existing_model_files=False
#     )

    # - no where obs where low temperature and coastal (NH)
    VarName = 'No outliers'
    bool1 = dfA[target] > utils.get_outlier_value(df=dfA, var2use=target)
    index2drop = dfA.loc[bool1, :].index
    df = dfA.drop(index2drop)
    # reset index of updated DataFrame (and save out the rm'd data prior)
    df2plot = dfA.drop(df.index).copy()
    df.index = np.arange(df.shape[0])
    # Reset the training/withhel data split
    features_used = df.columns.tolist()
    returned_vars = mk_iodide_test_train_sets(df=df.copy(),
                                              rand_20_80=False,
                                              rand_strat=True,
                                              features_used=features_used,
                                              )
    train_set, test_set, test_set_targets = returned_vars
    key_varname = 'Test set ({})'.format('strat. 20%')
    df[key_varname] = False
    df.loc[test_set.index, key_varname] = True
    df.loc[train_set.index, key_varname] = False
    # Print the size of the input set
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
    RFR_dict_d[VarName] = build_or_get_models_iodide(df=df,
                                              model_names=topmodels,
                                              save_model_to_disk=False,
                                              read_model_from_disk=False,
                                              delete_existing_model_files=False
                                              )

    # - No outliers or skaggerak
    VarName = 'No outliers \or Skagerrak'
    bool1 = dfA[target] > utils.get_outlier_value(df=dfA, var2use=target)
    bool2 = dfA['Data_Key'].values == 'Truesdale_2003_I'
    index2drop = dfA.loc[bool1 | bool2, :].index
    df = dfA.drop(index2drop)
    # reset index of updated DataFrame (and save out the rm'd data prior)
    df2plot = dfA.drop(df.index).copy()
    df.index = np.arange(df.shape[0])
    # Reset the training/withhel data split
    features_used = df.columns.tolist()
    returned_vars = mk_iodide_test_train_sets(df=df.copy(),
                                              rand_20_80=False,
                                              rand_strat=True,
                                              features_used=features_used,
                                              )
    train_set, test_set, test_set_targets = returned_vars
    key_varname = 'Test set ({})'.format('strat. 20%')
    df[key_varname] = False
    df.loc[test_set.index, key_varname] = True
    df.loc[train_set.index, key_varname] = False
    # Print the size of the input set
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
    RFR_dict_d[VarName] = build_or_get_models_iodide(df=df,
                                                     model_names=topmodels,
                                                     save_model_to_disk=False,
                                                     read_model_from_disk=False,
                                                     delete_existing_model_files=False
                                                     )

    # --- Include options that didn't improve things in PDF
    if plt_option_tried_but_only_slightly_helped:
        #                 - no where obs where low temperature and coastal (NH)
        #                 VarName = '{} '.format( Iaq ) +'<98$^{th}$'
        #                 bool1 = dfA[target] > np.percentile( df[target].values, 98 )
        #                 index2drop = dfA.loc[ bool1, : ].index
        #                 df = dfA.drop( index2drop )
        #                 reset index of updated DataFrame (and save out the rm'd data prior)
        #                 df2plot = dfA.drop( df.index ).copy()
        #                 df.index = np.arange(df.shape[0])
        #                 Reset the training/withhel data split
        #                 returned_vars = mk_iodide_test_train_sets(df=df.copy(),
        #                     rand_20_80=False, rand_strat=True,
        #                     features_used=df.columns.tolist(),
        #                     )
        #                 train_set, test_set, test_set_targets = returned_vars
        #                 key_varname = 'Test set ({})'.format( 'strat. 20%' )
        #                 df[key_varname] =  False
        #                 df.loc[ test_set.index,key_varname ] = True
        #                 df.loc[ train_set.index, key_varname ] = False
        #                 print the size of the input set
        #                 N = float(df.shape[0])
        #                 Nvals[ VarName ] = N
        #                 prt_str =  "N={:.0f} ({:.2f} % of total) for '{}'"
        #                 if verbose: print( prt_str.format( N, N/NA*100,VarName ) )
        #                 Test the locations?
        #                 if plt_excluded_obs_locations:
        #                     import seaborn as sns
        #                     sns.reset_orig()
        #                     lats = df2plot['Latitude'].values
        #                     lons = df2plot['Longitude'].values
        #                     title4plt = "Points excluded (N={}) for \n '{}'".format( int(NA-N), VarName )
        #                     AC.plot_lons_lats_spatial_on_map( lats=lats, lons=lons, title=title4plt )
        #                     savestr = 'Oi_prj_locations4data_split_{}'.format( VarName )
        #                     savestr = AC.rm_spaces_and_chars_from_str( savestr )
        #                     plt.savefig( savestr, dpi=320 )
        #                     plt.close()
        #                 rebuild (just the top models)
        #                 RFR_dict_d[VarName] = build_or_get_models_iodide( df=df,
        #                     model_names = topmodels,
        #                     save_model_to_disk=False,
        #                     read_model_from_disk=False,
        #                     delete_existing_model_files=False
        #                 )
        #
        #                 - no where obs where low temperature and coastal (NH)
        #                 VarName = '{}  + \n No Skaggerak'.format( Iaq ) +'<98$^{th}$'
        #                 bool1 = dfA[target] > np.percentile( df[target].values, 98 )
        #                 bool2 = dfA['Data_Key'].values == 'Truesdale_2003_I'
        #                 index2drop = dfA.loc[ bool1 | bool2, : ].index
        #                 df = dfA.drop( index2drop )
        #                 reset index of updated DataFrame (and save out the rm'd data prior)
        #                 df2plot = dfA.drop( df.index ).copy()
        #                 df.index = np.arange(df.shape[0])
        #                 Reset the training/withhel data split
        #                 returned_vars = mk_iodide_test_train_sets(df=df.copy(),
        #                     rand_20_80=False, rand_strat=True,
        #                     features_used=df.columns.tolist(),
        #                     )
        #                 train_set, test_set, test_set_targets = returned_vars
        #                 key_varname = 'Test set ({})'.format( 'strat. 20%' )
        #                 df[key_varname] =  False
        #                 df.loc[ test_set.index,key_varname ] = True
        #                 df.loc[ train_set.index, key_varname ] = False
        #                 print the size of the input set
        #                 N = float(df.shape[0])
        #                 Nvals[ VarName ] = N
        #                 prt_str =  "N={:.0f} ({:.2f} % of total) for '{}'"
        #                 if verbose: print( prt_str.format( N, N/NA*100,VarName ) )
        #                 Test the locations?
        #                 if plt_excluded_obs_locations:
        #                     import seaborn as sns
        #                     sns.reset_orig()
        #                     lats = df2plot['Latitude'].values
        #                     lons = df2plot['Longitude'].values
        #                     title4plt = "Points excluded (N={}) for \n '{}'".format( int(NA-N), VarName )
        #                     AC.plot_lons_lats_spatial_on_map( lats=lats, lons=lons, title=title4plt )
        #                     savestr = 'Oi_prj_locations4data_split_{}'.format( VarName )
        #                     savestr = AC.rm_spaces_and_chars_from_str( savestr )
        #                     plt.savefig( savestr, dpi=320 )
        #                     plt.close()
        #                 rebuild (just the top models)
        #                 RFR_dict_d[VarName] = build_or_get_models_iodide( df=df,
        #                     model_names = topmodels,
        #                     save_model_to_disk=False,
        #                     read_model_from_disk=False,
        #                     delete_existing_model_files=False
        #                 )
        #
        #                 Clean memory
        #                 gc.collect()

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
        returned_vars = mk_iodide_test_train_sets(df=df.copy(),
                                                  rand_20_80=False,
                                                  rand_strat=True,
                                                  features_used=df.columns.tolist(),
                                                  )
        train_set, test_set, test_set_targets = returned_vars
        key_varname = 'Test set ({})'.format('strat. 20%')
        df[key_varname] = False
        df.loc[test_set.index, key_varname] = True
        df.loc[train_set.index, key_varname] = False
        # Print the size of the input set
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
        RFR_dict_d[VarName] = build_or_get_models_iodide(df=df,
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
        returned_vars = mk_iodide_test_train_sets(df=df.copy(),
                                                  rand_20_80=False,
                                                  rand_strat=True,
                                                  features_used=df.columns.tolist(),
                                                  )
        train_set, test_set, test_set_targets = returned_vars
        key_varname = 'Test set ({})'.format('strat. 20%')
        df[key_varname] = False
        df.loc[test_set.index, key_varname] = True
        df.loc[train_set.index, key_varname] = False
        # Print the size of the input set
        N = float(df.shape[0])
        Nvals[VarName] = N
        prt_str = "N={:.0f} ({:.2f} % of total) for '{}'"
        if verbose:
            print(prt_str.format(N, N/NA*100, VarName))
        # rebuild (just the top models)
        RFR_dict_d[VarName] = build_or_get_models_iodide(df=df,
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
        returned_vars = mk_iodide_test_train_sets(df=df.copy(),
                                                  rand_20_80=False,
                                                  rand_strat=True,
                                                  features_used=df.columns.tolist(),
                                                  )
        train_set, test_set, test_set_targets = returned_vars
        key_varname = 'Test set ({})'.format('strat. 20%')
        df[key_varname] = False
        df.loc[test_set.index, key_varname] = True
        df.loc[train_set.index, key_varname] = False
        # Print the size of the input set
        N = float(df.shape[0])
        Nvals[VarName] = N
        prt_str = "N={:.0f} ({:.2f} % of total) for '{}'"
        if verbose:
            print(prt_str.format(N, N/NA*100, VarName))
        # rebuild (just the top models)
        RFR_dict_d[VarName] = build_or_get_models_iodide(df=df,
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
        returned_vars = mk_iodide_test_train_sets(df=df.copy(),
                                                  rand_20_80=False,
                                                  rand_strat=True,
                                                  features_used=df.columns.tolist(),
                                                  )
        train_set, test_set, test_set_targets = returned_vars
        key_varname = 'Test set ({})'.format('strat. 20%')
        df[key_varname] = False
        df.loc[test_set.index, key_varname] = True
        df.loc[train_set.index, key_varname] = False
        # Print the size of the input set
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
        RFR_dict_d[VarName] = build_or_get_models_iodide(df=df,
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
        returned_vars = mk_iodide_test_train_sets(df=df.copy(),
                                                  rand_20_80=False,
                                                  rand_strat=True,
                                                  features_used=df.columns.tolist(),
                                                  )
        train_set, test_set, test_set_targets = returned_vars
        key_varname = 'Test set ({})'.format('strat. 20%')
        df[key_varname] = False
        df.loc[test_set.index, key_varname] = True
        df.loc[train_set.index, key_varname] = False
        # Print the size of the input set
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
        RFR_dict_d[VarName] = build_or_get_models_iodide(df=df,
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
        returned_vars = mk_iodide_test_train_sets(df=df.copy(),
                                                  rand_20_80=False,
                                                  rand_strat=True,
                                                  features_used=df.columns.tolist(),
                                                  )
        train_set, test_set, test_set_targets = returned_vars
        key_varname = 'Test set ({})'.format('strat. 20%')
        df[key_varname] = False
        df.loc[test_set.index, key_varname] = True
        df.loc[train_set.index, key_varname] = False
        # Print the size of the input set
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
        RFR_dict_d[VarName] = build_or_get_models_iodide(df=df,
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
        returned_vars = mk_iodide_test_train_sets(df=df.copy(),
                                                  rand_20_80=False,
                                                  rand_strat=True,
                                                  features_used=df.columns.tolist(),
                                                  )
        train_set, test_set, test_set_targets = returned_vars
        key_varname = 'Test set ({})'.format('strat. 20%')
        df[key_varname] = False
        df.loc[test_set.index, key_varname] = True
        df.loc[train_set.index, key_varname] = False
        # Print the size of the input set
        N = float(df.shape[0])
        Nvals[VarName] = N
        prt_str = "N={:.0f} ({:.2f} % of total) for '{}'"
        if verbose:
            print(prt_str.format(N, N/NA*100, VarName))
        # rebuild (just the top models)
        RFR_dict_d[VarName] = build_or_get_models_iodide(df=df,
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
        returned_vars = mk_iodide_test_train_sets(df=df.copy(),
                                                  rand_20_80=False,
                                                  rand_strat=True,
                                                  features_used=df.columns.tolist(),
                                                  )
        train_set, test_set, test_set_targets = returned_vars
        key_varname = 'Test set ({})'.format('strat. 20%')
        df[key_varname] = False
        df.loc[test_set.index, key_varname] = True
        df.loc[train_set.index, key_varname] = False
        # Print the size of the input set
        N = float(df.shape[0])
        Nvals[VarName] = N
        prt_str = "N={:.0f} ({:.2f} % of total) for '{}'"
        if verbose:
            print(prt_str.format(N, N/NA*100, VarName))
        # rebuild (just the top models)
        RFR_dict_d[VarName] = build_or_get_models_iodide(df=df,
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
        returned_vars = mk_iodide_test_train_sets(df=df.copy(),
                                                  rand_20_80=False,
                                                  rand_strat=True,
                                                  features_used=df.columns.tolist(),
                                                  )
        train_set, test_set, test_set_targets = returned_vars
        key_varname = 'Test set ({})'.format('strat. 20%')
        df[key_varname] = False
        df.loc[test_set.index, key_varname] = True
        df.loc[train_set.index, key_varname] = False
        # Print the size of the input set
        N = float(df.shape[0])
        Nvals[VarName] = N
        prt_str = "N={:.0f} ({:.2f} % of total) for '{}'"
        if verbose:
            print(prt_str.format(N, N/NA*100, VarName))
        # rebuild (just the top models)
        RFR_dict_d[VarName] = build_or_get_models_iodide(df=df,
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
        returned_vars = mk_iodide_test_train_sets(df=df.copy(),
                                                  rand_20_80=False,
                                                  rand_strat=True,
                                                  features_used=df.columns.tolist(),
                                                  )
        train_set, test_set, test_set_targets = returned_vars
        key_varname = 'Test set ({})'.format('strat. 20%')
        df[key_varname] = False
        df.loc[test_set.index, key_varname] = True
        df.loc[train_set.index, key_varname] = False
        # Print the size of the input set
        N = float(df.shape[0])
        Nvals[VarName] = N
        prt_str = "N={:.0f} ({:.2f} % of total) for '{}'"
        if verbose:
            print(prt_str.format(N, N/NA*100, VarName))
        # rebuild (just the top models)
        RFR_dict_d[VarName] = build_or_get_models_iodide(df=df,
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
        returned_vars = mk_iodide_test_train_sets(df=df.copy(),
                                                  rand_20_80=False,
                                                  rand_strat=True,
                                                  features_used=df.columns.tolist(),
                                                  )
        train_set, test_set, test_set_targets = returned_vars
        key_varname = 'Test set ({})'.format('strat. 20%')
        df[key_varname] = False
        df.loc[test_set.index, key_varname] = True
        df.loc[train_set.index, key_varname] = False
        # Print the size of the input set
        N = float(df.shape[0])
        Nvals[VarName] = N
        prt_str = "N={:.0f} ({:.2f} % of total) for '{}'"
        if verbose:
            print(prt_str.format(N, N/NA*100, VarName))
        # rebuild (just the top models)
        RFR_dict_d[VarName] = build_or_get_models_iodide(df=df,
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
        returned_vars = mk_iodide_test_train_sets(df=df.copy(),
                                                  rand_20_80=False,
                                                  rand_strat=True,
                                                  features_used=df.columns.tolist(),
                                                  )
        train_set, test_set, test_set_targets = returned_vars
        key_varname = 'Test set ({})'.format('strat. 20%')
        df[key_varname] = False
        df.loc[test_set.index, key_varname] = True
        df.loc[train_set.index, key_varname] = False
        # Print the size of the input set
        N = float(df.shape[0])
        Nvals[VarName] = N
        prt_str = "N={:.0f} ({:.2f} % of total) for '{}'"
        if verbose:
            print(prt_str.format(N, N/NA*100, VarName))
        # rebuild (just the top models)
        RFR_dict_d[VarName] = build_or_get_models_iodide(df=df,
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
        returned_vars = mk_iodide_test_train_sets(df=df.copy(),
                                                  rand_20_80=False,
                                                  rand_strat=True,
                                                  features_used=df.columns.tolist(),
                                                  )
        train_set, test_set, test_set_targets = returned_vars
        key_varname = 'Test set ({})'.format('strat. 20%')
        df[key_varname] = False
        df.loc[test_set.index, key_varname] = True
        df.loc[train_set.index, key_varname] = False
        # Print the size of the input set
        N = float(df.shape[0])
        Nvals[VarName] = N
        prt_str = "N={:.0f} ({:.2f} % of total) for '{}'"
        if verbose:
            print(prt_str.format(N, N/NA*100, VarName))
        # rebuild (just the top models)
        RFR_dict_d[VarName] = build_or_get_models_iodide(df=df,
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
        returned_vars = mk_iodide_test_train_sets(df=df.copy(),
                                                  rand_20_80=False,
                                                  rand_strat=True,
                                                  features_used=df.columns.tolist(),
                                                  )
        train_set, test_set, test_set_targets = returned_vars
        key_varname = 'Test set ({})'.format('strat. 20%')
        df[key_varname] = False
        df.loc[test_set.index, key_varname] = True
        df.loc[train_set.index, key_varname] = False
        # Print the size of the input set
        N = float(df.shape[0])
        Nvals[VarName] = N
        prt_str = "N={:.0f} ({:.2f} % of total) for '{}'"
        if verbose:
            print(prt_str.format(N, N/NA*100, VarName))
        # rebuild (just the top models)
        RFR_dict_d[VarName] = build_or_get_models_iodide(df=df,
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
        returned_vars = mk_iodide_test_train_sets(df=df.copy(),
                                                  rand_20_80=False,
                                                  rand_strat=True,
                                                  features_used=df.columns.tolist(),
                                                  )
        train_set, test_set, test_set_targets = returned_vars
        key_varname = 'Test set ({})'.format('strat. 20%')
        df[key_varname] = False
        df.loc[test_set.index, key_varname] = True
        df.loc[train_set.index, key_varname] = False
        # Print the size of the input set
        N = float(df.shape[0])
        Nvals[VarName] = N
        prt_str = "N={:.0f} ({:.2f} % of total) for '{}'"
        if verbose:
            print(prt_str.format(N, N/NA*100, VarName))
        # rebuild (just the top models)
        RFR_dict_d[VarName] = build_or_get_models_iodide(df=df,
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
        returned_vars = mk_iodide_test_train_sets(df=df.copy(),
                                                  rand_20_80=False,
                                                  rand_strat=True,
                                                  features_used=df.columns.tolist(),
                                                  )
        train_set, test_set, test_set_targets = returned_vars
        key_varname = 'Test set ({})'.format('strat. 20%')
        df[key_varname] = False
        df.loc[test_set.index, key_varname] = True
        df.loc[train_set.index, key_varname] = False
        # Print the size of the input set
        N = float(df.shape[0])
        Nvals[VarName] = N
        prt_str = "N={:.0f} ({:.2f} % of total) for '{}'"
        if verbose:
            print(prt_str.format(N, N/NA*100, VarName))
        # rebuild (just the top models)
        RFR_dict_d[VarName] = build_or_get_models_iodide(df=df,
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
        returned_vars = mk_iodide_test_train_sets(df=df.copy(),
                                                  rand_20_80=False,
                                                  rand_strat=True,
                                                  features_used=df.columns.tolist(),
                                                  )
        train_set, test_set, test_set_targets = returned_vars
        key_varname = 'Test set ({})'.format('strat. 20%')
        df[key_varname] = False
        df.loc[test_set.index, key_varname] = True
        df.loc[train_set.index, key_varname] = False
        # Print the size of the input set
        N = float(df.shape[0])
        Nvals[VarName] = N
        prt_str = "N={:.0f} ({:.2f} % of total) for '{}'"
        if verbose:
            print(prt_str.format(N, N/NA*100, VarName))
        # rebuild (just the top models)
        RFR_dict_d[VarName] = build_or_get_models_iodide(df=df,
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
    # --- Plot these up
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
    # Loop and plot
    for n_key_, key_ in enumerate(vars2plot):
        # Print status to screen
        prt_str = "Plotting for {} @ {} and mk'ing Dataset object ({:.2f}%) - {}"
        Tnow = strftime("%c", gmtime())
        Pcent = n_key_/float(len(vars2plot))*100
        if verbose:
            print(prt_str.format(key_, res, Pcent, Tnow))
        # Plot up as a latitudeinal plot
        plot_predicted_iodide_vs_lat_figure_ENSEMBLE(ds=dssI[key_].copy(),
                                                     RFR_dict=RFR_dict,
                                                     res=res,
                                                     show_plot=False,
                                                     close_plot=False,
                                                     save2png=False,
                                                     topmodels=topmodels,
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
    """
    Explore sensitivty of iodide parameterisations to input variables
    """
    # --- Local variables
    # Get the core input variables
    folder = utils.get_file_locations('data_root') + '/data/'
    filename = 'Oi_prj_feature_variables_{}.nc'.format(res)
    ds = xr.open_dataset(folder + filename)
    # Get the models
    if isinstance(RFR_dict, type(None)):
        RFR_dict = build_or_get_models_iodide()
    topmodels = get_top_models(RFR_dict=RFR_dict,
                               vars2exclude=['DOC', 'Prod'], n=10)

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
        # Predict the values for the locations
        ds_tmp = mk_iodide_predictions_from_ancillaries(None,
                                                        dsA=dss[key_], RFR_dict=RFR_dict,
                                                       use_updated_predictor_NetCDF=False,
                                                        save2NetCDF=False,
                                                        topmodels=topmodels)
        # Add ensemble to ds
        ds_tmp = add_ensemble_avg_std_to_dataset(ds=ds_tmp,
                                                 RFR_dict=RFR_dict, topmodels=topmodels,
                                                 res=res,
                                                 save2NetCDF=False)
        # save to dict
        dssI[key_] = ds_tmp

    # --- Plot these up
    # setup a PDF
    savetitle = 'Oi_prj_test_impact_changing_input_features_Arctic_{}'
    savetitle = savetitle.format(res)
    pdff = AC.plot2pdfmulti(title=savetitle, open=True, dpi=dpi)
    # make sure BASE is first
    var_ = 'BASE'
    vars2plot = dssI.keys()
    del vars2plot[vars2plot.index(var_)]
    vars2plot = [var_] + list(sorted(vars2plot))
    # Loop and plot
    for key_ in vars2plot:
        # Plot up as a latitudeinal plot
        plot_predicted_iodide_vs_lat_figure_ENSEMBLE(ds=dssI[key_].copy(),
                                                     RFR_dict=RFR_dict,
                                                     res=res,
                                                     show_plot=False,
                                                     close_plot=False,
                                                     save2png=False,
                                                     topmodels=topmodels,
                                                     )
        plt.title("Obs.+Params. vs Lat for '{}'".format(key_))
        # Save to PDF and close plot
        AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
        plt.close()
    # Save entire pdf
    AC.plot2pdfmulti(pdff, savetitle, close=True, dpi=dpi)

    # Plot up up each splits prediction
    # (also plot the reference case for comparison)


def plot_predicted_iodide_PDF4region(dpi=320, extr_str='',
                                     plot_avg_as_median=False, RFR_dict=None,
                                     res='0.125x0.125', target='Iodide',
                                     show_plot=False, close_plot=True,
                                     save2png=False,
                                     folder=None, ds=None, topmodels=None):
    """
    Plot a figure of iodide vs laitude - showing all ensemble members

    Parameters
    -------
    target (str): name of the target variable being predicted by the feature variables
    dpi (int): resolution to use for saved image (dots per square inch)
    show_plot (bool): show the plot on screen
    save2png (bool): save the plot as png
    plot_avg_as_median (bool): use median as the average in plots
    close_plot (bool): close the plot?
    ds (xr.Dataset): xarray dataset to use for plotting

    Returns
    -------
    (None)
    """
    import seaborn as sns
    sns.set(color_codes=True)
    sns.set_context("paper", font_scale=0.75)
    # Get RFR_dict if not provide
    if isinstance(RFR_dict, type(None)):
        RFR_dict = build_or_get_models_iodide()
    # Get predicted values
    if isinstance(folder, type(None)):
        folder = utils.get_file_locations('data_root') + '/Iodide/outputs/'
    if isinstance(ds, type(None)):
        filename = 'Oi_prj_predicted_{}_{}{}.nc'.format(target, res, extr_str)
        ds = xr.open_dataset(folder + filename)
    # Rename to a more concise name
    print(ds.data_vars)
    # Get predicted values binned by latitude
    if res == '0.125x0.125':
        df = get_spatial_predictions_0125x0125_by_lat(ds=ds)
    else:
        df = get_stats_on_spatial_predictions_4x5_2x25_by_lat(res=res, ds=ds)
    # Params to pot
    if isinstance(topmodels, type(None)):
        topmodels = get_top_models(RFR_dict=RFR_dict,
                                   vars2exclude=['DOC', 'Prod'], n=10)
    params2plot = topmodels
    # Assign colors
    CB_color_cycle = AC.get_CB_color_cycle()
    CB_color_cycle += ['darkgreen']
    color_d = dict(zip(params2plot, CB_color_cycle))
    # - Plot up vs. lat
    fig, ax = plt.subplots()
    for param in params2plot:
        # Set color for param
        color = color_d[param]
        # Plot average
        if plot_avg_as_median:
            var2plot = '{} - median'.format(param)
        else:
            var2plot = '{} - mean'.format(param)
        # Get X
        X = df[var2plot].index.values
        # Plot as line
        plt.plot(X, df[var2plot].values, color=color, label=param)
        # Plot up quartiles tooo
        low = df['{} - 25%'.format(param)].values
        high = df['{} - 75%'.format(param)].values
        ax.fill_between(X, low, high, alpha=0.2, color=color)

    # Highlight coastal obs
    tmp_df = df_obs.loc[df_obs['Coastal'] == True, :]
    X = tmp_df['Latitude'].values
    Y = tmp_df[target].values
    plt.scatter(X, Y, color='k', marker='D', facecolor='none', s=3,
                label='Coastal obs.')
    # Non-coastal obs
    tmp_df = df_obs.loc[df_obs['Coastal'] == False, :]
    X = tmp_df['Latitude'].values
    Y = tmp_df[target].values
    plt.scatter(X, Y, color='k', marker='D', facecolor='k', s=3,
                label='Non-coastal obs.')
    # Limit plot y axis
    plt.ylim(-20, 420)
    plt.ylabel('[I$^{-}_{aq}$] (nM)')
    plt.xlabel('Latitude ($^{\\rm o}$N)')
    plt.legend()
    # save or show?
    filename = 'Oi_prj_global_predicted_vals_vs_lat_ENSEMBLE_{}{}'
    if save2png:
        plt.savefig(filename.format(res, extr_str), dpi=dpi)
    if show_plot:
        plt.show()
    if close_plot:
        plt.close()


def set_values_at_of_var_above_X_lat_2_avg(lat_above2set=65, ds=None,
                                           use_avg_at_lat=True,
                                           res='0.125x0.125',
                                           var2set=None,
                                           only_consider_water_boxes=True,
                                           fixed_value2use=None,
                                           save2NetCDF=True):
    """
    Set values above a latitude to the monthly lon average

    Parameters
    -------
    lat_above2set (float): latitude to set values above
    fixed_value2use (float): value to set selected latitudes (lat_above2set)
    var2set (str): variable in dataset to set to new value
    res (str): horizontal resolution of dataset (e.g. 4x5)
    only_consider_water_boxes (bool): only update non-water grid boxes
    ds (xr.Dataset): xarray dataset to use for plotting
    save2NetCDF (bool): save outputted dataset as a NetCDF file

    Returns
    -------
    (xr.Dataset)
    """
    print(var2set)
    # Local variables
    folder = utils.get_file_locations('data_root')+'/data/'
    # Get existing file
    if isinstance(ds, type(None)):
        filename = 'Oi_prj_feature_variables_{}.nc'.format(res)
        ds = xr.open_dataset(folder + filename)
    # Get the average value at lat
    avg = ds[var2set].sel(lat=lat_above2set, method='nearest')
    # Get index of lat to set values from
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
        # Add LWI to array
        if res == '0.125x0.125':
            folderLWI = utils.get_file_locations('AC_tools')
            folderLWI += '/data/LM/TEMP_NASA_Nature_run/'
            filenameLWI = 'ctm.nc'
            LWI = xr.open_dataset(folderLWI+filenameLWI)
            bool_water = LWI.to_array().values[0, :, idx, :] == 0.0
        else:
            LWI = AC.get_LWI_map(res=res)[..., 0]
            bool_water = (LWI[:, idx] == 0.0)
            # Use the annual value for ewvery month
            bool_water = np.ma.array([bool_water]*12)
        # Set land/ice values to NaN
        for n_month in range(12):
            avg[n_month, ~bool_water[n_month]] = np.NaN
    # Get the average over lon
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
            # Now replace values
            values[month, bool_, :] = arr
            del arr
    else:
        # Updated array of values
        arr = np.zeros(values[bool_, :].shape)
        arr[:] = np.nanmean(avg)
        # Now replace values
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
    """
    Driver to build NetCDF files with updates
    """
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



def do_analysis_processing_linked_to_depth_variable():
    """
    Function to do analysis specific to removing depth variable
    """
    from plotting_and_analysis import get_ensemble_predicted_iodide
    # Get the base topmodels
    vars2exclude = ['DOC', 'Prod', ]
    topmodels = get_top_models(RFR_dict=RFR_dict,
                               vars2exclude=vars2exclude, n=10 )
    topmodels_BASE = topmodels.copy()

    # Add the ensemble to the dataframe and over write this as the dictionary
    var2use='RFR(Ensemble)'
#     df = RFR_dict['df']
#     df = get_ensemble_predicted_iodide(df=df, RFR_dict=RFR_dict, topmodels=topmodels,
#                                        var2use=var2use)
#     RFR_dict['df'] =  df
#     # Now get the stats
#     mk_table_of_point_for_point_performance(RFR_dict=RFR_dict, inc_ensemble=True)
#
#     # - Now do the same thing, but calculate the prediction with depth
#     var2use = 'RFR(Ensemble_nDepth)'
    # Get topmodels without
    vars2exclude = ['DOC', 'Prod', 'DEPTH']
    topmodels = get_top_models(RFR_dict=RFR_dict,
                               vars2exclude=vars2exclude, n=10 )
    topmodels_DEPTH = topmodels.copy()
    # Now calculate the ensemble prediction
#     df = RFR_dict['df']
#     df = get_ensemble_predicted_iodide(df=df, RFR_dict=RFR_dict, topmodels=topmodels,
#                                        var2use=var2use)
#     RFR_dict['df'] =  df
#     # Now get the stats
#     mk_table_of_point_for_point_performance(RFR_dict=RFR_dict, df=df, inc_ensemble=True,
#                                             var2use=var2use)

    topmodels2use = topmodels_DEPTH + topmodels_BASE
    topmodels2use = list(set(topmodels2use))
    # Make a spatial prediction
    xsave_str = '_TEST_DEPTH_'
    # make NetCDF predictions from the main ancillary arrays
    save2NetCDF = True
    # resolution to use? (full='0.125x0.125', test at lower e.g. '4x5')
    res = '0.125x0.125'
#    res = '4x5'
#    res = '2x2.5'
#     mk_iodide_predictions_from_ancillaries(None, res=res, RFR_dict=RFR_dict,
#                                            use_updated_predictor_NetCDF=False,
#                                            save2NetCDF=save2NetCDF,
#                                            rm_Skagerrak_data=rm_Skagerrak_data,
#                                            topmodels=topmodels2use,
#                                            xsave_str=xsave_str, add_ensemble2ds=True)

    # Plot up the annual average predictions from the top models with depth
    filename = 'Oi_prj_predicted_Iodide_0.125x0.125_TEST_DEPTH__No_Skagerrak.nc'
    folder = './'
    ds = xr.open_dataset( folder+filename )
    # ... and without
    var2use4Ensemble = 'Ensemble_Monthly_mean'
    var2use4std = 'Ensemble_Monthly_std'
    ds = add_ensemble_avg_std_to_dataset(ds=ds, var2use4std=var2use4std,
                                         var2use4Ensemble=var2use4Ensemble,
                                         topmodels=topmodels_BASE,
                                         save2NetCDF=False
                                         )

    # Plot the same way for the no depth data
    var2use4Ensemble = 'Ensemble_Monthly_mean_nDepth'
    var2use4std = 'Ensemble_Monthly_std_nDepth'
    ds = add_ensemble_avg_std_to_dataset(ds=ds, var2use4std=var2use4std,
                                         var2use4Ensemble=var2use4Ensemble,
                                         topmodels=topmodels_DEPTH,
                                         save2NetCDF=False
                                         )

    # Save as a NetCDF to use for plotting
    ds.to_netcdf('Oi_temp_iodide_annual.nc')


def plot_spatial_figures_for_ML_paper_with_cartopy(target='Iodide'):
    """
    Plot up all the spatial figures for the ML paper with cartopy
    """
    # Add LWI to NEtCDF
    res ='0.125x0.125'
    ds = utils.add_LWI2array(ds=ds, res=res,
                             var2template='Ensemble_Monthly_mean')
    vars2mask = [
    'Ensemble_Monthly_mean_nDepth', 'Ensemble_Monthly_mean',
    'Ensemble_Monthly_std',
    'Chance2014_STTxx2_I', 'MacDonald2014_iodide',
    ]
    for var2mask in vars2mask:
        # set non water boxes to np.NaN
        ds[var2mask] = ds[var2mask].where(ds['IS_WATER'] == True)
    # Average over time
    ds = ds.mean(dim='time')
    ds.to_netcdf('Oi_temp_iodide.nc')
    # Variables for plotting
    ds = xr.open_dataset('Oi_temp_iodide.nc')
    dpi = 720
    projection = ccrs.PlateCarree()
    vmax = 240
    vmin = 0
    cbar_kwargs={
    'extend':'max', 'pad': 0.025, 'orientation':"vertical", 'label': 'nM',
#    'fraction' : 0.1
    'shrink':0.675,
    'ticks' : np.arange(vmin, vmax+1, 60),
    }
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = AC.get_colormap(arr=np.array([vmin,vmax]))

    # Now plot the core prediction
    var2use4Ensemble = 'Ensemble_Monthly_mean'
    title= 'Annual average sea-surface iodide (nM) predicted by RFR(Ensemble)'
    title = None # no title shown in paper's plots
    plotting.plot_spatial_data(ds=ds, var2plot=var2use4Ensemble, title=title,
                               vmin=0, vmax=240, extr_str=var2use4Ensemble,
                               target=target, cmap=cmap, projection=projection,
                               add_meridians_parallels=True,
                               cbar_kwargs=cbar_kwargs,
                               dpi=dpi,
                               )
    # And the one without depth
    var2use4Ensemble = 'Ensemble_Monthly_mean_nDepth'
    title= 'Annual average sea-surface iodide (nM) predicted by RFR(Ensemble-No_depth)'
    title = None # no title shown in paper's plots
    plotting.plot_spatial_data(ds=ds, var2plot=var2use4Ensemble, title=title,
                               vmin=0, vmax=240, extr_str=var2use4Ensemble,
                               target=target, cmap=cmap, projection=projection,
                               add_meridians_parallels=True,
                               cbar_kwargs=cbar_kwargs,
                               dpi=dpi,
                               )


    # -  Now plot up observations over the top
    var2use4Ensemble = 'Ensemble_Monthly_mean'
    title= 'Annual average sea-surface iodide (nM) predicted by RFR(Ensemble)'
    title = None # no title shown in paper's plots
    plotting.plot_spatial_data(ds=ds, var2plot=var2use4Ensemble, title=title,
                               vmin=0, vmax=240, extr_str=var2use4Ensemble,
                               target=target, cmap=cmap, projection=projection,
                               add_meridians_parallels=True,
                               cbar_kwargs=cbar_kwargs,
                               save_plot=False,
                               dpi=dpi,
                               )
    # Get the axis
    ax = plt.gca()
    # select dataframe with observations and predictions in it
    if isinstance(RFR_dict, type(None)):
        RFR_dict = build_or_get_models()
    df = RFR_dict['df']
    df = df.loc[df[target] <= utils.get_outlier_value(df=df, var2use=target),:]
    s = 15
    edgecolor = 'k'
    x = df[u'Longitude'].values
    y = df[u'Latitude'].values
    z = df[target].values
    ax.scatter(x, y, c=z, s=s, cmap=cmap, norm=norm, edgecolor=edgecolor,
               transform=projection, zorder=100, linewidth=0.05)
    # Now save
    extr_str = '_overlaid_with_obs'
    filename = 's2s_spatial_{}_{}.png'.format(target, extr_str)
    plt.savefig(filename, dpi=dpi, bbox_inches='tight', pad_inches=0.05)

    # -
    # Now plot the core prediction uncertainty (nM)
    var2use4Ensemble = 'Ensemble_Monthly_mean'
    var2use4std = 'Ensemble_Monthly_std'
    title= 'Spatial unceratainty in sea-surface iodide in predicted values (nM)'
    cbar_kwargs['ticks'] = np.arange(0, 30+1, 6)
    cbar_kwargs['label'] = 'nM'
    title = None # no title shown in paper's plots
    plotting.plot_spatial_data(ds=ds, var2plot=var2use4std, title=title,
                               vmin=0, vmax=30, extr_str=var2use4std,
                               target=target, cmap=cmap, projection=projection,
                               add_meridians_parallels=True,
                               cbar_kwargs=cbar_kwargs,
                               dpi=dpi,
                               )
    # Now plot the core prediction uncertainty (%)
    cbar_kwargs['ticks'] = np.arange(0, 25+1, 5)
    cbar_kwargs['label'] = '%'
    var2use4std_pcent = 'Ensemble_Monthly_std_pcent'
    ds[var2use4std_pcent] = ds[var2use4std] / ds[var2use4Ensemble] *100
    title= 'Spatial unceratainty in sea-surface iodide in predicted values (%)'
    title = None # no title shown in paper's plots
    plotting.plot_spatial_data(ds=ds, var2plot=var2use4std_pcent, title=title,
                               vmin=0, vmax=25, extr_str=var2use4std_pcent,
                               target=target, cmap=cmap, projection=projection,
                               add_meridians_parallels=True,
                               cbar_kwargs=cbar_kwargs,
                               dpi=dpi,
                               )

    # Now plot the existing parameterisations
    cbar_kwargs['ticks'] = np.arange(vmin, vmax+1, 60)
    cbar_kwargs['label'] = 'nM'
    cbar_kwargs['shrink'] = 0.85
    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(2, 1, 1, projection=projection, aspect='auto')
    var2use = 'Chance2014_STTxx2_I'
    title= '(A) Chance et al. (2014)'
    plotting.plot_spatial_data(ds=ds, var2plot=var2use, fig=fig, ax=ax1,
                               title=title,
                               vmin=0, vmax=240, extr_str=var2use,
                               target=target, cmap=cmap, projection=projection,
                               add_meridians_parallels=True,
                               cbar_kwargs=cbar_kwargs,
                               dpi=dpi, xticks=False,
                               save_plot=False,
                               )

    ax2 = fig.add_subplot(2, 1, 2, projection=projection, aspect='auto')
    var2use = 'MacDonald2014_iodide'
    title= '(B) MacDonald et al. (2014)'
    plotting.plot_spatial_data(ds=ds, var2plot=var2use, fig=fig, ax=ax2,
                               title=title,
                               vmin=0, vmax=240, extr_str=var2use,
                               target=target, cmap=cmap, projection=projection,
                               add_meridians_parallels=True,
                               cbar_kwargs=cbar_kwargs,
                               dpi=dpi,
                               save_plot=False,
                               )

    # Now save
    extr_str = '_existing_params'
    filename = 's2s_spatial_{}_{}.png'.format(target, extr_str)
    plt.savefig(filename, dpi=dpi, bbox_inches='tight', pad_inches=0.05)


def extract_4_nearest_points_in_iodide_NetCDF(lons=None, lats=None,
                                              target='Iodide',
                                              months=None,
                                           var2extract='Ensemble_Monthly_mean',
                                              rm_Skagerrak_data=False,
                                              verbose=True,
                                              debug=False):
    """
    Wrapper for extract4nearest_points_in_ds for iodide

    Parameters
    -------
    lons (np.array): list of Longitudes to use for spatial extraction
    lats (np.array): list of latitudes to use for spatial extraction
    months (np.array): list of months to use for temporal extraction
    var2extract (str): name of variable to extract data for
    rm_Skagerrak_data (bool): remove the data from the Skagerrak region
    verbose (bool): print out verbose output?
    debug (bool): print out debugging output?

    Returns
    -------
    (list)
    """
    # Get data from NetCDF as a xarray dataset
    folder = utils.get_file_locations('data_root') + '/Iodide/outputs/'
    filename = 'Oi_prj_predicted_{}_0.125x0.125{}.nc'.format(target)
    if rm_Skagerrak_data:
        filename = filename.format('_No_Skagerrak')
    else:
        filename = filename.format('')
    ds = xr.open_dataset(folder + filename)
    # Now extract the dataset
    extracted_vars = utils.extract4nearest_points_in_ds(ds=ds, lons=lons,
                                                        lats=lats,
                                                        months=months,
                                                       var2extract=var2extract,
                                                        verbose=verbose,
                                                        debug=debug)
    return extracted_vars
