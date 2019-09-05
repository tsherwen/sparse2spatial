#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Module to hold processing/analysis functions for CH3I workflow

Notes
----
ML = Machine Learning
target = the variable to be estimated and provided whilst training models
feature = a individual conpoinet of a predictor vector assigned to a target
( could be called an attribute )
"""

import numpy as np
import pandas as pd
import xarray as xr
import sparse2spatial as s2s
import sparse2spatial.utils as utils
#import sparse2spatial.ancillaries2grid_oversample as ancillaries2grid
#import sparse2spatial.archiving as archiving
from sparse2spatial.RFRbuild import mk_test_train_sets
import sparse2spatial.RFRbuild as build
import sparse2spatial.RFRanalysis as RFRanalysis
import sparse2spatial.analysis as analysis
import sparse2spatial.plotting as s2splotting
from sparse2spatial.RFRbuild import build_or_get_models

# Get CH3I specific functions
from observations import get_CH3I_obs
import observations as obs


def main():
    """
    Driver for module's man if run directly from command line. unhash
    functionalitliy to call.
    """
    # - Set core local variables
    target = 'CH3I'

    # - Get the observations? (Not needed for core workflow as also held in RFR_dict)
    # (This processese the observations and only needs to be done once)
#    df = get_dataset_processed4ML(target=target, rm_outliers=rm_outliers)

    # - build models with the observations
    RFR_dict = build_or_get_models_CH3I(rebuild=False, target=target)
    # build the models (just run once!)
#    RFR_dict = build_or_get_models_CH3I(rebuild=True, target=target)

    # Get stats ont these models
    stats = RFRanalysis.get_core_stats_on_current_models(RFR_dict=RFR_dict,
                                                         target=target, verbose=True,
                                                         debug=True)

    # Get the top ten models
    topmodels = build.get_top_models(RFR_dict=RFR_dict, stats=stats,
                                     vars2exclude=['DOC', 'Prod'], n=10)

    # --- Predict values globally (only use 0.125)
    # Extra string for NetCDF save name
    xsave_str = '_INITIAL'
    # Save predictions to NetCDF
    save2NetCDF = True
    # Resolution to use? (full='0.125x0.125', test at lower e.g. '4x5')
    res = '0.125x0.125'
#    res = '4x5'
    build.mk_predictions_for_3D_features(None, res=res, RFR_dict=RFR_dict,
                                         save2NetCDF=save2NetCDF, target=target,
                                         models2compare=topmodels,
                                         topmodels=topmodels,
                                         xsave_str=xsave_str, add_ensemble2ds=True)

    # --- Plot up the performance of the models
    df = RFR_dict['df']
    # Plot performance of models
    RFRanalysis.plt_stats_by_model(stats=stats, df=df, target=target )
    # Plot up also without derivative variables
    RFRanalysis.plt_stats_by_model_DERIV(stats=stats, df=df, target=target )

    # ---- Explore the predicted concentrations
    # Get the data
    ds = utils.get_predicted_3D_values(target=target)
    # plot up an annual mean
    s2splotting.plot_up_annual_averages_of_prediction(ds=ds, target=target)
    # Plot up the values by season
    s2splotting.plot_up_seasonal_averages_of_prediction(ds=ds, target=target)

    # ---- Check analysis for original emissions from Bell et al.
    # Check budgets

    # plot



def check_budgets_from_Bell_emiss():
    """
    """
    # target species for analysis
    target = 'CH3I'
    # Root data directory
    root = '/mnt/lustre/groups/chem-acm-2018/earth0_data/'
    root += 'GEOS//ExtData/HEMCO/'
    # Folder of CH3I emissions
    folder = root + 'CH3I/v2014-07/'
    # - Get the wetland emissions
    filename = 'ch4_wetl.geos.4x5.nc'
    dsW = xr.open_dataset( folder + filename )
    var2plot = 'CH4_SRCE__CH4cl'
    # Plot up the annual average emissions
    ds2plot = dsW[[var2plot]].mean(dim='time')
    # Set a title for the plot
    extr_str = 'Wetland_emissions_CH3I'
    units = dsW[var2plot].units
    title = "Wetland emissions of {} ({})".format(target, units)
    s2splotting.plot_spatial_data(ds=ds2plot, var2plot=var2plot, extr_str=extr_str,
                               fillcontinents=False,
                               target=target, title=title, units=units)

    # - Get the rice emissions
    filename = 'ch4_rice.geos.4x5.nc'
    dsR = xr.open_dataset( folder + filename )
    var2plot = 'CH4_SRCE__CH4an'
    # Plot up the annual average emissions
    ds2plot = dsR[[var2plot]].mean(dim='time')
    # Set a title for the plot
    extr_str = 'Rice_emissions_CH3I'
    units = dsR[var2plot].units
    title = "Rice emissions of {} ({})".format(target, units)
    s2splotting.plot_spatial_data(ds=ds2plot, var2plot=var2plot, extr_str=extr_str,
                               fillcontinents=False,
                               target=target, title=title, units=units)


    # - Get the ocean concentrations
    filename = 'ocean_ch3i.geos.4x5.nc'
    dsO = xr.open_dataset( folder + filename )
    var2plot = 'CH3I_OCEAN'
    # Plot up the annual average concentrations
    ds2plot = dsO[[var2plot]].mean(dim='time')
    # Set a title for the plot
    extr_str = 'Ocean_concs_CH3I'
    units = dsO[var2plot].units
    title = "Ocean concentrations of {} ({})".format(target, units)
    s2splotting.plot_spatial_data(ds=ds2plot, var2plot=var2plot, extr_str=extr_str,
                                target=target, title=title, units=units)


    # - Get the ocean concentrations (assuming no HEMCO conversion error)
    filename = 'ocean_ch3i.geos.4x5.nc'
    dsO = xr.open_dataset( folder + filename )
    var2plot = 'CH3I_OCEAN'
    # Plot up the annual average concentrations
    ds2plot = dsO[[var2plot]].mean(dim='time')
    #
    ds2plot[var2plot] = ds2plot[var2plot] / ( 141.9 * 1E-12 )
    # Set a title for the plot
    extr_str = 'Ocean_concs_CH3I_pM_fixed'
    vmin, vmax = 0, 15
    units = 'pM'
    title = "Ocean concentrations of {} ".format(target)
    title += "({})".format( 'pM')
    s2splotting.plot_spatial_data(ds=ds2plot, var2plot=var2plot, extr_str=extr_str,
                               vmin=vmin, vmax=vmax,
                               target=target, title=title, units=units)

   # - Get the ocean concentrations (assuming HEMCO conversion error)
    filename = 'ocean_ch3i.geos.4x5.nc'
    dsO = xr.open_dataset( folder + filename )
    var2plot = 'CH3I_OCEAN'
    # Plot up the annual average concentrations
    ds2plot = dsO[[var2plot]].mean(dim='time')
    # convert units
    ds2plot[var2plot] = ds2plot[var2plot] * 1E9 /10
    # Set a title for the plot
    extr_str = 'Ocean_concs_CH3I_ng_L'
#    vmin, vmax = 0, 15
    units = 'ng L$^{-1}$'
    title = "Ocean concentrations of {} ".format(target)
    title += "({} - x10 assumeing HEMCO conversion error)".format( units)
    s2splotting.plot_spatial_data(ds=ds2plot, var2plot=var2plot, extr_str=extr_str,
#                               vmin=vmin, vmax=vmax,
                               target=target, title=title, units=units)
    # Also plot seasonally
    vmin, vmax = 0, 2
#    vmin, vmax = 0, 1
#    extr_str = 'Ocean_concs_CH3I_ng_L_capped_at_1'
    ds2plot = dsO[[var2plot]]
    ds2plot[var2plot] = ds2plot[var2plot] * 1E9 /10
    s2splotting.plot_up_seasonal_averages_of_prediction(ds=ds2plot, target=target,
                                                     var2plot=var2plot,
                                                     var2plot_longname='Bell et al 2002',
                                                     version='_assume_x10_error',
                                                     vmin=vmin, vmax=vmax,
                                                     seperate_plots=False,
                                                     units=units)




    # - Plot the ocean concentrations from Ziska et al 2013
    folder3 = '/mnt/lustre/users/ts551/data/HalOcAt/Ziska_2013_SI_files/'
    filename = 'Ziska_CH3I.nc'
    dsZ = xr.open_dataset( folder3 + filename )
    dsZ = dsZ.rename(name_dict={'RF - ocean' : 'CH3I'})
    var2plot = 'CH3I'
    # Plot up the annual average concentrations (no time dimension so no action)
    ds2plot = dsZ[[var2plot]]
    # convert units (pM to ng/L)
    ds2plot[var2plot] = ds2plot[var2plot] /1E12 *141.9 * 1E9
    # Set a title for the plot
    extr_str = 'Ziska_Ocean_concs_CH3I_ng_L'
#    vmin, vmax = 0, 15
    units = 'ng L$^{-1}$'
    title = "Ziska et al (2013) ocean concentrations of {} ".format(target)
    title += "({})".format( units)
    s2splotting.plot_spatial_data(ds=ds2plot, var2plot=var2plot, extr_str=extr_str,
#                               vmin=vmin, vmax=vmax,
                               target=target, title=title, units=units)


    # - Get the ocean concentrations
#     filename = 'ocean_ch3i.geos.4x5.nc'
#     dsO = xr.open_dataset( folder + filename )
#     var2plot = 'CH3I_OCEAN'
#     # Plot up the annual average concentrations
#     ds2plot = dsO[[var2plot]].mean(dim='time')
#     #
#     ds2plot[var2plot] = ds2plot[var2plot] / ( 141.9 * 1E-9 ) *10
#     # Set a title for the plot
#     extr_str = 'Ocean_concs_CH3I_nM_Assume_error'
#     units = 'nM'
#     title = "Ocean concentrations of {} \n ".format(target)
#     title += "({} - assuming error of x10)".format('nM')
#     s2splotting.plot_spatial_data(ds=ds2plot, var2plot=var2plot, extr_str=extr_str,
#                                 target=target, title=title, units=units)


    # - plot up the new field  in pM concs.
    folder2 = '/users/ts551/scratch/data/s2s/CH3I/outputs/'
    filename = 'Oi_prj_predicted_CH3I_0.125x0.125.nc'
    dsML = xr.open_dataset( folder2 + filename )
    var2plot = 'Ensemble_Monthly_mean'
    # Plot up the annual average concentrations
    ds2plot = dsML[[var2plot]].mean(dim='time')
    # Set a title for the plot
    extr_str = 'Ocean_concs_CH3I_ML_fixed'
    vmin, vmax = 0, 15
    units = dsML[var2plot].units
    title = "ML ocean concentrations of {} ({})".format(target, 'pM')
    s2splotting.plot_spatial_data(ds=ds2plot, var2plot=var2plot, extr_str=extr_str,
                               vmin=vmin, vmax=vmax,
                               target=target, title=title, units=units)


    # - plot up the new fie3ld  in pM concs.
    folder2 = '/users/ts551/scratch/data/s2s/{}/outputs/'.format(target)
    filename = 'Oi_prj_predicted_{}_0.125x0.125.nc'.format(target)
    dsML = xr.open_dataset( folder2 + filename )
    var2plot = 'Ensemble_Monthly_mean'
    # Plot up the annual average concentrations
    ds2plot = dsML[[var2plot]].mean(dim='time') / 1E12 * 141.9 *1E9
    # Set a title for the plot
    extr_str = 'Ocean_concs_CH3I_ML_ng_L'
#    vmin, vmax = 0, 15
    units = dsML[var2plot].units
    title = "ML ocean concentrations of {} ({})".format(target, 'ng L$^{-1}$')
    s2splotting.plot_spatial_data(ds=ds2plot, var2plot=var2plot, extr_str=extr_str,
#                               vmin=vmin, vmax=vmax,
                               target=target, title=title, units=units)

    # Also plot seasonally
    vmin, vmax = 0, 2
#    vmin, vmax = 0, 1
#    extr_str = 'Ocean_concs_CH3I_ML_ng_L_capped_at_one'
    ds2plot = dsML[[var2plot]]
    ds2plot[var2plot] = ds2plot[var2plot] /1E12 *141.9 * 1E9
    units= 'ng L$^{-1}$'
    s2splotting.plot_up_seasonal_averages_of_prediction(ds=ds2plot, target=target,
                                                     var2plot=var2plot,
                                                     var2plot_longname='ML field',
                                                     version=extr_str,
                                                     vmin=vmin, vmax=vmax,
                                                     seperate_plots=False,
                                                     units=units)


    # - plot up the new kg/m3 concs.
    folder2 = '/users/ts551/scratch/data/s2s/{}}/outputs/'.format(target)
#    filename = 'Oi_prj_predicted_CH3I_0.125x0.125.nc'
    filename = 'Oi_prj_predicted_{}_0.125x0.125_kg_m3.nc'.format(target)
    dsML = xr.open_dataset( folder2 + filename )
    var2plot = 'Ensemble_Monthly_mean_kg_m3'
    # Plot up the annual average concentrations
    ds2plot = dsML[[var2plot]].mean(dim='time')
    # Set a title for the plot
    extr_str = 'Ocean_concs_CH3I_ML'
    units = dsML[var2plot].units
    title = "ML ocean concentrations of {} ({})".format(target, 'pM')
    s2splotting.plot_spatial_data(ds=ds2plot, var2plot=var2plot, extr_str=extr_str,
                               target=target, title=title, units=units)




def setup_ML_and_other_feilds():
    """
    Set up alternative fields to Bell et al (2002) to be read by HEMCO
    """
    # - Convert ML field for use in HEMCO GEOS-Chem

    # - Ziska et al 2013 fields
    folder3 = '/mnt/lustre/users/ts551/data/HalOcAt/Ziska_2013_SI_files/'
    filename = 'Ziska_CH3I.nc'
    dsZ = xr.open_dataset( folder3 + filename )
    dsZ = dsZ.rename(name_dict={'RF - ocean' : 'CH3I'})
    var2use = 'CH3I'
    new_var  = 'RF_CH3I_kg_m3'
    dsZ = utils.add_converted_field_pM_2_kg_m3( dsZ, new_var=new_var, var2use=var2use )
    # for lat...
    attrs_dict = dsZ['lat'].attrs
    attrs_dict['long_name'] = "latitude"
    attrs_dict['units'] = "degrees_north"
    attrs_dict["standard_name"] = "latitude"
    attrs_dict["axis"] = "Y"
    dsZ['lat'].attrs = attrs_dict
    # And lon...
    attrs_dict = dsZ['lon'].attrs
    attrs_dict['long_name'] = "longitude"
    attrs_dict['units'] = "degrees_east"
    attrs_dict["standard_name"] = "longitude"
    attrs_dict["axis"] = "X"
    dsZ['lon'].attrs = attrs_dict
    # save just the variable to use
    dsZ[[new_var]].to_netcdf( folder3+'Ziska_CH3I_kg_m3.nc' )


def compare_performance_of_parameters_against_observations():
    """
    """
    # - Get the observations and
    # Get the dataframe of observations and predictions
    df = RFR_dict['df']
    # Add ensemble to the df
    LatVar = 'Latitude'
    LonVar = 'Longitude'
    ds = utils.get_predicted_values_as_ds(target=target)
    vals = utils.extract4nearest_points_in_ds(ds=ds, lons=df[LonVar].values,
                                              lats=df[LatVar].values,
                                              months=df['Month'].values,
                                              var2extract='Ensemble_Monthly_mean',)
    var = 'RFR(Ensemble)'
    params = [var]

    df[var] = vals
    # Just withheld data?
    just_withheld_data = False
    if just_withheld_data:
        testset = 'Test set (strat. 20%)'
        df = df.loc[df[testset] == True, :]
    # Only consider the variables to be plotted
    obs_var = target
#    params2plot = [var,  'Chance2014_STTxx2_I', 'MacDonald2014_iodide',]
    params2plot = [var,  ]
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
    # loop and plot by region
    for region in regions:
        print(region)
        df = dfs[region]
        extr_str=region+' (withheld)'
#        extr_str=region
        # Now plot
        s2splotting.plt_X_vs_Y_for_obs_v_params(df=df, params2plot=params2plot,
                                             obs_var=obs_var,
                                             extr_str=extr_str)

    # Extract the Ziska et al dataset
    folder3 = '/mnt/lustre/users/ts551/data/HalOcAt/Ziska_2013_SI_files/'
    filename = 'Ziska_CH3I.nc'
    dsZ = xr.open_dataset( folder3 + filename )
    var2extract = 'RF - ocean'
    #
    var = 'Ziska13'
    params += [var]

    vals = utils.extract4nearest_points_in_ds(ds=dsZ, lons=df[LonVar].values,
                                              lats=df[LatVar].values,
                                              months=df[LatVar].values,
                                              select_within_time_dim=False,
                                              var2extract=var2extract,)
    df[var] = vals

    # Extract the Bell et al 2002 dataset
    # Root data directory
    root = '/mnt/lustre/groups/chem-acm-2018/earth0_data/'
    root += 'GEOS//ExtData/HEMCO/'
    # Folder of CH3I emissions
    folder = root + 'CH3I/v2014-07/'
    filename = 'ocean_ch3i.geos.4x5.nc'
    dsO = xr.open_dataset( folder + filename )
    var2extract = 'CH3I_OCEAN'
    # convert units
    # convert kg/m3 => ng/L
#    dsO[var2extract] = dsO[var2extract].copy() * 1E9
#    dsO[var2extract] = dsO[var2extract].copy() * 1E9 / 10 # assume a x10 error
    # convert kg/m3 to pM
    dsO[var2extract] = dsO[var2extract].copy() / ( 141.9 * 1E-12 )  / 10 # assume a x10 error
    #
#    var = 'Bell02 /10' # assume a x10 error
    var = 'Bell02'
    params += [var]
    vals = utils.extract4nearest_points_in_ds(ds=dsO, lons=df[LonVar].values,
                                              lats=df[LatVar].values,
                                              months=df['Month'].values,
                                              select_within_time_dim=True,
                                              var2extract=var2extract,)
    df[var] = vals

    # Get stats
    params = list(set(params))
    stats = utils.get_df_stats_MSE_RMSE(df=df, params=params, target=target, )

    # - Plot up an orthogonal distance regression.

    # ODr plot
    ylim = (0, 15)
    xlim = (0, 15)
    plot_ODR_window_plot(df=df, params=params, units='pM', target=target, ylim=ylim,
                         xlim=xlim)

    # Plot up a PDF of concs and bias
    ylim = (-3, 15)
    plot_up_PDF_of_obs_and_predictions_WINDOW(df=df, params=params, units='pM',
                                              target=target, xlim=xlim)
    #

    #


def plot_up_PDF_of_obs_and_predictions_WINDOW(show_plot=False, params=[],
                                              testset='Test set (strat. 20%)',
                                              target='Iodide', df=None,
                                              plot_up_CDF=False, units='pM',
                                              xlim=None,
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
    # Entire dataset
    dfs['Entire'] = df.copy()
    # Testdataset
    dfs['All (withheld)'] = df.loc[df[testset] == True, :].copy()
    # Coastal testdataset
    # maintain ordering of plotting
    datasets = dfs.keys()
    # setup color dictionary
    CB_color_cycle = AC.get_CB_color_cycle()
    color_d = dict(zip(params, CB_color_cycle))
    # plotting variables

    # set a PDF to save data to
    savetitle = 'Oi_prj_point_for_point_comparison_obs_vs_model_PDF_WINDOW'
    # --- Plot up CDF and PDF plots for the dataset and residuals
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
            axlabel = '[{}$_{}$] ({})'.format( target, '{aq}', units )
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
        # - Plot up PDF plots for the residual dataset
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


def plot_ODR_window_plot(params=[], show_plot=False, df=None,
                         testset='Test set (strat. 20%)', units='pM',
                         target='Iodide', context="paper", xlim=None, ylim=None,
                         dpi=720):
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
    # select dataframe with observations and predictions in it
    if isinstance(df, type(None)):
        print('Please provide DataFrame with data')

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
    # iodide in aq
    target_label = '[{}$_{}$]'.format(target, 'aq')
    # set location for alt_text
    f_size = 10
    N = int(df.shape[0])
    # split data into groups
    dfs = {}
    # Entire dataset
    dfs['Entire'] = df.copy()
    # Testdataset
    dfs['Withheld'] = df.loc[df[testset] == True, :].copy()
    dsplits = dfs.keys()
    # assign colors
    CB_color_cycle = AC.get_CB_color_cycle()
    color_d = dict(zip(dsplits, CB_color_cycle))
    # Intialise figure and axis
    fig, axs = plt.subplots(1, 3, sharex=True, sharey=True, dpi=dpi, \
                            #        figsize=(12, 5)
                            figsize=(11, 4)
                            )
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
        x_121 = np.arange(ylim[0]-(ylim[1]*0.05),ylim[1]*1.05 )
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



def add_field2HEMCO_in_kg_m3(ds, var2use='Ensemble_Monthly_mean', RMM=141.9,
                            target='CH3I', new_var='Ensemble_Monthly_mean_kg_m3'):
    """
    Convert the CH3I prediction from pM to kg/m3
    """
    # Convert pM to kg/m3
    ds[new_var] = ds[var2use].copy() /1E12 *RMM /1E3 *1E3
    # Add attributes (needed for HEMCO checks)
    attrs_dict = ds[new_var].attrs
    attrs_dict['units'] = "kg/m3"
    attrs_dict['units_longname'] = "kg({})/m3".format(target)
    ds[new_var].attrs = attrs_dict
    return ds


def plot_seasaonl_model_vs_obs(dpi=320, target='CH3I', use_hourly_files=True ):
    """
    Plot seasonal comparisons of observations and models
    """
    # Get observational data
    dfs = obs.get_ground_surface_CH3I_obs_DIRECT()
    sites = list(dfs.keys())
    # sort the sites by latitude
    sites = AC.sort_locs_by_lat(sites)
    # get model_data
    run_root = '/users/ts551/scratch/GC/rundirs/'
    run_str = 'geosfp_4x5_tropchem.v12.2.1.AQSA.'
    run_dict = {
#    'Ordonez2012': run_root + run_str + 'GFAS/',
    'ML (partial)': run_root + run_str  +'GFAS.CH3I.ML.repeat.II/',
#    'Bell2002 (partial)': run_root + run_str  +'GFAS.CH3I.repeat/',
    'Bell2002 (x10, partial)': run_root + run_str  +'GFAS.CH3I.repeat.III//',
    'Bell2002 (x10)': run_root + run_str  +'GFAS.CH3I.repeat.IV/',
    'Bell2002 (All fixes)': run_root + run_str  +'GFAS.CH3I.ALL/',
    'Bell2002 (AF+DMS Sc)': run_root + run_str  +'GFAS.CH3I.ALL.UseDMS_Sc/',
    'Ziska2013': run_root + run_str  +'GFAS.CH3I.Ziska13/',
    }
    runs = list(run_dict.keys())
    # Setup a colour dictionary
    colour_dicts = dict( zip(runs, AC.get_CB_color_cycle()[::-1] ))
    # loop and extract by site
    res='4x5'
    model_data = {}
    for run in runs:
        # Get data location and filename
        wd = run_dict[run]
        if use_hourly_files:
            filename = 'ts_ctm.nc'
        else:
            filename = 'ctm.nc'
        # open the file
        ds = xr.open_dataset(wd+filename)
        # select the nearest location to a given site
        df = pd.DataFrame()
        for site in sites:
            print(site)
            # Get: lat, lon, alt (m)
            lat, lon, alt, = AC.get_loc(site)
            # Nearest lat
            ds_tmp = ds['IJ_AVG_S__'+'CH3I'].sel(latitude=lat, method='nearest')
            # Nearest lon
            ds_tmp = ds_tmp.sel(longitude=lon, method='nearest')
            # take the bottom-most box if using 3D output
            if use_hourly_files:
                pass
            else:
                ds_tmp = ds_tmp.sel(model_level_number=1)
            # convert to pptv
            ds_tmp = ds_tmp *1E3
            # colaspe to a series and save to dataframe
            df[site] = ds_tmp.to_pandas().copy()
            del ds_tmp
        # Save the data to a DataFrame
        model_data[run] = df.copy()
        del df
        del ds


    # Now loop by site and plot up - one plot per site
#     for site in sites:
#         print(site)
#         fig = plt.figure()
#         ax = fig.add_subplot()
#         # Get Obs. for site
#         DateVar = 'Datetime'
#         DataVar = 'CH3I'
#         df = dfs[site][[DateVar, DataVar]].dropna()
#         dates = df[DateVar].values
#         data = df[DataVar].values
#         print(dates.shape, data.shape)
#         title = site
#         # Plot up observations
#         AC.BASIC_seasonal_plot(data=data, dates=dates, color='k', label='Obs.',
#                                title=title)
#         # plot up model data
#         for run in runs:
#             df = model_data[run]
#             #
#             dates = df.index.values
#             data = df[site].values
#             # Plot up observations
#             AC.BASIC_seasonal_plot(data=data, dates=dates, color=colour_dicts[run],
#                                 label=run)
#         #
#         filename = 's2s_{}_seasonal_cycle_{}'.format( target, site)
#         filename = AC.rm_spaces_and_chars_from_str(filename)
#         plt.savefig(filename)

    # Plot as a single mutiple panel figure
#    fig = plt.figure()
    fig = plt.figure(figsize=(16, 10))
    # leave a space for the figure caption
    subplots2use = [1,2, 3,4] +  list(np.arange(16)[6:] )
    units = 'pptv'
    for nsite, site in enumerate( sites ):
        print(site)
        #
        axn = (3, 5, subplots2use[nsite] )
        ax = fig.add_subplot(*axn)
        # Get Obs. for site
        DateVar = 'Datetime'
        DataVar = 'CH3I'
        df = dfs[site][[DateVar, DataVar]].dropna()
        dates = df[DateVar].values
        data = df[DataVar].values
        print(dates.shape, data.shape)
        title = AC.site_code2name(site)
        # Only have axis labels on bottom line and far left
        xlabel, ylabel, rm_yticks = False, None, False
        if subplots2use[nsite] in np.arange(1,16)[-5:]:
            xlabel = True
        if subplots2use[nsite] in np.arange(1,16)[::5]:
            ylabel = '{} ({})'.format(target, units )
            rm_yticks = False
        # Plot up observations
        AC.BASIC_seasonal_plot(data=data, dates=dates, color='k', label='Obs.',
                            title=title, xlabel=xlabel, ylabel=ylabel,
                            plot_Q1_Q3=True,
                            rm_yticks=rm_yticks)
        # Add location to plot as text
        lon, lat, alt = AC.get_loc(site)
        alt_text = '({:.2f}{}, {:.2f}{})'.format(lon, '$^{o}$E', lat, '$^{o}$N')
        plt.text(0.05, 0.95, alt_text, ha='left', va='center', color='black',
#             fontsize=15,
             alpha=0.75, transform=ax.transAxes)
        # Plot up model data
        for run in runs:
            df = model_data[run]
            #
            dates = df.index.values
            data = df[site].values
            # Plot up observations
            AC.BASIC_seasonal_plot(data=data, dates=dates, color=colour_dicts[run],
                                label=run,
                                plot_Q1_Q3=True,
                                xlabel=xlabel, ylabel=ylabel,
                                rm_yticks=rm_yticks )
    # Add a legend
    axn = (3, 5, 5 )
    ax = fig.add_subplot(*axn)
    #
    legend_text = 'Obs.'
    plt.text(0.5, 0.95, legend_text, ha='center', va='center', color='black',
             fontsize=15, transform=ax.transAxes)
    # Now add colours for model lines
    buffer = 0.085
    for nrun, run in enumerate( runs ):
        plt.text(0.5, 0.95-(buffer+(buffer*nrun)), '{}'.format(run),
                color=colour_dicts[run], fontsize=15,
                ha='center', va='center', transform=ax.transAxes)

#    plt.legend()
#    plt.legend(['First List','Second List'], loc='upper left')
    plt.axis('off')
    #
    filename = 's2s_{}_seasonal_cycle_ALL_ground_sites'.format( target )
    filename = AC.rm_spaces_and_chars_from_str(filename)
    plt.savefig(filename, dpi=dpi)
    plt.close('all')


def quick_check_of_CH3I_emissions(target='CH3I'):
    """
    Analyse the emissions of methyl iodide through HEMCO
    """
    #
    root = '/users/ts551/scratch/GC/rundirs/'
    file_str = 'geosfp_4x5_tropchem.v12.2.1.AQSA.{}'
    run_dict = {
    # intial test runs
#     'Ordonez2012' : root + file_str.format('.CH3I.Ordonez2012/spin_up/'),
#     'Bell2002' : root + file_str.format('.CH3I/spin_up/'),
#     'Bell2002x10' : root + file_str.format('.CH3I.x10/spin_up/'),
#     'ML'    : root + file_str.format('.CH3I.ML/spin_up/'),
    # repeat tests
#    'Ordonez2012' : root + file_str.format('GFAS.CH3I.Ordonez2012/spin_up/'),
#    'Bell2002' : root + file_str.format('GFAS.CH3I.repeat/'),
#    'ML'  : root + file_str.format( 'GFAS.CH3I.ML.repeat.II/' ),
    #
#    'Bell2002 (All fixes)': run_root + run_str  +'GFAS.CH3I.ALL/',
#    'Bell2002 (AF+DMS Sc)': run_root + run_str  +'GFAS.CH3I.ALL.UseDMS_Sc/',
#    'Ziska2013 ()': run_root + run_str  +'GFAS.CH3I.Ziska13/',
    }
    # use the run_dict from - obs.get_ground_surface_CH3I_obs_DIRECT
    wds = run_dict # for debugging...
    target = 'CH3I' # for testing
    #
    filename = 'HEMCO_diagnostics.201401010000.nc'
    # Get a dictionary of all the data
    dsDH = GetEmissionsFromHEMCONetCDFsAsDatasets(wds=run_dict)
    # - Analysis the totals
    # Extract the annual totals to a dataFrame
    df = pd.DataFrame()
    for run in dsDH.keys():
        print(run)
        # Get the values and variable names
        vars2use = [i for i in dsDH[run].data_vars if i != 'AREA']
        vars2use = list(sorted(vars2use))
        vals = [ dsDH[run][i].sum().values for i in vars2use ]
        # Save the summed values to the dataframe
        df[run] = pd.Series( dict(zip(vars2use, vals)) )
#        print( dsDH[run].sum() )
    # Print the DataFrame to screen
    print(df)

    # - Analysis the spatially the emissions
    # which variables to plot
    vars2plot = [
    'ML (partial)', 'Bell2002 (All fixes)', 'Bell2002 (AF+DMS Sc)', 'Ziska2013'
    ]
    #
    units2use = ['ng m$^{-2}$ s$^{-1}$', 'pmol m$^{-2}$ hr$^{-1}$']
    target='CH3I'
    # loop by variable and plot
    var2plot = 'EmisCH3I_SEAFLUX'
    for var in vars2plot:
        # Loop units too
        for units in units2use:
            # get the dataset as a temporary copy
            print(var, units)
            ds = dsDH[var].copy()
            # convert the values
            if units == 'ng m$^{-2}$ s$^{-1}$':
                # convert to /m2, then Gg => g, the g=>ng
                ds[var2plot] = ds[var2plot] / ds['AREA'] *1E9 *1E9
                # convert from /yr to /s
                ds[var2plot] = ds[var2plot] / 365/24/60/60
            elif units=='pmol m$^{-2}$ hr$^{-1}$':
                # convert to /m2, then Gg => g
                ds[var2plot] = ds[var2plot] / ds['AREA'] *1E9
                # convert from /yr to /hr
                ds[var2plot] = ds[var2plot] / 365/24
                # convert g to mol , then to pmol
                ds[var2plot] = ds[var2plot] / AC.species_mass(target) *1E12
#                / AC.constants('AVG')

            else:
                sys.exit()
                print('WARNING: unit conversion not setup')
            # plot the seasonally resoved flux
            s2splotting.plot_up_seasonal_averages_of_prediction(ds=ds, target=target,
                                                     var2plot=var2plot,
                                                     var2plot_longname=var,
                                                     version='{}_{}'.format(var, units),
                                                     vmin=None, vmax=None,
                                                     seperate_plots=False,
                                                     units=units)
            # plot the annual average resoved flux
            extr_str ='{}_{}'.format(var, units)
            # Set a title for the plot
            title = "Annual average '{}' ({})".format(var2plot, units)
            # Now plot
            s2splotting.plot_spatial_data(ds=ds[[var2plot]].mean(dim='time'),
                            var2plot=var2plot,
                              extr_str=extr_str, target=target,
                              title=title)
            plt.close()




def check_units_of_Bell2002_files():
    """
    convert kg/m3 to ng/L
    """
    #
    pass




def GetEmissionsFromHEMCONetCDFsAsDatasets(wds=None, average_over_time=False):
    """
    Get the emissions from the HEMCO NetCDF files as a dictionary of datasets.
    """
    import AC_tools as AC
    # Look at emissions through HEMCO
    # Get data locations and run names as a dictionary
    if isinstance(wds, type(None)):
        wds = get_run_dict4EGU_runs()
    runs = list(wds.keys())
    # variables to extract
    vars2use = [
        # Testing vars from v5.0
#         'TOTAL_CH3I_L', 'TOTAL_SEA_CH3I', 'TOTAL_CH3I_a', 'TOTAL_CH2BR2', 'TOTAL_CHBR3',
#         'TOTAL_CH2I2', 'TOTAL_CH2ICL', 'TOTAL_CH2IBR', 'TOTAL_HOI', 'TOTAL_I2',
#         'TOTAL_ISOP'
        #
        'EmisCH3I_ordonez', 'EmisCH3I_SEAFLUX', 'EmisCH3I_TOTAL',
        'EmisCH2I2_Ocean', 'EmisCH2ICl_Ocean', 'EmisCH2IBr_Ocean', 'EmisI2_Ocean',
        'EmisHOI_Ocean', 'EmisI2_Ocean_Total', 'EmisHOI_Ocean_Total',
        'EmisCH2Br2_Ocean', 'EmisCHBr3_Ocean',
        #
        'EmisCH3I_B02_RICE', 'EmisCH3I_B02_WETL', 'EmisCH3I_B02_BIOBURN',
        'EmisCH3I_B02_BIOFUEL',
        # Also get values for
#        'EmisACET_Ocean', 'EmisALD2_Ocean',
        'EmisDMS_Ocean',
        #
        'EmisCH2Br2_SEAFLUX', 'EmisCHBr3_SEAFLUX',
    ]
    # Make sure there are no double ups in the list
    vars2use = list(set(vars2use))
    # Loop and extract files
    dsDH = {}
    for run in runs:
        wd = wds[run]
        print(run, wd)
        dsDH[run] = AC.get_HEMCO_diags_as_ds(wd=wd)
    # Get actual species
    specs = [i.split('Emis')[-1].split('_')[0] for i in vars2use]
#     specs = [
#     'CH3I','CH3I', 'CH3I', 'CH2Br2', 'CHBr3', 'CH2I2', 'CH2ICl', 'CH2IBr',
#     'HOI', 'I2', 'ISOP'
#     ]
    var_species_dict = dict(zip(vars2use, specs))
    # Only include core species in dataset
    for key in dsDH.keys():
        ds = dsDH[key][ ['AREA'] ]
        # Try and add al of the requested variables too
        for var in vars2use:
            try:
                ds[var] = dsDH[ key ][ var ]
            except KeyError:
                print("WARNING: Skipped '{}' for '{}'".format(key, var) )
        dsDH[key] = ds

    # Convert to Gg
    for run in runs:
        dsREF = dsDH[run].copy()
        # Average over the time dimension?
        if average_over_time:
            ds = dsDH[run].copy().mean(dim='time')
            for var in dsREF.data_vars:
                ds[var].attrs = dsREF[var].attrs
            ds2use = ds
        else:
            ds2use = dsREF
        # Convert the units
        ds = AC.convert_HEMCO_ds2Gg_per_yr(ds2use, vars2convert=vars2use,
                                           var_species_dict=var_species_dict)
        # Update the dictionary
        dsDH[run] = ds
    return dsDH


def plot_up_obs_data_by_year( target='CH3I',):
    """
    Plot up the amount of observational in a given year, by region
    """
    # Get the observational data
    df = get_CH3I_obs()
    # Add the ocean that the data is in
    df = analysis.add_loc_ocean2df(df=df,  LatVar='Latitude', LonVar='Longitude')
    # Plot up the observational data by year
    plot_up_df_data_by_yr( df=df, target=target )


def plot_up_df_data_by_yr(df=None, Datetime_var='datetime', TimeWindow=5,
                          start_from_last_obs=False, drop_bins_without_data=True,
                          target='Iodide', dpi=320):
    """
    Plot up # of obs. data binned by region against year
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
    # Calculate the numer of points for each bin by region
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



def plt_X_vs_Y_for_regions(RFR_dict=None, df=None, params2plot=[], LatVar='lat',
                           LonVar='lon', target='CH3I',
                           obs_var='Obs.'):
    """
    Plot up the X vs. Y performance by region - using core s2s functions
    """
    # Get the dataframe of observations and predictions
    df = RFR_dict['df']
    # Add ensemble to the df
    LatVar = 'Latitude'
    LonVar = 'Longitude'
    ds = utils.get_predicted_values_as_ds(target=target)
    vals = utils.extract4nearest_points_in_ds(ds=ds, lons=df[LonVar].values,
                                              lats=df[LatVar].values,
                                              months=df['Month'].values,
                                              var2extract='Ensemble_Monthly_mean',)
    var = 'RFR(Ensemble)'
    df[var] = vals
    # Just withheld data?
    testset = 'Test set (strat. 20%)'
    df = df.loc[df[testset] == True, :]
    # Only consider the variables to be plotted
    obs_var = target
#    params2plot = [var,  'Chance2014_STTxx2_I', 'MacDonald2014_iodide',]
    params2plot = [var,  ]
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
    # loop and plot by region
    for region in regions:
        print(region)
        df = dfs[region]
        extr_str=region+' (withheld)'
#        extr_str=region
        # Now plot
        s2splotting.plt_X_vs_Y_for_obs_v_params(df=df, params2plot=params2plot,
                                             obs_var=obs_var,
                                             extr_str=extr_str)



def build_or_get_models_CH3I(target='CH3I',
                             rm_LOD_filled_data=False,
                             rm_outliers=True,
                             rebuild=False):
    """
    Wrapper call to build_or_get_models for sea-surface CH3I
    """
    # Get the dictionary  of model names and features (specific to iodide)
    model_feature_dict = utils.get_model_features_used_dict(rtn_dict=True)

    # Get the observational dataset prepared for ML pipeline
    df = get_dataset_processed4ML(target=target, rm_outliers=rm_outliers)

    if rebuild:
        RFR_dict = build_or_get_models(save_model_to_disk=True,
                                       model_feature_dict=model_feature_dict,
                                       df=df, target=target,
                                       read_model_from_disk=False,
                                       delete_existing_model_files=True)
    else:
        RFR_dict = build_or_get_models(save_model_to_disk=False,
                                       model_feature_dict=model_feature_dict,
                                       df=df, target=target,
                                       read_model_from_disk=True,
                                       delete_existing_model_files=False)
    return RFR_dict


def get_dataset_processed4ML(restrict_data_max=False, target='CH3I',
                             rm_outliers=True,
                             rm_LOD_filled_data=False):
    """
    Get dataset as a DataFrame with standard munging settings


    Parameters
    -------
    restrict_data_max (bool): restrict the obs. data to a maximum value?

    Returns
    -------
    (pd.DataFrame)

    Notes
    -----

    """
    from observations import add_extra_vars_rm_some_data
    from observations import get_processed_df_obs_mod
    # - Local variables
    features_used = None
    target = 'CH3I'
    target_name = [target]
    # - The following settings are set to False as default
    # settings for incoming feature data
    restrict_min_salinity = False
#    use_median_value_for_chlor_when_NaN = False
#    add_modulus_of_lat = False
    # Apply transforms to data?
    do_not_transform_feature_data = True
    # Just use the forest outcomes and do not optimise
    use_forest_without_optimising = True
    # - Get data as a DataFrame
    df = get_processed_df_obs_mod()  # NOTE this df contains values >400nM
    # Add extra vairables and remove some data.
    df = add_extra_vars_rm_some_data(df=df, target=target,
                                     restrict_data_max=restrict_data_max,
                                     restrict_min_salinity=restrict_min_salinity,
                                     rm_outliers=rm_outliers,
                                     )    # add
    # Re-index to a single contiguous index
    df['Original Index'] = df.index.copy()
    N1 = df.shape[0]
    df.index = np.arange(N1)
    print('WARNING: Reindexed to shape of DataFrame processed for ML ({})'.format(N1))

    # - Add test and training set assignment to columns
    # standard split vars?  (values=rand_20_80, rand_strat)
    ways2split_data = {
        'rn. 20%': (True, False),
        'strat. 20%': (False, True),
    }
    # Loop training/test split methods
    for key_ in ways2split_data.keys():
        # Get settings
        rand_20_80, rand_strat = ways2split_data[key_]
        # Copy a df for splitting
#        df_tmp = df['Iodide'].copy()
        # Now split using existing function
        returned_vars = mk_test_train_sets(df=df.copy(),
                                                 target=target,
                                                 rand_20_80=rand_20_80,
                                                 rand_strat=rand_strat,
                                                 features_used=df.columns.tolist(),
                                                 )
        train_set, test_set, test_set_targets = returned_vars
        # Now assign the values
        key_varname = 'Test set ({})'.format(key_)
        df[key_varname] = False
        df.loc[test_set.index, key_varname] = True
        df.loc[train_set.index, key_varname] = False
    return df


def check_flux_variables():
    """
    """
    filename = 'GCv12.2.1_GFAS.geos.log_TEST_4'
    folder = '/users/ts551/scratch/GC/rundirs'
    folder += '/geosfp_4x5_tropchem.v12.2.1.AQSA.GFAS.CH3I.'
    folder += 'ALL.test_other_sources.repeat.TESTING/'
    # variables to extract
    vars2use = [
    'Schmidt number in air', 'Drag coefficient', 'Friction velocity',
    'Airside resistance',
    'Schmidt number in water', 'Schmidt number of CO2', 'Waterside resistance',
    ]
    # lists for storing data
    lists = [[]]*len(vars2use)
    list_of_vars = []
    # loop by line and add to lists
    with open(folder+filename) as file:
        for line in file:
            for n, var in enumerate( vars2use ):
                if line.startswith(' '+var):
#                    lists[n].append(float(line.split(':')[-1].strip()) )
#                    lists[n].append( line )
                    list_of_vars.append( line )
#                else:
#                    pass
    # Now split off by variable
    list_of_lists = []
    for n, var in enumerate( vars2use ):
        new_list = []
        for nitem, item in enumerate( list_of_vars ):
                if item.startswith(' '+var):
#                    print( item, n, item.startswith(' '+var) )
#                    new_list.append( item  )
                    new_list.append( float(item.split(':')[-1].strip())  )

        list_of_lists += [new_list ]

    # turn into a single array
    a = np.array(list_of_lists)
    # Then a dataframe with labels
    df = pd.DataFrame( a.T )
    df.columns = vars2use
    # Now quickly plot these up.
    for var in vars2use:

        sns.distplot( df[var] )
        savename = 's2s_CH3I_{}'.format(var)
        savename = AC.rm_spaces_and_chars_from_str( savename )
        plt.savefig( savename, dpi=320 )

        plt.close()



if __name__ == "__main__":
    main()
