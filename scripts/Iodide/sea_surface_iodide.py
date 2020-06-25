#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Module to hold core processing/analysis functions for Ocean iodide (Oi!) project

Notes
----
ML = Machine Learning
target = the value aiming to be estimated or provided in training
feature = a induivual conpoinet of a predictor vector assigned to a target
( could be called an attribute )
predictor = vector assigned to a target value


Please see Paper(s) for more details:

Sherwen, T., Chance, R. J., Tinel, L., Ellis, D., Evans, M. J., and Carpenter, L. J.: A machine learning based global sea-surface iodide distribution, Earth Syst. Sci. Data Discuss., https://doi.org/10.5194/essd-2019-40, in review, 2019.


"""
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs

# import AC_tools (https://github.com/tsherwen/AC_tools.git)
import AC_tools as AC

import sparse2spatial as s2s
import sparse2spatial.utils as utils
import sparse2spatial.ancillaries2grid_oversample as ancillaries2grid
import sparse2spatial.archiving as archiving
import sparse2spatial.ancillaries as ancillaries
import sparse2spatial.RFRbuild as build
import sparse2spatial.archiving as archiving
import sparse2spatial.analysis as analysis
import sparse2spatial.RFRanalysis as RFRanalysis
import sparse2spatial.plotting as plotting
import sparse2spatial.plotting as s2splotting
#from sparse2spatial.RFRanalysis import get_stats_on_models
from sparse2spatial.analysis import add_ensemble_avg_std_to_dataset
from sparse2spatial.RFRbuild import get_top_models
from sparse2spatial.RFRbuild import build_or_get_models

# Get iodide specific functions
#from observations import get_dataset_processed4ML
import observations as obs
import project_misc as misc
import plotting_and_analysis as plt_analysis


def main():
    """
    Main driver if run directly from command line. unhash functionality to call.

    Notes
    -------
     - Calls for the full pipeline used for producing a new sea-surface iodide field
     are listed below. However, many of these now in the following functions in the
     same folder:
        plotting_and_analysis.py
        process_new_observations.py
        project_misc.py
        observations.py
        emissions.py
     -  simply unhash the functions to be run below.
    """
    # ---- ---- Over-arching settings
    #
    target='Iodide'
    # Setup the data directory structure (only needs to be done once))
    # NOTE: the locations of s2s and data are set in script/<target>'s *.rc file
#    utils.check_or_mk_directory_structure(target=target)

    # General settings
    rm_Skagerrak_data = True
    rebuild = False
#    rm_Skagerrak_data = False
    # Use top models from full dataset  ( now: nOutliers + nSkagerak
    RFR_dict = build_or_get_models_iodide(
        rebuild=rebuild,
        rm_Skagerrak_data=rm_Skagerrak_data)
#    RFR_dict = build_or_get_models_iodide( rm_Skagerrak_data=False )
#    topmodels = get_top_models(RFR_dict=RFR_dict, vars2exclude=['DOC', 'Prod'], n=10)
    print(RFR_dict.keys())
#    print(topmodels)
    # Check statistics on prediction
#    print(stats)
    # Get the dictionary of models and their features
    model_feature_dict = utils.get_model_features_used_dict(rtn_dict=True)
    print(model_feature_dict)
#    print(model_feature_dict['NO3+DOC+Phos'])

    # ---- ----- ----- ----- ----- ----- ----- ----- -----
    # ----- ----- Evaluating input datasets
    # General plots of all species
#    misc.get_diagnostic_plots_analysis4observations()

    # ---- ----- ----- ----- ----- ----- ----- ----- -----
    # ----- ----- Processing of observations (& extraction of ancillaries)
    # --- Get iodide observations?
#    df = obs.get_iodide_obs()
    # --- Re-process file?
#    df = obs.get_iodide_obs(process_new_iodide_obs_file=True)
    # --- Re-process ancillaries file?
#    obs.process_iodide_obs_ancillaries_2_csv()
#    get_core_Chance2014_obs()
    # --- Process MLD csv files? (Just for ease of use/data munging)
#    ancillaries.process_MLD_csv2NetCDF()

    # Check extracted data against observations.
#    misc.compare_obs_ancillaries_with_extracted_values()

    # ---- ----- ----- ----- ----- ----- ----- ----- -----
    # ----- ----- Build ancillary variable dataset file
    # ---
    # Build either a full or low res ancillary NetCDF file
#    res = '0.125x0.125'
#    res = '4x5' # low resolution to test extraction etc?
    # Get indicies to extract for variables in imported NetCDF
#    ancillaries2grid.mk_array_of_indices4locations4res( res=res )
    # Extract the variables to NetCDF
#    ancillaries2grid.extract_feature_variables2NetCDF( res=res )
    # Interpolate the fields at full resolution
#    ancillaries.interpolate_NaNs_in_feature_variables( res=res )

    # ---- ----- ----- ----- ----- ----- ----- ----- -----
    # ----- ----- Building new iodide field (inc. Machine learning)
    # ---
    # (Re-)Build all models
    # (random stats means this gives the same answer everytime)
#    build_or_get_models_iodide(rebuild=True,
#                              rm_Skagerrak_data=rm_Skagerrak_data )

    # --- Update the predictor array values
#     res='4x5'
#     plt_analysis.set_SAL_and_NIT_above_65N_to_avg(res=res)

    # --- Predict values globally (only use 0.125)
    # extra string for NetCDF save name
    xsave_str = 'TEST_'
    # make NetCDF predictions from the main array
    save2NetCDF = True
    # resolution to use? (full='0.125x0.125', test at lower e.g. '4x5')
#    res = '0.125x0.125'
#     res = '4x5'
#     mk_iodide_predictions_from_ancillaries(None, res=res, RFR_dict=RFR_dict,
#                                            use_updated_predictor_NetCDF=False,
#                                            save2NetCDF=save2NetCDF,
#                                            rm_Skagerrak_data=rm_Skagerrak_data,
#                                            topmodels=topmodels,
#                                            xsave_str=xsave_str,
#                                            add_ensemble2ds=True)

    # ---- ----- ----- ----- ----- ----- ----- ----- -----
    # ----- ----- Sensitivity testing of the new iodide field
    # ---
    # make NetCDF predictions from the updated arrays
# vars2use = [
# 'WOA_Nitrate', 'WOA_Salinity', 'WOA_Phosphate', 'WOA_TEMP_K', 'Depth_GEBCO',
# ]
#     folder = None
#     use_updated_predictor_NetCDF =  False
#     # #    for res in ['4x5', '2x2.5']:
#     for res in ['4x5']:
#     # #    for res in ['0.125x0.125',]:
#         # setup a pool to bulk process
#         p  = Pool( len(vars2use) )
#     #        for var2use in vars2use:
#             # now predict the arrays from this.
#         p.map( partial(mk_iodide_predictions_from_ancillaries, res=res,
#             RFR_dict=RFR_dict, folder=folder,
#             use_updated_predictor_NetCDF=use_updated_predictor_NetCDF,
#         save2NetCDF=save2NetCDF ),
#             vars2use
#         )
#         # close the pool
#         p.close()

    # --- test the updates predictions
#     res = '0.125x0.125'
#     folder = None
#     for var2use in vars2use:
# #        extr_str = '_INTERP_NEAREST_DERIVED'
#         extr_str = '_UPDATED_{}'.format( var2use )
#         plot_predicted_iodide_vs_lat_figure_ENSEMBLE( RFR_dict=RFR_dict,
#             extr_str=extr_str, res=res, folder=folder )
    # then test if depth is set to


    # ---- ----- ----- ----- ----- ----- ----- ----- -----
    # ----- ----- Plots / Analsis for sea-surface iodide ML paper
    # Get shared data
#    RFR_dict = build_or_get_models()

    # --- 2D analysis
    # Plot up spatial comparison of obs. and params
#    plt_analysis.plot_up_obs_spatially_against_predictions( RFR_dict=RFR_dict )
    # Test which plotting options to use (to display markers)
#    plt_analysis.plot_up_obs_spatially_against_predictions_options(
#          RFR_dict=RFR_dict )

    # plot up the input variables spatially
#    res = '0.125x0.125'
#    res = '4x5'
#   plt_analysis.plot_up_input_ancillaries_spatially( res=res, RFR_dict=RFR_dict,
#                                                    save2png=True)

    # Plot up the 2D differences in predictions
#    res= '0.125x0.125'
#    res= '4x5'
#    plt_analysis.plot_up_spatial_changes_in_predicted_values( res=res, window=True,
#                                                              f_size=30)

    # Get stats from the 4x5 and 2x2.5 predictions
#    analysis.get_stats_on_spatial_predictions_4x5_2x25()
#    analysis.get_stats_on_spatial_predictions_4x5_2x25_by_lat()

    # Get stats from the 0.125x0.125 prediction
#    analysis.get_stats_on_spatial_predictions_0125x0125()

    # Calculate the average predicted surface conc (only 4x5. 2x2.5 too? )
#    plt_analysis.calculate_average_predicted_surface_conc() # AGU calcs at 4x5

    # Plot up latitude vs. predicted iodide
#    plt_analysis.plot_predicted_iodide_vs_lat_figure()

    # Seasonal prediction of iodide by month
#    plt_analysis.plot_monthly_predicted_iodide( res='4x5' )
#    plt_analysis.plot_monthly_predicted_iodide( res='0.125x0.125' )
#    plt_analysis.plot_monthly_predicted_iodide_diff( res='0.125x0.125' )

    # explore the extracted data in the arctic and AnatArctic
#    plt_analysis.explore_extracted_data_in_Oi_prj_explore_Arctic_Antarctic_obs()

    # Check the sensitivity to input variables >= 65 N
#    plt_analysis.mk_PDFs_to_show_the_sensitivty_input_vars_65N_and_up(
#        save_str='TEST_V' )

    # --- Point-for-point analysis
    # build ODR plots for obs. vs. model
#    plt_analysis.analyse_X_Y_correlations_ODR( RFR_dict=RFR_dict, context='poster' )
#    plt_analysis.analyse_X_Y_correlations_ODR( RFR_dict=RFR_dict, context='paper' )

    # Analyse the X_Y correlations
#    plt_analysis.analyse_X_Y_correlations( RFR_dict=RFR_dict )

    # Get the importance of individual features for prediction
#    RFRanalysis.get_feature_importance( RFR_dict=RFR_dict )

    # Get general stats on the current models
#   RFRanalysis.get_stats_on_models( RFR_dict=RFR_dict )

    # Get tabulated performance
#    mk_table_of_point_for_point_performance(RFR_dict=RFR_dict)
#    mk_table_of_point_for_point_performance_ALL(RFR_dict=RFR_dict)
#    mk_table_of_point_for_point_performance_TESTSET(RFR_dict=RFR_dict)

    # Get CDF and PDF plots for test, training, entire, and residual
#    plt_analysis.plot_up_CDF_and_PDF_of_obs_and_predictions( df=RFR_dict['df'] )

    # Plot up various spatial plots for iodide concs + std.
#    plt_analysis.plot_up_ensemble_avg_and_std_spatially(
#        rm_Skagerrak_data=rm_Skagerrak_data
#    )

    # --- Spatial analysis for specific locations
    # explore the observational data in the Arctic
#    misc.explore_observational_data_in_Arctic_parameter_space( RFR_dict=RFR_dict )

    # plot up where decision points are
#    plt_analysis.plot_spatial_area4core_decisions( res='4x5' )
#    plt_analysis.plot_spatial_area4core_decisions( res='0.125x0.125' )

    # Explore the sensitivity to data denial
#    plt_analysis.explore_sensitivity_of_65N2data_denial( res='4x5' )
#    plt_analysis.explore_sensitivity_of_65N2data_denial( res='2x2.5' )
#    plt_analysis.explore_sensitivity_of_65N2data_denial( res='0.125x0.125' )

    # --- Analysis of models build
    # testset analysis
#    plt_analysis.test_model_sensitiivty2training_test_split() # driver not in use yet!
#    RFRanalysis.run_tests_on_testing_dataset_split_quantiles()
#    RFRanalysis.run_tests_on_testing_dataset_split()

    # selection of variables to build models

    # hyperparameter tuning of selected models

    # Analysis of the spatial variance of individual ensemble members
#    rm_Skagerrak_data = True
#    rm_Skagerrak_data = False
#     plt_analysis.analyse_dataset_error_in_ensemble_members( res='0.125x0.125', \
#         rebuild_models=False, remake_NetCDFs=False,
#         rm_Skagerrak_data=rm_Skagerrak_data,
#         topmodels=topmodels )
#    plt_analysis.analyse_dataset_error_in_ensemble_members( res='0.125x0.125', \
#       rebuild_models=True, remake_NetCDFs=True, \
#       rm_Skagerrak_data=rm_Skagerrak_data,
#       topmodels=topmodels
#        )
    # Common resolutions
#   archiving.regrid_output_to_common_res_as_NetCDFs(topmodels=topmodels,
#                                                    rm_Skagerrak_data=rm_Skagerrak_data)

    # --- Do tree by tree analysis
    # Extract trees to .dot files (used make make the single tree figures)
#    RFRanalysis.extract_trees4models( RFR_dict=RFR_dict, N_trees2output=50 )

    # Plot up interpretation of trees
    # Now in TreeSurgeon - see separate repository on github
    # https://github.com/wolfiex/TreeSurgeon

    # analysis of node spliting
#    RFRanalysis.analyse_nodes_in_models( RFR_dict=RFR_dict )
    # analysis of outputted trees
#    RFRanalysis.analyse_nodes_in_models()

    # --- Do futher analysis on the impact of the depth variable
    plt_analysis.do_analysis_processing_linked_to_depth_variable()

    # plot this up and other figures for the ML paper
    plt_analysis.plot_spatial_figures_for_ML_paper_with_cartopy()

    # - pass if no functions are uncommented
    pass


def plt_comparisons_of_Wadley2020_iodide_fields():
    """
    Make a comparison between the

    Notes
    -------
     - Wadley et al present day iodide fields are from this paper
    https://www.essoar.org/doi/10.1002/essoar.10502078.2
     - record on BODC
    """
    # - Get Wadley et al 2020's process-based iodide field
    folder = '/users/ts551/scratch/data/Oi/UEA/'
    filename = 'iodide_from_model_PRESENT_DAY_interp_0.125x0.125.nc'
    dsM = xr.open_dataset(folder+filename)
    #

    sns.reset_orig()

    # Monthly
    vmin, vmax = 0, 240
    version = 'Wadley_2020_ltd_cbar_regrid_0.125x0.125'
    var2plot = 'Present_Day_Iodide'
    target = 'Iodide'
    units = 'nM'
    var2plot_longname = 'Wadley2020'
    cmap = AC.get_colormap(np.arange(30))
    cbar_kwargs = {'extend': 'max'}
    s2splotting.plot_up_seasonal_averages_of_prediction(target=target,
#                                                        ds=ds,
                                                        ds=dsM,
                                                        version=version,
                                                        var2plot=var2plot,
                                         var2plot_longname=var2plot_longname,
                                                        vmin=vmin, vmax=vmax,
                                                        cmap=cmap,
                                                       cbar_kwargs=cbar_kwargs,
                                                        units=units)

    # Annual average too
    title = 'Annual average Iodide field from Wadley et al 2020'
    s2splotting.plot_up_annual_averages_of_prediction(target=target, ds=dsM,
                                                      title=title,
                                                      var2plot=var2plot,
                                                      vmin=vmin, vmax=vmax,
                                                      units=units,
                                                      cmap=cmap,
                                                      cbar_kwargs=cbar_kwargs,
                                                      )

    # Plot monthly values too...
    # TODO - this requires python 2 compatibility
#    plot_monthly_predicted_iodide
#    plt_analysis.plot_monthly_predicted_iodide_diff(ds=dsM, var2plot=var2plot)
    # TEMP - plot seasonal plot for ML output.
#     data_root = utils.get_file_locations('data_root')
#     folder = '{}/{}/outputs/'.format(data_root, target)
#     MLfilename = 's2s_predicted_Iodide_0.125x0.125.nc'
#     dsML = xr.open_dataset(folder+MLfilename)
#     var2plot = 'Ensemble Monthly mean'
#     var2plot_longname = 'Sherwen2019'
#     version = 'v8_5_1'
#     s2splotting.plot_up_seasonal_averages_of_prediction(target=target,
# #                                                        ds=ds,
#                                                         ds=dsML,
#                                                         version=version,
#                                                         var2plot=var2plot,
#                                          var2plot_longname=var2plot_longname,
#                                                         vmin=vmin, vmax=vmax,
#                                                         cmap=cmap,
#                                                        cbar_kwargs=cbar_kwargs,
#                                                         units=units)
#
#     # Annual average too
#     title = 'Annual average Iodide field from Sherwen et al 2019'
#     s2splotting.plot_up_annual_averages_of_prediction(target=target, ds=dsML,
#                                                       title=title,
#                                                       var2plot=var2plot,
#                                                       vmin=vmin, vmax=vmax,
#                                                       units=units,
#                                                       cmap=cmap,
#                                                       cbar_kwargs=cbar_kwargs,
#                                                       )


    # -
    # retrive models with the observations
    RFR_dict = build_or_get_models_iodide(rebuild=False)
    df = RFR_dict['df']
    # Add the ML values to this
    df = add_ensemble_prediction2df(df=df, target=target, var2extract='Ensemble Monthly mean')
    # Add the Lennetz values to this.
    LatVar = 'Latitude'
    LonVar = 'Longitude'
    MonthVar = 'Month'

    # Now add Martyn's values
    var2extract = 'Present_Day_Iodide'
    var2use = 'Wadley2020'
    vals = utils.extract4nearest_points_in_ds(ds=dsM, lons=df[LonVar].values,
                                              lats=df[LatVar].values,
                                              months=df[MonthVar].values,
                                              var2extract=var2extract)
    df[var2use] = vals

    # Now plot an ODR
    ylim = (0, 400)
    xlim = (0, 400)
    params =  [
    'RFR(Ensemble)', 'MacDonald2014_iodide', 'Wadley2020'
    ]
    # only points where there is data for both - THIS IS NOT NEEDED
#    df = df.loc[df[params].dropna().index, :]
    # ODR
    testset = 'Test set (strat. 20%)'
    df2plot = df[params+[target, testset]].copy()
    s2splotting.plot_ODR_window_plot(df=df2plot, params=params, units='nM',
                                     target=target,
                                     ylim=ylim, xlim=xlim)

    # Plot up a PDF of concs and bias
    ylim = (0, 400)
#    ylim = (0, 400)
    s2splotting.plot_up_PDF_of_obs_and_predictions_WINDOW(df=df2plot,
                                                         params=params,
                                                          units='nM',
                                                          target=target,
                                                          xlim=xlim)

    # Summurise states
    stats = RFRanalysis.get_core_stats_on_current_models(RFR_dict=RFR_dict,
                                                         target=target,
                                                         param_names=params,
                                                         verbose=True,
                                                         debug=True)




def add_ensemble_prediction2df(df=None, LatVar='Latitude', LonVar='Longitude',
                               target='Iodide', version='_v0_0_0',
                               var='RFR(Ensemble)', MonthVar='Month',
                               var2extract='Ensemble_Monthly_mean'):
    """
    Wrapper function to add the ensemble prediction to a dataframe from NetCDF
    Parameters
    -------
    target (str): Name of the target variable (e.g. iodide)
    LatVar (str): variable name in DataFrame for latitude
    LonVar (str): variable name in DataFrame for longitude
    MonthVar (str): variable name in DataFrame for month
    version (str): Version number or string (present in NetCDF names etc)
    var2extract (str): variable to extract from the
    Returns
    -------
    (pd.DataFrame)
    """
    # Get the 3D prediction as a dataset
    ds = utils.get_predicted_values_as_ds(target=target, version=version)
    # extract the nearest values
    vals = utils.extract4nearest_points_in_ds(ds=ds, lons=df[LonVar].values,
                                              lats=df[LatVar].values,
                                              months=df[MonthVar].values,
                                              var2extract=var2extract)
    df[var] = vals
    return df


def run_tests_on_testing_dataset_split(model_name=None, n_estimators=500,
                                       features_used=None, target='Iodide',
                                       df=None):
    """
    Run tests on the sensitivity of model to test/training choices

    Parameters
    -------
    target (str): Name of the target variable (e.g. iodide)
    df (pd.DataFrame): dataframe containing target and feature variables
    n_estimators (int), number of estimators (decision trees) to use
    features_used (list): list of the features within the model_name model
    model_name (str): name of model to build

    Returns
    -------
    (None)
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.externals import joblib
    # target='Iodide'
    # ----- Local variables
    # Get unprocessed input data at observation points
    if isinstance(df, type(None)):
        df = get_processed_df_obs_mod()  # NOTE this df contains values >400nM
    # ---- get the data
    # Which "features" (variables) to use
    if isinstance(features_used, type(None)):
        #        model_name = 'ALL'
        model_name = 'RFR(TEMP+DEPTH+SAL)'
        features_used = utils.get_model_features_used_dict(model_name)

    # --- local variables
    # dictionary of test set variables
    random_split_var = 'rn. 20%'
    strat_split_var = 'strat. 20%'
    # set a basis for filenames to saved as
    save_filename_str = 'Oi_prj_test_training_selection'
#    random_states = [38, 39, 40, 41, 42, 43, 44 ]
#    random_states = [36, 37, 38, 39, 40, 41, 42, ]
#    random_states = np.arange(33, 43, 1)
    random_states = np.arange(25, 45, 1)
    # Formatted variable name for iodide
    Iaq = '[I$^{-}_{aq}$]'
    # --- set testset to evaulte
    TSETS = {}
    TSETS_N = {}
    TSETS_nsplits = {}
    # - no vals above 400
    Tname = 'All'
    tmp_ts = df[features_used+[target]].copy()
    TSETS_N[Tname] = tmp_ts.shape[0]
    TSETS[Tname] = tmp_ts
    TSETS_nsplits[Tname] = 4
    # - no vals above 400
#     Tname = '{}<400'.format( Iaq )
#     tmp_ts = df.loc[ df['Iodide']<400 ][ features_used+[target] ].copy()
#     TSETS_N[Tname] = tmp_ts.shape[0]
#     TSETS[Tname] = tmp_ts
#     TSETS_nsplits[Tname] = 4
#     # - no vals above 450
#     Tname = '{}<450'.format( Iaq )
#     tmp_ts = df.loc[ df['Iodide']<450 ][ features_used+[target] ].copy()
#     TSETS_N[Tname] = tmp_ts.shape[0]
#     TSETS[Tname] = tmp_ts
#     TSETS_nsplits[Tname] = 4
    # - no vals above 350
#     Tname = '{}<350'.format( Iaq )
#     tmp_ts = df.loc[ df['Iodide']<350 ][ features_used+[target] ].copy()
#     TSETS_N[Tname] = tmp_ts.shape[0]
#     TSETS[Tname] = tmp_ts
#     TSETS_nsplits[Tname] = 4
    # - remove estuarine (no values < 30 salinity?) values
    Tname = 'SAL>=30 \n & no outliers'
    bool1 = df['WOA_Salinity'] >= 30
    # also remove outliers
    bool2 = df['Iodide'] < utils.get_outlier_value(df=df, var2use='Iodide')
    tmp_ts = df.loc[bool1 & bool2][features_used+[target]].copy()
    TSETS_N[Tname] = tmp_ts.shape[0]
    TSETS[Tname] = tmp_ts
    TSETS_nsplits[Tname] = 4
    # - remove estuarine (no values < 30 salinity?) values
#     Tname = 'SAL>=30 \n & {}'.format( Iaq ) + '<98$^{th}$'
#     bool1 =  df['WOA_Salinity']>=30
#     bool2 = df['Iodide'] < np.percentile( df['Iodide'].values, 98 )
#     # also remove values where iodide <400
#     tmp_ts = df.loc[ bool1 & bool2  ][ features_used+[target] ].copy()
#     TSETS_N[Tname] = tmp_ts.shape[0]
#     TSETS[Tname] = tmp_ts
#     TSETS_nsplits[Tname] = 4
    # - remove estuarine (no values < 30 salinity?) values
#     Tname = 'SAL>=30 \n & {}'.format( Iaq ) + '<98$^{th}$'
#     bool1 =  df['WOA_Salinity']>=30
#     bool2 = df['Iodide'] < np.percentile( df['Iodide'].values, 98 )
#     # also remove values where iodide <400
#     tmp_ts = df.loc[ bool1 & bool2  ][ features_used+[target] ].copy()
#     TSETS_N[Tname] = tmp_ts.shape[0]
#     TSETS[Tname] = tmp_ts
#     TSETS_nsplits[Tname] = 4
    # - Just coastal
    Tname = 'Just coastal\n& no outliers'
    bool1 = df['Coastal'] == 1
    # also remove outliers
    bool2 = df['Iodide'] < utils.get_outlier_value(df=df, var2use='Iodide')
    tmp_ts = df.loc[bool1 & bool2][features_used+[target]].copy()
    TSETS_N[Tname] = tmp_ts.shape[0]
    TSETS[Tname] = tmp_ts
    TSETS_nsplits[Tname] = 4
    # - Just coastal
#     Tname = 'Coastal \n & {}'.format( Iaq )+ '<98$^{th}$'
#     bool1 =  df['Coastal'] ==1
#     # also remove values where iodide <98
#     bool2 = df['Iodide'] < np.percentile( df['Iodide'].values, 98 )
#     tmp_ts = df.loc[ bool1 & bool2  ][ features_used+[target] ].copy()
#     TSETS_N[Tname] = tmp_ts.shape[0]
#     TSETS[Tname] = tmp_ts
#     TSETS_nsplits[Tname] = 4
    # - non-coastal
    Tname = 'Just non-coastal\n& no outliers'
    bool1 = df['Coastal'] == 0
    # also remove outliers
    bool2 = df['Iodide'] < utils.get_outlier_value(df=df, var2use='Iodide')
    tmp_ts = df.loc[bool1 & bool2][features_used+[target]].copy()
    TSETS_N[Tname] = tmp_ts.shape[0]
    TSETS[Tname] = tmp_ts
    TSETS_nsplits[Tname] = 4
    # - non-coastal
#     Tname = 'Non Coastal \n & {}'.format( Iaq )+ '<98$^{th}$'
#     bool1 =  df['Coastal'] == 0
#     # also remove values where iodide <98
#     bool2 = df['Iodide'] < np.percentile( df['Iodide'].values, 98 )
#     tmp_ts = df.loc[ bool1 & bool2  ][ features_used+[target] ].copy()
#     TSETS_N[Tname] = tmp_ts.shape[0]
#     TSETS[Tname] = tmp_ts
#     TSETS_nsplits[Tname] = 4
    # - only that < 98th
#     Tname = '{} '.format( Iaq ) +'<98$^{th}$'
#     bool_ = df['Iodide'] < np.percentile( df['Iodide'].values, 98 )
#     tmp_ts = df.loc[ bool_ ][ features_used+[target] ].copy()
#     # also remove values where iodide <400
#     tmp_ts = tmp_ts.loc[ df['Iodide']<400  ]
#     TSETS_N[Tname] = tmp_ts.shape[0]
#     TSETS[Tname] = tmp_ts
#     TSETS_nsplits[Tname] = 4
    # - only that < 99th
#     Tname = '{} '.format( Iaq ) + '<99$^{th}$'
#     bool_ = df['Iodide'] >= np.percentile( df['Iodide'].values, 99 )
#     tmp_ts = df.loc[ bool_ ][ features_used+[target] ].copy()
#     # also remove values where iodide <400
#     tmp_ts = tmp_ts.loc[ df['Iodide']<400  ]
#     TSETS_N[Tname] = tmp_ts.shape[0]
#     TSETS[Tname] = tmp_ts
#     TSETS_nsplits[Tname] = 4
    # - No Skagerrak
# 	Tname = 'No Skagerrak'
# 	bool_ = df['Data_Key'].values != 'Truesdale_2003_I'
# 	tmp_ts = df.loc[ bool_ ][ features_used+[target] ].copy()
# 	# also remove values where iodide <400
# 	TSETS_N[Tname] = tmp_ts.shape[0]
# 	TSETS[Tname] = tmp_ts
# 	TSETS_nsplits[Tname] = 4
    # - No Skagerrak
#     Tname = 'No Skagerrak \n or {}'.format( Iaq )+ '>98$^{th}$'
#     bool1 = df['Data_Key'].values == 'Truesdale_2003_I'
#     bool2 = df['Iodide'] > np.percentile( df['Iodide'].values, 98 )
#     index2drop = df.loc[ bool1 | bool2, : ].index
#     tmp_ts = df.drop( index2drop )[ features_used+[target] ].copy()
    # also remove values where iodide <400
    TSETS_N[Tname] = tmp_ts.shape[0]
    TSETS[Tname] = tmp_ts
    TSETS_nsplits[Tname] = 4
    # - No outliers
    Tname = 'No outliers'
    bool_ = df['Iodide'] < utils.get_outlier_value(df=df, var2use='Iodide')
    tmp_ts = df.loc[bool_][features_used+[target]].copy()
    # also remove values where iodide <400
    TSETS_N[Tname] = tmp_ts.shape[0]
    TSETS[Tname] = tmp_ts
    TSETS_nsplits[Tname] = 4
    # - No Skagerrak
    Tname = 'No Skagerrak \n or outliers'
    bool1 = df['Data_Key'].values == 'Truesdale_2003_I'
    bool2 = df['Iodide'] > utils.get_outlier_value(df=df, var2use='Iodide')
    index2drop = df.loc[bool1 | bool2, :].index
    tmp_ts = df.drop(index2drop)[features_used+[target]].copy()
    # also remove values where iodide <400
    TSETS_N[Tname] = tmp_ts.shape[0]
    TSETS[Tname] = tmp_ts
    TSETS_nsplits[Tname] = 4

    # ---  build models using testsets
    # Get
    RMSE_df = pd.DataFrame()
    # Now loop TSETS
    for Tname in TSETS.keys():
        # Initialise lists to store data in
        RMSE_l = []
        # get random state to use
        for random_state in random_states:
            print('Using: random_state={}'.format(random_state))
            # Get testset
            df_tmp = TSETS[Tname].copy()
            # force index to be a range
            df_tmp.index = range(df_tmp.shape[0])
            print(Tname, df_tmp.shape)
            # Stratified split by default, unless random var in name
            rand_strat = True
            rand_20_80 = False
            # get the training and test set
            returned_vars = mk_iodide_test_train_sets(df=df_tmp,
                                                      rand_20_80=rand_20_80,
                                                     random_state=random_state,
                                                  nsplits=TSETS_nsplits[Tname],
                                                      rand_strat=rand_strat,
                                                   features_used=features_used,
                                                      )
            train_set, test_set, test_set_targets = returned_vars
            # set the training and test sets
            train_features = df_tmp[features_used].loc[train_set.index]
            train_labels = df_tmp[[target]].loc[train_set.index]
            test_features = df_tmp[features_used].loc[test_set.index]
            test_labels = df_tmp[[target]].loc[test_set.index]
            # build the model - NOTE THIS MUST BE RE-DONE!
            # ( otherwise the model is being re-trained )
            model = RandomForestRegressor(random_state=random_state,
                                          n_estimators=n_estimators,
                                          criterion='mse')
            # Fit the model
            model.fit(train_features, train_labels)
            # Predict the values
            df_tmp[Tname] = model.predict(df_tmp[features_used].values)
            # Get the stats against the test group
            df_tmp = df_tmp[[Tname, target]].loc[test_set.index]
            # Get MSE and RMSE
            MSE = (df_tmp[target]-df_tmp[Tname])**2
            MSE = np.mean(MSE)
            std = np.std(df_tmp[Tname].values)
            # Return stats on bias and variance
            # (just use RMSE and std dev. for now)
            RMSE_l += [np.sqrt(MSE)]
            del df_tmp, train_features, train_labels, test_features, test_labels
            del model
        # Save the results to a dictionary
        RMSE_df[Tname] = RMSE_l

    # --- Get stats on the ensemble values
    # Get general stats on ensemble
    RMSE_stats = pd.DataFrame(RMSE_df.describe().copy()).T
    # ad number of samples
    RMSE_stats['N'] = pd.Series(TSETS_N)
    # sort to order by mean
    RMSE_stats.sort_values(by='mean', inplace=True)
    # sort the main Dataframe by the magnitude of the mean
    RMSE_df = RMSE_df[list(RMSE_stats.index)]
    # work out the deviation from mean of the ensemble
    pcent_var = '% from mean'
    means = RMSE_stats['mean']
    pcents = ((means - means.mean()) / means.mean() * 100).values
    RMSE_stats[pcent_var] = pcents
    # update order of columns
    first_cols = ['mean', 'N']
    order4cols = [i for i in RMSE_stats.columns if i not in first_cols]
    RMSE_stats = RMSE_stats[first_cols + order4cols]
    # print to screen
    print(RMSE_stats)
    pstr = '{:<13} - mean: {:.2f} (% from ensemble mean: {:.2f})'
    for col in RMSE_stats.T.columns:
        vals2print = RMSE_stats.T[col][['mean', pcent_var]].values
        print(pstr.format(col.replace("\n", ""), *vals2print))
        # remove the '\n' symbols etc from the column names
    RMSE_stats2save = RMSE_stats.copy()
    RMSE_stats2save.index = [i.replace('\n', '') for i in RMSE_stats.index]
    # save to csv
    RMSE_stats2save.to_csv(save_filename_str+'.csv')

    # ---- Do some further analysis and save this to a text file
    a = open(save_filename_str+'_analysis.txt', 'w')
    # Set a header
    print('This file contains analysis of the training set selection', file=a)
    print('\n', file=a)
    # which files are being analysed?
    print('---- Detail range of RMSE values by build', file=a)
    for test_ in RMSE_stats.T.columns:
        df_tmp = RMSE_stats.T[test_].T
        min_ = df_tmp['min']
        max_ = df_tmp['max']
        range_ = max_ - min_
        test_ = test_.replace("\n", "")
        # print range for test_
        ptr_str = "range for '{:<20}' : {:.3g} ({:.5g}-{:.5g})"
        print(ptr_str.format(test_, range_, min_, max_), file=a)
        # print this as a % of the mean
        mean_ = df_tmp['mean']
        prange_ = range_ / mean_ * 100
        pmin_ = min_ / mean_ * 100
        pmax_ = max_ / mean_ * 100
        ptr_str = "range as % of mean ({:.3g}) for'{:<20}':"
        ptr_str += ": {:.3g} % ({:.5g} % -{:.5g} %)"
        print(ptr_str.format(mean_, test_, prange_, pmin_, pmax_), file=a)
    a.close()

    # --- Setup the datafframes for plotting ( long form needed )
    RMSE_df = RMSE_df.melt()
    # rename columns
    ylabel_str = 'RMSE (nM)'
    RMSE_df.rename(columns={'value': ylabel_str}, inplace=True)

    # --- Plot up the test runs
    CB_color_cycle = AC.get_CB_color_cycle()
    import seaborn as sns
    sns.set(color_codes=True)
    sns.set_context("paper")
    dpi = 320

    # --- plot up the results as violin plots
    fig, ax = plt.subplots(figsize=(10, 3.5), dpi=dpi)
    # plot up these values
    ax = sns.violinplot(x='variable', y=ylabel_str, data=RMSE_df,
                        palette=CB_color_cycle, ax=ax)
    # remove the variable label from the x axis
    ax.xaxis.label.set_visible(False)
    # force yaxis extent
    ymax = AC.myround(RMSE_df[ylabel_str].max(), base=25, round_up=True)
    ax.set_ylim(-15, ymax+25+10)
    # add N value to plot
    f_size = 10
    xlabels = [i.get_text() for i in ax.get_xticklabels()]
    # set locations for N lael
    if len(xlabels) == 7:
        x_l = np.linspace(0.041, 0.9025, len(xlabels))
    if len(xlabels) == 6:
        x_l = np.linspace(0.055, 0.89, len(xlabels))
    else:
        x_l = np.linspace(0.035, 0.9025, len(xlabels))
    # loop and add N value
    for xlabel_n, xlabel in enumerate(xlabels):
        N = TSETS_N[xlabel]
        # Set location for label
        alt_text_x = x_l[xlabel_n]
        alt_text_y = 0.035
        # Setup label and plot
        alt_text = 'N={}'.format(N)
        ax.annotate(alt_text, xy=(alt_text_x, alt_text_y),
                    textcoords='axes fraction', )
    # Adjust positions of subplot
    bottom = 0.095
    top = 0.975
    left = 0.075
    right = 0.975
    fig.subplots_adjust(bottom=bottom, top=top, left=left, right=right,)
    # save the plot
    plt.savefig(save_filename_str+'_sensitivity_violin.png', dpi=dpi)
    plt.close()


# ---------------------------------------------------------------------------
# ---------- Functions to generate/predict modelled field -------------------
# ---------------------------------------------------------------------------
def mk_iodide_predictions_from_ancillaries(var2use, res='4x5', target='Iodide',
                                           models_dict=None,
                                           features_used_dict=None,
                                           RFR_dict=None, dsA=None,
                                           stats=None, folder=None,
                                           use_updated_predictor_NetCDF=False,
                                           save2NetCDF=False, plot2check=False,
                                           models2compare=[], topmodels=None,
                                           rm_Skagerrak_data=False,
                                           xsave_str='',
                                           add_ensemble2ds=False,
                                           verbose=True, debug=False):
    """
    Make a NetCDF file of predicted vairables for a given resolution

    Parameters
    -------
    var2use (str): var to use as main model prediction
    rm_Skagerrak_data (bool): remove the data from the Skagerrak region
    RFR_dict (dict): dictionary of core variables and data
    target (str): Name of the target variable (e.g. iodide)
    res (str): horizontal resolution of dataset (e.g. 4x5)
    models_dict (dict): dictionary of models (values) and their names (keys)
    features_used_dict (dict): dictionary of models (keys) and their features (values)
    use_updated_predictor_NetCDF (bool):

    Returns
    -------

    Notes
    -----
    """
    # -local variables
    # extract the models...
    if isinstance(RFR_dict, type(None)):
        RFR_dict = build_or_get_models(
            rm_Skagerrak_data=rm_Skagerrak_data
        )
    # set models to always predict values for
    if (len(models2compare) == 0):
        models2compare = [
            # Ones using all variable options
            'RFR(TEMP+DEPTH+SAL+NO3+DOC)', 'RFR(TEMP+DOC+Phos)',
            'RFR(TEMP+DEPTH+SAL+Prod)',
            # ones just using variable options
            'RFR(TEMP+SAL+NO3)', 'RFR(TEMP+DEPTH+SAL+Phos)',
            'RFR(TEMP+SWrad+NO3+MLD+SAL)', 'RFR(TEMP+DEPTH+SAL)',
            # Temperature for zeroth order
            'RFR(TEMP)',
            # ones in v8.1 topmodels
            'RFR(TEMP+DEPTH+SAL+SWrad)', 'RFR(TEMP+DEPTH+NO3+SWrad)',
            'RFR(TEMP+NO3+MLD+SAL)', 'RFR(TEMP+DEPTH+SAL+NO3)',
            'RFR(TEMP+DEPTH+SAL+ChlrA)', 'RFR(TEMP+DEPTH+NO3)', 'RFR(TEMP+NO3)',
            # ones in topmodels_nSkagerrak
            'RFR(TEMP+DEPTH+SAL)', 'RFR(SWrad+SAL+DEPTH)', 'RFR(TEMP+SAL)'
        ]
    # Make sure the top 10 models are included
    # ( with derivative variables )
    if isinstance(topmodels, type(None)):
        # Get stats on models in RFR_dict
        if isinstance(stats, type(None)):
            stats = get_stats_on_models(RFR_dict=RFR_dict,
                                        analysis4coastal=True,
                                        verbose=False)
        topmodels = get_top_models(RFR_dict=RFR_dict, stats=stats,
                                   vars2exclude=['DOC', 'Prod'])
    models2compare += topmodels
    # Remove any double ups
    models2compare = list(set(models2compare))
    # Get the variables required here
    if isinstance(models_dict, type(None)):
        models_dict = RFR_dict['models_dict']
    if isinstance(features_used_dict, type(None)):
        features_used_dict = RFR_dict['features_used_dict']
    # Get location to save file and set filename
    if isinstance(folder, type(None)):
        folder = utils.get_file_locations('data_root')+'/data/'

    extr_str = '_INTERP_NEAREST_DERIVED'
    # Add lines to save strings
    if use_updated_predictor_NetCDF:
        xsave_str += '_UPDATED_{}'.format(var2use)
        extr_str += xsave_str
    if rm_Skagerrak_data:
        xsave_str += '_No_Skagerrak'
    if isinstance(dsA, type(None)):
        filename = 'Oi_prj_feature_variables_{}.nc'.format(res)
        dsA = xr.open_dataset(folder + filename)
    # --- Make a da for each model
    ds_l = []
    for modelname in models2compare:
        print(modelname)
        # get model
        model = models_dict[modelname]
        # get testinng features
        features_used = utils.get_model_features_used_dict(modelname)
        # Make a DataSet of predicted values
        ds_tmp = utils.mk_da_of_predicted_values(model=model,
                                                 modelname=modelname,
                                                 res=res,
                                                 features_used=features_used,
                                                 dsA=dsA)
        #  Add attributes to the prediction
        ds_tmp = add_attrs2iodide_ds(ds_tmp, add_global_attrs=False,
                                     varname=modelname)
        # Savea
        ds_l += [ds_tmp]
    # Combine datasets
    ds = xr.merge(ds_l)
    # - Also get values for parameterisations
    # Chance et al (2013)
    param = u'Chance2014_STTxx2_I'
    arr = utils.calc_I_Chance2014_STTxx2_I(dsA['WOA_TEMP'].values)
    ds[param] = ds[modelname]  # use existing array as dummy to fill
    ds[param].values = arr
    # MacDonald et al (2013)
    param = 'MacDonald2014_iodide'
    arr = utils.calc_I_MacDonald2014(dsA['WOA_TEMP'].values)
    ds[param] = ds[modelname]  # use existing array as dummy to fill
    ds[param].values = arr
    # Add ensemble to ds too
    if add_ensemble2ds:
        print('WARNING: Using topmodels for ensemble as calculated here')
        ds = add_ensemble_avg_std_to_dataset(ds=ds,
                                             RFR_dict=RFR_dict,
                                             topmodels=topmodels,
                                             res=res,
                                             save2NetCDF=False)
    # - Do a quick diagnostic plot
    if plot2check:
        for var_ in ds.data_vars:
            # plot an annual average
            arr = ds[var_].mean(dim='time')
            AC.map_plot(arr, res=res)
            plt.title(var_)
            plt.show()
        # Add global variables
    ds = add_attrs2iodide_ds(ds, add_varname_attrs=False)
    # - Save to NetCDF
    if save2NetCDF:
        filename = 'Oi_prj_predicted_{}_{}{}.nc'.format(target, res, xsave_str)
        ds.to_netcdf(filename)
    else:
        return ds


def mk_table_of_point_for_point_performance(RFR_dict=None, df=None,
                                            testset='Test set (strat. 20%)',
                                            inc_ensemble=False,
                                            var2use='RFR(Ensemble)',
                                            target='Iodide'):
    """
    Make a table to summarise point-for-point performance

    Parameters
    -------
    target (str): Name of the target variable (e.g. iodide)
    var2use (str): variable name to use for ensemble prediction
    testset (str): Testset to use, e.g. stratified sampling over quartiles for 20%:80%
    inc_ensemble (bool), include the ensemble (var2use) in the analysis
    RFR_dict (dict): dictionary of core variables and data
    df (pd.DataFrame): dataframe containing target and feature variables

    Returns
    -------
    (None)
    """
    # Get data objects as dictionary and extract dataframe if not provided.
    if isinstance(RFR_dict, type(None)):
        RFR_dict = build_or_get_models()
    if isinstance(df, type(None)):
        df = RFR_dict['df']
    # Get stats on model tuns runs
    stats = get_stats_on_models(RFR_dict=RFR_dict, df=df,
                                analysis4coastal=True,
                                var2use=var2use,
                                inc_ensemble=inc_ensemble, verbose=False)
    # Select param values of interest (and give updated title names )
    rename_titles = {u'Chance2014_STTxx2_I': 'Chance et al. (2014)',
                     u'MacDonald2014_iodide': 'MacDonald et al. (2014)',
                     var2use : var2use,
                     'Iodide': 'Obs.',
                     }
    # Set the stats to use
    first_columns = [
        'mean', 'std', '25%', '50%', '75%',
        'RMSE ({})'.format(testset),  'RMSE (all)',
    ]
    stats = stats[first_columns]
    # Rename columns (50% to median and ... )
    cols2rename = {
        '50%': 'median', 'std': 'std. dev.',
        'RMSE ({})'.format(testset): 'RMSE (withheld)'
    }
    stats.rename(columns=cols2rename,  inplace=True)
    # Only select params of interest
    stats = stats.T[rename_titles.values()].T
    # Rename
    stats.rename(index=rename_titles, inplace=True)
    # Set filename and save detail on models
    csv_name = 'Oi_prj_point_for_point_comp4tabale.csv'
    stats.round(1).to_csv(csv_name)


def mk_table_of_point_for_point_performance_TESTSET(RFR_dict=None, df=None,
                                                    testset='Test set (strat. 20%)',
                                                    inc_ensemble=False,
                                                    var2use='RFR(Ensemble)',
                                                    target='Iodide'):
    """
    Make a table to summarise point-for-point performance within testset

    Parameters
    -------
    target (str): Name of the target variable (e.g. iodide)
    var2use (str): variable name to use for ensemble prediction
    testset (str): Testset to use, e.g. stratified sampling over quartiles for 20%:80%
    inc_ensemble (bool), include the ensemble (var2use) in the analysis
    RFR_dict (dict): dictionary of core variables and data
    df (pd.DataFrame): dataframe containing target and feature variables

    Returns
    -------
    (None)
    """
    # Get data objects as dictionary and extract dataframe if not provided.
    if isinstance(RFR_dict, type(None)):
        RFR_dict = build_or_get_models()
    if isinstance(df, type(None)):
        df = RFR_dict['df']
    # Just select the testing dataset
    df = df.loc[df[testset] == True, :]
    # Get stats on model tuns runs
    stats = get_stats_on_models(RFR_dict=RFR_dict, df=df,
                                analysis4coastal=True,
                                inc_ensemble=inc_ensemble, verbose=False)
    # Select param values of interest (and give updated title names )
    rename_titles = {u'Chance2014_STTxx2_I': 'Chance et al. (2014)',
                     u'MacDonald2014_iodide': 'MacDonald et al. (2014)',
                     'RFR(Ensemble)': 'RFR(Ensemble)',
                     'Iodide': 'Obs.',
                     }
    # Set the stats to use for in csv output
    first_columns = [
        'mean', 'std', '25%', '50%', '75%',
        'RMSE ({})'.format(testset),  'RMSE (all)',
    ]
    stats = stats[first_columns]
    # Rename columns (50% to median and ... )
    cols2rename = {
        '50%': 'median', 'std': 'std. dev.',
        'RMSE ({})'.format(testset): 'RMSE (withheld)'
    }
    stats.rename(columns=cols2rename,  inplace=True)
    # Only select params of interest
    stats = stats.T[rename_titles.values()].T
    # Rename
    stats.rename(index=rename_titles, inplace=True)
    # Set filename and save detail on models
    csv_name = 'Oi_prj_point_for_point_comp4tabale_TESTSET.csv'
    stats.round(1).to_csv(csv_name)


def mk_table_of_point_for_point_performance_ALL(RFR_dict=None, df=None,
                                                testset='Test set (strat. 20%)',
                                                var2use='RFR(Ensemble)',
                                                inc_ensemble=False,
                                                target='Iodide'):
    """
    Make a table to summarise point-for-point performance for all datapoints

    Parameters
    -------
    target (str): Name of the target variable (e.g. iodide)
    var2use (str): variable name to use for ensemble prediction
    testset (str): Testset to use, e.g. stratified sampling over quartiles for 20%:80%
    inc_ensemble (bool), include the ensemble (var2use) in the analysis
    RFR_dict (dict): dictionary of core variables and data
    df (pd.DataFrame): dataframe containing target and feature variables

    Returns
    -------
    (None)
    """
    # Get data objects as dictionary and extract dataframe if not provided.
    if isinstance(RFR_dict, type(None)):
        RFR_dict = build_or_get_models()
    if isinstance(df, type(None)):
        df = RFR_dict['df']
    # Get stats on model tuns runs
    stats = get_stats_on_models(RFR_dict=RFR_dict, df=df,
                                analysis4coastal=True,
                                verbose=False)
    # Select param values of interest (and give updated title names )
    rename_titles = {u'Chance2014_STTxx2_I': 'Chance et al. (2014)',
                     u'MacDonald2014_iodide': 'MacDonald et al. (2014)',
                     'RFR(Ensemble)': 'RFR(Ensemble)',
                     target: 'Obs.',
                     }
    # Set the stats to use
    first_columns = [
        'mean', 'std', '25%', '50%', '75%',
        'RMSE ({})'.format(testset),  'RMSE (all)',
    ]
    stats = stats[first_columns]
    # Rename columns to more standard names for stats (e.g. 50% to median and ... )
    cols2rename = {
        '50%': 'median', 'std': 'std. dev.',
        'RMSE ({})'.format(testset): 'RMSE (withheld)'
    }
    stats.rename(columns=cols2rename,  inplace=True)
    # Rename the columns
    stats.rename(index=rename_titles, inplace=True)
    # Set filename and save detail on models
    csv_name = 'Oi_prj_point_for_point_comp4tabale_ALL.csv'
    stats.round(1).to_csv(csv_name)
    # Also save a .csv of values without derived values
    index2use = [i for i in stats.index if all(
        [ii not in i for ii in derived])]
    stats = stats.T
    stats = stats[index2use]
    stats = stats.T
    csv_name = 'Oi_prj_point_for_point_comp4tabale_ALL_NO_DERIV.csv'
    stats.round(1).to_csv(csv_name)


def get_dataset_processed4ML(restrict_data_max=False,
                             rm_Skagerrak_data=False, rm_outliers=True,
                             rm_LOD_filled_data=False):
    """
    Get dataset as a DataFrame with standard munging settings

    Parameters
    -------
    restrict_data_max (bool): restrict the obs. data to a maximum value?
    rm_Skagerrak_data (bool): remove the data from the Skagerrak region
    rm_outliers (bool): remove the outliers from the observational dataset

    Returns
    -------
    (pd.DataFrame)
    """
    from observations import add_extra_vars_rm_some_data
    from observations import get_processed_df_obs_mod
    # - Local variables
    features_used = None
    target = 'Iodide'
    # - The following settings are set to False as default
    # settings for incoming feature data
    restrict_min_salinity = False
    use_median4chlr_a_NaNs = False
    add_modulus_of_lat = False
    # Apply transforms to data?
    do_not_transform_feature_data = True
    # Just use the forest outcomes and do not optimise
    use_forest_without_optimising = True
    # KLUDGE - this is for N=85
    median_4MLD_when_NaN_or_less_than_0 = False  # This is no longer needed?
    # KLUDGE -  this is for depth values greater than zero
    median_4depth_when_greater_than_0 = False
    # - Get data as a DataFrame
    df = get_processed_df_obs_mod()  # NOTE this df contains values >400nM
    # Add extra vairables and remove some data.
    df = add_extra_vars_rm_some_data(df=df,
                                     restrict_data_max=restrict_data_max,
                                     restrict_min_salinity=restrict_min_salinity,
                                     rm_Skagerrak_data=rm_Skagerrak_data,
                                     rm_outliers=rm_outliers,
                                     rm_LOD_filled_data=rm_LOD_filled_data,
                                     )    # add

    # - Add test and training set assignment to columns
#    print( 'WARNING - What testing had been done on training set selection?!' )
    # Choose a sub set of data to exclude from the input data...
#     from sklearn.model_selection import train_test_split
#     targets = df[ [target] ]
#     # Use a standard 20% test set.
#     train_set, test_set =  train_test_split( targets, test_size=0.2, \
#         random_state=42 )
    # standard split vars?  (values=  rand_20_80, rand_strat )
    ways2split_data = {
        'rn. 20%': (True, False),
        'strat. 20%': (False, True),
    }
    # Loop training/test split methods
    features_used = df.columns.tolist()
    for key_ in ways2split_data.keys():
        # Get settings
        rand_20_80, rand_strat = ways2split_data[key_]
        # Copy a df for splitting
#        df_tmp = df['Iodide'].copy()
        # Now split using existing function
        returned_vars = build.mk_test_train_sets(df=df.copy(), target=target,
                                                 rand_20_80=rand_20_80,
                                                 rand_strat=rand_strat,
                                                 features_used=features_used,
                                                 )
        train_set, test_set, test_set_targets = returned_vars
        # Now assign the values
        key_varname = 'Test set ({})'.format(key_)
        df[key_varname] = False
        df.loc[test_set.index, key_varname] = True
        df.loc[train_set.index, key_varname] = False
    return df


# ---------------------------------------------------------------------------
# ---------- Wrappers for s2s -------------
# ---------------------------------------------------------------------------
def build_or_get_models_iodide(rm_Skagerrak_data=True,
                               rm_LOD_filled_data=False,
                               rm_outliers=True,
                               rebuild=False):
    """
    Wrapper call to build_or_get_models for sea-surface iodide

    Parameters
    -------
    rm_Skagerrak_data (bool): remove the data from the Skagerrak region
    rm_LOD_filled_data (bool): remove the observational values below LOD
    add_modulus_of_lat (bool): add the modulus of lat to dataframe
    rm_outliers (bool): remove the observational outliers from the dataframe

    Returns
    -------
    (dict)
    """
    # Get the dictionary  of model names and features (specific to iodide)
    model_feature_dict = utils.get_model_features_used_dict(rtn_dict=True)

    # Get the observational dataset prepared for ML pipeline
    df = get_dataset_processed4ML(
        rm_Skagerrak_data=rm_Skagerrak_data,
        rm_outliers=rm_outliers,
    )
    # Exclude data from the Skaggerakk data?
    if rm_Skagerrak_data:
        model_sub_dir = '/TEMP_MODELS_No_Skagerrak/'
    else:
        model_sub_dir = '/TEMP_MODELS/'

    if rebuild:
        RFR_dict = build_or_get_models(save_model_to_disk=True,
                                       model_feature_dict=model_feature_dict,
                                       df=df,
                                       read_model_from_disk=False,
                                       model_sub_dir=model_sub_dir,
                                       delete_existing_model_files=True)
    else:
        RFR_dict = build_or_get_models(save_model_to_disk=False,
                                       model_feature_dict=model_feature_dict,
                                       df=df,
                                       read_model_from_disk=True,
                                       model_sub_dir=model_sub_dir,
                                       delete_existing_model_files=False)
    return RFR_dict

def get_stats_on_models(df=None, testset='Test set (strat. 20%)',
                        target='Iodide', inc_ensemble=False,
                        analysis4coastal=False, var2use='RFR(Ensemble)',
                        plot_up_model_performance=True, RFR_dict=None,
                        add_sklean_metrics=False, verbose=True, debug=False):
    """
    Analyse the stats on of params and obs.

    Parameters
    -------
    analysis4coastal (bool): include analysis of data split by coastal/non-coastal
    target (str): Name of the target variable (e.g. iodide)
    testset (str): Testset to use, e.g. stratified sampling over quartiles for 20%:80%
    inc_ensemble (bool): include the ensemble (var2use) in the analysis
    var2use (str): var to use as main model prediction
    debug (bool): print out debugging output?
    add_sklean_metrics (bool): include core sklearn metrics

    Returns
    -------
    (pd.DataFrame)
    """
    # --- Get data
    if isinstance(RFR_dict, type(None)):
        RFR_dict = build_or_get_models()
    # select dataframe with observations and predictions in it
    if isinstance(df, type(None)):
        df = RFR_dict['df']
    # model names
    model_names = RFR_dict['model_names']
    features_used_dict = RFR_dict['features_used_dict']
    N_features_used = RFR_dict['N_features_used']
    oob_scores = RFR_dict['oob_scores']
    # - Evaluate performance of models (e.g. Root Mean Square Error (RMSE) )
    # Also evaluate parameterisations
    param_names = []
    if target == 'Iodide':
        param_names += [u'Chance2014_STTxx2_I', u'MacDonald2014_iodide',
                        # u'Chance2014_Multivariate',
                        ]
    # Aslo include the ensemble parameters
    if inc_ensemble:
        param_names += [var2use]
    # Calculate performance
    stats = calc_performance_of_params(df=df,
                                       params=param_names+model_names)
    # Just test on test set
    df_tmp = df.loc[df[testset] == True, :]
    stats_sub1 = utils.get_df_stats_MSE_RMSE(params=param_names+model_names,
                                       df=df_tmp[[target]+model_names+param_names],
                                       dataset_str=testset,
                                       target=target,
                                       add_sklean_metrics=add_sklean_metrics).T
    stats2concat = [stats, stats_sub1]
    if analysis4coastal:
        # Add testing on coastal
        dataset_split = 'Coastal'
        df_tmp = df.loc[(df['Coastal'] == 1), :]
        stats_sub2 = utils.get_df_stats_MSE_RMSE(params=param_names+model_names,
                                           df=df_tmp[[target]+model_names+param_names],
                                           target=target,
                                           dataset_str=dataset_split,
                                           add_sklean_metrics=add_sklean_metrics).T
        # Add testing on non-coastal
        dataset_split = 'Non coastal'
        df_tmp = df.loc[(df['Coastal'] == 0), :]
        stats_sub3 = utils.get_df_stats_MSE_RMSE(params=param_names+model_names,
                                           df=df_tmp[[target]+model_names+param_names],
                                           target=target,
                                           dataset_str=dataset_split,
                                           add_sklean_metrics=add_sklean_metrics).T
        # Add testing on coastal
        dataset_split = 'Coastal ({})'.format(testset)
        df_tmp = df.loc[(df['Coastal'] == 1) & (df[testset] == True), :]
        stats_sub4 = utils.get_df_stats_MSE_RMSE(params=param_names+model_names,
                                           df=df_tmp[[target]+model_names+param_names],
                                           target=target,
                                           dataset_str=dataset_split,
                                           add_sklean_metrics=add_sklean_metrics).T
        # Add testing on non-coastal
        dataset_split = 'Non coastal ({})'.format(testset)
        df_tmp = df.loc[(df['Coastal'] == 0) & (df[testset] == True), :]
        stats_sub5 = utils.get_df_stats_MSE_RMSE(params=param_names+model_names,
                                           df=df_tmp[[target]+model_names+param_names],
                                           target=target,
                                           dataset_str=dataset_split,
                                           add_sklean_metrics=add_sklean_metrics).T
        # Statistics to concat
        stats2concat += [stats_sub2, stats_sub3, stats_sub4, stats_sub5, ]
    # Combine all stats (RMSE and general stats)
    stats = pd.concat(stats2concat)
    # Add number of features too
    stats = stats.T
    feats = pd.DataFrame(index=model_names)
    N_feat_Var = '# features'
    feats[N_feat_Var] = [N_features_used[i] for i in model_names]
    # and the feature names
    feat_Var = 'features_used'
    feats[feat_Var] = [features_used_dict[i] for i in model_names]
    # and the oob score
    feats['OOB score'] = [oob_scores[i] for i in model_names]
    # combine with the rest of the stats
    stats = pd.concat([stats, feats], axis=1)
    # which vars to sort by
#    var2sortby = ['RMSE (all)', N_feat]
#    var2sortby = 'RMSE (all)'
    var2sortby = 'RMSE ({})'.format(testset)
    # print useful vars to screen
    vars2inc = [
        'RMSE (all)', 'RMSE ({})'.format(testset),
        #    'MSE ({})'.format(testset),'MSE (all)',
    ]
    vars2inc += feats.columns.tolist()
    # Sort df by RMSE
    stats.sort_values(by=var2sortby, axis=0, inplace=True)
    # sort columns
    first_columns = [
        'mean', 'std', '25%', '50%', '75%',
        'RMSE ({})'.format(testset),  'RMSE (all)',
    ]
    rest_of_columns = [i for i in stats.columns if i not in first_columns]
    stats = stats[first_columns + rest_of_columns]
    # Rename columns (50% to median and ... )
    df.rename(columns={'50%': 'median', 'std': 'std. dev.'})
    # Set filename and save detail on models
    csv_name = 'Oi_prj_stats_on_{}_models_built_at_obs_points'.format(target)
    stats.round(2).to_csv(csv_name+'.csv')
    # Also print to screen
    if verbose:
        print(stats[vars2inc+[N_feat_Var]])
    if verbose:
        print(stats[vars2inc])
    # Without testing features
    vars2inc.pop(vars2inc.index('features_used'))
    if verbose:
        print(stats[vars2inc])
    if verbose:
        print(stats[['RMSE ({})'.format(testset), 'OOB score', ]])
    # Save statistics to csv
    csv_name += '_selected'
    stats[vars2inc].round(2).to_csv(csv_name+'.csv')

    # - also save a version that doesn't include the derived dataset
    params2inc = stats.T.columns
    params2inc = [i for i in params2inc if 'DOC' not in i]
    params2inc = [i for i in params2inc if 'Prod' not in i]
    # select these variables from the list
    tmp_stats = stats.T[params2inc].T
    # save a reduced csv
    vars2inc_REDUCED = [
        'mean', 'std', '25%', '50%', '75%',
        'RMSE ({})'.format(testset),  'RMSE (all)',
    ]
    # add the coastal testsets to the data?
    if analysis4coastal:
        vars2inc_REDUCED += [
            u'RMSE (Coastal)', u'RMSE (Non coastal)',
            'RMSE (Coastal (Test set (strat. 20%)))',
            u'RMSE (Non coastal (Test set (strat. 20%)))',
        ]
    # Save a csv with reduced infomation
    csv_name = 'Oi_prj_models_built_stats_on_models_at_obs_points'
    csv_name += '_REDUCED_NO_DERIVED.csv'
    tmp_stats[vars2inc_REDUCED].round(2).to_csv(csv_name)

    # - plot up model performance against the testset
    if plot_up_model_performance:
        #
        rename_titles = {u'Chance2014_STTxx2_I': 'Chance et al. (2014)',
                         u'MacDonald2014_iodide': 'MacDonald et al. (2014)',
                         'Ensemble_Monthly_mean': 'RFR(Ensemble)',
                         'Iodide': 'Obs.',
                         }
        # also compare existing parameters
        params = [
            'Chance et al. (2014)',
            'MacDonald et al. (2014)',
        ]
        # Plot performance of models
        plt_stats_by_model_DERIV(stats=stats, df=df, target=target,
                                 params=params,
                                 rename_titles=rename_titles )
        # Plot up also without derivative variables
        plt_stats_by_model_DERIV(stats=stats, df=df, target=target,
                                 params=params,
                                 rename_titles=rename_titles )


def test_performance_of_params(target='Iodide', features_used=None):
    """
    Test the performance of the parameters

    Parameters
    -------
    target (str): Name of the target variable (e.g. iodide)
    features_used (list): list of the features within the model_name model

    Returns
    -------
    (None)
    """
    # - Get the data
    # get processed data
    # settings for incoming feature data
    restrict_data_max = False
    restrict_min_salinity = False
    use_median4chlr_a_NaNs = True
    add_modulus_of_lat = False
    # apply transforms to  data?
    do_not_transform_feature_data = True
    # just use the forest out comes
    use_forest_without_optimising = True
    # Which "features" (variables) to use
    if isinstance(features_used, type(None)):
        features_used = [
            'WOA_TEMP_K',
            'WOA_Salinity',
            'Depth_GEBCO',
        ]
    # Local variables for use in this function
    param_rename_dict = {
        u'Chance2014_STTxx2_I': 'Chance2014',
        u'MacDonald2014_iodide': 'MacDonald2014',
        u'Iodide': 'Obs.',
    }
    param_names = param_rename_dict.keys()
    param_names.pop(param_names.index(target))
    # Set-up a dictionary of test set variables
    random_split_var = 'rn. 20%'
    strat_split_var = 'strat. 20%'
    model_names_dict = {
        'TEMP+DEPTH+SAL (rs)': random_split_var,
        'TEMP+DEPTH+SAL': strat_split_var,
    }
    model_names = model_names_dict.keys()
    # Get data as a DataFrame
    df = get_processed_df_obs_mod()  # NOTE this df contains values >400nM
    # Add extra variables and remove some data.
    df = add_extra_vars_rm_some_data(df=df,
                                     restrict_data_max=restrict_data_max,
                                     restrict_min_salinity=restrict_min_salinity,
                                     add_modulus_of_lat=add_modulus_of_lat,
                                     use_median4chlr_a_NaNs=use_median4chlr_a_NaNs,
                                     )    # add
    # add boolean for test and training dataset
    splits_dict = {
        random_split_var: (True, False), strat_split_var: (False, True),
    }
    # Loop test sets
    for test_split in splits_dict.keys():
        rand_20_80, rand_strat = splits_dict[test_split]
        # Select just the features used and the target variable
        df_tmp = df[features_used+[target]].copy()
        # split into the training and test sets
        returned_vars = build.mk_test_train_sets(df=df_tmp,
                                                 rand_20_80=rand_20_80,
                                                 rand_strat=rand_strat,
                                                 features_used=features_used,
                                                 )
        train_set, test_set, test_set_targets = returned_vars
        # Add this to the dataframe using the passed shape as a template
        dummy = np.zeros(df.shape[0])
        dummy[test_set.index] = True
        df['test ({})'.format(test_split)] = dummy

    # Add model predictions
    for model_name in model_names:
        df[model_name] = get_model_predictions4obs_point(df=df,
                                                         model_name=model_name)

    # - Get stats on whole dataset?
    stats = calc_performance_of_params(df=df,
                                       params=param_names+model_names)

    # - Get stats for model on just its test set dataset
    model_stats = []
    for modelname in model_names:
        test_set = model_names_dict[modelname]
        dataset_str = 'test ({})'.format(test_set)
        print(modelname, test_set, dataset_str)
        df_tmp = df.loc[df[dataset_str] == True]
        print(df_tmp.shape, df_tmp[target].mean())
        model_stats.append(utils.get_df_stats_MSE_RMSE(
            df=df_tmp[[target, modelname]+param_names],
            params=param_names+[modelname],
            dataset_str=test_set, target=target).T)
    # Add these to core dataset
    stats = pd.concat([stats] + model_stats)

    # - get stats for coastal values
    # Just ***NON*** coastal values
    df_tmp = df.loc[df['coastal_flagged'] == False]
    test_set = '>30 Salinty'
    print(df_tmp.shape)
    # Calculate...
    stats_open_ocean = utils.get_df_stats_MSE_RMSE(
        df=df_tmp[[target]+model_names+param_names],
        params=param_names+model_names,
        dataset_str=test_set, target=target).T
    # Just  ***coastal*** values
    df_tmp = df.loc[df['coastal_flagged'] == True]
    test_set = '<30 Salinty'
    print(df_tmp.shape)
    # Calculate...
    stats_coastal = utils.get_df_stats_MSE_RMSE(
        df=df_tmp[[target]+model_names+param_names],
        params=param_names+model_names,
        dataset_str=test_set, target=target).T
    # Add these to core dataset
    stats = pd.concat([stats] + [stats_coastal, stats_open_ocean])

    # - Minor processing and save
    # rename the columns for re-abliiity
    stats.rename(columns=param_rename_dict, inplace=True)
    # round the columns to one dp.
    stats = stats.round(1)
    print(stats)
    # Save as a csv
    stats.to_csv('Oi_prj_param_performance.csv')

# ---------------------------------------------------------------------------
# ---------- Wrappers for s2s -------------
# ---------------------------------------------------------------------------

def build_or_get_models_iodide(rm_Skagerrak_data=True,
                                       rm_LOD_filled_data=False,
                                       rm_outliers=True,
                                       rebuild=False ):
    """
    Wrapper call to build_or_get_models for sea-surface iodide
    """
    # Get the dictionary  of model names and features (specific to iodide)
    model_feature_dict = utils.get_model_features_used_dict(rtn_dict=True)

    # Get the observational dataset prepared for ML pipeline
    df = get_dataset_processed4ML(
        rm_Skagerrak_data=rm_Skagerrak_data,
        rm_LOD_filled_data=rm_LOD_filled_data,
        rm_outliers=rm_outliers,
        )
    #
    if rm_Skagerrak_data:
        model_sub_dir = '/TEMP_MODELS_No_Skagerrak/'
#     elif rm_LOD_filled_data:
#         temp_model_dir = wrk_dir+'/TEMP_MODELS_no_LOD_filled/'
    else:
        model_sub_dir = '/TEMP_MODELS/'

    if rebuild:
        RFR_dict = build_or_get_models(save_model_to_disk=True,
#                                    rm_Skagerrak_data=rm_Skagerrak_data,
                                    model_feature_dict=model_feature_dict,
                                    df=df,
                                    read_model_from_disk=False,
                                    delete_existing_model_files=True )
    else:
        RFR_dict = build_or_get_models(save_model_to_disk=True,
#                                    rm_Skagerrak_data=rm_Skagerrak_data,
                                    model_feature_dict=model_feature_dict,
                                    df=df,
                                    read_model_from_disk=True,
                                    delete_existing_model_files=False )
    return RFR_dict


def mk_iodide_test_train_sets(df=None, target='Iodide',
                              rand_strat=True, features_used=None,
                              random_state=42, rand_20_80=False,
                              nsplits=4, verbose=True, debug=False):
    """
    Wrapper for build.mk_test_train_sets for iodide code

    Parameters
    -------
    rand_strat (bool), split the data in a random way using stratified sampling
    rand_20_80 (bool), split the data in a random way
    df (pd.DataFrame): dataframe containing target and feature variables
    target (str): Name of the target variable (e.g. iodide)
    nsplits (int), number of ways to split the data
    random_state (int), seed value to use as random seed for reproducible analysis
    debug (bool): print out debugging output?
    verbose (bool): print out verbose output?

    Returns
    -------
    (list)
    """
    # Call the s2s function with some presets
    returned_vars = build.mk_test_train_sets(df=df, target=target,
                                             nsplits=nsplits,
                                             rand_strat=rand_strat,
                                             features_used=features_used,
                                             random_state=random_state,
                                             rand_20_80=rand_20_80,
                                             verbose=verbose, debug=debug)
    return returned_vars


def add_attrs2iodide_ds(ds, convert_to_kg_m3=False,
                        varname='Ensemble_Monthly_mean',
                        add_global_attrs=True, add_varname_attrs=True,
                        rm_spaces_from_vars=False,
                        convert2HEMCO_time=False):
    """
    wrapper for add_attrs2target_ds for iodine scripts

    Parameters
    -------
    varname (str): variable name to make changes to
    rm_spaces_from_vars (bool), remove spaces from variable names
    convert_to_kg_m3 (bool), convert the output units to kg/m3
    global_attrs_dict (dict), dictionary of global attributes
    convert2HEMCO_time (bool), convert to a HEMCO-compliant time format
    add_global_attrs (bool), add global attributes to dataset
    add_varname_attrs (bool), add variable attributes to dataset

    Returns
    -------
    (xr.Dataset)
    """
    # Set variable attribute dictionary variables
    attrs_dict = {}
    attrs_dict['long_name'] = "sea-surface iodide concentration"
    # Set global attribute dictionary variables
    title_str = "A parameterisation of sea-surface iodide on a monthly basis"
    global_attrs_dict = {
        'Title': title_str,
        'Author': 'Tomas Sherwen (tomas.sherwen@york.ac.uk)',
        'Notes': 'This is a parameterisation of sea-surface iodide on a monthly basis. The NetCDF was made using xarray (xarray.pydata.org).',
        'DOI': '10.5285/02c6f4eea9914e5c8a8390dd09e5709a.',
        'Citation': "A machine learning based global sea-surface iodide distribution, T. Sherwen , et al., in review, 2019 ; Data reference: Sherwen, T., Chance, R., Tinel, L., Ellis, D., Evans, M., and Carpenter, L.: Global predicted sea-surface iodide concentrations v0.0.0., https://doi.org/10.5285/02c6f4eea9914e5c8a8390dd09e5709a., 2019.",
        'references': "Paper Reference: A machine learning based global sea-surface iodide distribution, T. Sherwen , et al., in review, 2019 ; Data reference: Sherwen, T., Chance, R., Tinel, L., Ellis, D., Evans, M., and Carpenter, L.: Global predicted sea-surface iodide concentrations v0.0.0., https://doi.org/10.5285/02c6f4eea9914e5c8a8390dd09e5709a., 2019.",
    }
    #  Call s2s function
    ds = utils.add_attrs2target_ds(ds, convert_to_kg_m3=False,
                                   attrs_dict=attrs_dict,
                                   global_attrs_dict=global_attrs_dict,
                                   varname=varname,
                                   add_global_attrs=True,
                                   add_varname_attrs=True,
                                   rm_spaces_from_vars=False,
                                   convert2HEMCO_time=False)
    return ds


if __name__ == "__main__":
    main()
