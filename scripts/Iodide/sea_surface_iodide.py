#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Module to hold processing/analysis functions for Ocean iodide (Oi!) project

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

# from multiprocessing import Pool
# from functools import partial

import sparse2spatial as s2s
import sparse2spatial.utils as utils
import sparse2spatial.ancillaries2grid_oversample as ancillaries2grid
import sparse2spatial.archiving as archiving
import sparse2spatial.RFRbuild as build
import sparse2spatial.RFRanalysis as analysis
import sparse2spatial.plotting as plotting
from sparse2spatial.RFRanalysis import get_stats_on_models
from sparse2spatial.RFRanalysis import add_ensemble_avg_std_to_dataset
from sparse2spatial.RFRbuild import get_top_models
from sparse2spatial.RFRbuild import mk_test_train_sets
from sparse2spatial.RFRbuild import build_or_get_models
from sparse2spatial.utils import get_model_features_used_dict
from sparse2spatial.utils import get_outlier_value

# import AC_tools (https://github.com/tsherwen/AC_tools.git)
import AC_tools as AC

# temp
from sparse2spatial.RFRanalysis import get_core_stats_on_current_models

# Get iodide specific functions
#from observations import get_dataset_processed4ML


def main():
    """
    Driver for module's man if run directly from command line. unhash
    functionalitliy to call.
    """
    model_feature_dict = utils.get_model_features_used_dict(rtn_dict=True)
    print(model_feature_dict)
#    print(model_feature_dict['NO3+DOC+Phos'])
    # ---- ---- Over-arching settings
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

    # ---- ----- ----- ----- ----- ----- ----- ----- -----
    # ----- ----- Evaluating input datasets
    # General plots of all species
#    get_diagnostic_plots_analysis4observations()

    # ---- ----- ----- ----- ----- ----- ----- ----- -----
    # ----- ----- Processing of observations (& extraction of ancillaries)
    # --- Get iodide observations?
#    get_iodide_obs()
    # --- Re-process file?
#    get_iodide_obs(process_new_iodide_obs_file=True)
    # --- Re-process ancillaries file?
#    process_iodide_obs_ancillaries_2_csv()
#    get_core_rosie_obs()
    # --- Process MLD csv files? (Just for ease of use/data munging)
#    process_MLD_csv2NetCDF()

    # Check extracted data against observations.
#    compare_obs_ancillaries_with_extracted_values()

    # ---- ----- ----- ----- ----- ----- ----- ----- -----
    # ----- ----- Build ancillary variable dataset file
    # ---
    # Build either a full or low res ancillary NetCDF file
#    res = '0.125x0.125'
#    res = '4x5' # low resolution to test extraction etc?
    # Get indicies to extract for variables in imported NetCDF
#    mk_array_of_indices4locations4res( res=res )
    # Extract the variables to NetCDF
#    extract_feature_variables2NetCDF( res=res )
    # Interpolate the fields at full resolution
#    interpolate_NaNs_in_feature_variables( res=res )

    # ---- ----- ----- ----- ----- ----- ----- ----- -----
    # ----- ----- Building new iodide field (inc. Machine learning)
    # ---
    # (Re-)Build all models
    # (random stats means this gives the same answer everytime)
#    build_or_get_models_iodide(rebuild=True,
#                                       rm_Skagerrak_data=rm_Skagerrak_data )

    # --- Update the predictor array values
#     res='4x5'
#     set_SAL_and_NIT_above_65N_to_avg(res=res)

    # --- Predict values globally (only use 0.125)
    # extra string for NetCDF save name
#    xsave_str = ''
    # make NetCDF predictions from the main array
#    save2NetCDF = True
    # resolution to use? (full='0.125x0.125', test at lower e.g. '4x5')
#    res = '0.125x0.125'
#     res = '4x5'
#     mk_iodide_predictions_from_ancillaries(None, res=res, RFR_dict=RFR_dict,
#                                            use_updated_predictor_NetCDF=False,
#                                            save2NetCDF=save2NetCDF,
#                                            rm_Skagerrak_data=rm_Skagerrak_data,
#                                            topmodels=topmodels,
#                                            xsave_str=xsave_str, add_ensemble2ds=False)

    # ---- ----- ----- ----- ----- ----- ----- ----- -----
    # ----- ----- Sensitive test the new iodide field (
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
    # ----- ----- Plotting of output
#     res = '4x5'
#     res = '2x2.5'
#     extr_str = 'tree_X_JUST_TEMP_K_GEBCO_SALINTY'
#     get_diagnostic_plots_analysis4model(extr_str=extr_str, res=res)

#    extr_str = 'tree_X_JUST_TEMP_K_GEBCO_SALINTY'
#    extr_str='tree_X_JUST_TEMP_K_GEBCO_SALINTY_REPEAT'
#    extr_str='tree_X_STRAT_JUST_TEMP_K_GEBCO_SALINTY'
#     extr_strs = (
#     'tree_X_JUST_TEMP_K_GEBCO_SALINTY_REPEAT',
#     'tree_X_STRAT_JUST_TEMP_K_GEBCO_SALINTY',
#     )
#     for res in ['4x5', '2x2.5']:
#         for extr_str in extr_strs:
#              get_diagnostic_plots_analysis4model(extr_str=extr_str, res=res)

    # ---- ----- ----- ----- ----- ----- ----- ----- -----
    # ----- ----- Plots for AGU poster
    # get_analysis_numbers_for_AGU17_poster()
    # get_plots_for_AGU_poster()

    # ---- ----- ----- ----- ----- ----- ----- ----- -----
    # ----- ----- Plots / Analsis for Oi! paper
    # Get shared data
#    RFR_dict = build_or_get_models()

    # --- 2D analysis
    # Plot up spatial comparison of obs. and params
#    plot_up_obs_spatially_against_predictions( RFR_dict=RFR_dict )
    # Test which plotting options to use (to display markers)
#    plot_up_obs_spatially_against_predictions_options(
#          RFR_dict=RFR_dict )

    # plot up the input variables spatially
#    res = '0.125x0.125'
#    res = '4x5'
#   plot_up_input_ancillaries_spatially( res=res, RFR_dict=RFR_dict,
#    save2png=True)

    # Plot up the 2D differences in predictions
#    res= '0.125x0.125'
#    res= '4x5'
#    plot_up_spatial_changes_in_predicted_values( res=res, window=True, f_size=30)

    # Plot the locations with greatest variation in the predictions - REDUNDENT!
    # (Use plot_up_ensemble_avg_and_std_spatially instead )
#     plot_up_spatial_uncertainty_predicted_values(res=res,
#                                                  rm_Skagerrak_data=rm_Skagerrak_data )

    # Get stats from the 4x5 and 2x2.5 predictions
#    get_stats_on_spatial_predictions_4x5_2x25()

    # Get stats from the 0.125x0.125 prediction
#    get_stats_on_spatial_predictions_0125x0125()

    # Calculate the average predicted surface conc (only 4x5. 2x2.5 too? )
#    calculate_average_predicted_surface_conc() # AGU calcs at 4x5

    # Plot up latitude vs. predicted iodide
#    plot_predicted_iodide_vs_lat_figure()

    # Seasonal prediction of iodide by month
#    plot_monthly_predicted_iodide( res='4x5' )
#    plot_monthly_predicted_iodide( res='0.125x0.125' )
#    plot_monthly_predicted_iodide_diff( res='0.125x0.125' )

    # explore the extracted data in the arctic and AnatArctic
#    explore_extracted_data_in_Oi_prj_explore_Arctic_Antarctic_obs()

    # Check the sensitivity to input variables >= 65 N
#    mk_PDFs_to_show_the_sensitivty_input_vars_65N_and_up(
#        save_str='TEST_V' )

    # --- Point-for-point analysis
    # build ODR plots for obs. vs. model
#    analyse_X_Y_correlations_ODR( RFR_dict=RFR_dict, context='poster' )
#    analyse_X_Y_correlations_ODR( RFR_dict=RFR_dict, context='paper' )

    # Analyse the X_Y correlations
#    analyse_X_Y_correlations( RFR_dict=RFR_dict )

    # Get the importance of individual features for prediction
#    get_feature_importance( RFR_dict=RFR_dict )

    # Get general stats on the current models
#    get_stats_on_models( RFR_dict=RFR_dict )

    # Get tabulated performance
#    mk_table_of_point_for_point_performance(RFR_dict=RFR_dict)
#    mk_table_of_point_for_point_performance_ALL(RFR_dict=RFR_dict)
#    mk_table_of_point_for_point_performance_TESTSET(RFR_dict=RFR_dict)

    # Get CDF and PDF plots for test, training, entire, and residual
#    plot_up_CDF_and_PDF_of_obs_and_predictions( df=RFR_dict['df'] )

    # Plot up various spatial plots for iodide concs + std.
#    plot_up_ensemble_avg_and_std_spatially(
#        rm_Skagerrak_data=rm_Skagerrak_data
#    )

    # --- Spatial analysis for specific locations
    # explore the observational data in the Arctic
#    explore_observational_data_in_Arctic_parameter_space( RFR_dict=RFR_dict )

    # plot up where decision points are
#    plot_spatial_area4core_decisions( res='4x5' )
#    plot_spatial_area4core_decisions( res='0.125x0.125' )

    # Explore the sensitivity to data denial
#    explore_sensitivity_of_65N2data_denial( res='4x5' )
#    explore_sensitivity_of_65N2data_denial( res='2x2.5' )
#    explore_sensitivity_of_65N2data_denial( res='0.125x0.125' )

    # --- Analysis of models build
    # testset analysis
#    test_model_sensitiivty2training_test_split() # driver not in use yet!
#    run_tests_on_testing_dataset_split_quantiles()
#    run_tests_on_testing_dataset_split()

    # selection of variables to build models

    # hyperparameter tuning of selected models

    # Analysis of the spatial variance of individual ensemble members
#    rm_Skagerrak_data = True
#    rm_Skagerrak_data = False
#     analyse_dataset_error_in_ensemble_members( res='0.125x0.125', \
#         rebuild_models=False, remake_NetCDFs=False,
#         rm_Skagerrak_data=rm_Skagerrak_data,
#         topmodels=topmodels )
#    analyse_dataset_error_in_ensemble_members( res='0.125x0.125', \
#       rebuild_models=True, remake_NetCDFs=True, \
#       rm_Skagerrak_data=rm_Skagerrak_data,
#       topmodels=topmodels
#        )
    # Common resolutions
#   regrid_output_to_common_res_as_NetCDFs(topmodels=topmodels,
#                                         rm_Skagerrak_data=rm_Skagerrak_data)

    # --- Do tree by tree analysis
    # Extract trees to .dot files (used make make the single tree figures)
#    extract_trees4models( RFR_dict=RFR_dict, N_trees2output=50 )

    # Plot up interpretation of trees
    # Now in TreeSurgeon - see separate repository on github
    # https://github.com/wolfiex/TreeSurgeon

    # analysis of node spliting
#    analyse_nodes_in_models( RFR_dict=RFR_dict )
    # analysis of outputted trees
#    analyse_nodes_in_models()

    # Now make Ensemble diagram - REDUNDENT
#    mk_ensemble_diagram()

    # --- Do futher analysis on the impact of the depth variable
    do_analysis_processing_linked_to_depth_variable()

    # plot this up and other figures for the ML paper
    plot_spatail_figures_for_ML_paper_with_cartopy()

    # - pass if no functions are uncommented
    pass


def do_analysis_processing_linked_to_depth_variable():
    """
    """
    from plotting_and_analysis import get_ensemble_predicted_iodide
    # Get the base topmodels
    vars2exclude = ['DOC', 'Prod', ]
    topmodels = get_top_models(RFR_dict=RFR_dict, vars2exclude=vars2exclude, n=10 )
    topmodels_BASE = topmodels.copy()

    # add the ensemble to the dataframe and over write this as the dictionary
    var2use='RFR(Ensemble)'
#     df = RFR_dict['df']
#     df = get_ensemble_predicted_iodide(df=df, RFR_dict=RFR_dict, topmodels=topmodels,
#                                        var2use=var2use)
#     RFR_dict['df'] =  df
#     # now get the stats
#     mk_table_of_point_for_point_performance(RFR_dict=RFR_dict, inc_ensemble=True)
#
#     # - Now do the same thing, but calculate the prediction with depth
#     var2use = 'RFR(Ensemble_nDepth)'
    # Get topmodels without
    vars2exclude = ['DOC', 'Prod', 'DEPTH']
    topmodels = get_top_models(RFR_dict=RFR_dict, vars2exclude=vars2exclude, n=10 )
    topmodels_DEPTH = topmodels.copy()
    # Now calculate the ensemble prediction
#     df = RFR_dict['df']
#     df = get_ensemble_predicted_iodide(df=df, RFR_dict=RFR_dict, topmodels=topmodels,
#                                        var2use=var2use)
#     RFR_dict['df'] =  df
#     # now get the stats
#     mk_table_of_point_for_point_performance(RFR_dict=RFR_dict, df=df, inc_ensemble=True,
#                                             var2use=var2use)

    topmodels2use = topmodels_DEPTH + topmodels_BASE
    topmodels2use = list(set(topmodels2use))
    # Make a spatial prediction
    xsave_str = '_TEST_DEPTH_'
    # make NetCDF predictions from the main array
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


def plot_spatail_figures_for_ML_paper_with_cartopy():
    """
    Plot up all the spatial figures for the ML paper with cartopy
    """


    # Add LWI to NEtCDF
    res ='0.125x0.125'
    ds = add_LWI2array(ds=ds, res=res, var2template='Ensemble_Monthly_mean')
    vars2mask = [
    'Ensemble_Monthly_mean_nDepth', 'Ensemble_Monthly_mean', 'Ensemble_Monthly_std',
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
    target = 'Iodide'
    dpi = 720
    projection = ccrs.PlateCarree()
    cbar_kwargs={
    'extend':'max', 'pad': 0.025, 'orientation':"vertical", 'label': 'nM',
#    'fraction' : 0.1
    'shrink':0.675,
    'ticks' : np.arange(vmin, vmax+1, 60),
    }
    vmax = 240
    vmin = 0
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = AC.get_colormap(arr=np.array([vmin,vmax]))

    # Now plot the core prediction
    var2use4Ensemble = 'Ensemble_Monthly_mean'
    title= 'Annual average sea-surface iodide (nM) predicted by RFR(Ensemble)'
    title = None # no title shown in paper's plots
    plot_spatial_data_TEST(ds=ds, var2plot=var2use4Ensemble, title=title,
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
    plot_spatial_data_TEST(ds=ds, var2plot=var2use4Ensemble, title=title,
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
    plot_spatial_data_TEST(ds=ds, var2plot=var2use4Ensemble, title=title,
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
    df = df.loc[df['Iodide'] <= get_outlier_value(df=df, var2use='Iodide'), :]
    s = 15
    edgecolor = 'k'
    x = df[u'Longitude'].values
    y = df[u'Latitude'].values
    z = df['Iodide'].values
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
    plot_spatial_data_TEST(ds=ds, var2plot=var2use4std, title=title,
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
    plot_spatial_data_TEST(ds=ds, var2plot=var2use4std_pcent, title=title,
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
    title= '(A) MacDonald et al. (2014)'
    plot_spatial_data_TEST(ds=ds, var2plot=var2use, fig=fig, ax=ax1,
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
    plot_spatial_data_TEST(ds=ds, var2plot=var2use, fig=fig, ax=ax2,
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


def run_tests_on_testing_dataset_split(model_name=None, n_estimators=500,
                                       features_used=None, target='Iodide', df=None):
    """
    Run tests on the sensitivity of model to test/training choices

    Parameters
    -------

    Returns
    -------

    Notes
    -----
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
        features_used = get_model_features_used_dict(model_name)

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
    bool2 = df['Iodide'] < get_outlier_value(df=df, var2use='Iodide')
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
    bool2 = df['Iodide'] < get_outlier_value(df=df, var2use='Iodide')
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
    bool2 = df['Iodide'] < get_outlier_value(df=df, var2use='Iodide')
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
    bool_ = df['Iodide'] < get_outlier_value(df=df, var2use='Iodide')
    tmp_ts = df.loc[bool_][features_used+[target]].copy()
    # also remove values where iodide <400
    TSETS_N[Tname] = tmp_ts.shape[0]
    TSETS[Tname] = tmp_ts
    TSETS_nsplits[Tname] = 4
    # - No Skagerrak
    Tname = 'No Skagerrak \n or outliers'
    bool1 = df['Data_Key'].values == 'Truesdale_2003_I'
    bool2 = df['Iodide'] > get_outlier_value(df=df, var2use='Iodide')
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
    #         if random_split_var in Tname:
    #             rand_strat = False
    #             rand_20_80 = True
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
                                          n_estimators=n_estimators, criterion='mse')
    #            , oob_score=oob_score)
            # fit the model
            model.fit(train_features, train_labels)
            # predict the values
            df_tmp[Tname] = model.predict(df_tmp[features_used].values)
            # get the stats against the test group
            df_tmp = df_tmp[[Tname, target]].loc[test_set.index]
            # get MSE and RMSE
            MSE = (df_tmp[target]-df_tmp[Tname])**2
            MSE = np.mean(MSE)
            std = np.std(df_tmp[Tname].values)

            # return stats on bias and variance
            # (just use RMSE and std dev. for now)
            RMSE_l += [np.sqrt(MSE)]
    #        s = pd.Series( {'MSE': MSE, 'RMSE': np.sqrt(MSE), 'std':std } )
    #        RMSE_df[Tname] = s
            #
            del df_tmp, train_features, train_labels, test_features, test_labels
            del model
#        gc.collect()
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
                                           models_dict=None, features_used_dict=None,
                                           RFR_dict=None, dsA=None,
                                           stats=None, folder=None,
                                           use_updated_predictor_NetCDF=False,
                                           save2NetCDF=False, plot2check=False,
                                           models2compare=[], topmodels=None,
                                           rm_Skagerrak_data=False, xsave_str='',
                                           add_ensemble2ds=False,
                                           verbose=True, debug=False):
    """
    Make a NetCDF file of predicted vairables for a given resolution

    Parameters
    -------

    Returns
    -------

    Notes
    -----
    """
    # --- local variables
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
            stats = get_stats_on_models(RFR_dict=RFR_dict, analysis4coastal=True,
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
        features_used = get_model_features_used_dict(modelname)
        # Make a DataSet of predicted values
        ds_tmp = utils.mk_da_of_predicted_values(model=model, modelname=modelname,
                                           res=res, features_used=features_used,
                                           dsA=dsA)
        #  Add attributes to the prediction
        ds_tmp = add_attrs2iodide_ds(ds_tmp, add_global_attrs=False,
                                     varname=modelname)
        # Savea
        ds_l += [ds_tmp]
    # Combine datasets
    ds = xr.merge(ds_l)
    # -- Also get values for parameterisations
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
                                             RFR_dict=RFR_dict, topmodels=topmodels,
                                             res=res,
                                             save2NetCDF=False)
    # ---- Do a quick diagnostic plot
    if plot2check:
        for var_ in ds.data_vars:
            # plot an annual average
            arr = ds[var_].mean(dim='time')
            AC.map_plot(arr, res=res)
            plt.title(var_)
            plt.show()
        # Add global variables
    ds = add_attrs2iodide_ds(ds, add_varname_attrs=False)
    # --- Save to NetCDF
    if save2NetCDF:
        filename = 'Oi_prj_predicted_{}_{}{}.nc'.format(target, res, xsave_str)
        ds.to_netcdf(filename)
    else:
        return ds


# def mk_ensemble_diagram(dpi=1000):
#     """
#     Make a schematic image from outtputted trees to show the workflow
#     """
#     from matplotlib import gridspec
#     import matplotlib.image as mpimg
#     import glob
#     # --- Local fixed variablse
#     depth2investigate = 6
#     fontname = 'Montserrat'
#     # Use an external file for the braces?
# #    file = 'Oi_prj_Braces_01'
# #    file = 'Oi_prj_Braces_02'
#     # Use local font
#     from matplotlib import font_manager as fm, rcParams
#     import matplotlib.pyplot as plt
#     fpath = 'Montserrat-ExtraLight.ttf'
# #    model2use = 'RFR(TEMP+DEPTH+NO3+SWrad)'
#     model2use = 'RFR(TEMP+DEPTH+NO3)'
#
#     # --- Setup the figure / subfigures for plotting
#     fig = plt.figure(figsize=(6, 1.4), dpi=dpi)
#     # Use grid specs to setup plot with nrows with set "height ratios"
#     nrows = 1000
#     ncols = 1000
#     G = gridspec.GridSpec(nrows, ncols)
#     # Setup axis
#     Swidth = 100
#     Lwidth = 200
#     buffer = 10
#     # Setup rows
#     RowSizes = np.linspace(1, 999, 6)
#     RowSizes = [int(i) for i in RowSizes]
#     # Setup large axis subplots (for single tree and single forest)
#     colsizes = (
#         (0, 0+Lwidth+buffer),
#         #    (300, 300+Swidth+buffer),
#         (500, 500+Lwidth+buffer),
#         #    (670, 670+Swidth+buffer),
#     )
#     Laxes = [plt.subplot(G[:, i[0]:i[-1]]) for i in colsizes]
#     # switch off axis etc
#     for ax in Laxes:
#         ax.set_xticks([])
#         ax.set_yticks([])
#         ax.patch.set_visible(False)
#         ax.axis('off')
#
#     # --- Single tree subplot
#     ax = Laxes[0]
#     # Open and use an arbitrary single tree
#     filename = glob.glob('tree_{}*.png'.format(model2use))[0]
#     img = mpimg.imread(filename)
#     imgplot = ax.imshow(img)
#
#     # --- Plot an arrow sign from this to many (~10) trees of the forest's 500
#     # Axis to use?
#     ax = Laxes[0]
#     # Arrow
#     alt_text = r'$\Rightarrow$'
#     fontsize = 30
#     prop = fm.FontProperties(fname=fpath, size=fontsize)
#     ax.annotate(alt_text, (0.85, .50), textcoords='axes fraction',
#                 fontproperties=prop,
#                 #    	fontname=fontname, fontsize=fontsize,
#                 )
#
#     # --- Plot ten single trees (of the forest's 500 )
#     # Setup Small Laxes
#     startx = 50+Lwidth+buffer+buffer
#     # tree files
# #    file_str = 'tree_*{}*.png'.format(model2use)
# #    file_str = file_str.replace(')', '\\)').replace('(', '\\(')
#     tree_files = glob.glob('tree_*{}*.png'.format(model2use))
#     if len(tree_files):
#         tree_files += tree_files
#     # loop by file to plot
#     for n_plot in range(10):
#         # Set locations for subplot
#         Hspacer = 0
#         Swidth = 75
#         Hspacer = 5
#         Xleft = startx
#         XRight = startx+Swidth
#         # Set Xright depending on plot number
#         if n_plot in range(10)[0::2]:
#             Xleft = XRight
#             XRight = Xleft+Swidth+Hspacer
#         # Get the row number
#         Nrow = int(n_plot/2.)
#         # make subplot
#         print(RowSizes[Nrow], RowSizes[Nrow+1], Xleft, XRight)
#         ax = plt.subplot(G[RowSizes[Nrow]:RowSizes[Nrow+1], Xleft:XRight])
#         # Remove lines for subplots
#         ax.set_xticks([])
#         ax.set_yticks([])
#         ax.patch.set_visible(False)
#         ax.axis('off')
#         # get tree
#         img = mpimg.imread(tree_files[n_plot])
#         imgplot = ax.imshow(img)
#
#     # --- Use mathematical braces etc to show combining of trees
#     # Axis to use?
#     ax = Laxes[0]
#     # X... X500
#     alt_text = r'$\bar \}$'
#     fontsize = 70
#     prop = fm.FontProperties(fname=fpath, size=fontsize)
# #    ax.text(x=3000,y=0.25, s=alt_text, fontsize=fontsize )
#     ax.annotate(alt_text, (2.05, 0.3), textcoords='axes fraction',
#                 fontsize=fontsize,
#                 #    	fontweight='ultralight', style='italic',
#                 fontname=fontname)
#     # now add x values
#     fontsize = 5
#     prop = fm.FontProperties(fname=fpath, size=fontsize)
#     alt_text = r'$\bar X = X_i,...X_{500}$'
#     ax.annotate(alt_text, (2.05, 0.05), textcoords='axes fraction',
#                 fontproperties=prop,
#                 #        fontweight='light', fontname=fontname, fontsize=fontsize,
#                 )
#
#     # --- Single forest
#     ax = Laxes[1]
#     # Open and use tree
#     filename = 'Oi_prj_features_of_{}_for_depth_{}_white.png'
#     filename = filename.format(model2use, depth2investigate)
#     img = mpimg.imread(filename)
#     imgplot = ax.imshow(img)
#
#     # --- Use mathematical braces etc to show combining of forests
#     ax = Laxes[1]
#     # X... X10
#     alt_text = r'$\bar \}$'
#     fontsize = 70
#     prop = fm.FontProperties(fname=fpath, size=fontsize)
# #    ax.text(x=3000,y=0.25, s=alt_text, fontsize=fontsize )
#     ax.annotate(alt_text, (2.2, 0.3), textcoords='axes fraction',
#                 fontproperties=prop,
#                 #    	fontweight='ultralight', style='italic',  fontname=fontname,
#                 #		fontsize=fontsize,
#                 )
#     # now add x values
#     fontsize = 5
#     prop = fm.FontProperties(fname=fpath, size=fontsize)
#     alt_text = r'$\bar X = X_i,...X_{10}$'
#     ax.annotate(alt_text, (2.2, 0.05), textcoords='axes fraction',
#                 fontproperties=prop,
#                 #        fontweight='light', fontname=fontname, fontsize=fontsize,
#                 )
#
#     # --- Arrow sign
#     fontsize = 30
#     prop = fm.FontProperties(fname=fpath, size=fontsize)
#     alt_text = r'$\Rightarrow$'
#     ax.annotate(alt_text, (0.85, .50), textcoords='axes fraction',
#                 fontproperties=prop,
#                 #    	fontname=fontname, fontsize=fontsize,
#                 )
#
#     # --- forest
# #    ax = axes[3]
#     # --- Ten single trees
#     # Get names of trees
#     import glob
#     files = glob.glob('*{}*white*png'.format(depth2investigate))
#     # Setup Small Laxes
#     startx = 570+Lwidth+buffer+buffer
# #    starty =
#     for n_plot in range(10):
#
#         # if
#         Hspacer = 0
#         Swidth = 75
#         Hspacer = 5
#         Xleft = startx
#         XRight = startx+Swidth
#
#         if n_plot in range(10)[0::2]:
#             Xleft = XRight
#             XRight = Xleft+Swidth+Hspacer
#         # Get the row number
#         Nrow = int(n_plot/2.)
#         # make subplot
#         print(RowSizes[Nrow], RowSizes[Nrow+1], Xleft, XRight)
#         ax = plt.subplot(G[RowSizes[Nrow]:RowSizes[Nrow+1], Xleft:XRight])
#         # remove lines for subplots
#         ax.set_xticks([])
#         ax.set_yticks([])
#         ax.patch.set_visible(False)
#         ax.axis('off')
#
#         # get tree
# #        filename = 'Oi_prj_features_of_RFR(TEMP+DEPTH+NO3+SWrad)_for_depth_7_white.png'
#         img = mpimg.imread(files[n_plot])
#         imgplot = ax.imshow(img)
#
#     # --- Add final text ("prediction")
#     ax = Laxes[1]
# #    ax = Label_ax
# #    Label_ax = plt.subplot()
# #    ax = plt.subplot()
# #    ax.set_xticks([])
# #    ax.set_yticks([])
# #    ax.patch.set_visible(False)
# #    ax.axis('off')
# #    ax.patch.set_facecolor( None )
# #    ax.patch.set_alpha(0.0)
#
#     # Arrow?
#
#     # Predcition
#     alt_text = 'Prediction'
#     fontsize = 18
#     prop = fm.FontProperties(fname=fpath, size=fontsize)
# #    ax.annotate( alt_text, (2.7, .90), textcoords='axes fraction',
#     ax.annotate(alt_text, (2.7, .875), textcoords='axes fraction',
#                 fontproperties=prop,
#                 #    	fontname=fontname, fontsize=fontsize,
#                 rotation=90)
# #    ax.text( 0.75, .50, alt_text, ha="center", va="center", fontsize=fontsize,
# #        rotation=90)
#
#     # ----  Update spacings
#     left = 0.025
#     bottom = 0.025
#     top = 1 - bottom
# #    hspace = 0.05
#     fig.subplots_adjust(bottom=bottom, top=top,
#                         left=left,
#                         #         right=right,
#                         #        hspace=hspace
#                         #        , wspace=wspace
#                         )
#     # Save figure
#     plt.savefig('Oi_prj_ensemble_workflow_image.png', dpi=dpi)


def mk_table_of_point_for_point_performance(RFR_dict=None, df=None,
                                            testset='Test set (strat. 20%)',
                                            inc_ensemble=False,
                                            var2use='RFR(Ensemble)',
                                            target='Iodide'):
    """
    Make a table to summarise point-for-point performance

    Parameters
    -------
    target (str), Name of the target variable (e.g. iodide)
    var2use (str), variable name to use for ensemble prediction
    testset (str), Testset to use, e.g. stratified sampling over quartiles for 20%:80%
    inc_ensemble (bool), include the ensemble (var2use) in the analysis

    Returns
    -------

    Notes
    -----
    """
    # Get data objects as dictionary and extract dataframe if not provided.
    if isinstance(RFR_dict, type(None)):
        RFR_dict = build_or_get_models()
    if isinstance(df, type(None)):
        df = RFR_dict['df']
    # Get stats on model tuns runs
    stats = get_stats_on_models(RFR_dict=RFR_dict, df=df, analysis4coastal=True,
                                var2use=var2use,
                                inc_ensemble=inc_ensemble, verbose=False)
    # Select param values of interest (and give updated title names )
    rename_titles = {u'Chance2014_STTxx2_I': 'Chance et al. (2014)',
                     u'MacDonald2014_iodide': 'MacDonald et al. (2014)',
#                     'RFR(Ensemble)': 'RFR(Ensemble)',
                     var2use : var2use,
                     'Iodide': 'Obs.',
                     #                    u'Chance2014_Multivariate': 'Chance et al. (2014) (Multi)',
                     }
    # Set the stats to use
    first_columns = [
        'mean', 'std', '25%', '50%', '75%',
        'RMSE ({})'.format(testset),  'RMSE (all)',
    ]
#    rest_of_columns = [i for i in stats.columns if i not in first_columns]
    stats = stats[first_columns]
    # rename columns (50% to median and ... )
    cols2rename = {
        '50%': 'median', 'std': 'std. dev.',
        'RMSE ({})'.format(testset): 'RMSE (withheld)'
    }
    stats.rename(columns=cols2rename,  inplace=True)
    # only select params of interest
    stats = stats.T[rename_titles.values()].T
    # rename
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
    target (str), Name of the target variable (e.g. iodide)
    var2use (str), variable name to use for ensemble prediction
    testset (str), Testset to use, e.g. stratified sampling over quartiles for 20%:80%
    inc_ensemble (bool), include the ensemble (var2use) in the analysis

    Returns
    -------

    Notes
    -----
    """
    # Get data objects as dictionary and extract dataframe if not provided.
    if isinstance(RFR_dict, type(None)):
        RFR_dict = build_or_get_models()
    if isinstance(df, type(None)):
        df = RFR_dict['df']
    # Just select the testing dataset
    df = df.loc[df[testset] == True, :]
    # Get stats on model tuns runs
    stats = get_stats_on_models(RFR_dict=RFR_dict, df=df, analysis4coastal=True,
                                inc_ensemble=inc_ensemble, verbose=False)
    # Select param values of interest (and give updated title names )
    rename_titles = {u'Chance2014_STTxx2_I': 'Chance et al. (2014)',
                     u'MacDonald2014_iodide': 'MacDonald et al. (2014)',
                     'RFR(Ensemble)': 'RFR(Ensemble)',
                     'Iodide': 'Obs.',
                     #                     u'Chance2014_Multivariate': 'Chance et al. (2014) (Multi)',
                     }
    # Set the stats to use for in csv output
    first_columns = [
        'mean', 'std', '25%', '50%', '75%',
        'RMSE ({})'.format(testset),  'RMSE (all)',
    ]
#    rest_of_columns = [i for i in stats.columns if i not in first_columns]
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
    target (str), Name of the target variable (e.g. iodide)
    var2use (str), variable name to use for ensemble prediction
    testset (str), Testset to use, e.g. stratified sampling over quartiles for 20%:80%
    inc_ensemble (bool), include the ensemble (var2use) in the analysis

    Returns
    -------

    Notes
    -----
    """
    # Get data objects as dictionary and extract dataframe if not provided.
    if isinstance(RFR_dict, type(None)):
        RFR_dict = build_or_get_models()
    if isinstance(df, type(None)):
        df = RFR_dict['df']
    # Get stats on model tuns runs
    stats = get_stats_on_models(RFR_dict=RFR_dict, df=df, analysis4coastal=True,
                                verbose=False)
    # Select param values of interest (and give updated title names )
    rename_titles = {u'Chance2014_STTxx2_I': 'Chance et al. (2014)',
                     u'MacDonald2014_iodide': 'MacDonald et al. (2014)',
                     'RFR(Ensemble)': 'RFR(Ensemble)',
                     target: 'Obs.',
                     #                     u'Chance2014_Multivariate': 'Chance et al. (2014) (Multi)',
                     }
    # Set the stats to use
    first_columns = [
        'mean', 'std', '25%', '50%', '75%',
        'RMSE ({})'.format(testset),  'RMSE (all)',
    ]
#    rest_of_columns = [i for i in stats.columns if i not in first_columns]
    stats = stats[first_columns]
    # rename columns (50% to median and ... )
    cols2rename = {
        '50%': 'median', 'std': 'std. dev.',
        'RMSE ({})'.format(testset): 'RMSE (withheld)'
    }
    stats.rename(columns=cols2rename,  inplace=True)
    # rename
    stats.rename(index=rename_titles, inplace=True)
    # Set filename and save detail on models
    csv_name = 'Oi_prj_point_for_point_comp4tabale_ALL.csv'
    stats.round(1).to_csv(csv_name)
    # also save a .csv of values without derived values
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
                                     #                                     add_modulus_of_lat=add_modulus_of_lat,
                                     rm_Skagerrak_data=rm_Skagerrak_data,
                                     rm_outliers=rm_outliers,
                                     rm_LOD_filled_data=rm_LOD_filled_data,
                                     #                use_median4chlr_a_NaNs=use_median4chlr_a_NaNs,
                                     #                median_4MLD_when_NaN_or_less_than_0=median_4MLD_when_NaN_or_less_than_0,
                                     #                    median_4depth_when_greater_than_0=median_4depth_when_greater_than_0,
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
    for key_ in ways2split_data.keys():
        # Get settings
        rand_20_80, rand_strat = ways2split_data[key_]
        # Copy a df for splitting
#        df_tmp = df['Iodide'].copy()
        # Now split using existing function
        returned_vars = mk_test_train_sets(df=df.copy(), target=target,
                                                 rand_20_80=rand_20_80,
                                                 rand_strat=rand_strat,
                                                 features_used=df.columns.tolist(),
                                                 #                                                   features_used=features_used,
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

    Returns
    -------

    Notes
    -----
    """
    # Get the dictionary  of model names and features (specific to iodide)
    model_feature_dict = get_model_features_used_dict(rtn_dict=True)

    # Get the observational dataset prepared for ML pipeline
    df = get_dataset_processed4ML(
        rm_Skagerrak_data=rm_Skagerrak_data,
#        rm_LOD_filled_data=rm_LOD_filled_data,
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


def mk_iodide_test_train_sets(df=None, target='Iodide',
                              rand_strat=True, features_used=None,
                              random_state=42, rand_20_80=False,
                              nsplits=4, verbose=True, debug=False):
    """
    Wrapper for mk_test_train_sets for iodide code

    Parameters
    -------
    rand_strat (bool), split the data in a random way using stratified sampling
    rand_20_80 (bool), split the data in a random way
    """
    # Call the s2s function with some presets
    returned_vars = mk_test_train_sets(df=df, target=target, nsplits=nsplits,
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

    Returns
    -------

    Notes
    -----
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
                                   add_global_attrs=True, add_varname_attrs=True,
                                   rm_spaces_from_vars=False,
                                   convert2HEMCO_time=False)
    return ds


if __name__ == "__main__":
    main()
