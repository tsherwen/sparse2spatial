
#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Module to hold processing/analysis functions for OCS work

Notes
----
ML = Machine Learning
target = the value aiming to be estimated or provided in training
feature = a induivual conpoinet of a predictor vector assigned to a target
( could be called an attribute )
predictor = vector assigned to a target value
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
import sparse2spatial.plotting as s2splotting
from sparse2spatial.RFRbuild import mk_test_train_sets
from sparse2spatial.RFRbuild import build_or_get_models
from sparse2spatial.RFRbuild import get_top_models
#from sparse2spatial.RFRanalysis import get_stats_on_models
#from sparse2spatial.RFRanalysis import get_stats_on_multiple_global_predictions
# Local modules specific to OCS work
import observations as obs

def main():
    """
    Driver for module's man if run directly from command line. unhash
    functionalitliy to call.
    """
    # - Set core local variables
    target = 'OCS'

    # - build models with the observations
    RFR_dict = build_or_get_models_OCS(rebuild=False, target=target)
    # Get stats ont these models
    stats = RFRanalysis.get_core_stats_on_current_models(RFR_dict=RFR_dict,
                                                      target=target, verbose=True,
                                                      debug=True)
    # Get the top ten models
    topmodels = build.get_top_models(RFR_dict=RFR_dict, stats=stats,
                                     vars2exclude=['DOC', 'Prod'], n=10)

    # --- Predict values globally (only use 0.125)
    # extra strig for NetCDF save name
    xsave_str = '_TEST'
    # make NetCDF predictions from the main array
    save2NetCDF = True
    # resolution to use? (full='0.125x0.125', test at lower e.g. '4x5')
    res = '0.125x0.125'
#    res = '4x5'
#    res='2x2.5'
    build.mk_predictions_for_3D_features(None, res=res, RFR_dict=RFR_dict,
                                         use_updated_predictor_NetCDF=False,
                                         save2NetCDF=save2NetCDF, target=target,
                                         models2compare=topmodels,
                                         topmodels=topmodels,
                                         xsave_str=xsave_str, add_ensemble2ds=True)


    # - Plot up the predicted field
    # get the predicted data as saved offline
    ds = utils.get_predicted_values_as_ds(target=target, )
    # annual average
    s2splotting.plot_up_annual_averages_of_prediction(target=target, ds=ds)
    # seasonally resolved average
    s2splotting.plot_up_seasonal_averages_of_prediction(target=target, ds=ds)

    # --- Plot up the performance of the models
    df = RFR_dict['df']
    #
    df = add_ensemble_prediction2df(df=df, target=target)

    # Plot performance of models
    RFRanalysis.plt_stats_by_model(stats=stats, df=df, target=target )
    # Plot up also without derivative variables
    RFRanalysis.plt_stats_by_model_DERIV(stats=stats, df=df, target=target )

    # - Plot comparisons against observations
    # Plot up an orthogonal distance regression (ODR) plot
    ylim = (0, 80)
    xlim = (0, 80)
#    xlim, ylim =  None, None
    params =  ['RFR(Ensemble)']
    s2splotting.plot_ODR_window_plot(df=df, params=params, units='pM', target=target,
                                     ylim=ylim, xlim=xlim)

    # Plot up a PDF of concs and bias
    ylim = (0, 80)
    s2splotting.plot_up_PDF_of_obs_and_predictions_WINDOW(df=df, params=params,
                                                          units='pM',
                                                          target=target,
                                                          xlim=xlim)
    #
    LonVar = 'Longitude'
    LatVar = 'Latitude'
    s2splotting.plt_X_vs_Y_for_regions(df=df, target=target, LonVar=LonVar, LatVar=LatVar)


    # --- Save out the field in kg/m3 for use in models
    version = 'v0_0_0'
    folder = '/users/ts551/scratch/data/s2s/{}/outputs/'.format(target)
    filename = 'Oi_prj_predicted_{}_0.125x0.125_{}'.format(target, version)
    ds = xr.open_dataset( folder + filename+'.nc' )
    # Convert to kg/m3
    RMM = 60.08
    new_var = 'Ensemble_Monthly_mean_kg_m3'
    ds = utils.add_converted_field_pM_2_kg_m3(ds=ds, var2use='Ensemble_Monthly_mean',
                                        target=target, RMM=RMM,
                                        new_var=new_var)
    # Save with just the kg/m3 field to a NetCDF file
    ds = ds[[new_var]]
    ds = ds.rename(name_dict={new_var:'Ensemble_Monthly_mean'})
    ds.to_netcdf( folder + filename+'{}.nc'.format('_kg_m3') )




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
                                              var2extract=var2extract,)
    df[var] = vals
    return df


def explore_values_per_hour(df, target='OCS', dpi=320, debug=False):
    """
    Explore the concentrations of OCS on a hourly basis
    """
    import seaborn as sns
    sns.set(style="whitegrid")
    # - get the data
    df = obs.get_OCS_obs()
    N0 = float(df.shape[0])
    # drop the NaNs
    df.dropna()
    # - plot up the values by hour
    hrs = df['Hour'].value_counts(dropna=True)
    hrs = hrs.sort_index()
    N = float(hrs.sum())
    ax = hrs.plot.bar(x='hour of day', y='#', rot=0)
    title_str = '{} data that includes measured hour \n (N={}, {:.2f}% of all data)'
    plt.title( title_str.format(target, int(N), N/N0*100 ) )
    # Update the asthetics
    time_labels = hrs.index.values
    # make sure the values with leading zeros drop these
    index = [float(i) for i in time_labels]
    # Make the labels into strings of integers
    time_labels = [str(int(i)) for i in time_labels]
    if len(index) < 6:
        ax.set_xticks(index)
        ax.set_xticklabels(time_labels)
    else:
        ax.set_xticks(index[2::3])
        ax.set_xticklabels(time_labels[2::3])
    xticks = ax.get_xticks()
    if debug:
        print((xticks, ax.get_xticklabels()))
    # Save the plot
    plt.savefig( 's2s_obs_data_by_hour_{}'.format(target), dpi=dpi)
    plt.close('all')

    # - Plot the data X vs. Y for the obs vs. hour
    x_var, y_var = 'Hour', 'OCS'
    df_tmp = df[ [x_var, y_var] ]
    X = df_tmp[x_var].values
    Y = df_tmp[y_var].values
    # Drop NaNs

    # Plot all data
    fig = plt.figure(dpi=dpi, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(111)
    alpha = 0.5
    plt.scatter(X, Y, color='red', s=3, facecolor='none', alpha=alpha)
    # plot linear fit line


    # Now plot
#    AC.plt_df_X_vs_Y( df=df_tmp, x_var=x_var, y_var=y_var, save_plot=True )
    png_filename = 'X_vs_Y_{}_vs_{}'.format(x_var, y_var)
    png_filename = AC.rm_spaces_and_chars_from_str(png_filename)
    plt.savefig(png_filename, dpi=dpi)


    plt.close('all')


def build_or_get_models_OCS(rm_Skagerrak_data=True, target='OCS',
                               rm_LOD_filled_data=False,
                               rm_outliers=True,
                               rebuild=False):
    """
    Wrapper call to build_or_get_models for sea-surface OCS

    Parameters
    -------
    target (str): Name of the target variable (e.g. iodide)
    rm_outliers (bool): remove the outliers from the observational dataset
    rm_LOD_filled_data (bool): remove the limit of detection (LOD) filled values?
    rebuild (bool): rebuild the models or just read them from disc?

    Returns
    -------
    (pd.DataFrame)
    """
    # Get the dictionary  of model names and features (specific to OCS)
    model_feature_dict = utils.get_model_features_used_dict(rtn_dict=True)
    # Get the observational dataset prepared for ML pipeline
    df = get_dataset_processed4ML(target=target, rm_outliers=rm_outliers)
    # Now extract built models or build new models
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


def get_dataset_processed4ML(restrict_data_max=False, target='OCS',
                             rm_Skagerrak_data=False, rm_outliers=True,
                             rm_LOD_filled_data=False):
    """
    Get dataset as a DataFrame with standard munging settings

    Parameters
    -------
    restrict_data_max (bool): restrict the obs. data to a maximum value?
    target (str): Name of the target variable (e.g. iodide)
    rm_outliers (bool): remove the outliers from the observational dataset
    rm_LOD_filled_data (bool): remove the limit of detection (LOD) filled values?

    Returns
    -------
    (pd.DataFrame)
    """
    from observations import add_extra_vars_rm_some_data
    from observations import get_processed_df_obs_mod
    # - Local variables
    features_used = None
    target = 'OCS'
    target_name = [target]
    # - The following settings are set to False as default
    # settings for incoming feature data
    restrict_min_salinity = False
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
#        df_tmp = df['OCS'].copy()
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




if __name__ == "__main__":
    main()

