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
import sparse2spatial.plotting as plotting
from sparse2spatial.RFRbuild import build_or_get_models

# Get CH3I specific functions


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
    stats = analysis.get_core_stats_on_current_models(RFR_dict=RFR_dict,
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
#    res = '0.125x0.125'
    res = '4x5'
    build.mk_predictions_for_3D_features(None, res=res, RFR_dict=RFR_dict,
                                         save2NetCDF=save2NetCDF, target=target,
                                         models2compare=topmodels,
                                         topmodels=topmodels,
                                         xsave_str=xsave_str, add_ensemble2ds=True)

    #


def plt_X_vs_Y_for_regions(RFR_dict=None, df=None, params2plot=[], LatVar='lat',
                           LonVar='lon', target='CH3I',
                           obs_var='Obs.')
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
#    testset = 'Test set (strat. 20%)'
#    df = df.loc[df[testset] == True, :]
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
#        extr_str=region+' (withheld)'
        extr_str=region
        # Now plot
        plotting.plt_X_vs_Y_for_obs_v_params(df=df, params2plot=params2plot,
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


if __name__ == "__main__":
    main()
