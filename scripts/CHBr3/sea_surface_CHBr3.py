#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Module to hold processing/analysis functions for CHBr3 work

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
import xarray as xr
import sparse2spatial as s2s
import sparse2spatial.utils as utils
#import sparse2spatial.ancillaries2grid_oversample as ancillaries2grid
#import sparse2spatial.archiving as archiving
from sparse2spatial.RFRbuild import mk_test_train_sets
import sparse2spatial.RFRbuild as build
import sparse2spatial.RFRanalysis as analysis
import sparse2spatial.plotting as plotting
#import sparse2spatial.RFRanalysis as analysis
from sparse2spatial.RFRbuild import build_or_get_models

# Get CHBr3 specific functions
from observations import get_CHBr3_obs


def main():
    """
    Driver for module's man if run directly from command line. unhash
    functionalitliy to call.
    """
    # - Set core local variables
    target = 'CHBr3'
    # Setup the data directory structure (only needs to be done once))
    # NOTE: the locations of s2s and data are set in script/<target>'s *.rc file
#    utils.check_or_mk_directory_structure(target=target)

    # - Get the observations? (Not needed for core workflow as also held in RFR_dict)
#    df = get_dataset_processed4ML(target=target, rm_outliers=rm_outliers)

    # - build models with the observations
    RFR_dict = build_or_get_models_CHBr3(rebuild=False, target=target)
    # Get stats ont these models
    stats = analysis.get_core_stats_on_current_models(RFR_dict=RFR_dict,
                                                      target=target, verbose=True,
                                                      debug=True)
    # Get the top ten models
    topmodels = build.get_top_models(RFR_dict=RFR_dict, stats=stats,
                                     vars2exclude=['DOC', 'Prod'], n=10)

    # --- Predict values globally (only use 0.125)
    # extra strig for NetCDF save name
    xsave_str = '_v_0_0_0'
    # make NetCDF predictions from the main array
    save2NetCDF = True
    # resolution to use? (full='0.125x0.125', test at lower e.g. '4x5')
    res = '0.125x0.125'
#    res='4x5'
#    res='2x2.5'
    build.mk_predictions_for_3D_features(None, res=res, RFR_dict=RFR_dict,
                                         save2NetCDF=save2NetCDF, target=target,
                                         models2compare=topmodels,
                                         topmodels=topmodels,
                                         xsave_str=xsave_str, add_ensemble2ds=True)

    # --- Plot up the performance of the models
    df = RFR_dict['df']
    # Plot performance of models
    analysis.plt_stats_by_model(stats=stats, df=df, target=target )
    # Plot up also without derivative variables
    analysis.plt_stats_by_model_DERIV(stats=stats, df=df, target=target )

    # --- Save out the field in kg/m3 for use in models
    version = 'v0_0_0'
    folder = '/users/ts551/scratch/data/s2s/{}/outputs/'.format(target)
    filename = 'Oi_prj_predicted_{}_0.125x0.125_{}'.format(target, version)
    ds = xr.open_dataset( folder + filename+'.nc' )
    # Convert to kg/m3
    RMM = 252.73
    new_var = 'Ensemble_Monthly_mean_kg_m3'
    ds = add_converted_field_pM_2_kg_m3(ds=ds, var2use='Ensemble_Monthly_mean',
                                        target=target, RMM=RMM,
                                        new_var=new_var)
    # Save with just the kg/m3 field to a NetCDF file
    ds = ds[[new_var]]
    ds = ds.rename(name_dict={new_var:'Ensemble_Monthly_mean'})
    ds.to_netcdf( folder + filename+'{}.nc'.format('_kg_m3') )


def build_or_get_models_CHBr3(target='CHBr3',
                              rm_outliers=True,
                              rebuild=False):
    """
    Wrapper call to build_or_get_models for sea-surface CHBr3
    """
    # Get the dictionary  of model names and features (specific to CHBr3)
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
    version = '_v0_0_0' # NOTE: this string is different to one in main()
    ds = utils.get_predicted_values_as_ds(target=target, version=version)
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
#    params2plot = [var,  'Chance2014_STTxx2_I', 'MacDonald2014_CHBr3',]
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
#        extr_str=region
        # Now plot
        plotting.plt_X_vs_Y_for_obs_v_params(df=df, params2plot=params2plot,
                                             obs_var=obs_var,
                                             extr_str=extr_str)



def get_dataset_processed4ML(restrict_data_max=False, target='CHBr3',
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
    target = 'CHBr3'
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
                                     )
    # Re-index to a single contiguous index
    df['Original Index'] = df.index.copy()
    N1 = df.shape[0]
    df.index = np.arange(N1)
    print('WARNING: Reindexed to shape of DataFrame processed for ML ({})'.format(N1))

    # - Add test and training set assignment to columns
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
#        df_tmp = df['CHBr3'].copy()
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
