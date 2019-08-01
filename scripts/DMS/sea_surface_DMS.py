#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
processing/analysis functions for DMS observations
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

# Get DMS specific functions
from observations import get_DMS_obs


def main():
    """
    Driver for module's man if run directly from command line. unhash
    functionalitliy to call.
    """
    # - Set core local variables
    target = 'DMS'
    # Setup the data directory structure (only needs to be done once))
    # NOTE: the locations of s2s and data are set in script/<target>'s *.rc file
#    utils.check_or_mk_directory_structure(target=target)

    # - Get the observations? (Not needed for core workflow as also held in RFR_dict)
    # (This processese the observations and only needs to be done once)
    rm_outliers = True
#    df = get_dataset_processed4ML(target=target, rm_outliers=rm_outliers)

    # - build models with the observations
    RFR_dict = build_or_get_models_DMS(rebuild=False, target=target)
    # build the models (just run once!)
#    RFR_dict = build_or_get_models_DMS(rebuild=True, target=target)
    # Get stats ont these models
    stats = RFRanalysis.get_core_stats_on_current_models(RFR_dict=RFR_dict,
                                                         target=target, verbose=True,
                                                         debug=True)
    # Get the top ten models
    topmodels = build.get_top_models(RFR_dict=RFR_dict, stats=stats,
                                     vars2exclude=['DOC', 'Prod'], n=10)

    # --- Predict values globally (only use 0.125)
    # Extra string for NetCDF save name
    xsave_str = '_v0_0_0'
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
    analysis.plt_stats_by_model(stats=stats, df=df, target=target )
    # Plot up also without derivative variables
    analysis.plt_stats_by_model_DERIV(stats=stats, df=df, target=target )

    # ----
    # Explore the predicted concentrations
    # Get the data
    ds = utils.get_predicted_3D_values(target=target)
    # plot up an annual mean
    plotting.plot_up_annual_averages_of_prediction(ds=ds, target=target)
    # Plot up the values by season
    plotting.plot_up_seasonal_averages_of_prediction(ds=ds, target=target)



def build_or_get_models_DMS(target='DMS',
                            rm_LOD_filled_data=False,
                            rm_outliers=True,
                            rebuild=False):
    """
    Wrapper call to build_or_get_models for sea-surface DMS

    Parameters
    -------
    target (str): Name of the target variable (e.g. DMS)
    rm_outliers (bool): remove the outliers from the observational dataset
    rm_LOD_filled_data (bool): remove the limit of detection (LOD) filled values?
    rebuild (bool): rebuild the models or just read them from disc?

    Returns
    -------
    (pd.DataFrame)
    """
    # Get the dictionary  of model names and features (specific to DMS)
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



def get_dataset_processed4ML(restrict_data_max=False, target='DMS',
                             rm_outliers=True,
                             rm_LOD_filled_data=False):
    """
    Get dataset as a DataFrame with standard munging settings

    Parameters
    -------
    restrict_data_max (bool): restrict the obs. data to a maximum value?
    target (str): Name of the target variable (e.g. DMS)
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
    target = 'DMS'
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
#        df_tmp = df['DMS'].copy()
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
