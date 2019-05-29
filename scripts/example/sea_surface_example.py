#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Module to hold processing/analysis functions for example workflow

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
from sparse2spatial.RFRbuild import mk_ML_testing_and_training_set
import sparse2spatial.RFRbuild as build
import sparse2spatial.RFRanalysis as analysis
from sparse2spatial.RFRbuild import build_or_get_current_models


def main():
    """
    Driver for module's man if run directly from command line. unhash
    functionalitliy to call.
    """
    # Set core local variables
    target = 'example'

    # Get the observations?
    # NOTE(S): This processes the observations and only needs to be done once.
    #          Not needed for core workflow as also held in RFR_dict.
#    df = get_dataset_processed4ML(target=target, rm_outliers=rm_outliers)

    # Build models with the observations and ancillary variables
    # build the models (just run once!)
#    RFR_dict = build_or_get_current_models_example(rebuild=True, target=target)
    # afterwards, just read in the models saved to disk
    RFR_dict = build_or_get_current_models_example(rebuild=False, target=target)

    # Get stats ont these models
    stats = analysis.get_core_stats_on_current_models(RFR_dict=RFR_dict,
            target=target, verbose=True, debug=True)

    # Get the top ten models
    topmodels = build.get_top_models(RFR_dict=RFR_dict, stats=stats,
            NO_DERIVED=True, n=10)

    # - Predict values globally (only use 0.125)
    # extra strig for NetCDF save name
    xsave_str = '_INITIAL'
    # make NetCDF predictions from the main array
    save2NetCDF = True
    # resolution to use? (full='0.125x0.125', test at lower e.g. '4x5')
#    res = '0.125x0.125'
    res='4x5'
#    res='2x2.5'
    build.mk_predictions_from_ancillaries(None, res=res, RFR_dict=RFR_dict,
                                         save2NetCDF=save2NetCDF, target=target,
                                         models2compare=topmodels,
                                         topmodels=topmodels,
                                         xsave_str=xsave_str, add_ensemble2ds=True )


def build_or_get_current_models_example(target='example', rm_outliers=True,
                                       rebuild=False ):
    """
    Wrapper call to build_or_get_current_models for sea-surface example
    """
    # Get the dictionary  of model names and features (specific to iodide)
    model_feature_dict = utils.get_model_testing_features_dict(rtn_dict=True)
    # Get the observational dataset prepared for ML pipeline
    df = get_dataset_processed4ML(target=target, rm_outliers=rm_outliers)
    # load the models or build them from scratch
    if rebuild:
        RFR_dict = build_or_get_current_models(save_model_to_disk=True,
                                    model_feature_dict=model_feature_dict,
                                    df=df, target=target,
                                    read_model_from_disk=False,
                                    delete_existing_model_files=True )
    else:
        RFR_dict = build_or_get_current_models(save_model_to_disk=False,
                                    model_feature_dict=model_feature_dict,
                                    df=df, target=target,
                                    read_model_from_disk=True,
                                    delete_existing_model_files=False )
    return RFR_dict


def get_dataset_processed4ML(target='example', rm_outliers=True):
    """
    Get dataset as a DataFrame with standard munging settings

    Parameters
    -------
    rm_outliers (boolean): remove outliers from the input data

    Returns
    -------
    (pd.DataFrame)

    Notes
    -----

    """
    from observations import add_extra_vars_rm_some_data
    from observations import get_processed_df_obs_mod
    # Local variables
    testing_features = None
    target = 'example'
    target_name = [target]
    # Apply transforms to data?
    do_not_transform_feature_data = True
    # Just use the forest outcomes and do not optimise
    use_forest_without_optimising = True
    # Get observational and ancillary data as a single DataFrame
    df = get_processed_df_obs_mod()  # NOTE this df contains values >400nM
    # Add extra vairables and remove some data.
    df = add_extra_vars_rm_some_data(df=df, target=target,rm_outliers=rm_outliers)
    # Re-index to a single contiguous index
    df['Original Index'] = df.index.copy()
    N1 = df.shape[0]
    df.index = np.arange( N1 )
    print('WARNING: Reindexed to shape of DataFrame processed for ML ({})'.format(N1))
    # - Add test and training set assignment to columns
    # Add both random and standard stratified split for nows
    # (values=random_20_80_split, random_strat_split)
    ways2split_data = {
        'rn. 20%': (True, False),
        'strat. 20%': (False, True),
    }
    # Loop training/test split methods
    for key_ in ways2split_data.keys():
        # Get settings
        random_20_80_split, random_strat_split = ways2split_data[key_]
        # Now split using existing function
        returned_vars = mk_ML_testing_and_training_set(df=df.copy(),
                                                    target=target,
                                                    random_20_80_split=random_20_80_split,
                                                    random_strat_split=random_strat_split,
                                                    testing_features=df.columns.tolist(),
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
