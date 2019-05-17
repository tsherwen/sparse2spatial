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
import sparse2spatial as s2s
import sparse2spatial.utils as utils
#import sparse2spatial.ancillaries2grid_oversample as ancillaries2grid
a#import sparse2spatial.archiving as archiving
import sparse2spatial.RTRbuild as build
import sparse2spatial.RFRanalysis as analysis
from sparse2spatial.RTRbuild import build_or_get_current_models

# Get iodide specific functions
from observations import get_dataset_processed4ML


def main():
    """
    Driver for module's man if run directly from command line. unhash
    functionalitliy to call.
    """
    # - Set core local variables
    target = 'CH3Br'

    # - Get the observations
    #






def build_or_get_current_models_CHBr3(rm_Skagerrak_data=True, target='CH3Br',
                                       rm_LOD_filled_data=False,
                                       rm_outliers=True,
                                       rebuild=False ):
    """
    Wrapper call to build_or_get_current_models for sea-surface CHBr3
    """
    # Get the dictionary  of model names and features (specific to iodide)
    model_feature_dict = get_model_testing_features_dict(rtn_dict=True)

    # Get the observational dataset prepared for ML pipeline
    df = get_dataset_processed4ML(
#        rm_Skagerrak_data=rm_Skagerrak_data,
        rm_LOD_filled_data=rm_LOD_filled_data,
        rm_outliers=rm_outliers,
        )

    if rebuild:
        RFR_dict = build_or_get_current_models(save_model_to_disk=True,
#                                    rm_Skagerrak_data=rm_Skagerrak_data,
                                    model_feature_dict=model_feature_dict,
                                    df=df,
                                    read_model_from_disk=False,
                                    delete_existing_model_files=True )
    else:
        RFR_dict = build_or_get_current_models(save_model_to_disk=True,
#                                    rm_Skagerrak_data=rm_Skagerrak_data,
                                    model_feature_dict=model_feature_dict,
                                    df=df,
                                    read_model_from_disk=True,
                                    delete_existing_model_files=False )
    return RFR_dict



def get_dataset_processed4ML(restrict_data_max=False, target='CH3Br',
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

    Notes
    -----

    """
    from observations import add_extra_vars_rm_some_data
    from observations import get_processed_df_obs_mod
    # - Local variables
    testing_features = None
    target = 'CHBr3'
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
    # KLUDGE - this is for N=85
#    median_4MLD_when_NaN_or_less_than_0 = False  # This is no longer needed?
    # KLUDGE -  this is for depth values greater than zero
#    median_4depth_when_greater_than_0 = False
    # - Get data as a DataFrame
    df = get_processed_df_obs_mod()  # NOTE this df contains values >400nM
    # Add extra vairables and remove some data.
    df = add_extra_vars_rm_some_data(df=df, target=target,
                                     restrict_data_max=restrict_data_max,
                                     restrict_min_salinity=restrict_min_salinity,
#                                     add_modulus_of_lat=add_modulus_of_lat,
#                                     rm_Skagerrak_data=rm_Skagerrak_data,
                                     rm_outliers=rm_outliers,
#                                     rm_LOD_filled_data=rm_LOD_filled_data,
                                     )    # add

    # - Add test and training set assignment to columns
#    print( 'WARNING - What testing had been done on training set selection?!' )
    # Choose a sub set of data to exclude from the input data...
#     from sklearn.model_selection import train_test_split
#     targets = df[ target_name ]
#     # Use a standard 20% test set.
#     train_set, test_set =  train_test_split( targets, test_size=0.2, \
#         random_state=42 )
    # standard split vars?  (values=  random_20_80_split, random_strat_split )
    ways2split_data = {
        'rn. 20%': (True, False),
        'strat. 20%': (False, True),
    }
    # Loop training/test split methods
    for key_ in ways2split_data.keys():
        # Get settings
        random_20_80_split, random_strat_split = ways2split_data[key_]
        # Copy a df for splitting
#        df_tmp = df['Iodide'].copy()
        # Now split using existing function
        returned_vars = mk_iodide_ML_testing_and_training_set(df=df.copy(),
                                                    random_20_80_split=random_20_80_split,
                                                    random_strat_split=random_strat_split,
                                                    testing_features=df.columns.tolist(),
#                                                   testing_features=testing_features,
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