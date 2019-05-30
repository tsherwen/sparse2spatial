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
from sparse2spatial.RFRbuild import mk_testing_training_sets
import sparse2spatial.RFRbuild as build
import sparse2spatial.RFRanalysis as analysis
from sparse2spatial.RFRbuild import build_or_get_models

# Get iodide specific functions

def main():
    """
    Driver for module's man if run directly from command line. unhash
    functionalitliy to call.
    """
    # - Set core local variables
    target= 'CHBr3'

    # - Get the observations? (Not needed for core workflow as also held in RFR_dict)
#    df = get_dataset_processed4ML(target=target, rm_outliers=rm_outliers)

    # - build models with the observations
    RFR_dict = build_or_get_models_CHBr3(rebuild=False, target=target)
    #

    # Get stats ont these models
    stats = analysis.get_core_stats_on_current_models(RFR_dict=RFR_dict,
            target=target, verbose=True, debug=True)

    # Get the top ten models
    topmodels = build.get_top_models(RFR_dict=RFR_dict, stats=stats,
            NO_DERIVED=True, n=10)

    # --- Predict values globally (only use 0.125)
    # extra strig for NetCDF save name
    xsave_str = '_v_0_0_0'
    # make NetCDF predictions from the main array
    save2NetCDF = True
    # resolution to use? (full='0.125x0.125', test at lower e.g. '4x5')
    res = '0.125x0.125'
#    res='4x5'
#    res='2x2.5'
    mk_predictions_for_3D_features(None, res=res, RFR_dict=RFR_dict,
                                    use_updated_predictor_NetCDF=False,
                                    save2NetCDF=save2NetCDF, target=target,
#                                           rm_Skagerrak_data=rm_Skagerrak_data,
                                    models2compare=topmodels,
                                    topmodels=topmodels,
                                    xsave_str=xsave_str, add_ensemble2ds=True )




def build_or_get_models_CHBr3(rm_Skagerrak_data=True, target='CHBr3',
                                       rm_LOD_filled_data=False,
                                       rm_outliers=True,
                                       rebuild=False ):
    """
    Wrapper call to build_or_get_models for sea-surface CHBr3
    """
    # Get the dictionary  of model names and features (specific to iodide)
    model_feature_dict = utils.get_model_features_used_dict(rtn_dict=True)

    # Get the observational dataset prepared for ML pipeline
    df = get_dataset_processed4ML(target=target, rm_outliers=rm_outliers)

    if rebuild:
        RFR_dict = build_or_get_models(save_model_to_disk=True,
#                                    rm_Skagerrak_data=rm_Skagerrak_data,
                                    model_feature_dict=model_feature_dict,
                                    df=df, target=target,
                                    read_model_from_disk=False,
                                    delete_existing_model_files=True )
    else:
        RFR_dict = build_or_get_models(save_model_to_disk=False,
#                                    rm_Skagerrak_data=rm_Skagerrak_data,
                                    model_feature_dict=model_feature_dict,
                                    df=df, target=target,
                                    read_model_from_disk=True,
                                    delete_existing_model_files=False )
    return RFR_dict



def get_dataset_processed4ML(restrict_data_max=False, target='CHBr3',
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
    features_used = None
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
    # Re-index to a single contiguous index
    df['Original Index' ] = df.index.copy()
    N1 = df.shape[0]
    df.index = np.arange( N1 )
    print('WARNING: Reindexed to shape of DataFrame processed for ML ({})'.format(N1))

    # - Add test and training set assignment to columns
#    print( 'WARNING - What testing had been done on training set selection?!' )
    # Choose a sub set of data to exclude from the input data...
#     from sklearn.model_selection import train_test_split
#     targets = df[ target_name ]
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
        returned_vars = mk_testing_training_sets(df=df.copy(),
                                                    target=target,
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


def mk_predictions_for_3D_features(dsA=None, RFR_dict=None, res='4x5',
                                    models_dict=None, features_used_dict=None,
                                    stats=None, folder=None, target='Iodide',
                                    use_updated_predictor_NetCDF=False,
                                    save2NetCDF=False, plot2check=False,
                                    models2compare=[], topmodels=None,
#                                           rm_Skagerrak_data=False,
                                    xsave_str='',
                                    add_ensemble2ds=False,
                                    verbose=True, debug=False):
    """
    Make a NetCDF file of predicted vairables for a given resolution
    """
    # Make sure the core dictionary is provided
    assert (type(RFR_dict) == dict), 'Core variables must be provided as dict (RFR_dict)'
    # Make sure a full list of models was provided
    assert (len(models2compare) > 0), 'List of models to must be provided!'
    # Inc. all the topmodels in the list of models to compare if they have been provided.
    if isinstance(topmodels, type(list)):
        models2compare += topmodels
    # Remove any double ups in list of of models to predict
    models2compare = list(set(models2compare))
    # Get the variables required here
    if isinstance(models_dict, type(None)):
       models_dict = RFR_dict['models_dict']
    if isinstance(features_used_dict, type(None)):
       features_used_dict = RFR_dict['features_used_dict']
    # Get location to save file and set filename
    if isinstance(folder, type(None)):
        folder = utils.get_file_locations('data_root')
    if isinstance(dsA, type(None)):
        filename = 'Oi_prj_feature_variables_{}.nc'.format(res)
        dsA = xr.open_dataset(folder + filename)
    # - Make a dataset of predictions for each model
    ds_l = []
    for modelname in models2compare:
        # get model
        model = models_dict[modelname]
        # get testinng features
        features_used = utils.get_model_features_used_dict(modelname)
        # Make a DataSet of predicted values
        ds_tmp = utils.mk_da_of_predicted_values(dsA=dsA, model=model, res=res,
                                                 modelname=modelname,
                                                 features_used=features_used)
        #  Add attributes to the prediction
        ds_tmp = utils.add_attrs2target_ds(ds_tmp, add_global_attrs=False,
                                           varname=modelname)
        # Save to list
        ds_l += [ds_tmp]
    # Combine datasets
    ds = xr.merge(ds_l)
    # - Also get values for parameterisations
    if target == 'Iodide':
        # Chance et al (2013)
        param = u'Chance2014_STTxx2_I'
        arr = utils.calc_iodide_chance2014_STTxx2_I(dsA['WOA_TEMP'].values)
        ds[param] = ds[modelname]  # use existing array as dummy to fill
        ds[param].values = arr
        # MacDonald et al (2013)
        param = 'MacDonald2014_iodide'
        arr = utils.calc_iodide_MacDonald2014(dsA['WOA_TEMP'].values)
        ds[param] = ds[modelname]  # use existing array as dummy to fill
        ds[param].values = arr
    # Add ensemble to ds too
    if add_ensemble2ds:
        print('WARNING: Using topmodels for ensemble as calculated here')
        var2template = list(ds.data_vars)[0]
        ds = analysis.add_ensemble_avg_std_to_dataset(ds=ds, res=res, target=target,
                                                      RFR_dict=RFR_dict,
                                                      topmodels=topmodels,
                                                      var2template=var2template,
                                                      save2NetCDF=False)
    # Do a quick diagnostic plot?
    if plot2check:
        for var_ in ds.data_vars:
            # plot an annual average
            arr = ds[var_].mean(dim='time')
            AC.map_plot(arr, res=res)
            plt.title(var_)
            plt.show()
    # Add global variables
    ds = utils.add_attrs2target_ds(ds, add_varname_attrs=False)
    # Save to NetCDF
    if save2NetCDF:
        filename = 'Oi_prj_predicted_{}_{}{}.nc'.format(target, res, xsave_str)
        ds.to_netcdf(filename)
    else:
        return ds




if __name__ == "__main__":
    main()
