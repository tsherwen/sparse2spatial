#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
processing/analysis functions for 'example' observations
"""
import pandas as pd
import numpy as np
import sparse2spatial as s2s
import sparse2spatial.utils as utils
import sparse2spatial.ancillaries2grid_oversample as ancillaries2grid


def get_example_obs(target='example', limit_depth_to=20):
    """
    Get the raw sparse observations from a database...

    Parameters
    -------
    target (str), Name of the target variable (e.g. iodide)
    limit_depth_to (float), depth from sea surface to include data (metres)

    Returns
    -------
    (pd.DataFrame)
    """
    # File to use (example name string...)
    filename = 'HC_seawater_concs_above_{}m.csv'.format(limit_depth_to)
    # Where is the file?
    s2s_root = utils.get_file_locations('s2s_root')
    folder = '{}/{}/inputs/'.format(s2s_root, target)
    df = pd.read_csv(folder+filename)
    # Variable name?
    Varname = 'example (pM)'
    # Assume using coord variables for now
    LatVar1 = '<native latitude name (+ve N)>'
    LonVar1 = '<native longitude name (+ve E)>'
    # Add time
    TimeVar1 = 'native Date and time (UTC)'
    month_var = 'Month'
    dt = pd.to_datetime(
        df[TimeVar1], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    df['datetime'] = dt

    def get_month(x):
        return x.month
    df[month_var] = df['datetime'].map(get_month)
    # Make sure all values are numeric
    for var in [Varname]+[LatVar1, LonVar1]:
        df.loc[:, var] = pd.to_numeric(df[var].values, errors='coerce')
        # replace flagged values with NaN
        df.replace(999, np.NaN, inplace=True)
        df.replace(-999, np.NaN, inplace=True)
    # Update names to use
    cols2use = ['datetime', 'Month', LatVar1, LonVar1, Varname]
    name_dict = {
        LatVar1: 'Latitude', LonVar1: 'Longitude', month_var: 'Month', Varname: target
    }
    df = df[cols2use].rename(columns=name_dict)
    # Add a unique identifier
    df['NEW_INDEX'] = range(1, df.shape[0]+1)
    # Set to a unique string instead of a number

    def get_unique_Data_Key_ID(x):
        return 'HC_{:0>6}'.format(int(x))
    df['Data_Key_ID'] = df['NEW_INDEX'].map(get_unique_Data_Key_ID)
    # Remove all the NaNs and print to screen the change in dataset size
    t0_shape = df.shape[0]
    df = df.dropna()
    if t0_shape != df.shape[0]:
        pstr = 'WARNING: Dropped obs. (#={}), now have #={} (had #={})'
        print(pstr.format(t0_shape-df.shape[0], df.shape[0], t0_shape))
    return df


def process_obs_and_ancillaries_2_csv(target='example', version='v0_0_0'
                                      file_and_path='./sparse2spatial.rc'):
    """
    Process the observations and extract ancillary variables for these locations

    Parameters
    -------
    target (str), Name of the target variable (e.g. iodide)
    version (str), version name/number (e.g. semantic version - https://semver.org/)
    file_and_path (str), folder and filename with location settings as single str

    Returns
    -------
    (None)
    """
    # Get the base observations
    df = get_example_obs()
    # Extract the ancillary values for these locations
    df = ancillaries2grid.extract_ancillaries_from_compiled_file(df=df)
    # Save the intermediate file
    folder = utils.get_file_locations('s2s_root', file_and_path=file_and_path)
    folder += '/{}/inputs/'.format(target)
    filename = 's2s_{}_obs_ancillaries_{}.csv'.format(target, version)
    df.to_csv(folder+filename, encoding='utf-8')


def get_processed_df_obs_mod(target='example', file_and_path='./sparse2spatial.rc'):
    """
    Get the processed observation and model output

    Parameters
    -------
    target (str), Name of the target variable (e.g. iodide)
    file_and_path (str), folder and filename with location settings as single str

    Returns
    -------
    (pd.DataFrame)

    Notes
    -----
    """
    # Read in processed csv file of observations and ancillaries
    folder = utils.get_file_locations('s2s_root', file_and_path=file_and_path)
    folder += '/{}/inputs/'.format(target)
    filename = 's2s_{}_obs_ancillaries.csv'.format(target)
    df = pd.read_csv(folder+filename, encoding='utf-8')
    # Add SST in Kelvin too
    if 'WOA_TEMP_K' not in df.columns:
        df['WOA_TEMP_K'] = df['WOA_TEMP_K'].values + 273.15
    return df


def add_extra_vars_rm_some_data(df=None, target='example', rm_outliers=False,
                                verbose=True, debug=False):
    """
    Add, process, or remove (requested) derivative variables for use with ML code

    Parameters
    -------
    target (str), Name of the target variable (e.g. iodide)
    rm_outliers (bool), remove all the observational points above outlier definition


    Returns
    -------
    (pd.DataFrame)
    """
    # --- Apply choices & Make user aware of choices applied to data
    Shape0 = str(df.shape)
    N0 = df.shape[0]
    # remove the outlier values
    if rm_outliers:
        Outlier = utils.get_outlier_value(
            df, var2use=target, check_full_df_used=False)
        bool = df[target] < Outlier
        df_tmp = df.loc[bool]
        prt_str = 'Removing outlier {} values. (df {}=>{},{})'
        N = int(df_tmp.shape[0])
        if verbose:
            print(prt_str.format(target, Shape0, str(df_tmp.shape), N0-N))
        df = df_tmp
    return df
