#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
processing/analysis functions for DMS observations
"""

import sparse2spatial.utils as utils
import sparse2spatial as s2s
import pandas as pd
import numpy as np

from sparse2spatial.ancillaries2grid_oversample import extract_ancillaries_from_compiled_file


def read_in_DMS_file_and_select_data(limit_depth_to=20):
    """
    Read in the DMS file
    """
    # File to use
    filename = 'dms_o5y852.dat'
    skiprows = 10
    sep = '\t'
    encoding = "ISO-8859-1"
    encoding = "utf-8"
    # Where is the file?
    data_root = utils.get_file_locations('data_root')
    folder = '{}/{}/inputs/'.format(data_root, target)
    df = pd.read_csv(folder+filename, skiprows=skiprows, sep=sep, encoding=encoding)
    # Use sdepth for depth
    # sdepth: The depth in meters of the sample that was analyzed for seawater DMS,
    # if given, if not given, a 'missing value' of �999 is given. All samples in
    # this database are assumed to be 'surface' samples that were collected the
    # top 20 meters of the water column.
    df = df.loc[ df['sdepth'] < limit_depth_to , :]
    # save the data
    savename = 'NOAA_seawater_concs_above_{}m.csv'.format(limit_depth_to)
    df.to_csv(folder+savename)


def get_DMS_obs(target='DMS', limit_depth_to=20,):
    """
    Get the raw observations from HalOcAt database
    """
    # File to use
    filename = 'NOAA_seawater_concs_above_{}m.csv'.format(limit_depth_to)
    # Where is the file?
    data_root = utils.get_file_locations('data_root')
    folder = '{}/{}/inputs/'.format(data_root, target)
    df = pd.read_csv(folder+filename)
    # Variable name? - Just use one of the values for now
    Varname = 'swDMS'
    # Assume using coord variables for now
    LatVar1 = 'Lat'
    LonVar1 = 'Lon'
    # Add time
    TimeVar1 = 'DateTime'
    month_var = 'Month'
    format = '%Y-%m-%d %H:%M:%S'
    dt = pd.to_datetime(df[TimeVar1], format=format, errors='coerce')
    df['datetime'] = dt
    # Get month by mapping a local helper function
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
    # Kludge for now to just a name then number
    def get_unique_Data_Key_ID(x):
        return 'NOAA_{:0>6}'.format(int(x))
    df['Data_Key_ID'] = df['NEW_INDEX'].map(get_unique_Data_Key_ID)
    # Remove all the NaNs
    t0_shape = df.shape[0]
    df = df.dropna()
    if t0_shape != df.shape[0]:
        pstr = 'WARNING: Dropped obs. (#={}), now have #={} (had #={})'
        print(pstr.format(t0_shape-df.shape[0], df.shape[0], t0_shape))
    return df


def process_obs_and_ancillaries_2_csv(target='DMS',
                                      file_and_path='./sparse2spatial.rc'):
    """
    Process the observations and extract ancillary variables for these locations
    """
    # Get the bass observations
    df = get_DMS_obs()
    # Extract the ancillary values for these locations
    df = extract_ancillaries_from_compiled_file(df=df)
    # Save the intermediate file
    folder = utils.get_file_locations('data_root', file_and_path=file_and_path)
    folder += '/{}/inputs/'.format(target)
    filename = 's2s_{}_obs_ancillaries_v0_0_0.csv'.format(target)
    df.to_csv(folder+filename, encoding='utf-8')


def get_processed_df_obs_mod(reprocess_params=False, target='DMS',
                             filename='s2s_DMS_obs_ancillaries.csv',
                             rm_Skagerrak_data=False,
                             file_and_path='./sparse2spatial.rc',
                             verbose=True, debug=False):
    """
    Get the processed observation and model output

    Parameters
    -------

    Returns
    -------
    (pd.DataFrame)

    Notes
    -----
    """
    # Read in processed csv file
    folder = utils.get_file_locations('data_root', file_and_path=file_and_path)
    folder += '/{}/inputs/'.format(target)
    filename = 's2s_{}_obs_ancillaries.csv'.format(target)
    df = pd.read_csv(folder+filename, encoding='utf-8')
    # Add SST in Kelvin too
    if 'WOA_TEMP_K' not in df.columns:
        df['WOA_TEMP_K'] = df['WOA_TEMP_K'].values + 273.15
    return df


def add_extra_vars_rm_some_data(df=None, target='DMS',
                                restrict_data_max=False, restrict_min_salinity=False,
                                rm_outliers=False, verbose=True, debug=False):
    """
    Add, process, or remove (requested) derivative variables for use with ML code

    Parameters
    -------

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
