"""
processing/analysis functions for CH2Br2 observations
"""

import sparse2spatial.utils as utils
import sparse2spatial as s2s
import pandas as pd
import numpy as np

from sparse2spatial.ancillaries2grid_oversample import extract_ancillaries_from_compiled_file


def get_CH2Br2_obs(target='CH2Br2', limit_depth_to=20,):
    """
    Get the raw observations from HalOcAt database
    """
    # File to use
    filename = 'HC_seawater_concs_above_{}m.csv'.format(limit_depth_to)
    # Where is the file?
    s2s_root = utils.get_file_locations('s2s_root')
    folder = '{}/{}/inputs/'.format(s2s_root, target)
    df = pd.read_csv(folder+filename)
    # Variable name? - Just use one of the values for now
    Varname = 'CH2Br2 (pM)'
    # Assume using coord variables for now
    LatVar1 = 'Sample start latitude (+ve N)'
    LonVar1 = 'Sample start longitude (+ve E)'
    # Add time
    TimeVar1 = 'Date (UTC) and time'
    TimeVar2 = 'Sampling date/time (UT)'
    month_var = 'Month'
    dt = pd.to_datetime(
        df[TimeVar1], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    df['datetime'] = dt

    def get_month(x):
        return x.month
    df[month_var] = df['datetime'].map(get_month)
    # make sure all values are numeric
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
        return 'HC_{:0>6}'.format(int(x))
    df['Data_Key_ID'] = df['NEW_INDEX'].map(get_unique_Data_Key_ID)
    # Remove all the NaNs
    t0_shape = df.shape[0]
    df = df.dropna()
    if t0_shape != df.shape[0]:
        pstr = 'WARNING: Dropped obs. (#={}), now have #={} (had #={})'
        print(pstr.format(t0_shape-df.shape[0], df.shape[0], t0_shape))
    return df


def process_obs_and_ancillaries_2_csv(target='CH2Br2',
                                      file_and_path='./sparse2spatial.rc'):
    """
    Process the observations and extract ancillary variables for these locations
    """
    # Get the bass observations
    df = get_CH2Br2_obs()
    # Extract the ancillary values for these locations
    df = extract_ancillaries_from_compiled_file(df=df)
    # Save the intermediate file
    folder = utils.get_file_locations('s2s_root', file_and_path=file_and_path)
    folder += '/{}/inputs/'.format(target)
    filename = 's2s_{}_obs_ancillaries_v0_0_0.csv'.format(target)
    df.to_csv(folder+filename, encoding='utf-8')


def get_processed_df_obs_mod(reprocess_params=False, target='CH2Br2',
                             filename='s2s_CH2Br2_obs_ancillaries.csv',
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
    folder = utils.get_file_locations('s2s_root', file_and_path=file_and_path)
    folder += '/{}/inputs/'.format(target)
    filename = 's2s_{}_obs_ancillaries.csv'.format(target)
    df = pd.read_csv(folder+filename, encoding='utf-8')
    # Add SST in Kelvin too
    if 'WOA_TEMP_K' not in df.columns:
        df['WOA_TEMP_K'] = df['WOA_TEMP_K'].values + 273.15
    return df


def add_extra_vars_rm_some_data(df=None, target='CH2Br2',
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
