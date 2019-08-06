#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
processing/analysis functions for CH3I observations
"""

import sparse2spatial.utils as utils
import sparse2spatial as s2s
import pandas as pd
import numpy as np
import glob

from sparse2spatial.ancillaries2grid_oversample import extract_ancillaries_from_compiled_file


def get_CH3I_obs(target='CH3I', limit_depth_to=20,):
    """
    Get the raw observations from HalOcAt database
    """
    # File to use
    filename = 'HC_seawater_concs_above_{}m.csv'.format(limit_depth_to)
    # Where is the file?
    data_root = utils.get_file_locations('data_root')
    folder = '{}/{}/inputs/'.format(data_root, target)
    df = pd.read_csv(folder+filename)
    # Variable name? - Just use one of the values for now
    Varname = 'CH3I (pM)'
    # Assume using coord variables for now
    LatVar1 = 'Sample start latitude (+ve N)'
    LonVar1 = 'Sample start longitude (+ve E)'
    # Add time
    TimeVar1 = 'Date (UTC) and time'
    TimeVar2 = 'Sampling date/time (UT)'
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
        return 'HC_{:0>6}'.format(int(x))
    df['Data_Key_ID'] = df['NEW_INDEX'].map(get_unique_Data_Key_ID)
    # Remove all the NaNs
    t0_shape = df.shape[0]
    df = df.dropna()
    if t0_shape != df.shape[0]:
        pstr = 'WARNING: Dropped obs. (#={}), now have #={} (had #={})'
        print(pstr.format(t0_shape-df.shape[0], df.shape[0], t0_shape))
    return df


def process_obs_and_ancillaries_2_csv(target='CH3I',
                                      file_and_path='./sparse2spatial.rc'):
    """
    Process the observations and extract ancillary variables for these locations
    """
    # Get the bass observations
    df = get_CH3I_obs()
    # Extract the ancillary values for these locations
    df = extract_ancillaries_from_compiled_file(df=df)
    # Save the intermediate file
    folder = utils.get_file_locations('data_root', file_and_path=file_and_path)
    folder += '/{}/inputs/'.format(target)
    filename = 's2s_{}_obs_ancillaries_v0_0_0.csv'.format(target)
    df.to_csv(folder+filename, encoding='utf-8')


def get_processed_df_obs_mod(reprocess_params=False, target='CH3I',
                             filename='s2s_CH3I_obs_ancillaries.csv',
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


def get_ground_surface_CH3I_obs(file_and_path='./sparse2spatial.rc',):
    """
    Get the NOAA/ESRL observations for CH3I (and other species)
    """
    # Location of files
    folder = utils.get_file_locations('data_root', file_and_path=file_and_path)
    folder += '/../NOAA/ground_based/'
    # Get files
    prefix = 'Montzka_sitenumber_'
    files = glob.glob( '{}{}*.txt'.format(folder, prefix) )
    # extract sites from file names
    sites = [i.split(prefix)[-1].split('_')[-1][:-4] for i in files]
    # Update site names to be the same as used elsewhere
    NOAA_name2GAW = {
        'CapeGrim': 'CGO',
        'Alert': 'Alert',
        'MaceHead': 'MHD',
        'PalmerStation': 'Palmer Station',
        'Summit': 'Summit',
        'CapeKumuhaki': 'Cape Kumukahi',
        'MaunaLoa': 'MLO',
        'ParkFallsWisconsin': 'Park Falls Wisconsin',
        'Barrow': 'BRW',
        'SouthPole': 'SPO',
        'TrinidadHead': 'THD',
        'NiwotRidge': 'Niwot Ridge'

    }
    for n in range(len(sites)):
        sites[n] = NOAA_name2GAW[sites[n]]
    # Extract the observations to a DataFrame
    dfs ={}
    for n,site in enumerate(sites):
#        df = pd.read_csv(files[n], delimiter=';')
        df = pd.read_csv(files[n], delimiter=';', header=None, skiprows=1)
        #
        with open(files[n], 'r') as lines:
            header = [i for i in lines][0]
            print(header )
            header = header.split(';')

        dfs[site] = df
    # load summit as a separate .csv file
    site = 'Summit'
    dfs[site] = pd.read_csv(folder +'Montzka_sitenumber_2_Summit.csv')
    # Set flagged values to NaNs
    for n,site in enumerate(sites):
        df = dfs[site]
        df[df==-999] = np.NaN
        dfs[site] = df
    # set a function to map a datetime.datetime from columns
    def add_dt2_df(year=None, month=None, day=None, debug=False
#                   hour=None, min=None
                   ):
        """
        compile a datetime from DataFrame columns
        """
        if debug:
            print( year, month, day)
#        print( year, month, day, hour, min)
        # Set the Date to the middle of the month (#=15) if not known value
        if isinstance(day, type(None)) or ~np.isfinite(day):
            Day = 15
        return datetime.datetime(int(year), int(month), int(day),
#                                int(hour), int(min)
                                )

    # Add a date string
    DateVar= 'Datetime'
    for n,site in enumerate(sites):
        print(site)
        df = dfs[site]
        #
        try:
            year_var = 'year'
            month_var = 'month'
            day_var = 'day'
#            hour_var = 'hour'
#            min_var = 'min'
#            df_tmp = df[[year_var,month_var,day_var,hour_var,min_var]]
            df_tmp = df[[year_var,month_var,day_var]]

            # Now apply the function on the whole dataframe
            df[DateVar] = df_tmp.apply(lambda x: add_dt2_df(year=x[year_var],
                                                            month=x[month_var],
                                                            day=x[day_var],
#                                                            hour=x[hour_var],
#                                                            min=x[min_var],
                                                            ),axis=1)

        except:
            year_var = 'Date'
            month_var = 'Gear'
            day_var = 'Sampling'
#            hour_var = '(UTC)'
#            min_var = 'and'
#            df_tmp = df[[year_var,month_var,day_var,hour_var,min_var]]
            df_tmp = df[[year_var,month_var,day_var]]

            # Now apply the function on the whole dataframe
            df[DateVar] = df_tmp.apply(lambda x: add_dt2_df(year=x[year_var],
                                                        month=x[month_var],
                                                        day=x[day_var],
#                                                        hour=x[hour_var],
#                                                        min=x[min_var],
                                                        ),axis=1)

        #
        dfs[site] = df.copy()
        del df

    return dfs


def expand_NOAA_name(input, invert=True):
    """
    Get the expanded NOAA name of observation station
    """
    NOAA_name2GAW = {
        'CapeGrim': 'CGO',
        'Alert': 'Alert',
        'MaceHead': 'MHD',
        'PalmerStation': 'Palmer Station',
        'Summit': 'Summit',
        'CapeKumuhaki': 'Cape Kumukahi',
        'MaunaLoa': 'MLO',
        'ParkFallsWisconsin': 'Park Falls Wisconsin',
        'Barrow': 'BRW',
        'SouthPole': 'SPO',
        'TrinidadHead': 'THD',
        'NiwotRidge': 'Niwot Ridge'

    }
    # invert
    if invert:
        NOAA_name2GAW = {v: k for k, v in list(NOAA_name2GAW.items())}

    if rtn_dict:
        return NOAA_name2GAW
    else:
        return NOAA_name2GAW[input]


def add_extra_vars_rm_some_data(df=None, target='CH3I',
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
