"""

processing/analysis functions for CHBr3 observations

"""

import sparse2spatial.utils as utils
import sparse2spatial as s2s
import pandas as pd
import numpy as np

from sparse2spatial.ancillaries2grid_oversample import extract_ancillary_obs_from_COMPILED_file

def get_CHBr3_obs(target='CHBr3', limit_depth_to=20,):
    """
    Get the raw observations from HalOcAt database
    """
    # File to use
    filename = 'HC_seawater_concs_above_{}m.csv'.format(limit_depth_to)
    # Where is the file?
    s2s_root = utils.get_file_locations('s2s_root')
    folder = '{}/{}/inputs/'.format(s2s_root, target)
    df = pd.read_csv( folder+filename )
    # Variable name? - Just use one of the values for now
    Varname = 'CHBr3 (pM)'
    # Assume using coord variables for now
    LatVar1 = 'Sample start latitude (+ve N)'
    LonVar1 = 'Sample start longitude (+ve E)'
    # Add time
    TimeVar1 = 'Date (UTC) and time'
    TimeVar2 = 'Sampling date/time (UT)'
    month_var = 'Month'
    dt = pd.to_datetime(df[TimeVar1], format='%Y-%m-%d %H:%M:%S', errors='coerce' )
    df['datetime'] = dt
    def get_month(x):
        return x.month
    df[month_var] = df['datetime'].map(get_month)
    # make sure all values are numeric
    for var in [Varname]+[LatVar1, LonVar1]:
        df.loc[:, var] = pd.to_numeric(df[var].values, errors='coerce')
        # replace flagged values with NaN
        df.replace(999,np.NaN, inplace=True)
        df.replace(-999,np.NaN, inplace=True)
    # Update names to use
    cols2use = ['datetime', 'Month', LatVar1, LonVar1, Varname ]
    name_dict = {
    LatVar1: 'Latitude', LonVar1: 'Longitude', month_var: 'Month', Varname: target
    }
    df = df[cols2use].rename(columns=name_dict)
    # Add a unique identifier
    df['NEW_INDEX'] = range(1, df.shape[0]+1 )
    # Kludge for now to just a name then number
    def get_unique_Data_Key_ID(x):
        return 'HC_{:0>6}'.format( int(x) )
    df['Data_Key_ID'] = df['NEW_INDEX'].map(get_unique_Data_Key_ID)
    # Remove all the NaNs
    t0_shape = df.shape[0]
    df = df.dropna()
    if t0_shape != df.shape[0]:
        pstr = 'WARNING: Dropped obs. (#={}), now have #={} (had #={})'
        print( pstr.format(t0_shape-df.shape[0], df.shape[0], t0_shape) )
    return df


def process_obs_and_ancillaries_2_csv(target='CHBr3',
                                      file_and_path='./sparse2spatial.rc'):
    """
    Process the observations and extract ancillary variables for these locations
    """
    # Get the bass observations
    df = get_CHBr3_obs()
    # Extract the ancillary values for these locations
    df = extract_ancillary_obs_from_COMPILED_file(df=df)
    # Save the intermediate file
    folder = utils.get_file_locations('s2s_root', file_and_path=file_and_path)
    folder += '/{}/inputs/'.format(target)
    filename = 's2s_{}_obs_ancillaries_v0_0_0.csv'.format(target)
    df.to_csv(folder+filename, encoding='utf-8')


def get_processed_df_obs_mod(reprocess_params=False, target='CHBr3',
                             filename='s2s_CHBr3_obs_ancillaries.csv',
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
    # Kludge (temporary) - make Chlorophyll values all floats
#     def mk_float_or_nan(input):
#         try:
#             return float(input)
#         except:
#             return np.nan
#     df['SeaWIFs_ChlrA'] = df['SeaWIFs_ChlrA'].map(mk_float_or_nan)
    # Add ln of iodide too
#    df['ln(Iodide)'] = df['Iodide'].map(np.ma.log)
    # Add SST in Kelvin too
    if 'WOA_TEMP_K' not in df.columns:
        df['WOA_TEMP_K'] = df['WOA_TEMP_K'].values + 273.15
    # Add a flag for coastal values
#     coastal_flagged = 'coastal_flagged'
#     if coastal_flagged not in df.columns:
#         df = get_coastal_flag(df=df)
    # Make sure month is numeric (if not given)
#     month_var = 'Month'
#     NaN_months_bool = ~np.isfinite(df[month_var].values)
#     NaN_months_df = df.loc[NaN_months_bool, :]
#     N_NaN_months = NaN_months_df.shape[0]
#     if N_NaN_months > 1:
#         print_str = 'DataFrame contains NaNs for {} months - '
#         print_str += 'Replacing these with month # 3 months '
#         print_str += 'before (hemispheric) summer solstice'
#         if verbose:
#             print(print_str.format(N_NaN_months))
#         NaN_months_df[month_var] = NaN_months_df.apply(lambda x:
#                                                        set_backup_month_if_unkonwn(
#                                                            lat=x['Latitude'],
#                                                            #main_var=var2use,
#                                                            #var2use=var2use,
#                                                            #
#                                                            #Data_key_ID_=Data_key_ID_,
#                                                            debug=False), axis=1)
#         # Add back into DataFrame
#         df.loc[NaN_months_bool, month_var] = NaN_months_df[month_var].values
    # Re-process the parameterisations (Chance et al etc + ensemble)?
#     if reprocess_params:
#                 # Add predictions from literature
#         df = get_literature_predicted_iodide(df=df)
#         # Add ensemble prediction
#         df = get_ensemble_predicted_iodide(
#             rm_Skagerrak_data=rm_Skagerrak_data
#         )
    return df



def add_extra_vars_rm_some_data(df=None, target='CHBr3',
                                restrict_data_max=False, restrict_min_salinity=False,
#                                use_median_value_for_chlor_when_NaN=False,
#                                median_4MLD_when_NaN_or_less_than_0=False,
#                                median_4depth_when_greater_than_0=False,
                                rm_LOD_filled_data=False,
#                                add_modulus_of_lat=False,
#                                rm_Skagerrak_data=False,
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
        Outlier = utils.get_outlier_value(df, var2use=target, check_full_df_used=False)
        bool = df[target] < Outlier
        df_tmp = df.loc[bool]
        prt_str = 'Removing outlier {} values. (df {}=>{},{})'
        N = int(df_tmp.shape[0])
        if verbose:
            print(prt_str.format(target, Shape0, str(df_tmp.shape), N0-N))
        df = df_tmp
    return df
