"""

Extract Ancillaries onto a common grid using an oversampling approach

"""
import numpy as np
import pandas as pd
import xarray as xr
import gc

# import AC_tools (https://github.com/tsherwen/AC_tools.git)
import AC_tools as AC

# specific s2s imports
from sparse2spatial.utils import get_file_locations
from sparse2spatial.utils import set_backup_month_if_unknown


def extract_ancillaries_from_external_files(obs_data_df=None,
                                            obs_metadata_df=None,
                                            fill_error_strings_with_NaNs=True,
                                            buffer_CORDS=3, debug=False):
    """
    Get ancillary data for each datapoint in observational dataset - REDUNDENT

    Parameters
    -------
    obs_data_df (pd.DataFrame), DataFrame of observational data
    obs_metadata_df (pd.DataFrame), DataFrame of metadata for observational data
    fill_error_strings_with_NaNs (boolean), fill the error strings with NaNs?
    buffer_CORDS (int), number of buffer coordinates to use for interpolation
    debug (boolean), perform debugging and verbose printing?

    Returns
    -------
    (pd.DataFrame)

    Notes
    -----
     - This function is redundant. Ancillaries should just be extracted directly from
       the ancillary NetCDF. use "extract_ancillaries_from_compiled_file" instead!
    """
    # NOTE: If a new index axis is created, then the index info is lost
#    Data_key_ID = list(set(obs_data_df[ u'Data_Key_ID']))
    Data_key_ID = obs_data_df['Data_Key_ID'].values
    print(len(Data_key_ID), obs_data_df.shape)
    # for testing use only first few...
#    Data_key_ID = Data_key_ID[:15]
    # Intialise master list to sort data
    master_l = []
    # Loop unique data indentifiers
    for n, Data_key_ID_ in enumerate(Data_key_ID):
        # - Get Data_key location
        tmp_df = obs_data_df[obs_data_df['Data_Key_ID'] == Data_key_ID_]
        # Get Obs Lat
        tmp_lat = tmp_df['Latitude'].values[0]
        # Get Obs Lon
        tmp_lon = tmp_df['Longitude'].values[0]
        # Get Obs month
        tmp_month = tmp_df['Month'].values[0]
        try:
            tmp_month = int(tmp_month)
            assert_str = 'Month # must be between 1 and 12'
            assert (tmp_month > 0) and (tmp_month < 13), assert_str
        except:
            # use annual mean
            tmp_month = 0
#            tmp_month = 3
        # Get Obs date
        tmp_date = tmp_df['Date'].values[0]
        # Pring to screen to debug...
        if debug:
            ptr_str = '{} (Lon={:.2f},Lat={:.2f},month={},date={},'.format(
                Data_key_ID_, tmp_lat, tmp_lon, tmp_month, tmp_date
            )
            ptr_str += '%=({:.2f}))'.format(((n+1.)/len(Data_key_ID))*100)
            print(ptr_str)
        # - Now extract Ancillary data
        sub_l = []
        labels = []
        var_ex_str = '>'*5 + ' Getting {} for {}'
        var_st_str = '-'*5 + ' Status of {} extraction: flagged={}'
        # Get surface ocean temperature from WOA climatology
        var_ = 'WOA_TEMP'
        if debug:
            print(var_ex_str.format(var_, Data_key_ID_))
        value, flagged = get_WOA_TEMP_4_loc(lat=tmp_lat, lon=tmp_lon,
                                            month=tmp_month, Data_key_ID_=Data_key_ID_,
                                            buffer_CORDS=buffer_CORDS)
        sub_l += [value]
        labels += [var_]
        sub_l += [flagged]
        labels += [var_+'_flagged']
        if debug:
            print(var_st_str.format(var_, flagged))
        # Get Nitrate from WOA climatology
        var_ = 'WOA_Nitrate'
        if debug:
            print(var_ex_str.format(var_, Data_key_ID_))
        value, flagged = get_WOA_Nitrate_4_loc(lat=tmp_lat, lon=tmp_lon,
                                               month=tmp_month,
                                               Data_key_ID_=Data_key_ID_,
                                               buffer_CORDS=buffer_CORDS)
        sub_l += [value]
        labels += [var_]
        sub_l += [flagged]
        labels += [var_+'_flagged']
        if debug:
            print(var_st_str.format(var_, flagged))
        # Get Salinity from WOA climatology
        var_ = 'WOA_Salinity'
        if debug:
            print(var_ex_str.format(var_, Data_key_ID_))
        value, flagged = get_WOA_Salinity_4_loc(lat=tmp_lat, lon=tmp_lon,
                                                month=tmp_month,
                                                Data_key_ID_=Data_key_ID_,
                                                buffer_CORDS=buffer_CORDS)
        sub_l += [value]
        labels += [var_]
        sub_l += [flagged]
        labels += [var_+'_flagged']
        if debug:
            print(var_st_str.format(var_, flagged))
        # Get Dissolved O2 from WOA climatology
        var_ = 'WOA_Dissolved_O2'
        if debug:
            print(var_ex_str.format(var_, Data_key_ID_))
        value, flagged = get_WOA_Dissolved_O2_4_loc(lat=tmp_lat, lon=tmp_lon,
                                                    month=tmp_month,
                                                    Data_key_ID_=Data_key_ID_,
                                                    buffer_CORDS=buffer_CORDS)
        sub_l += [value]
        labels += [var_]
        sub_l += [flagged]
        labels += [var_+'_flagged']
        if debug:
            print(var_st_str.format(var_, flagged))
        # Get Phosphate from WOA climatology
        var_ = 'WOA_Phosphate'
        if debug:
            print(var_ex_str.format(var_, Data_key_ID_))
        value, flagged = get_WOA_Phosphate_4_loc(lat=tmp_lat, lon=tmp_lon,
                                                 month=tmp_month,
                                                 Data_key_ID_=Data_key_ID_,
                                                 buffer_CORDS=buffer_CORDS)
        sub_l += [value]
        labels += [var_]
        sub_l += [flagged]
        labels += [var_+'_flagged']
        if debug:
            print(var_st_str.format(var_, flagged))
        # Get Silicate from WOA climatology
        var_ = 'WOA_Silicate'
        if debug:
            print(var_ex_str.format(var_, Data_key_ID_))
        value, flagged = get_WOA_Silicate_4_loc(lat=tmp_lat, lon=tmp_lon,
                                                month=tmp_month,
                                                Data_key_ID_=Data_key_ID_,
                                                buffer_CORDS=buffer_CORDS)
        sub_l += [value]
        labels += [var_]
        sub_l += [flagged]
        labels += [var_+'_flagged']
        # Get Bathymetry values from GEBCO (annual avg., accessed via BODC)
        var_ = 'Depth_GEBCO'
        if debug:
            print(var_ex_str.format(var_, Data_key_ID_))
        value, flagged = get_GEBCO_depth_4_loc(lat=tmp_lat, lon=tmp_lon,
                                               month=tmp_month,
                                               Data_key_ID_=Data_key_ID_,
                                               buffer_CORDS=buffer_CORDS/2.)
        sub_l += [value]
        labels += [var_]
        sub_l += [flagged]
        labels += [var_+'_flagged']
        if debug:
            print(var_st_str.format(var_, flagged))
        # Get SeaWIFs/NASA (Oceancolor) climatological values for Chlorophyll
        var_ = 'SeaWIFs_ChlrA'
        if debug:
            print(var_ex_str.format(var_, Data_key_ID_))
        value, flagged = get_SeaWIFs_ChlrA_4_loc(lat=tmp_lat, lon=tmp_lon,
                                                 month=tmp_month,
                                                 Data_key_ID_=Data_key_ID_,
                                                 buffer_CORDS=buffer_CORDS/2)
        sub_l += [value]
        labels += [var_]
        sub_l += [flagged]
        labels += [var_+'_flagged']
        if debug:
            print(var_st_str.format(var_, flagged))
        # Get (monthly) MLD (pt) values from WOA climatology
        var_ = 'WOA_MLDpt'
        if debug:
            print(var_ex_str.format(var_, Data_key_ID_))
        value, flagged = get_MLD_4_loc(lat=tmp_lat, lon=tmp_lon,
                                       month=tmp_month, var2use='pt',
                                       Data_key_ID_=Data_key_ID_,
                                       buffer_CORDS=buffer_CORDS)
        sub_l += [value]
        labels += [var_]
        sub_l += [flagged]
        labels += [var_+'_flagged']
        if debug:
            print(var_st_str.format(var_, flagged))
        # Get (sum and max) MLD (pt) values from WOA climatology
        if debug:
            print(var_ex_str.format(var_+'sum&max', Data_key_ID_))
        listed_output = get_MLD_4_loc(lat=tmp_lat, lon=tmp_lon,
                                      month=tmp_month, var2use='pt',
                                      Data_key_ID_=Data_key_ID_,
                                      buffer_CORDS=buffer_CORDS,
                                      get_max_and_sum_of_values=True)
        # Now max values
        sub_l += [listed_output[0]]
        labels += [var_+'_max']
        sub_l += [listed_output[1]]
        labels += [var_+'_max_flagged']
        # Now sum values
        sub_l += [listed_output[2]]
        labels += [var_+'_sum']
        sub_l += [listed_output[3]]
        labels += [var_+'_sum_flagged']
        if debug:
            print(var_st_str.format(var_, 'list:'), listed_output)
        # - Get MLD (pd) values...
        var_ = 'WOA_MLDpd'
        if debug:
            print(var_ex_str.format(var_, Data_key_ID_))
        value, flagged = get_MLD_4_loc(lat=tmp_lat, lon=tmp_lon,
                                       month=tmp_month, var2use='pd',
                                       Data_key_ID_=Data_key_ID_,
                                       buffer_CORDS=buffer_CORDS)
        sub_l += [value]
        labels += [var_]
        sub_l += [flagged]
        labels += [var_+'_flagged']
        if debug:
            print(var_st_str.format(var_, Data_key_ID_))
        # Get (sum and max) MLD (pd) values from WOA climatology
        if debug:
            print(var_ex_str.format(var_+'sum&max', Data_key_ID_))
        listed_output = get_MLD_4_loc(lat=tmp_lat, lon=tmp_lon,
                                      month=tmp_month, var2use='pd',
                                      Data_key_ID_=Data_key_ID_,
                                      buffer_CORDS=buffer_CORDS,
                                      get_max_and_sum_of_values=True)
        # Now max values
        sub_l += [listed_output[0]]
        labels += [var_+'_max']
        sub_l += [listed_output[1]]
        labels += [var_+'_max_flagged']
        # Now sum values
        sub_l += [listed_output[2]]
        labels += [var_+'_sum']
        sub_l += [listed_output[3]]
        labels += [var_+'_sum_flagged']
        if debug:
            print(var_st_str.format(var_, 'list:'), listed_output)
        # Get MLD (vd) values from WOA climatology
        var_ = 'WOA_MLDvd'
        if debug:
            print(var_ex_str.format(var_, Data_key_ID_))
        value, flagged = get_MLD_4_loc(lat=tmp_lat, lon=tmp_lon,
                                       month=tmp_month, var2use='vd',
                                       Data_key_ID_=Data_key_ID_,
                                       buffer_CORDS=buffer_CORDS)
        sub_l += [value]
        labels += [var_]
        sub_l += [flagged]
        labels += [var_+'_flagged']
        # Get (sum and max) MLD (vd) values...
        if debug:
            print(var_ex_str.format(var_+'sum&max', Data_key_ID_))
        listed_output = get_MLD_4_loc(lat=tmp_lat, lon=tmp_lon,
                                      month=tmp_month, var2use='vd',
                                      Data_key_ID_=Data_key_ID_,
                                      buffer_CORDS=buffer_CORDS,
                                      get_max_and_sum_of_values=True)
        # Now max values
        sub_l += [listed_output[0]]
        labels += [var_+'_max']
        sub_l += [listed_output[1]]
        labels += [var_+'_max_flagged']
        # Now sum values
        sub_l += [listed_output[2]]
        labels += [var_+'_sum']
        sub_l += [listed_output[3]]
        labels += [var_+'_sum_flagged']
        if debug:
            print(var_st_str.format(var_, 'list:'), listed_output)
        # Get DOC values from WOA climatology
        var_ = 'DOC'
        if debug:
            print(var_ex_str.format(var_, Data_key_ID_))
        value, flagged = get_DOC_4_loc(lat=tmp_lat, lon=tmp_lon,
                                       month=tmp_month, var2use='DOCmdl_avg',
                                       Data_key_ID_=Data_key_ID_,
                                       buffer_CORDS=buffer_CORDS)
        sub_l += [value]
        labels += [var_]
        sub_l += [flagged]
        labels += [var_+'_flagged']
        if debug:
            print(var_st_str.format(var_, Data_key_ID_))
        # Get DOC values from WOA climatology
        var_ = 'DOCaccum'
        if debug:
            print(var_ex_str.format(var_, Data_key_ID_))
        value, flagged = get_DOC_accum_4_loc(lat=tmp_lat, lon=tmp_lon,
                                             month=tmp_month, var2use='DOCaccum_avg',
                                             Data_key_ID_=Data_key_ID_,
                                             buffer_CORDS=buffer_CORDS)
        sub_l += [value]
        labels += [var_]
        sub_l += [flagged]
        labels += [var_+'_flagged']
        if debug:
            print(var_st_str.format(var_, Data_key_ID_))
        # Get Productivity values from WOA climatology
        var_ = 'Prod'
        if debug:
            print(var_ex_str.format(var_, Data_key_ID_))
        value, flagged = get_Prod_4_loc(lat=tmp_lat, lon=tmp_lon,
                                        month=tmp_month, var2use='vgpm',
                                        Data_key_ID_=Data_key_ID_,
                                        buffer_CORDS=buffer_CORDS)
        sub_l += [value]
        labels += [var_]
        sub_l += [flagged]
        labels += [var_+'_flagged']
        if debug:
            print(var_st_str.format(var_, Data_key_ID_))
        # Get Productivity values from WOA climatology
        var_ = 'SWrad'
        if debug:
            print(var_ex_str.format(var_, Data_key_ID_))
        value, flagged = get_RAD_4_loc(lat=tmp_lat, lon=tmp_lon,
                                       month=tmp_month, var2use='SWDN',
                                       buffer_CORDS=buffer_CORDS,
                                       Data_key_ID_=Data_key_ID_)
        sub_l += [value]
        labels += [var_]
        sub_l += [flagged]
        labels += [var_+'_flagged']
        if debug:
            print(var_st_str.format(var_, Data_key_ID_))
        # - To extract other database... just add here with a label
        # NOTE: lists (label, sub_l) must be the same length for pd processing.
        # - Add databases to list...
        # save sub_l to master list
        master_l += [sub_l]
        # Garbage collect
        gc.collect()
    # Garbage collect
    gc.collect()
    # - Construct into Dataframe and combine
    df = pd.DataFrame(master_l)
    df.columns = labels
    df = df.T
    df.columns = Data_key_ID
    df = df.T
    df.to_csv('Extracted_ancillary_DATA.csv')
    # Use the dataframe's own Data_key_ID list.
    obs_data_df.index = obs_data_df['Data_Key_ID']
    # Drop the 'Data_key_ID' from obs_data_df| to remove issues with indexing
    obs_data_df = obs_data_df.drop(['Data_Key_ID'], axis=1)
    # Combine new rows and old rows...
    obs_data_df = pd.concat([df, obs_data_df], axis=1, join_axes=[df.index])
    # Restore the 'Data_key_ID' column for referencing...
    obs_data_df['Data_Key_ID'] = obs_data_df.index
    # Restore index to just be a list of numbers (to allow save out)
    obs_data_df.index = list(range(len(obs_data_df)))
    # fill "--"/ error strings with NaNs.
    if fill_error_strings_with_NaNs:
        vars2fill = ('SeaWIFs_ChlrA', )
        for fill_var in vars2fill:
            df[fill_var] = pd.to_numeric(df[fill_var], errors='coerce')
    return obs_data_df


def get_ancillaries4df_locs(df=None,
                            get_Chance_multi_vars=False,
                            df_lar_var='lat', df_lon_var='lon',
                            df_time_var='month'):
    """
    Extract ancillary variables for a given lat, lon, and time

    Parameters
    -------
    df (pd.DataFrame), DataFrame of observational data
    get_Chance_multi_vars (boolean), get the extra ancillary values needed?
    fill_error_strings_with_NaNs (boolean), fill the error strings with NaNs?
    df_lar_var (str), variable name in DataFrame for latitude
    df_lon_var (str), variable name in DataFrame for longitude
    df_time_var (str), variable name in DataFrame for time (month)

    Returns
    -------
    (pd.DataFrame)
    """
    # To extract ancillary variables only lat, lon, and time needed
    # -- Local variables
    TEMP_K_var = 'WOA_TEMP_K'
    TEMP_var = 'WOA_TEMP'
    # - Map function to extract each required variable
    # dictionary of functions and names
    funcs2cycle = {
        TEMP_var: get_WOA_TEMP_4_loc,
        'WOA_Nitrate': get_WOA_Nitrate_4_loc,
        'WOA_Salinity': get_WOA_Salinity_4_loc,
        'Depth_GEBCO': get_GEBCO_depth_4_loc,
        'SeaWIFs_ChlrA': get_SeaWIFs_ChlrA_4_loc,
    }
    # Loop dictionary of functions
    for key_ in list(funcs2cycle.keys()):
        # Apply function to DataFrame.
        df[key_] = df.apply(lambda x: funcs2cycle[key_](
            lat=x['lat'], lon=x['lon'], month=int(x['month']), rtn_flag=False
        ), axis=1)
    # Add WOA_TEMP_K
    if TEMP_K_var not in df.columns:
        df[TEMP_K_var] = df[TEMP_var].values + 273.15
    # Also consider variables for
    if consider_extra_vars4Rosies_multivariate_eqn:
        sumMLDpt_sum_var = 'WOA_MLDpt_sum'
        #
        df[sumMLDpt_sum_var] = df.apply(lambda x: get_MLD_4_loc(
            just_return_sum=True,
            lat=x['lat'], lon=x['lon'], month=int(x['month']), rtn_flag=False
        ), axis=1)
    # Return updated df
    return df


def mk_predictor_variable_csv(res='4x5', month=9,
                              df_lar_var='lat', df_lon_var='lon', df_time_var='month',
                              get_Chance_multi_vars=False):
    """
    Make a predictor array to pass as input for a statistical model

    Parameters
    -------
    df (pd.DataFrame), DataFrame of observational data
    get_Chance_multi_vars (boolean), get the extra ancillary values needed?
    fill_error_strings_with_NaNs (boolean), fill the error strings with NaNs?
    df_lar_var (str), variable name in DataFrame for latitude
    df_lon_var (str), variable name in DataFrame for longitude
    df_time_var (str), variable name in DataFrame for time (month)
    month (int): month number to use (1=jan, 12=dec)
    res (str), horizontal resolution of dataset (e.g. 4x5)

    Returns
    -------
    (pd.DataFrame)
    """
    # - Local variables
    # Make array of lon, lat, time
    df = AC.get_2D_df_of_lon_lats_and_time(df_lar_var=df_lar_var, res=res,
                                           df_lon_var=df_lon_var, df_time_var=df_time_var,
                                           month=month)
    # - Ocean consider ocean grid boxes (e.g. mask for values not in ocean )
    # get Land / Water /Ice fraction
    df['LWI'] = df.apply(lambda x: AC.get_LWI(lat=x[df_lar_var],
                                              lon=x[df_lon_var], date=x[df_time_var],
                                              res=res), axis=1)
    # Drop values not over ocean
    df = df[df['LWI'] == 0]
    # Remove LWI from df
    columns = list(df.columns)
    columns.pop(columns.index('LWI'))
    df = df[columns]
    # - Extract Ancillary values for lat, lons, and times
    df = get_ancillaries4df_locs(df=df,
                                 get_Chance_multi_vars=get_Chance_multi_vars)
    # - Save csv
    filename = 'Oi_prj_predictor_values_{}_month_num_{}.csv'.format(res, month)
    df.to_csv(filename)


# ---------------------------------------------------------------------------
# ------------------- Function to bulk extract ancillaries for NetCDF -------
# ---------------------------------------------------------------------------
def extract_ancillaries_from_compiled_file(df=None, debug=False):
    """
    Get ancillary data for each datapoint in observational dataset

    Parameters
    -------
    df (pd.DataFrame), DataFrame of observational data
    debug (boolean), perform debugging and verbose printing?

    Returns
    -------
    (pd.DataFrame)
    """
    # --- local variables
    # file ancillary data as a xarray Dataset
    res = '0.125x0.125'  # Use Nature res. run for analysis
    data_root = get_file_locations('data_root')
    filename = 'Oi_prj_feature_variables_{}.nc'.format(res)
    dsA = xr.open_dataset(data_root + filename)
    # Get list of site IDs...
    # NOTE: if a new index axis is created, then the index info is lost
    Data_key_ID = df['Data_Key_ID'].values
    print(len(Data_key_ID), df.shape)
    # for testing use only first few...
#    Data_key_ID = Data_key_ID[:15]
    # Initialise a DataFrame to sort ancillary data
    dfA = pd.DataFrame()
    # Loop unique data indentifiers
    for n, Data_key_ID_ in enumerate(Data_key_ID):
        # - Get Data_key location
        tmp_df = df[df['Data_Key_ID'] == Data_key_ID_]
        # Get Obs Lat
        tmp_lat = tmp_df['Latitude'].values[0]
        # Get Obs Lon
        tmp_lon = tmp_df['Longitude'].values[0]
        # Get Obs month
        tmp_month = tmp_df['Month'].values[0]
        try:
            tmp_month = int(tmp_month)
            assert_str = 'Month # must be between 1 and 12'
            assert (tmp_month > 0) and (tmp_month < 13), assert_str
        except:
            # use annual mean
            tmp_month = 0
        # Get Obs date or datetime
        try:
            tmp_date = tmp_df['Date'].values[0]
        except:
            tmp_date = tmp_df['datetime'].values[0]
        # Print to screen to if debugging...
        if debug:
            ptr_str = '{:<20} (Lon={:.2f},Lat={:.2f},month={},date={},'
            ptr_str = ptr_str.format(
                Data_key_ID_, tmp_lat, tmp_lon, tmp_month, tmp_date
            )
            ptr_str += '%=({:.2f}))'.format(((n+1.)/len(Data_key_ID))*100)
            print(ptr_str)
        # - Now extract Ancillary data
        # use the mid season month if not provided
        if tmp_month == 0:
            tmp_month = set_backup_month_if_unknown(lat=tmp_lat,)
        # Select data for month
        ds = dsA.sel(time=(dsA['time.month'] == tmp_month))
        # Select for location
        ds = ds.sel(lon=tmp_lon, lat=tmp_lat, method='nearest')
        # Remove time (all values have only 1 time (confirm with asssert)
        assert len(ds.time) == 1, 'Only 1 time should be selected!'
        ds = ds.mean(dim='time')
        # Convert to pandas series and achieve to DataFrame
        dfA[Data_key_ID_] = ds.to_array().to_pandas()
    gc.collect()
    # - Construct into Dataframe and combine
    dfA = dfA.T
    dfA.to_csv('Extracted_ancillary_DATA.csv')
    # Use dataframe's own Data_key_ID list.
    df.index = df['Data_Key_ID']
    # Drop the 'Data_key_ID' from df| to remove issues with indexing
    df = df.drop(['Data_Key_ID'], axis=1)
    # combine new rows and old rows...
    df = pd.concat([dfA, df], axis=1, join_axes=[dfA.index])
    # Restore the 'Data_key_ID' column for referencing...
    df['Data_Key_ID'] = df.index
    # Restore index to just be a list of numbers (to allow save out)
    df.index = list(range(len(df)))
    # fill "--"/ error strings with NaNs.
    return df


def mk_array_of_indices4locations4res(res='4x5', df_lar_var='lat', df_lon_var='lon',
                                      df_time_var='month'):
    """
    Make a .csv to store indices to extract location data from

    Parameters
    -------
    df (pd.DataFrame), DataFrame of observational data
    df_lar_var (str), variable name in DataFrame for latitude
    df_lon_var (str), variable name in DataFrame for longitude
    df_time_var (str), variable name in DataFrame for time (month)
    res (str), horizontal resolution of dataset (e.g. 4x5)

    Returns
    -------
    (None)
    """
    # - Get all locations to extract for
    if res == '0.5x0.5':
        lons, lats, alt = AC.get_latlonalt4res(wd=wd, res=res,
                                               filename='EMEP.geos.1x1.nc',)
    #
    else:
        lons, lats, alt = AC.get_latlonalt4res(res=res)

    # - Make an array with all lats and lons of interest (for all months)
    df = AC.get_2D_df_of_lon_lats_and_time(df_lar_var=df_lar_var, res=res,
                                           df_lon_var=df_lon_var,
                                           df_time_var=df_time_var,
                                           lats=lats, lons=lons,
                                           add_all_months=False)
    del df['month']
    # Add nearest 1x1 WOA indices to array (Nitrate is 1x1)
    lon2ind, lat2ind = get_WOA_array_1x1_indices(lons=lons, lats=lats)
    df['WOA_1x1_LON'] = df['lon'].map(lon2ind)
    df['WOA_1x1_LAT'] = df['lat'].map(lat2ind)
    # Add nearest 0.25x0.25 WOA indices to array (TEMP+SAL are 0.25x0.25)
    lon2ind, lat2ind = get_WOA_array_025x025_indices(lons=lons, lats=lats)
    df['WOA_025x025_LON'] = df['lon'].map(lon2ind)
    df['WOA_025x025_LAT'] = df['lat'].map(lat2ind)
    # Add DOC
    lon2ind, lat2ind = get_DOC_array_1x1_indices(lons=lons, lats=lats)
    df['DOC_1x1_LON'] = df['lon'].map(lon2ind)
    df['DOC_1x1_LAT'] = df['lat'].map(lat2ind)
    # DOCaccum
    lon2ind, lat2ind = get_DOC_accum_1x1_indices(lons=lons, lats=lats)
    df['DOCaccum_1x1_LON'] = df['lon'].map(lon2ind)
    df['DOCaccum_1x1_LAT'] = df['lat'].map(lat2ind)
    # Add GEBCO
    lon2ind, lat2ind = get_GEBCO_array_1min_indices(lons=lons, lats=lats)
    df['GEBCO_1min_LON'] = df['lon'].map(lon2ind)
    df['GEBCO_1min_LAT'] = df['lat'].map(lat2ind)
    # MLD
    lon2ind, lat2ind = get_WOA_MLD_array_1x1_indices(lons=lons, lats=lats)
    df['WOA_MLD_1x1_LON'] = df['lon'].map(lon2ind)  # is this correct
    df['WOA_MLD_1x1_LAT'] = df['lat'].map(lat2ind)
    # Add (SeaWIFs) Chlorophyll
    lon2ind, lat2ind = get_SeaWIFs_ChlrA_array_9x9km_indices(lons=lons,
                                                             lats=lats)
    df['SeaWIFs_ChlrA_9km_LON'] = df['lon'].map(lon2ind)
    df['SeaWIFs_ChlrA_9km_LAT'] = df['lat'].map(lat2ind)
    # Production
    lon2ind, lat2ind = get_Prod_array_1min_indices(lons=lons, lats=lats)
    df['Prod_1min_LON'] = df['lon'].map(lon2ind)
    df['Prod_1min_LAT'] = df['lat'].map(lat2ind)
    # Phosphate , dissolved oxygen? - just use WOA 1x1
    # SWrad
    lon2ind, lat2ind = get_RAD_array_1_9x1_9_indices(lons=lons, lats=lats)
    df['SWrad_1_9_LON'] = df['lon'].map(lon2ind)
    df['SWrad_1_9_LAT'] = df['lat'].map(lat2ind)
    #  - Save the dataframes of indieces to disk
    df.to_csv('Oi_prj_indices4feature_variable_inputs_{}.csv'.format(res))


def get_WOA_array_1x1_indices(lons=None, lats=None, month=9, debug=False):
    """
    Get the indices for given lats and lons in 1x1 WAO files

    Parameters
    -------
    lons (np.array), list of Longitudes to use for spatial extraction
    lats (np.array), list of latitudes to use for spatial extraction
    month (int): month number to use (1=jan, 12=dec)
    debug (boolean), perform debugging and verbose printing?

    Returns
    -------
    (None)
    """
    # Set folder that files are in (using nitrate arrays)
    folder = get_file_locations('WOA_2013') + '/Nitrate_1x1/'
    # Select the correct file (abituaryily using September )
    filename = 'woa13_all_n{:0>2}_01.nc'.format(month)
    # Open file
    with Dataset(folder+filename, 'r') as rootgrp:
        # Get index of cell mid point closest to obs lat
        #        half_grid_cell = (rootgrp['lat_bnds'][0][0]-rootgrp['lat_bnds'][0][1])/2
        #        file_latc = rootgrp['lat']+abs(half_grid_cell)
        # Use lower-left coordinate system
        file_lat = rootgrp['lat']
        # Get index of cell mid point closest to obs lat
#        half_grid_cell = (rootgrp['lon_bnds'][0][0]-rootgrp['lon_bnds'][0][1])/2
#        file_lonc = rootgrp['lon']+abs(half_grid_cell)
        # Use lower-left coordinate system
        file_lon = rootgrp['lon']
        # THIS IS THE SLOW BIT - OPTIMISE?
#        lat_ind = [ AC.find_nearest_value( file_latc, i ) for i in lats ]
#        lon_ind = [ AC.find_nearest_value( file_lonc, i ) for i in lons ]
        # Use lower-left coordinate system
        lat_ind = [AC.find_nearest_value(file_lat, i) for i in lats]
        lon_ind = [AC.find_nearest_value(file_lon, i) for i in lons]
        # Check the extraction
        if debug:
            prt_str = 'LAT={}({},IND={})'.format(
                lat, file_latc[lat_ind], lat_ind)
            prt_str += 'LON={}({},IND={})'.format(lon,
                                                  file_lonc[lon_ind], lon_ind)
            print(prt_str, rootgrp[var2use].shape, depth)
    # Convert to dictionaries, then return
    lat2ind = dict(zip(lats, lat_ind))
    lon2ind = dict(zip(lons, lon_ind))
    return lon2ind, lat2ind


def get_GEBCO_array_1min_indices(lons=None, lats=None, month=9, debug=False):
    """
    Get the indices for given lats and lons in 7km GEBCO files

    Parameters
    -------
    lats (np.array): array of latitude degrees north
    lons (np.array): array of longitude degrees east
    month (int): number of month in year (1-12)
    debug (bool): print debug statements

    Returns
    -------
    (list, list)

    Notes
    -----
     - Using the "One Minute Grid", but a 30 second grid is availible.
     - The time resolution for depth NetCDF is annual.
    """
    # var2use='elevation'; buffer_CORDS=2; rtn_flag=True; debug=True
    # Directory?
    folder = get_file_locations('data_root') + '/BODC/'
    # Filename as string
    filename = 'GRIDONE_2D.nc'
    # - Extract data
    # Open file
    with Dataset(folder+filename, 'r') as rootgrp:
        # - get indices of the data array for the provided lat and lon
        # Get index of cell mid point closest to obs lat
        #        latitude_step = abs(rootgrp['lat'][-1])-abs(rootgrp['lat'][-2])
        #        file_latc = rootgrp['lat']+(latitude_step/2)
        # Use lower-left coordinate system
        file_lat = rootgrp['lat']
        # Get index of cell mid point closest to obs lat
#        longitude_step = abs(rootgrp['lon'][-1])-abs(rootgrp['lon'][-2])
#        file_lonc = rootgrp['lon']+(longitude_step/2)
        # Use lower-left coordinate system
        file_lon = rootgrp['lon']
        # THIS IS THE SLOW BIT - OPTIMISE?
#        lat_ind = [ AC.find_nearest_value( file_latc, i ) for i in lats ]
#        lon_ind = [ AC.find_nearest_value( file_lonc, i ) for i in lons ]
        # Use lower-left coordinate system
        lat_ind = [AC.find_nearest_value(file_lat, i) for i in lats]
        lon_ind = [AC.find_nearest_value(file_lon, i) for i in lons]
    # Convert to dictionaries, then return
    lat2ind = dict(zip(lats, lat_ind))
    lon2ind = dict(zip(lons, lon_ind))
    return lon2ind, lat2ind


def get_WOA_array_025x025_indices(lons=None, lats=None, month=9, debug=False):
    """
    Get the indices for given lats and lons in 1x1 WAO files

    Parameters
    -------
    lats (np.array): array of latitude degrees north
    lons (np.array): array of longitude degrees east
    month (int): number of month in year (1-12)
    debug (bool): print debug statements

    Returns
    -------
    (list, list)

    Notes
    -----
    """
    # Set folder that files are in (using temperatures arrays)
    folder = get_file_locations('WOA_2013') + '/Temperature_025x025/'
    # Select the correct file (abituaryily using September )
    # (The file below is a decadal average ("decav"))
    filename = 'woa13_decav_t{:0>2}_04v2.nc'.format(month)
    # Open file
    with Dataset(folder+filename, 'r') as rootgrp:
        # Get index of cell mid point closest to obs lat
        #        half_grid_cell = (rootgrp['lat_bnds'][0][0]-rootgrp['lat_bnds'][0][1])/2
        #        file_latc = rootgrp['lat']+abs(half_grid_cell)
        # Use lower-left coordinate system
        file_lat = rootgrp['lat']
        # Get index of cell mid point closest to obs lat
#        half_grid_cell = (rootgrp['lon_bnds'][0][0]-rootgrp['lon_bnds'][0][1])/2
#        file_lonc = rootgrp['lon']+abs(half_grid_cell)
        # Use lower-left coordinate system
        file_lon = rootgrp['lon']
        # THIS IS THE SLOW BIT - OPTIMISE?
#        lat_ind = [ AC.find_nearest_value( file_latc, i ) for i in lats ]
#        lon_ind = [ AC.find_nearest_value( file_lonc, i ) for i in lons ]
        # Use lower-left coordinate system
        lat_ind = [AC.find_nearest_value(file_lat, i) for i in lats]
        lon_ind = [AC.find_nearest_value(file_lon, i) for i in lons]
        # Check the extraction
        if debug:
            prt_str = 'LAT={}({},IND={})'.format(
                lat, file_latc[lat_ind], lat_ind)
            prt_str += 'LON={}({},IND={})'.format(lon,
                                                  file_lonc[lon_ind], lon_ind)
            print(prt_str, rootgrp[var2use].shape, depth)
    # Convert to dictionaries, then return
    lat2ind = dict(zip(lats, lat_ind))
    lon2ind = dict(zip(lons, lon_ind))
    return lon2ind, lat2ind


def get_RAD_array_1_9x1_9_indices(lons=None, lats=None, month=9, debug=False):
    """
    Get indieces to extract shortwave (SW) radiation

    Parameters
    -------
    lats (np.array): array of latitude degrees north
    lons (np.array): array of longitude degrees east
    month (int): number of month in year (1-12)
    debug (bool): print debug statements

    Returns
    -------
    (list, list)

    Notes
    -----
    """
    # Directory?
    folder = get_file_locations('data_root') + '/GFDL/'
    # Filename as string
    file_str = 'ncar_rad.15JUNE2009_TMS_EDIT.nc'
    # - Open file
    # Using compiled NetCDFs (data was only available as csv.)
    with Dataset(folder+file_str, 'r') as rootgrp:
        # - Cet indices of the data array for the provided lat and lon
        # Get index of cell mid point closest to obs lat
        # NOTE: latitude is -90 to 90
        # (this doesn't effect nearest point extraction approach taken here)
        #        latitude_step = abs(rootgrp['LAT'][-1])-abs(rootgrp['LAT'][-2])
        #        latitude_step = np.diff(rootgrp['LAT'][:], n=1)
        #        file_latc = rootgrp['LAT']+(latitude_step/2)
        # Use lower-left coordinate system
        file_lat = rootgrp['LAT']
        # Get index of cell mid point closest to obs lon
#        longitude_step = abs(rootgrp['LON'][-1])-abs(rootgrp['LON'][-2])
#        file_lonc = rootgrp['LON']+(longitude_step/2)
        # Use lower-left coordinate system
        file_lon = rootgrp['LON']
        # adjust "file_loncs" to actual values (e.g. 0=>360, to -180=>180)
#        adj_f_lonc = file_lonc[:]-180
        adj_f_lon = file_lon[:]
        adj_f_lon[adj_f_lon > 180] = adj_f_lon[adj_f_lon > 180] - 360
        # THIS IS THE SLOW BIT - OPTIMISE?
#        lat_ind = [ AC.find_nearest_value( file_latc, i ) for i in lats ]
#        lon_ind = [ AC.find_nearest_value( adj_f_lonc, i ) for i in lons ]
        # Use lower-left coordinate system
        lat_ind = [AC.find_nearest_value(file_lat, i) for i in lats]
        lon_ind = [AC.find_nearest_value(adj_f_lon, i) for i in lons]
    # Convert to dictionaries, then return
    lat2ind = dict(zip(lats, lat_ind))
    lon2ind = dict(zip(lons, lon_ind))
    return lon2ind, lat2ind


def get_SeaWIFs_ChlrA_array_9x9km_indices(lons=None, lats=None, month=9,
                                          resolution='9km', debug=False):
    """
    Extract SeaWIFS (WOA) climatology value for Chlorophyll A (mg m^-3)

    Parameters
    -------
    lats (np.array): array of latitude degrees north
    lons (np.array): array of longitude degrees east
    month (int): number of month in year (1-12)
    debug (bool): print debug statements
    resolution (str): resolution of SeaWIFs ChlrA files to use

    Returns
    -------
    (list, list)

    Notes
    -----
     - Using 9km files. 4km files are available.
    """
    # - Extract files
    # Detail on naming convention of files:
    # For a Level-3 binned data product, the form of the name of the main file
    # is iyyyydddyyyyddd.L3b_ttt, where where i is the instrument identifier
    # (S for SeaWiFS, A for Aqua MODIS, T for Terra MODIS, O for OCTS, C for
    # CZCS), yyyydddyyyyddd are the concatenated digits for the GMT year and
    # day of the year of the start and end days of the binning period, and ttt
    # is a code for the binning period length, resolution and product. Binning
    # period codes are DAY, 8D, MO, and YR. For daily products, only the year
    # and day of the data are used; i.e., yyyyddd. Subordinate files have an
    # extension xff appended to the name, where ff is a file number starting
    # from 00, with on subordinate file for each geophysical parameter. Note
    # that the "day of the year represents the dataday. (full details at URL)
    # https://oceancolor.gsfc.nasa.gov/docs/format/Ocean_Level-3_Binned_Data_Products.pdf
    # Kludge - no annual values, so just use a fix SH/NH month for now.
    if month == 0:
        month_ = set_backup_month_if_unknown(lat=lat, main_var='ChlrA',
                                             var2use=var2use, Data_key_ID_=Data_key_ID_)
    else:
        month_ = month
    # Directory?
    folder = get_file_locations('data_root') + '/SeaWIFS/'
    # Filename as string
    file_Str = 'S*.L3m_MC_*{}*'.format(resolution)
    # get SeaWIFS Files
    files = glob.glob(folder+file_Str)
#    print files
    # Loop
    dates_for_files = []
    for file in files:
        #
        filename = file.split('/')[-1]
        # extract start date and calculate datetime
        start_year = int(filename[1:5])
        start_doy = int(filename[5:8])
        sdate = AC.add_days(datetime.datetime(start_year, 1, 1), start_doy-1)
        # extract end date and calculate datetime
        end_year = int(filename[8:12])
        end_doy = int(filename[12:15])
        edate = AC.add_days(datetime.datetime(end_year, 1, 1), end_doy-1)
        # add dates to list
        dates_for_files += [[sdate, edate]]
    dates = dates_for_files
    months = [i[0].month for i in dates_for_files]
    dict_month2filename = dict(list(zip(months, files)))
    # - Extract data for correct month.
    filename = dict_month2filename[month_]
    # Open file
    with Dataset(filename, 'r') as rootgrp:
        # - get indices of the data array for the provided lat and lon
        # Get index of cell mid point closest to obs lat
        #        file_latc = rootgrp['lat']+(rootgrp.latitude_step/2)
        # Use lower-left coordinate system
        file_lat = rootgrp['lat']
        # Get index of cell mid point closest to obs lat
#        file_lonc = rootgrp['lon']+(rootgrp.longitude_step/2)
        # Use lower-left coordinate system
        file_lon = rootgrp['lon']
        # now extract data  (CF array dims, lat, lon)
        # THIS IS THE SLOW BIT - OPTIMISE?
#        lat_ind = [ AC.find_nearest_value( file_latc, i ) for i in lats ]
#        lon_ind = [ AC.find_nearest_value( file_lonc, i ) for i in lons ]
        # Use lower-left coordinate system
        lat_ind = [AC.find_nearest_value(file_lat, i) for i in lats]
        lon_ind = [AC.find_nearest_value(file_lon, i) for i in lons]
    # Convert to dictionaries, then return
    lat2ind = dict(zip(lats, lat_ind))
    lon2ind = dict(zip(lons, lon_ind))
    return lon2ind, lat2ind


def get_DOC_array_1x1_indices(lons=None, lats=None, month=9, debug=False):
    """
    Extract DOC data from annual climatology

    Parameters
    -------
    lats (np.array): array of latitude degrees north
    lons (np.array): array of longitude degrees east
    month (int): number of month in year (1-12)
    debug (bool): print debug statements

    Returns
    -------
    (list, list)

    Notes
    -----
    """
    # Directory?
    folder = get_file_locations('data_root') + '/DOC/'
    # Filename as string
#    file_str = 'DOCmodelSR.nc'
    file_str = 'DOCmodelSR_TMS_EDIT.nc'
#    print folder+file_str
    # - Open file
    # Using compiled NetCDFs (data was only available as csv.)
    with Dataset(folder+file_str, 'r') as rootgrp:
        # - get indices of the data array for the provided lat and lon
        # Get index of cell mid point closest to obs lat
        #        latitude_step = abs(rootgrp['lat'][-1])-abs(rootgrp['lat'][-2])
        #        file_latc = rootgrp['lat']+(latitude_step/2)
        # Use lower-left coordinate system
        file_lat = rootgrp['lat']
        # Get index of cell mid point closest to obs lon
#         longitude_step = np.diff(rootgrp['lon'][:], n=1, axis=-1)
#         if len( set(longitude_step) ) == 1:
#             longitude_step = list(set(longitude_step))[0] / 2
#         else:
#             longitude_step = np.ma.mean(longitude_step) / 2
#             prt_str = 'Warning: unevening steps for {} in {}! - using mean!'
#             print( prt_str.format(var2use, file_str ) )
#        file_lonc = rootgrp['lon']+(longitude_step/2)
        # Use lower-left coordinate system
        file_lon = rootgrp['lon']
        # now extract data  (dims: depth, lon, lat)
        # THIS IS THE SLOW BIT - OPTIMISE?
#        lat_ind = [ AC.find_nearest_value( file_latc, i ) for i in lats ]
#        lon_ind = [ AC.find_nearest_value( file_lonc, i ) for i in lons ]
        # Use lower-left coordinate system
        lat_ind = [AC.find_nearest_value(file_lat, i) for i in lats]
        lon_ind = [AC.find_nearest_value(file_lon, i) for i in lons]
    # Convert to dictionaries, then return
    lat2ind = dict(zip(lats, lat_ind))
    lon2ind = dict(zip(lons, lon_ind))
    return lon2ind, lat2ind


def get_Prod_array_1min_indices(lons=None, lats=None, month=9, debug=False):
    """
    Extract Productivity from Behrenfeld and Falkowski (1997) files (*.csv => NetCDF)

    Parameters
    -------
    lats (np.array): array of latitude degrees north
    lons (np.array): array of longitude degrees east
    month (int): number of month in year (1-12)
    debug (bool): print debug statements

    Returns
    -------
    (list, list)

    Notes
    -----

    Notes: Data extracted from OCRA and extrapolated to poles by Martin Wadley. NetCDF contructed using xarray (xarray.pydata.org) by Tomas Sherwen.
 NOTES from oringal site (http://orca.science.oregonstate.edu/) from 'based on the standard vgpm algorithm. npp is based on the standard vgpm, using modis chl, sst4, and par as input; clouds have been filled in the input data using our own gap-filling software. For citation, please reference the original vgpm paper by Behrenfeld and Falkowski, 1997a as well as the Ocean Productivity site for the data.'
    """
    # Directory?
    folder = get_file_locations('data_root') + '/Productivity/'
    # Filename as string
    filename = 'productivity_behrenfeld_and_falkowski_1997_extrapolated.nc'
    # - Extract data
    # which month to use?
    # Kludge - no annual values, so just use a fix SH/NH month for now.
    if month == 0:
        month_ = set_backup_month_if_unknown(lat=lat, main_var=var2use,
                                             var2use=var2use, Data_key_ID_=Data_key_ID_)
    else:
        month_ = month
    # Open file
    with Dataset(folder+filename, 'r') as rootgrp:
        # - get indices of the data array for the provided lat and lon
        # Get index of cell mid point closest to obs lat
        #        latitude_step = abs(rootgrp['lat'][-1])-abs(rootgrp['lat'][-2])
        #        file_latc = rootgrp['lat']+(latitude_step/2)
        # Use lower-left coordinate system
        file_lat = rootgrp['lat']
        # Get index of cell mid point closest to obs lat
#        longitude_step = abs(rootgrp['lon'][-1])-abs(rootgrp['lon'][-2])
#        file_lonc = rootgrp['lon']+(longitude_step/2)
        # Use lower-left coordinate system
        file_lon = rootgrp['lon']
        # NOTE: pythonic index starts at 0, therefore month index = n_month-1
        # THIS IS THE SLOW BIT - OPTIMISE?
#        lat_ind = [ AC.find_nearest_value( file_latc, i ) for i in lats ]
#        lon_ind = [ AC.find_nearest_value( file_lonc, i ) for i in lons ]
        # Use lower-left coordinate system
        lat_ind = [AC.find_nearest_value(file_lat, i) for i in lats]
        lon_ind = [AC.find_nearest_value(file_lon, i) for i in lons]
    # Convert to dictionaries, then return
    lat2ind = dict(zip(lats, lat_ind))
    lon2ind = dict(zip(lons, lon_ind))
    return lon2ind, lat2ind


def get_WOA_MLD_array_1x1_indices(var2use='pt', lons=None, lats=None, month=9,
                                  debug=False):
    """
    sub-function for get_MLD_4_loc to extract by month.

    Parameters
    -------
    lat (float): latitude degrees north
    lon (float): latitude degress east
    month (int): month number (1-12)
    Data_key_ID_ (str): ID for input data point
    debug (boolean): print out debug information?
    _fill_value (float): fill value in NetCDF
    buffer_CORDS (int): value (degrees lat/lon) added to loc define edges of
        rectangle used for interpolation.
    var2use (str): var to extract from NetCDF

    Returns
    -------
    (float), (str)
    (or list of two sets of above variables if get_max_and_sum_of_values==True)
    """
    # Directory?
    folder = get_file_locations('data_root') + '/WOA_1994/'
    # Filename as string
    file_str = 'WOA94_MLD_1x1_{}_1x1.nc'.format(var2use)
#    print folder+file_str
    # - Open file
    # Using compiled NetCDFs (data was only available as csv.)
    with Dataset(folder+file_str, 'r') as rootgrp:
        # - get indices of the data array for the provided lat and lon
        # Get index of cell mid point closest to obs lat
        #        latitude_step = abs(rootgrp['lat'][-1])-abs(rootgrp['lat'][-2])
        #        file_latc = rootgrp['lat']+(latitude_step/2)
        # Use lower-left coordinate system
        file_lat = rootgrp['lat']
#            print file_latc[:], lat
        # Get index of cell mid point closest to obs lon
#        longitude_step = abs(rootgrp['lon'][-1])-abs(rootgrp['lon'][-2])
#        file_lonc = rootgrp['lon']+(longitude_step/2)
        # Use lower-left coordinate system
        file_lon = rootgrp['lon']
        # adjust "file_loncs" to actual values (e.g. 0=>360, to -180=>180)
#        adj_f_lonc = file_lonc-180
        adj_f_lon = file_lon[:]
        adj_f_lon[adj_f_lon > 180] = adj_f_lon[adj_f_lon > 180] - 360
        # NOTE: pythonic index starts at 0, therefore month index = n_month-1
        # THIS IS THE SLOW BIT - OPTIMISE?
#        lat_ind = [ AC.find_nearest_value( file_latc, i ) for i in lats ]
#        lon_ind = [ AC.find_nearest_value( adj_f_lonc, i ) for i in lons ]
        # Use lower-left coordinate system
        lat_ind = [AC.find_nearest_value(file_lat, i) for i in lats]
        lon_ind = [AC.find_nearest_value(adj_f_lon, i) for i in lons]
    # Convert to dictionaries, then return
    lat2ind = dict(zip(lats, lat_ind))
    lon2ind = dict(zip(lons, lon_ind))
    return lon2ind, lat2ind


def get_DOC_accum_1x1_indices(var2use='DOCaccum_avg', lons=None, lats=None,
                              month=9, debug=False):
    """
    Extract DOC accumulation data from annual climatology

    Parameters
    -------
    lat (float): latitude degrees north
    lon (float): latitude degress east
    depth (float): index of float to extract
    month (int): month number (1-12)
    Data_key_ID_ (str): ID for input data point
    debug (boolean): print out debug information?
    buffer_CORDS (int): value (degrees lat/lon) added to loc define edges of
        rectangle used for interpolation.
    var2use (str): var to extract from NetCDF

    Returns
    -------
    (float), (str)

    Notes
    -----
    """
    # var2use='DOCaccum_avg'; verbose=True; debug=False
    # Only annual values in dataset
    month_ = 0
    # Directory?
    folder = get_file_locations('data_root') + '/DOC/'
    # Filename as string
    file_str = 'DOC_Accum_rate_SR_TMS_EDIT.nc'
    if debug:
        print(folder+file_str)
    # - Open file
    # Using compiled NetCDFs (data was only available as csv.)
    with Dataset(folder+file_str, 'r') as rootgrp:
        # - get indices of the data array for the provided lat and lon
        # Get index of cell mid point closest to obs lat
        #        latitude_step = abs(rootgrp['lat'][-1])-abs(rootgrp['lat'][-2])
        #        file_latc = rootgrp['lat']+(latitude_step/2)
        # Use lower-left coordinate system
        file_lat = rootgrp['lat']
        # Get index of cell mid point closest to obs lat
#        longitude_step = abs(rootgrp['lon'][-1])-abs(rootgrp['lon'][-2])
#        file_lonc = rootgrp['lon']+(longitude_step/2)
        # Use lower-left coordinate system
        file_lon = rootgrp['lon']
        # adjust "file_loncs" to actual values (e.g. 0=>360, to -180=>180)
#        adjusted_file_lonc = file_lonc[:]-180
        adj_f_lon = file_lon[:]
        adj_f_lon[adj_f_lon > 180] = adj_f_lon[adj_f_lon > 180] - 360
        # now extract data  (dims: depth, lon, lat)
        if debug:
            print(rootgrp[var2use].shape,  lat_ind, lon_ind)
        # Add mask for filler values ('--') - N/A (already masked)
#        file_data = np.ma.array(file_data, mask=(file_data == '--') )
        # Select location
        # NOTE: array is LON, LAT (not LAT, LON)!
        # THIS IS THE SLOW BIT - OPTIMISE?
        lat_ind = [AC.find_nearest_value(file_lat, i) for i in lats]
        lon_ind = [AC.find_nearest_value(adj_f_lon, i) for i in lons]
    # Convert to dictionaries, then return
    lat2ind = dict(zip(lats, lat_ind))
    lon2ind = dict(zip(lons, lon_ind))
    return lon2ind, lat2ind


def extract_feature_variables2NetCDF(res='4x5',
                                     interpolate_nans=True,
                                     add_derivative_vars=True):
    """
    Construct a NetCDF of feature variables for testing

    Parameters
    -------
    res (str), horizontal resolution of dataset (e.g. 4x5)
    interpolate_nans (boolean), interpolate to fill the NaN values
    add_derivative_vars (boolean), add the derivative feature variables

    Returns
    -------
    (None)
    """
    import xarray as xr
    # --- Local variables
    # Temp vars variables
    TEMP_K_var = 'WOA_TEMP_K'
    TEMP_var = 'WOA_TEMP'
    # Which vairables to extract (map variables to indexing to use)
    vars2use = {
        TEMP_var: 'WOA_025x025',
        'WOA_Nitrate': 'WOA_1x1',
        'WOA_Phosphate': 'WOA_1x1',
        'WOA_Salinity': 'WOA_025x025',
        'Depth_GEBCO': 'GEBCO_1min',
        'SeaWIFs_ChlrA': 'SeaWIFs_ChlrA_9km',
        'SWrad': 'SWrad_1_9',
        'DOC': 'DOC_1x1',
        'DOCaccum': 'DOCaccum_1x1',
        'Prod': 'Prod_1min',
        'WOA_MLDvd': 'WOA_MLD_1x1',
        'WOA_MLDpt': 'WOA_MLD_1x1',
        'WOA_MLDpd': 'WOA_MLD_1x1',
        'WOA_Dissolved_O2': 'WOA_1x1',
        'WOA_Silicate': 'WOA_1x1',
    }
    # Functions to extract these values?
    funcs4vars = {
        TEMP_var: get_WOA_TEMP4indices,
        'WOA_Nitrate': get_WOA_Nitrate4indices,
        'WOA_Phosphate': get_WOA_Phosphate4indices,
        'WOA_Salinity': get_WOA_Salinity4indices,
        'Depth_GEBCO': get_Depth_GEBCO4indices,
        'SeaWIFs_ChlrA': get_SeaWIFs_ChlrA4indices,
        'SWrad': get_RAD4indices,
        'DOC': get_DOC4indices,
        'DOCaccum': get_DOC_accum4indices,
        'Prod': get_Prod4indices,
        'WOA_MLDvd': extract_MLD_file4indices,
        'WOA_MLDpt': extract_MLD_file4indices,
        'WOA_MLDpd': extract_MLD_file4indices,
        'WOA_Dissolved_O2': get_WOA_Dissolved_O2_4indices,
        'WOA_Silicate': get_WOA_Silicate4indices,
    }
    # Are some of these values only availibe for annual period
    annual_data_only = 'Depth_GEBCO'
    months = range(1, 13)
    dates = [datetime.datetime(1970, i, 1, 0, 0) for i in months]
    # Get indicies for boxes (calculated offline)
    filename = 'Oi_prj_indices4feature_variable_inputs_{}.csv'.format(res)
    df_IND = pd.read_csv(filename)
    # --- Loop variables and add values to NetCDF
    # Loop variables and extract en masse
    ds_vars = []
    for var in vars2use:
        # Get lon and lat indices
        lon_idx = df_IND[vars2use[var]+'_LON']
        lat_idx = df_IND[vars2use[var]+'_LAT']
        # if data is monthly, then extract by month
        if (var not in annual_data_only):
            # Loop and extract by month
            data4var = []
            for n_month, month in enumerate(months):
                # extract data for indicies
                if 'WOA_MLD' in var:
                    vals = funcs4vars[var](lon_idx=lon_idx, lat_idx=lat_idx,
                                           month=month, var2use=var.split('WOA_MLD')[-1])
                else:
                    vals = funcs4vars[var](lon_idx=lon_idx, lat_idx=lat_idx,
                                           month=month)
                # construct DataFrame by unstacking
                lat = df_IND['lat']
                lon = df_IND['lon']
                df = pd.DataFrame(vals, index=[lat, lon]).unstack()
                # convert to Dataset
                lon4ds = list(df.columns.levels[1])
                lat4ds = list(df.index)
                arr = df.values[None, ...]
                date = dates[n_month]
                data4var += [
                    xr.Dataset(
                        data_vars={var: (['time', 'lat', 'lon', ], arr)},
                        coords={'lat': lat4ds, 'lon': lon4ds, 'time': [date]})
                ]
            # Combine months into a single dataframe
            ds_var = xr.concat(data4var, dim='time')
        else:
            # Extract for a single month
            vals = funcs4vars[var](lon_idx=lon_idx, lat_idx=lat_idx,
                                   month=month)
            # construct DataFrame by unstacking
            lat = df_IND['lat']
            lon = df_IND['lon']
            df = pd.DataFrame(vals, index=[lat, lon]).unstack()
            # convert to Dataset
            lon4ds = list(df.columns.levels[1])
            lat4ds = list(df.index)
            arr = df.values
            # save without time dimension
            ds_var = xr.Dataset(data_vars={var: (['lat', 'lon', ], arr)},
                                coords={'lat': lat4ds, 'lon': lon4ds})
        # Save variable to list
        ds_vars += [ds_var]
    # Combine all variables into a single Dataset
    ds = xr.merge(ds_vars)
    # save NetCDF of feature vairables
    ds.to_netcdf('Oi_prj_feature_variables_{}_TEST.nc'.format(res))
    # Interpolate NaNs?
    if interpolate_nans:
        ds = interpolate_NaNs_in_feature_variables(ds, res=res,
                                                   save2NetCDF=False)
        # save interpolated version
        ext_str = '_INTERP_NEAREST'
        filename = 'Oi_prj_feature_variables_{}{}.nc'.format(res, ext_str)
        ds.to_netcdf(filename)
    # Add derived variables?
    if add_derivative_vars:
        ds = add_derivitive_variables(ds)
        # save interpolated version
        ext_str = '_INTERP_NEAREST_DERIVED'
        filename = 'Oi_prj_feature_variables_{}{}.nc'.format(res, ext_str)
        ds.to_netcdf(filename)


# ---------------------------------------------------------------------------
# ------------------- External ancillary data extractors --------------------
# ---------------------------------------------------------------------------
def check_where_extraction_fails(verbose=True, dpi=320, debug=False):
    """
    Check locations where extraction fails - REDUNENT (Now extracting all points)
    """
    # --- Get extracted and observational data
    df = get_processed_df_obs_mod()  # NOTE this df contains values >400nM
    # --- Local variables
    flag4flagging = '_flagged'
    lat_var = 'Latitude'
    lon_var = 'Longitude'
    Data_key_var = 'Data_Key'
    # get flag variables
    flagged_vars = [i for i in df.columns if flag4flagging in i]
    # remove named variables
    flagged_vars.pop(flagged_vars.index('coastal_flagged'))
    # remove the max and sum values for now
    flagged_vars = [i for i in flagged_vars if 'max' not in i]
    flagged_vars = [i for i in flagged_vars if 'sum' not in i]
    # --- Loop and save
    # setup a PDF to save to
    savetitle = 'Oi_prj_check_for_flags_in_extracted_values'
    pdff = AC.plot2pdfmulti(title=savetitle, open=True, dpi=dpi)
    # Now loop
    for f_var in flagged_vars:
        var_ = f_var.split(flag4flagging)[0]
        # Get vars
        df_tmp = df[[var_, f_var, lat_var, lon_var, Data_key_var]]
        #
        flagged_vals = list(set(set(df_tmp[f_var].values)))
        # If more than just null flat (all OK), then plot up
        if len(flagged_vals) > 0:
            if verbose:
                print(f_var, var_, len(flagged_vals),  flagged_vals)
            # remove null flag and loop
            flagged_vals = [i for i in flagged_vals if i != False]
            flagged_vals = [i for i in flagged_vals if i != 'False']
            print(flagged_vals)
            for flag in flagged_vals:
                # plot and figure...
                fig, ax = plt.subplots()
                # Select data
                df_flag = df_tmp.loc[df[f_var] == flag, :]
                lats = df_flag[lat_var].values.tolist()
                lons = df_flag[lon_var].values.tolist()
                N_flag = df_flag.shape[0]
                if debug:
                    print(var_, flag, N_flag, lats, lons)
                title_str = "'{}' flagged as \n '{}' (N={})"
                title = title_str.format(var_, flag, N_flag)
                # plot up
                AC.plot_lons_lats_spatial_on_map(lats=lats, lons=lons,
                                                 title=title,
                                                 split_title_if_too_long=False,
                                                 f_size=10)
                # Save to PDF and close plot
                AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
                plt.close()
    # Save entire pdf
    AC.plot2pdfmulti(pdff, savetitle, close=True, dpi=dpi)


def get_Prod4indices(lat_idx=None, lon_idx=None, month=None,
                     var2use='vgpm', depth=1, verbose=True, debug=False):
    """
    Extract Productivity from Behrenfeld and Falkowski (1997) files (*.csv => NetCDF)

    Parameters
    -------
    lat_idx (list): indicies for latitude
    lon_idx (list): indicies for longitude
    Data_key_ID_ (str): ID for input data point
    debug (boolean): print out debug information?
    var2use (str): var to extract from NetCDF

    Returns
    -------
    (float), (str)

    NOTES
    ---
    Notes: Data extracted from OCRA and extrapolated to poles by Martin Wadley. NetCDF contructed using xarray (xarray.pydata.org) by Tomas Sherwen.
 NOTES from oringal site (http://orca.science.oregonstate.edu/) from 'based on the standard vgpm algorithm. npp is based on the standard vgpm, using modis chl, sst4, and par as input; clouds have been filled in the input data using our own gap-filling software. For citation, please reference the original vgpm paper by Behrenfeld and Falkowski, 1997a as well as the Ocean Productivity site for the data.'
    """
    # Directory?
    folder = get_file_locations('data_root') + '/Productivity/'
    # Filename as string
    filename = 'productivity_behrenfeld_and_falkowski_1997_extrapolated.nc'
    # - Extract data
    # which month to use?
    # Kludge - no annual values, so just use a fix SH/NH month for now.
    if month == 0:
        month_ = set_backup_month_if_unknown(lat=lat, main_var=var2use,
                                             var2use=var2use, Data_key_ID_=Data_key_ID_)
    else:
        month_ = month
    # Open file
    with Dataset(folder+filename, 'r') as rootgrp:
        # NOTE: pythonic index starts at 0, therefore month index = n_month-1
        file_data = rootgrp[var2use][month_-1, ...]
        file_data_ = file_data[lat_idx, lon_idx]
    return file_data_


def get_Prod_4_loc(lat=None, lon=None, month=None, Data_key_ID_=None,
                   var2use='vgpm', buffer_CORDS=5, rtn_flag=True,
                   verbose=True, debug=False):
    """
    Extract Productivity from Behrenfeld and Falkowski (1997) files (*.csv => NetCDF)

    Parameters
    -------
    lat (float): latitude degrees north
    lon (float): latitude degress east
    month (int): month number (1-12)
    Data_key_ID_ (str): ID for input data point
    debug (boolean): print out debug information?
    _fill_value (float): fill value in NetCDF
    buffer_CORDS (int): value (degrees lat/lon) added to loc define edges of
        rectangle used for interpolation.
    var2use (str): var to extract from NetCDF

    Returns
    -------
    (float), (str)

    NOTES
    ---
    Notes: Data extracted from OCRA and extrapolated to poles by Martin Wadley. NetCDF contructed using xarray (xarray.pydata.org) by Tomas Sherwen.
 NOTES from oringal site (http://orca.science.oregonstate.edu/) from 'based on the standard vgpm algorithm. npp is based on the standard vgpm, using modis chl, sst4, and par as input; clouds have been filled in the input data using our own gap-filling software. For citation, please reference the original vgpm paper by Behrenfeld and Falkowski, 1997a as well as the Ocean Productivity site for the data.'
    """
    # Directory?
    folder = get_file_locations('data_root') + '/Productivity/'
    # Filename as string
    filename = 'productivity_behrenfeld_and_falkowski_1997_extrapolated.nc'
    # - Extract data
    # which month to use?
    # Kludge - no annual values, so just use a fix SH/NH month for now.
    if month == 0:
        month_ = set_backup_month_if_unknown(lat=lat, main_var=var2use,
                                             var2use=var2use, Data_key_ID_=Data_key_ID_)
    else:
        month_ = month
    # Open file
    with Dataset(folder+filename, 'r') as rootgrp:
        # - get indices of the data array for the provided lat and lon
        # Get index of cell mid point closest to obs lat
        #        latitude_step = abs(rootgrp['lat'][-1])-abs(rootgrp['lat'][-2])
        #        file_latc = rootgrp['lat']+(latitude_step/2)
        # Use lower-left coordinate system
        file_lat = rootgrp['lat']
        lat_ind = AC.find_nearest_value(file_lat, lat)
        # Get index of cell mid point closest to obs lat
#        longitude_step = abs(rootgrp['lon'][-1])-abs(rootgrp['lon'][-2])
#        file_lonc = rootgrp['lon']+(longitude_step/2)
        # Use lower-left coordinate system
        file_lon = rootgrp['lon']
        lon_ind = AC.find_nearest_value(file_lon, lon)
        # now extract data  (CF array dims, lat, lon)
        if debug:
            print(rootgrp[var2use].shape,  lat_ind, lon_ind, depth)
        # NOTE: pythonic index starts at 0, therefore month index = n_month-1
        file_data = rootgrp[var2use][month_-1, ...]
        file_data_ = file_data[lat_ind, lon_ind]
        # - Return value if present, else interpolate from nearby values...
        if isinstance(file_data_, np.ma.core.MaskedConstant):
            ptr_str = 'interpolating to get {} for {} (buffer_CORDS={})'
            if verbose:
                print(ptr_str.format(var2use, Data_key_ID_, buffer_CORDS))
            # Make sure input data is X vs. Y (e.g. (lon, lat)) !
            # (CF netCDF will be lat, lon)!
            # e.g. (360, 180) for 1x1 vs. (180, 360)
            file_data = file_data.T

            # Try Radial Basis Function Interpolation / Kernel Smoothing
            try:
                # also this dataset's file are lat numebered -90=>90
                # shouldn't this be 90=>-90 according to CF standard?
                # Just reverse lat dimension and index for now.
                #                file_data = file_data[:,::-1]
                #                file_latc = file_latc[::-1]
                # commented out as it won't effect the interpolation
                # Now interpolate and extract data
                file_data_ = AC.interpolate_sparse_grid2value(
                    X_CORDS=file_lonc, Y_CORDS=file_latc, X=lon, Y=lat,
                    XYarray=file_data, buffer_CORDS=buffer_CORDS)
                # ADD flag to say values are interpolated.
                flagged = 'Interpolated (buffer={})'.format(buffer_CORDS)

            # Flag value if interpolated
            except:
                flagged = 'FAILED INTERPOLATION!'
        else:
            flagged = 'False'
    if rtn_flag:
        return file_data_, flagged
    else:
        return file_data_


def get_DOC4indices(lat_idx=None, lon_idx=None, month=None,
                    var2use='DOCmdl_avg', depth=1, verbose=True, debug=False):
    """
    Extract "Elevation relative to sea level" from General Bathymetric Chart
    of the Oceans (GEBCO).

    Parameters
    -------
    lat_idx (list): indicies for latitude
    lon_idx (list): indicies for longitude
    Data_key_ID_ (str): ID for input data point
    debug (boolean): print out debug information?
    var2use (str): var to extract from NetCDF

    Returns
    -------
    (array)

    NOTES
    ---
     - Using the "One Minute Grid", but a 30 second grid is availible.
     - The time resolution for depth NetCDF is annual.
    """
    # Directory?
    folder = get_file_locations('data_root') + '/DOC/'
    # Filename as string
    file_str = 'DOCmodelSR_TMS_EDIT.nc'
    if debug:
        print(folder+file_str)
    # - Open file
    with Dataset(folder+file_str, 'r') as rootgrp:
        # Now extract data (dims: depth, lon, lat)
        file_data = rootgrp[var2use][:][depth, ...]
        # NOTE: array is LON, LAT (not LAT, LON)!
        file_data_ = file_data[lon_idx, lat_idx]
    return file_data_


def get_DOC_4_loc(var2use='DOCmdl_avg', lat=None, lon=None, month=None,
                  depth=1, buffer_CORDS=5, Data_key_ID_=None, verbose=True, debug=False):
    """
    Extract DOC data from annual climatology

    Parameters
    -------
    lat (float): latitude degrees north
    lon (float): latitude degress east
    depth (float): index of float to extract
    month (int): month number (1-12)
    Data_key_ID_ (str): ID for input data point
    debug (boolean): print out debug information?
    buffer_CORDS (int): value (degrees lat/lon) added to loc define edges of
        rectangle used for interpolation.
    var2use (str): var to extract from NetCDF

    Returns
    -------
    (float), (str)

    Notes
    -----
    """
    # Only annual values in dataset
    month_ = 0
    # Directory?
    folder = get_file_locations('data_root') + '/DOC/'
    # Filename as string
#    file_str = 'DOCmodelSR.nc'
    file_str = 'DOCmodelSR_TMS_EDIT.nc'
#    print folder+file_str
    # - Open file
    # Using compiled NetCDFs (data was only available as csv.)
    with Dataset(folder+file_str, 'r') as rootgrp:
        # - get indices of the data array for the provided lat and lon
        # Get index of cell mid point closest to obs lat
        #         latitude_step = abs(rootgrp['lat'][-1])-abs(rootgrp['lat'][-2])
        #         file_latc = rootgrp['lat']+(latitude_step/2)
        # Use lower-left coordinate system
        file_lat = rootgrp['lat']
        lat_ind = AC.find_nearest_value(file_lat, lat)
        # Get index of cell mid point closest to obs lat
#         longitude_step = np.diff(rootgrp['lon'][:], n=1, axis=-1)
#         if len( set(longitude_step) ) == 1:
#             longitude_step = list(set(longitude_step))[0] / 2
#         else:
#             longitude_step = np.ma.mean(longitude_step) / 2
#             prt_str = 'Warning: unevening steps for {} in {}! - using mean!'
#             print( prt_str.format(var2use, file_str ) )
#        longitude_step = abs(rootgrp['lon'][-1])-abs(rootgrp['lon'][-2])
#        file_lonc = rootgrp['lon']+(longitude_step/2)
        # Use lower-left coordinate system
        file_lon = rootgrp['lon']
        lon_ind = AC.find_nearest_value(file_lon, lon)
        # now extract data  (dims: depth, lon, lat)
        if debug:
            print(rootgrp[var2use].shape,  lat_ind, lon_ind, depth)
        file_data = rootgrp[var2use][:][depth, ...]
        file_data_ = file_data[lon_ind, lat_ind]
        # - Return value if present, else interpolate from nearby values...
        if isinstance(file_data_, np.ma.core.MaskedConstant):
            ptr_str = 'interpolating to get {} for {} (buffer_CORDS={})'
            if verbose:
                print(ptr_str.format(var2use, Data_key_ID_, buffer_CORDS))
            # Make sure input data is (lon, lat)! (CF netCDF will be lat, lon)!
            # e.g. (360, 180) for 1x1 vs. (180, 360)
            # N/A dims are lon., lat
            # Try Radial Basis Function Interpolation / Kernel Smoothing
            try:
                # Now interpolate and extract data
                file_data_ = AC.interpolate_sparse_grid2value(
                    X_CORDS=file_lonc, Y_CORDS=file_latc,
                    X=lon, Y=lat,
                    XYarray=file_data, buffer_CORDS=buffer_CORDS)
                # ADD flag to say values are interpolated.
                flagged = 'Interpolated (buffer={})'.format(buffer_CORDS)
                # Try again but with twice the buffer_CORDS
            except:
                try:
                    debug_ = debug
                    # Now interpolate and extract data
                    file_data_ = AC.interpolate_sparse_grid2value(
                        X_CORDS=file_lonc, Y_CORDS=file_latc, X=lon, Y=lat,
                        XYarray=file_data, buffer_CORDS=buffer_CORDS*2,
                        debug=debug_)
                    # Flag value if interpolated
                    flagged = 'Interpolated 2nd time(buffer={})'
                    flagged = flagged.format(buffer_CORDS/2)
                # Flag value if not interpolated
                except:
                    # Flag value if interpolated
                    flagged = 'FAILED INTERPOLATION!'
#                    print '!'*15, flagged, '!'*15
        else:
            flagged = 'False'
    return file_data_, flagged


def get_DOC_accum4indices(lat_idx=None, lon_idx=None, month=None,
                          var2use='DOCaccum_avg', depth=1, verbose=True, debug=False):
    """
    Extract DOC accumulation data from annual climatology

    Parameters
    -------
    lat_idx (list): indicies for latitude
    lon_idx (list): indicies for longitude
    Data_key_ID_ (str): ID for input data point
    debug (boolean): print out debug information?
    var2use (str): var to extract from NetCDF

    Returns
    -------
    (float), (str)

    Notes
    -----
    """
    # Directory?
    folder = get_file_locations('data_root') + '/DOC/'
    # Filename as string
    file_str = 'DOC_Accum_rate_SR_TMS_EDIT.nc'
    if debug:
        print(folder+file_str)
    # Open file
    with Dataset(folder+file_str, 'r') as rootgrp:
        file_data = rootgrp[var2use][:]
        # NOTE: array is LON, LAT (not LAT, LON)!
        file_data_ = file_data[lon_idx, lat_idx]
    return file_data_


def get_DOC_accum_4_loc(var2use='DOCaccum_avg', lat=None, lon=None, month=None,
                        buffer_CORDS=5, Data_key_ID_=None, verbose=True, debug=False):
    """
    Extract DOC accumulation data from annual climatology

    Parameters
    -------
    lat (float): latitude degrees north
    lon (float): latitude degress east
    depth (float): index of float to extract
    month (int): month number (1-12)
    Data_key_ID_ (str): ID for input data point
    debug (boolean): print out debug information?
    buffer_CORDS (int): value (degrees lat/lon) added to loc define edges of
        rectangle used for interpolation.
    var2use (str): var to extract from NetCDF

    Returns
    -------
    (float), (str)

    Notes
    -----
    """
    # var2use='DOCaccum_avg'; buffer_CORDS=5; Data_key_ID_=None; verbose=True; debug=False
    # Only annual values in dataset
    month_ = 0
    # Directory?
    folder = get_file_locations('data_root') + '/DOC/'
    # Filename as string
    file_str = 'DOC_Accum_rate_SR_TMS_EDIT.nc'
#    print folder+file_str
    # - Open file
    # Using compiled NetCDFs (data was only available as csv.)
    with Dataset(folder+file_str, 'r') as rootgrp:
        # - get indices of the data array for the provided lat and lon
        # Get index of cell mid point closest to obs lat
        #        latitude_step = abs(rootgrp['lat'][-1])-abs(rootgrp['lat'][-2])
        #        file_latc = rootgrp['lat']+(latitude_step/2)
        # Use lower-left coordinate system
        file_lat = rootgrp['lat']
        lat_ind = AC.find_nearest_value(file_lat, lat)
        # Get index of cell mid point closest to obs lat
#        longitude_step = abs(rootgrp['lon'][-1])-abs(rootgrp['lon'][-2])
#        file_lonc = rootgrp['lon']+(longitude_step/2)
        # Use lower-left coordinate system
        file_lon = rootgrp['lon']
        # adjust "file_loncs" to actual values (e.g. 0=>360, to -180=>180)
#        adjusted_file_lonc = file_lonc[:]-180
        adj_f_lon = file_lon[:]
        adj_f_lon[adj_f_lon > 180] = adj_f_lon[adj_f_lon > 180] - 360
        lon_ind = AC.find_nearest_value(adj_f_lon, lon)
        # now extract data  (dims: depth, lon, lat)
        if debug:
            print(rootgrp[var2use].shape,  lat_ind, lon_ind)
        file_data = rootgrp[var2use][:]
        # Add mask for filler values ('--') - N/A (already masked)
#        file_data = np.ma.array(file_data, mask=(file_data == '--') )
        # Select location
        file_data_ = file_data[lon_ind, lat_ind]
        # - Return value if present, else interpolate from nearby values...
        if isinstance(file_data_, np.ma.core.MaskedConstant):
            # also this dataset's file are lat numebered -90=>90
            # shouldn't this be 90=>-90 according to CF standard?
            # Just reverse lat dimension and index for now.
            #            file_data = file_data[:,::-1]
            #            file_latc = file_latc[::-1]
            # commented out as it won't effect the interpolation
            # recalculate lat index
            #            lat_ind = AC.find_nearest_value( file_latc, lat )
            ptr_str = 'interpolating to get {} for {} (buffer_CORDS={})'
            if verbose:
                print(ptr_str.format(var2use, Data_key_ID_, buffer_CORDS))
            # Make sure input data is (lon, lat)! (CF netCDF will be lat, lon)!
            # e.g. (360, 180) for 1x1 vs. (180, 360)
            # N/A dims are lon., lat
            # Try Radial Basis Function Interpolation / Kernel Smoothing
            try:
                # Now interpolate and extract data
                file_data_ = AC.interpolate_sparse_grid2value(
                    X_CORDS=adjusted_file_lonc, Y_CORDS=file_latc,
                    X=lon, Y=lat,
                    XYarray=file_data, buffer_CORDS=buffer_CORDS)
                # ADD flag to say values are interpolated.
                flagged = 'Interpolated (buffer={})'.format(buffer_CORDS)
            # Flag value if interpolated
            except:
                try:
                    # Now interpolate and extract data
                    file_data_ = AC.interpolate_sparse_grid2value(
                        X_CORDS=adjusted_file_lonc, Y_CORDS=file_latc,
                        X=lon, Y=lat,
                        XYarray=file_data, buffer_CORDS=buffer_CORDS*2)
                    # ADD flag to say values are interpolated.
                    flagged = 'Interpolated 2nd time (buffer={})'
                    flagged = flagged.format(buffer_CORDS*2)
                except:
                    try:
                        # Now interpolate and extract data
                        file_data_ = AC.interpolate_sparse_grid2value(
                            X_CORDS=adjusted_file_lonc, Y_CORDS=file_latc,
                            X=lon, Y=lat,
                            XYarray=file_data, buffer_CORDS=buffer_CORDS*5)
                        # ADD flag to say values are interpolated.
                        flagged = 'Interpolated 3rd time (buffer={})'
                        flagged = flagged.format(buffer_CORDS*5)
                    except:
                        flagged = 'FAILED INTERPOLATION!'
        else:
            flagged = 'False'
    return file_data_, flagged


def get_RAD4indices(lat_idx=None, lon_idx=None, month=None,
                    var2use='SWDN', verbose=True, debug=False):
    """
    Extract shortwave (SW) radiation for indices

    Parameters
    -------
    lat_idx (list): indicies for latitude
    lon_idx (list): indicies for longitude
    Data_key_ID_ (str): ID for input data point
    debug (boolean): print out debug information?
    var2use (str): var to extract from NetCDF

    Returns
    -------
    (array)
    """
    # Directory?
    folder = get_file_locations('data_root') + '/GFDL/'
    # Filename as string
    file_str = 'ncar_rad.15JUNE2009_TMS_EDIT.nc'
    # Open file
    with Dataset(folder+file_str, 'r') as rootgrp:
        # now extract data  (dims: time, lat, lon)
        # NOTE: pythonic index starts at 0, therefore month index = n_month-1
        file_data = rootgrp[var2use][:][month-1, ...]
        file_data_ = file_data[lat_idx, lon_idx]
    return file_data_


def get_RAD_4_loc(var2use='SWDN', lat=None, lon=None, month=None,
                  Data_key_ID_=None, buffer_CORDS=5, _fill_value=-1000.0,
                  verbose=True, debug=False):
    """
    Extract radiative flux data for given month at surface

    Parameters
    -------
    lat (float): latitude degrees north
    lon (float): latitude degress east
    month (int): month number (1-12)
    Data_key_ID_ (str): ID for input data point
    debug (boolean): print out debug information?
    _fill_value (float): fill value in NetCDF
    buffer_CORDS (int): value (degrees lat/lon) added to loc define edges of
        rectangle used for interpolation.
    var2use (str): var to extract from NetCDF

    Returns
    -------
    (float), (str)
    (or list of two sets of above variables if get_max_and_sum_of_values==True)
    """
    # Directory?
    folder = get_file_locations('data_root') + '/GFDL/'
    # Filename as string
    file_str = 'ncar_rad.15JUNE2009_TMS_EDIT.nc'
    # - Which month to use?
    # Kludge - no annual values, so just use a fix SH/NH month for now.
    if month == 0:
        month_ = set_backup_month_if_unknown(lat=lat, main_var=var2use,
                                             var2use=var2use, Data_key_ID_=Data_key_ID_)
    else:
        month_ = month
    # - Open file
    # Using compiled NetCDFs (data was only available as csv.)
    with Dataset(folder+file_str, 'r') as rootgrp:
        # - get indices of the data array for the provided lat and lon
        # Get index of cell mid point closest to obs lat
        # NOTE: latitude is -90 to 90
        # (this doesn't effect nearest point extraction approach taken here)
        #        latitude_step = abs(rootgrp['LAT'][-1])-abs(rootgrp['LAT'][-2])
        #        latitude_step = np.diff(rootgrp['LAT'][:], n=1)
        #        file_latc = rootgrp['LAT']#+(latitude_step/2)
        #        lat_ind = AC.find_nearest_value( file_latc, lat )
        # Use lower-left coordinate system
        file_lat = rootgrp['LAT']  # +(latitude_step/2)
        lat_ind = AC.find_nearest_value(file_lat, lat)
        # Get index of cell mid point closest to obs lon
#        longitude_step = abs(rootgrp['LON'][-1])-abs(rootgrp['LON'][-2])
#        file_lonc = rootgrp['LON']#+(longitude_step/2)
        # Use lower-left coordinate system
        file_lon = rootgrp['LON']
        # adjust "file_loncs" to actual values (e.g. 0=>360, to -180=>180)
#        adjusted_file_lonc = file_lonc[:]-180
        # Use lower-left coordinate system
        adj_f_lon = file_lon[:]
        adj_f_lon[adj_f_lon > 180] = adj_f_lon[adj_f_lon > 180] - 360
        lon_ind = AC.find_nearest_value(adj_f_lon, lon)
        # now extract data  (dims: time, lat, lon)
        if debug:
            print(rootgrp[var2use].shape,  lat_ind, lon_ind, depth)
        # NOTE: pythonic index starts at 0, therefore month index = n_month-1
        file_data = rootgrp[var2use][:][month_-1, ...]
        file_data_ = file_data[lat_ind, lon_ind]
        # - Return value if present, else interpolate from nearby values...
        if isinstance(file_data_, np.ma.core.MaskedConstant):
            ptr_str = 'interpolating to get {} for {} (buffer_CORDS={})'
            if verbose:
                print(ptr_str.format(var2use, Data_key_ID_, buffer_CORDS))
            # Make sure input data is (lon, lat)! (CF netCDF will be lat, lon)!
            # e.g. (360, 180) for 1x1 vs. (180, 360)
            # N/A dims are lon., lat
            # Try Radial Basis Function Interpolation / Kernel Smoothing
            try:
                # Now interpolate and extract data
                file_data_ = AC.interpolate_sparse_grid2value(
                    X_CORDS=adjusted_file_lonc, Y_CORDS=file_latc,
                    X=lon, Y=lat,
                    XYarray=file_data, buffer_CORDS=buffer_CORDS)
                # ADD flag to say values are interpolated.
                flagged = 'Interpolated (buffer={})'.format(buffer_CORDS)
            # Flag value if interpolated
            except:
                flagged = 'FAILED INTERPOLATION!'
        else:
            flagged = 'False'
    return file_data_, flagged


def get_MLD_4_loc(var2use='pt', lat=None, lon=None, month=None,
                  buffer_CORDS=5, _fill_value=-99.9, Data_key_ID_=None,
                  get_max_and_sum_of_values=False, just_return_sum=False,
                  rtn_flag=True, verbose=True, debug=False):
    """
    Get mixed layer depth from WOA climatology in (m?)

    Parameters
    -------
    lat (float): latitude degrees north
    lon (float): latitude degress east
    month (int): month number (1-12)
    Data_key_ID_ (str): ID for input data point
    debug (boolean): print out debug information?
    get_max_and_sum_of_values (boolean): return annual sum/max instead of
        month value
    _fill_value (float): fill value in NetCDF
    buffer_CORDS (int): value (degrees lat/lon) added to loc define edges of
        rectangle used for interpolation.
    var2use (str): var to extract from NetCDF

    Returns
    -------
    (float), (str)
    (or list of two sets of above variables if get_max_and_sum_of_values==True)

    Notes
    -----
    """
    # Use annual max or sum of values?
    if get_max_and_sum_of_values or just_return_sum:
        # loop months
        file_data_l, flagged_l = [], []
        for month_ in range(1, 13):
            file_data_, flagged = extract_MLD_file_4_loc(var2use=var2use,
                                                         lat=lat, lon=lon, month=month_,
                                                         Data_key_ID_=Data_key_ID_,
                                                         buffer_CORDS=buffer_CORDS,
                                                         _fill_value=_fill_value,
                                                         debug=debug)
            #
            file_data_l += [file_data_]
            flagged_l += [flagged]
        # print warning to log file if values are flagged.
        if any([(i != 'False') for i in flagged_l]):
            flagged_str = 'WARNING {} annualised value inc. flagged data: {}'
            print(flagged_str.format(Data_key_ID_, flagged_l))
        # setup list of values to return (max, then sum)
        rtn_list = []
        # Calculate max...
        rtn_list += [np.ma.max(file_data_l), ','.join(flagged_l)]
        # Calculate sum...
        rtn_list += [np.ma.sum(file_data_l), ','.join(flagged_l)]
        # and return values (or just sum of values)
        if just_return_sum:
            return np.ma.sum(file_data_l)
        else:
            return rtn_list
    else:
        # Kludge - no annual values, so just use a fix SH/NH month for now.
        if month == 0:
            month_ = set_backup_month_if_unknown(lat=lat, main_var='MLD',
                                                 var2use=var2use,
                                                 Data_key_ID_=Data_key_ID_)
        else:
            month_ = month
        # Now extract
        file_data_, flagged = extract_MLD_file_4_loc(var2use=var2use,
                                                     lat=lat, lon=lon, month=month_,
                                                     buffer_CORDS=buffer_CORDS,
                                                     _fill_value=_fill_value,
                                                     debug=debug)
        # return just these two values.
        if rtn_flag:
            return file_data_, flagged
        else:
            return file_data_


def extract_MLD_file4indices(var2use='pt', lat_idx=None, lon_idx=None,
                             month=None, _fill_value=-99.9, Data_key_ID_=None,
                             verbose=True,
                             debug=False):
    """
    sub-function for get_MLD_4_loc to extract by month.

    Parameters
    -------
    lat_idx (list): indicies for latitude
    lon_idx (list): indicies for longitude
    Data_key_ID_ (str): ID for input data point
    debug (boolean): print out debug information?
    var2use (str): var to extract from NetCDF

    Returns
    -------
    (float), (str)
    (or list of two sets of above variables if get_max_and_sum_of_values==True)
    """
    # Directory?
    folder = get_file_locations('data_root') + '/WOA_1994/'
    # Filename as string
    file_str = 'WOA94_MLD_1x1_{}_1x1.nc'.format(var2use)
    if debug:
        print(folder+file_str)
    # - Open file
    # Using compiled NetCDFs (data was only available as csv.)
    with Dataset(folder+file_str, 'r') as rootgrp:
        # NOTE: pythonic index starts at 0, therefore month index = n_month-1
        file_data = rootgrp[var2use][month-1, ...]
        # Select values for indices
        file_data_ = file_data[lat_idx, lon_idx]
    return file_data_


def extract_MLD_file_4_loc(var2use='pt', lat=None, lon=None, month=None,
                           buffer_CORDS=5, _fill_value=-99.9, Data_key_ID_=None,
                           verbose=True, debug=False):
    """
    sub-function for get_MLD_4_loc to extract by month.

    Parameters
    -------
    lat (float): latitude degrees north
    lon (float): latitude degress east
    month (int): month number (1-12)
    Data_key_ID_ (str): ID for input data point
    debug (boolean): print out debug information?
    _fill_value (float): fill value in NetCDF
    buffer_CORDS (int): value (degrees lat/lon) added to loc define edges of
        rectangle used for interpolation.
    var2use (str): var to extract from NetCDF

    Returns
    -------
    (float), (str)
    (or list of two sets of above variables if get_max_and_sum_of_values==True)
    """
    # Directory?
    folder = get_file_locations('data_root') + '/WOA_1994/'
    # Filename as string
    file_str = 'WOA94_MLD_1x1_{}_1x1.nc'.format(var2use)
    # - Open file
    # Using compiled NetCDFs (data was only available as csv.)
    with Dataset(folder+file_str, 'r') as rootgrp:
        # - get indices of the data array for the provided lat and lon
        # Get index of cell mid point closest to obs lat
        latitude_step = abs(rootgrp['lat'][-1])-abs(rootgrp['lat'][-2])
        file_latc = rootgrp['lat']+(latitude_step/2)
#            print file_latc[:], lat
        lat_ind = AC.find_nearest_value(file_latc, lat)
        # Get index of cell mid point closest to obs lon
        longitude_step = abs(rootgrp['lon'][-1])-abs(rootgrp['lon'][-2])
        file_lonc = rootgrp['lon']+(longitude_step/2)
        # Adjust "file_loncs" to actual values (e.g. 0=>360, to -180=>180)
        adj_f_lonc = file_lonc-180
        adj_f_lonc[adj_f_lonc > 180] = adj_f_lonc[adj_f_lonc > 180] - 360
        lon_ind = AC.find_nearest_value(adj_f_lonc, lon)
        # Now extract data  (CF array dims, lat, lon)
        if debug:
            print(rootgrp[var2use].shape,  lat_ind, lon_ind)
        # NOTE: pythonic index starts at 0, therefore month index = n_month-1
        file_data = rootgrp[var2use][month-1, ...]
        file_data_ = file_data[lat_ind, lon_ind]
        # Round up value ( to remove noise )
        file_data_rounded_ = AC.myround(file_data_, 0.1, integer=False)
        # Setup a mask for the data
        file_data = np.ma.array(file_data, mask=file_data < 0)
        # - Return value if present, else interpolate from nearby values...
        if isinstance(file_data_, np.ma.core.MaskedConstant) or  \
                (file_data_rounded_ == _fill_value):
            ptr_str = 'interpolating to get {} for {} (buffer_CORDS={})'
            if verbose:
                print(ptr_str.format(var2use, Data_key_ID_, buffer_CORDS))
            # Make sure input data is (lon, lat)! (CF netCDF will be lat, lon)!
            # e.g. (360, 180) for 1x1 vs. (180, 360)
            file_data = file_data.T
            # Try Radial Basis Function Interpolation / Kernel Smoothing
            try:
                # set values with fill value to NaN (round for numeric noise)
                file_data_rounded = np.around(file_data, decimals=2)
                file_data[file_data_rounded == _fill_value] = np.NaN
                # Now interpolate and extract data
                file_data_ = AC.interpolate_sparse_grid2value(
                    X_CORDS=adjusted_file_lonc, Y_CORDS=file_latc,
                    X=lon, Y=lat,
                    XYarray=file_data, buffer_CORDS=buffer_CORDS)
                # If still equal to _fill_value, then raise error
                file_data_rounded_ = AC.myround(file_data_, 0.1, integer=False)
                if (file_data_rounded_ == _fill_value):
                    raise ValueError
                # ADD flag to say values are interpolated.
                flagged = 'Interpolated (buffer={})'.format(buffer_CORDS)
                # Try again but with twice the buffer_CORDS
            except:
                try:
                    debug_ = debug
                    # Now interpolate and extract data
                    file_data_ = AC.interpolate_sparse_grid2value(
                        X_CORDS=adjusted_file_lonc, Y_CORDS=file_latc, X=lon,
                        Y=lat,
                        XYarray=file_data, buffer_CORDS=buffer_CORDS*2,
                        debug=debug_)
                    # If still equal to _fill_value, then raise error
                    file_data_rounded_ = AC.myround(file_data_, 0.1,
                                                    integer=False)
                    if (file_data_rounded_ == _fill_value):
                        raise ValueError
                    # Flag value if interpolated
                    flagged = 'Interpolated 2nd time(buffer={})'
                    flagged = flagged.format(buffer_CORDS*2)
                # Flag value if not interpolated
                except:
                    try:
                        debug_ = debug
                        # Now interpolate and extract data
                        file_data_ = AC.interpolate_sparse_grid2value(
                            X_CORDS=adjusted_file_lonc,
                            Y_CORDS=file_latc,
                            X=lon, Y=lat,
                            XYarray=file_data, buffer_CORDS=buffer_CORDS*3,
                            debug=debug_)
                        # If still equal to _fill_value, then raise error
                        file_data_rounded_ = AC.myround(file_data_, 0.1,
                                                        integer=False)
                        if (file_data_rounded_ == _fill_value):
                            raise ValueError
                        # ADD flag to say values are interpolated.
                        flagged = 'Interpolated 3rd time (buffer={})'
                        flagged = flagged.format(buffer_CORDS*3)
                    # Flag value if not interpolated
                    except:
                        # Flag value if interpolated
                        flagged = 'FAILED INTERPOLATION!'
        else:
            flagged = 'False'
    # Add an extra Gotcha for negative MLD
    if file_data_ < 0:
        file_data_ = np.NaN
    return file_data_, flagged


def get_SeaWIFs_ChlrA4indices(resolution='9km', lat_idx=None, lon_idx=None,
                              month=None, var2use='chlor_a', verbose=True, debug=False):
    """
    Extract SeaWIFS (WOA) climatology value for Chlorophyll A (mg m^-3)

    Parameters
    -------
    lat_idx (list): indicies for latitude
    lon_idx (list): indicies for longitude
    Data_key_ID_ (str): ID for input data point
    debug (boolean): print out debug information?
    var2use (str): var to extract from NetCDF

    Returns
    -------
    (float), (str)

    NOTES
    ---
     - Using 9km files. 4km files are available.
    """
    # - Extract files
    # Detail on naming convention of files:
    # For a Level-3 binned data product, the form of the name of the main file
    # is iyyyydddyyyyddd.L3b_ttt, where where i is the instrument identifier
    # (S for SeaWiFS, A for Aqua MODIS, T for Terra MODIS, O for OCTS, C for
    # CZCS), yyyydddyyyyddd are the concatenated digits for the GMT year and
    # day of the year of the start and end days of the binning period, and ttt
    # is a code for the binning period length, resolution and product. Binning
    # period codes are DAY, 8D, MO, and YR. For daily products, only the year
    # and day of the data are used; i.e., yyyyddd. Subordinate files have an
    # extension xff appended to the name, where ff is a file number starting
    # from 00, with on subordinate file for each geophysical parameter. Note
    # that the "day of the year represents the dataday. (full details at URL)
    # https://oceancolor.gsfc.nasa.gov/docs/format/Ocean_Level-3_Binned_Data_Products.pdf
    # Kludge - no annual values, so just use a fix SH/NH month for now.
    if month == 0:
        month_ = set_backup_month_if_unknown(lat=lat, main_var='ChlrA',
                                             var2use=var2use, Data_key_ID_=Data_key_ID_)
    else:
        month_ = month
    # Directory?
    folder = get_file_locations('data_root') + '/SeaWIFS/'
    # Filename as string
    file_Str = 'S*.L3m_MC_*{}*'.format(resolution)
    # get SeaWIFS Files
    files = glob.glob(folder+file_Str)
    if debug:
        print(files)
    # Loop
    dates_for_files = []
    for file in files:
        # Get filename
        filename = file.split('/')[-1]
        # extract start date and calculate datetime
        start_year = int(filename[1:5])
        start_doy = int(filename[5:8])
        sdate = AC.add_days(datetime.datetime(start_year, 1, 1), start_doy-1)
        # extract end date and calculate datetime
        end_year = int(filename[8:12])
        end_doy = int(filename[12:15])
        edate = AC.add_days(datetime.datetime(end_year, 1, 1), end_doy-1)
        # add dates to list
        dates_for_files += [[sdate, edate]]
    dates = dates_for_files
    months = [i[0].month for i in dates_for_files]
    dict_month2filename = dict(list(zip(months, files)))
    # - Extract data for correct month.
    filename = dict_month2filename[month_]
    # Open file
    with Dataset(filename, 'r') as rootgrp:
        # Select values for indices
        # now extract data  (CF array dims, lat, lon)
        file_data = rootgrp[var2use][:]
        file_data_ = file_data[lat_idx, lon_idx]
    return file_data_


def get_SeaWIFs_ChlrA_4_loc(resolution='9km', var2use='chlor_a', lat=None,
                            lon=None, month=None, buffer_CORDS=5, Data_key_ID_=None,
                            rtn_flag=True,
                            verbose=True, debug=False):
    """
    Extract SeaWIFS (WOA) climatology value for Chlorophyll A (mg m^-3)

    Parameters
    -------
    lat (float): latitude degrees north
    lon (float): latitude degress east
    month (int): month number (1-12)
    Data_key_ID_ (str): ID for input data point
    debug (boolean): print out debug information?
    _fill_value (float): fill value in NetCDF
    buffer_CORDS (int): value (degrees lat/lon) added to loc define edges of
        rectangle used for interpolation.
    var2use (str): var to extract from NetCDF

    Returns
    -------
    (float), (str)

    NOTES
    ---
    using 9km files. 4km files are available.
    """
    # - Extract files
    # Detail on naming convention of files:
    # For a Level-3 binned data product, the form of the name of the main file
    # is iyyyydddyyyyddd.L3b_ttt, where where i is the instrument identifier
    # (S for SeaWiFS, A for Aqua MODIS, T for Terra MODIS, O for OCTS, C for
    # CZCS), yyyydddyyyyddd are the concatenated digits for the GMT year and
    # day of the year of the start and end days of the binning period, and ttt
    # is a code for the binning period length, resolution and product. Binning
    # period codes are DAY, 8D, MO, and YR. For daily products, only the year
    # and day of the data are used; i.e., yyyyddd. Subordinate files have an
    # extension xff appended to the name, where ff is a file number starting
    # from 00, with on subordinate file for each geophysical parameter. Note
    # that the "day of the year represents the dataday. (full details at URL)
    # https://oceancolor.gsfc.nasa.gov/docs/format/Ocean_Level-3_Binned_Data_Products.pdf
    # Kludge - no annual values, so just use a fix SH/NH month for now.
    if month == 0:
        month_ = set_backup_month_if_unknown(lat=lat, main_var='ChlrA',
                                             var2use=var2use, Data_key_ID_=Data_key_ID_)
    else:
        month_ = month
    # Directory?
    folder = get_file_locations('data_root') + '/SeaWIFS/'
    # Filename as string
    file_Str = 'S*.L3m_MC_*{}*'.format(resolution)
    # get SeaWIFS Files
    files = glob.glob(folder+file_Str)
    # Loop
    dates_for_files = []
    for file in files:
        # Get filename
        filename = file.split('/')[-1]
        # extract start date and calculate datetime
        start_year = int(filename[1:5])
        start_doy = int(filename[5:8])
        sdate = AC.add_days(datetime.datetime(start_year, 1, 1), start_doy-1)
        # extract end date and calculate datetime
        end_year = int(filename[8:12])
        end_doy = int(filename[12:15])
        edate = AC.add_days(datetime.datetime(end_year, 1, 1), end_doy-1)
        # add dates to list
        dates_for_files += [[sdate, edate]]
    dates = dates_for_files
    months = [i[0].month for i in dates_for_files]
    dict_month2filename = dict(list(zip(months, files)))
    # - Extract data for correct month.
    filename = dict_month2filename[month_]
    # Open file
    with Dataset(filename, 'r') as rootgrp:
        # - get indices of the data array for the provided lat and lon
        # Get index of cell mid point closest to obs lat
        #        file_latc = rootgrp['lat']+(rootgrp.latitude_step/2)
        #        lat_ind = AC.find_nearest_value( file_latc, lat )
        # Use lower-left coordinate system
        file_lat = rootgrp['lat']
        lat_ind = AC.find_nearest_value(file_lat, lat)
        # Get index of cell mid point closest to obs lat
#        file_lonc = rootgrp['lon']+(rootgrp.longitude_step/2)
#        lon_ind = AC.find_nearest_value( file_lonc, lon )
        # Use lower-left coordinate system
        file_lon = rootgrp['lon']
        lon_ind = AC.find_nearest_value(file_lon, lon)
        # now extract data  (CF array dims, lat, lon)
        if debug:
            print(rootgrp[var2use].shape,  lat_ind, lon_ind, depth)
        if debug:
            print(lat, file_latc[lat_ind], lon, file_lonc[lon_ind])
        file_data = rootgrp[var2use][:]
        file_data_ = file_data[lat_ind, lon_ind]
        # - Return value if present, else interpolate from nearby values...
        if isinstance(file_data_, np.ma.core.MaskedConstant):
            ptr_str = 'interpolating to get {} for {} (buffer_CORDS={})'
            if verbose:
                print(ptr_str.format(var2use, Data_key_ID_, buffer_CORDS))
            # Make sure input data is (lon, lat)! (CF netCDF will be lat, lon)!
            # e.g. (360, 180) for 1x1 vs. (180, 360)
            file_data = file_data.T
            # Try Radial Basis Function Interpolation / Kernel Smoothing
            try:
                # also this dataset's file are lat numebered -90=>90
                # shouldn't this be 90=>-90 according to CF standard?
                # Just reverse lat dimension and index for now.
                #                file_data = file_data[:,::-1]
                #                file_latc = file_latc[::-1]
                # commented out as it won't effect the interpolation
                # Now interpolate and extract data
                file_data_ = AC.interpolate_sparse_grid2value(
                    X_CORDS=file_lonc, Y_CORDS=file_latc, X=lon, Y=lat,
                    XYarray=file_data, buffer_CORDS=buffer_CORDS)
                # ADD flag to say values are interpolated.
                flagged = 'Interpolated (buffer={})'.format(buffer_CORDS)
            except:
                try:
                    # Now interpolate and extract data (using a smaller grid)
                    file_data_ = AC.interpolate_sparse_grid2value(
                        X_CORDS=file_lonc, Y_CORDS=file_latc, X=lon, Y=lat,
                        XYarray=file_data, buffer_CORDS=buffer_CORDS/6)
                    # ADD flag to say values are interpolated.
                    flagged = 'Interpolated (buffer={})'.format(buffer_CORDS/6)
                    # Raise error if value greater than zero
                    if isinstance(file_data_, np.ma.core.MaskedConstant):
                        raise ValueError
#                except MemoryError:
                except:
                    try:
                        # Now interpolate and extract data
                        # (using a smaller grid)
                        file_data_ = AC.interpolate_sparse_grid2value(
                            X_CORDS=file_lonc, Y_CORDS=file_latc, X=lon, Y=lat,
                            XYarray=file_data, buffer_CORDS=buffer_CORDS/3)
                        # ADD flag to say values are interpolated.
                        flagged = 'Interpolated (buffer={})'
                        flagged = flagged.format(buffer_CORDS/3)
                        # Raise error if value greater than zero
                        if isinstance(file_data_, np.ma.core.MaskedConstant):
                            raise ValueError
                    except:
                        try:
                            # Now interpolate and extract data
                            # (using a smaller grid)
                            file_data_ = AC.interpolate_sparse_grid2value(
                                X_CORDS=file_lonc, Y_CORDS=file_latc,
                                X=lon, Y=lat,
                                XYarray=file_data,
                                buffer_CORDS=buffer_CORDS*1.5)
                            # ADD flag to say values are interpolated.
                            flagged = 'Interpolated (buffer={})'
                            flagged = flagged.format(buffer_CORDS*1.5)
                            # Raise error if value greater than zero
                            if isinstance(file_data_,
                                          np.ma.core.MaskedConstant):
                                raise ValueError
                        except:
                            flagged = 'FAILED INTERPOLATION!'
        else:
            flagged = 'False'
    if rtn_flag:
        return file_data_, flagged
    else:
        return file_data_


def get_Depth_GEBCO4indices(lat_idx=None, lon_idx=None, month=None,
                            var2use='elevation', verbose=True, debug=False):
    """
    Extract "Elevation relative to sea level" from General Bathymetric Chart
    of the Oceans (GEBCO).

    Parameters
    -------
    lat_idx (list): indicies for latitude
    lon_idx (list): indicies for longitude
    Data_key_ID_ (str): ID for input data point
    debug (boolean): print out debug information?
    var2use (str): var to extract from NetCDF

    Returns
    -------
    (array)

    NOTES
    ---
     - Using the "One Minute Grid", but a 30 second grid is availible.
     - The time resolution for depth NetCDF is annual.
    """
    # Directory?
    folder = get_file_locations('data_root') + '/BODC/'
    # Filename as string
    filename = 'GRIDONE_2D.nc'
    # Open file and extract data
    with Dataset(folder+filename, 'r') as rootgrp:
        # Select values for indices
        file_data = rootgrp[var2use][:]
        file_data_ = file_data[lat_idx, lon_idx]
    return file_data_


def get_GEBCO_depth_4_loc(lat=None, lon=None, month=None,
                          var2use='elevation', buffer_CORDS=2, rtn_flag=True,
                          Data_key_ID_=None,
                          verbose=True, debug=False):
    """
    Extract "Elevation relative to sea level" from General Bathymetric Chart
    of the Oceans (GEBCO).

    Parameters
    -------
    lat (float): latitude degrees north
    lon (float): latitude degress east
    month (int): month number (1-12)
    Data_key_ID_ (str): ID for input data point
    debug (boolean): print out debug information?
    _fill_value (float): fill value in NetCDF
    buffer_CORDS (int): value (degrees lat/lon) added to loc define edges of
        rectangle used for interpolation.
    var2use (str): var to extract from NetCDF

    Returns
    -------
    (float), (str)

    NOTES
    ---
     - Using the "One Minute Grid", but a 30 second grid is availible.
     - The time resolution for depth NetCDF is annual.
    """
    # var2use='elevation'; buffer_CORDS=2; rtn_flag=True; debug=True
    # Directory?
    folder = get_file_locations('data_root') + '/BODC/'
    # Filename as string
    filename = 'GRIDONE_2D.nc'
    # - Extract data
    # Open file
    with Dataset(folder+filename, 'r') as rootgrp:
        # - get indices of the data array for the provided lat and lon
        # Get index of cell mid point closest to obs lat
        #        latitude_step = abs(rootgrp['lat'][-1])-abs(rootgrp['lat'][-2])
        #        file_latc = rootgrp['lat']+(latitude_step/2)
        #        lat_ind = AC.find_nearest_value( file_latc, lat )
        # Use lower-left coordinate system
        file_lat = rootgrp['lat']
        lat_ind = AC.find_nearest_value(file_lat, lat)
        # Get index of cell mid point closest to obs lat
#        longitude_step = abs(rootgrp['lon'][-1])-abs(rootgrp['lon'][-2])
#        file_lonc = rootgrp['lon']+(longitude_step/2)
#        lon_ind = AC.find_nearest_value( file_lonc, lon )
        # Use lower-left coordinate system
        file_lon = rootgrp['lon']
        lon_ind = AC.find_nearest_value(file_lon, lon)
        # now extract data  (CF array dims, lat, lon)
        if debug:
            print(rootgrp[var2use].shape,  lat_ind, lon_ind)
        file_data = rootgrp[var2use][:]
        file_data_ = file_data[lat_ind, lon_ind]
        # - Return value if present, else interpolate from nearby values...
        if isinstance(file_data_, np.ma.core.MaskedConstant) or (file_data_ > 0):
            ptr_str = 'interpolating to get {} for {} (buffer_CORDS={})'
            if verbose:
                print(ptr_str.format(var2use, Data_key_ID_, buffer_CORDS))
            # Make sure input data is X vs. Y (e.g. (lon, lat)) !
            # (CF netCDF will be lat, lon)!
            # e.g. (360, 180) for 1x1 vs. (180, 360)
            file_data = file_data.T
            # Convert to floats (Interpolation is setup only for floats)
            file_data = file_data.astype(np.float64)
            # Mask all values with depth greater than zero (expect -ve!)
#            file_data[ np.isnan(file_data) ] = 9.99999E10
#            bool_greater_than_zero = np.where(file_data> 0) ]
            file_data = np.ma.array(file_data, mask=(file_data > 0))
#            file_data[bool_greater_than_zero].mask = True
            # Try Radial Basis Function Interpolation / Kernel Smoothing
            try:
                # also this dataset's file are lat numebered -90=>90
                # shouldn't this be 90=>-90 according to CF standard?
                # Just reverse lat dimension and index for now.
                # NOTE: orientation has not impact on selected value as this is
                # done via nearest value in coordinate space, it just effects
                # the east to read the map outputted during debugging
                #                file_data = file_data[:,::-1]
                #                file_latc = file_latc[::-1]
                # commented out as it won't effect the interpolation
                # Now interpolate and extract data
                file_data_ = AC.interpolate_sparse_grid2value(
                    X_CORDS=file_lonc, Y_CORDS=file_latc, X=lon, Y=lat,
                    XYarray=file_data, buffer_CORDS=buffer_CORDS)
                # ADD flag to say values are interpolated.
                flagged = 'Interpolated (buffer={})'.format(buffer_CORDS)
                # Raise error if value greater than zero
                if file_data_ > 0:
                    raise ValueError
#            except MemoryError :
            except:
                try:
                    # Now interpolate and extract data (using a smaller grid)
                    file_data_ = AC.interpolate_sparse_grid2value(
                        X_CORDS=file_lonc, Y_CORDS=file_latc, X=lon, Y=lat,
                        XYarray=file_data, buffer_CORDS=buffer_CORDS/3)
                    # ADD flag to say values are interpolated.
                    flagged = 'Interpolated (buffer={})'.format(buffer_CORDS/3)
                    # Raise error if value greater than zero
                    if file_data_ > 0:
                        raise ValueError
#                except MemoryError:
                except:
                    flagged = 'FAILED INTERPOLATION!'
        else:
            flagged = 'False'
    if rtn_flag:
        return file_data_, flagged
    else:
        return file_data_


def get_WOA_TEMP4indices(lat_idx=None, lon_idx=None, month=None,
                         var2use='t_an', verbose=True, debug=False):
    """
    Extract World ocean atlas (WOA) climatology value for temperature

    Parameters
    -------
    lat_idx (list): indicies for latitude
    lon_idx (list): indicies for longitude
    Data_key_ID_ (str): ID for input data point
    debug (boolean): print out debug information?
    var2use (str): var to extract from NetCDF

    Returns
    -------
    (array)

    NOTES
    ---
     - default var to use is "t_an"
     - t_an:long_name = "Objectively analyzed mean fields for
     sea_water_temperature at standard depth levels." ;
     - t_mn:long_name = "Average of all unflagged interpolated values at each
      standard depth level for sea_water_temperature in each grid-square which
       contain at least one measurement." ;
    """
    # Set folder that files are in
    folder = get_file_locations('data_root') + '/WOA_2013/Temperature_025x025/'
    # Select the correct file
    # (The file below is a decadal average ("decav"))
    filename = 'woa13_decav_t{:0>2}_04v2.nc'.format(month)
    # Fix depth = 0 for now...
    depth = 0
    if debug:
        print(folder+filename)
    # Open file
    with Dataset(folder+filename, 'r') as rootgrp:
        # Select values for indices
        # array dims = (time lat lon depth) - (!!!incorrect in file!!!!)
        # lon== 1440, lat=720 so time, depth, lat, lon
        file_data = rootgrp[var2use][0, depth, ...]
        if debug:
            print(file_data.shape, type(file_data))
        file_data_ = file_data[lat_idx, lon_idx]
    return file_data_


def get_WOA_TEMP_4_loc(lat=None, lon=None, month=None, var2use='t_an',
                       buffer_CORDS=5, rtn_flag=True, Data_key_ID_=None,
                       verbose=True, debug=False):
    """
    Extract World ocean atlas (WOA) climatology value for temperature

    Parameters
    -------
    lat (float): latitude degrees north
    lon (float): latitude degress east
    month (int): month number (1-12)
    Data_key_ID_ (str): ID for input data point
    debug (boolean): print out debug information?
    _fill_value (float): fill value in NetCDF
    buffer_CORDS (int): value (degrees lat/lon) added to loc define edges of
        rectangle used for interpolation.
    var2use (str): var to extract from NetCDF

    Returns
    -------
    (float), (str)

    NOTES
    ---
     - default var to use is "t_an"
     - t_an:long_name = "Objectively analyzed mean fields for
     sea_water_temperature at standard depth levels." ;
     - t_mn:long_name = "Average of all unflagged interpolated values at each
      standard depth level for sea_water_temperature in each grid-square which
       contain at least one measurement." ;
    """
    if debug:
        print(locals())
    # Set folder that files are in
    folder = get_file_locations('data_root') + '/WOA_2013/Temperature_025x025/'
    # Select the correct file
    # (The file below is a decadal average ("decav"))
    filename = 'woa13_decav_t{:0>2}_04v2.nc'.format(month)
    if debug:
        print(folder+filename)
    # Open file
    with Dataset(folder+filename, 'r') as rootgrp:
        # Get index of cell mid point closest to obs lat
        #        half_grid_cell = (rootgrp['lat_bnds'][0][0]-rootgrp['lat_bnds'][0][1])/2
        #        file_latc = rootgrp['lat']+abs(half_grid_cell)
        #        lat_ind = AC.find_nearest_value( file_latc, lat )
        # Use lower-left coordinate system
        file_lat = rootgrp['lat']
        lat_ind = AC.find_nearest_value(file_lat, lat)
        # Get index of cell mid point closest to obs lat
#        half_grid_cell = (rootgrp['lon_bnds'][0][0]-rootgrp['lon_bnds'][0][1])/2
#        file_lonc = rootgrp['lon']+abs(half_grid_cell)
#        lon_ind = AC.find_nearest_value( file_lonc, lon )
        # Use lower-left coordinate system
        file_lon = rootgrp['lon']
        lon_ind = AC.find_nearest_value(file_lon, lon)
        # Fix depth = 0 for now...
        depth = 0
        # get indices of the data array for the provided lat and lon
        # now extract data
        # array dims = (time lat lon depth) - (!!!incorrect in file!!!!)
        # lon== 1440, lat=720 so time, depth, lat, lon
        if debug:
            prt_str = 'LAT={}({},IND={})'.format(
                lat, file_latc[lat_ind], lat_ind)
            prt_str += 'LON={}({},IND={})'.format(lon,
                                                  file_lonc[lon_ind], lon_ind)
            print(prt_str, rootgrp[var2use].shape, depth)
        file_data = rootgrp[var2use][0, depth, ...]
#        print file_data.shape, type( file_data )
        file_data_ = file_data[lat_ind, lon_ind]
        # What is the data is masked for this location?
        if isinstance(file_data_, np.ma.core.MaskedConstant):
            ptr_str = 'interpolating to get {} for {} (buffer_CORDS={})'
            if verbose:
                print(ptr_str.format(var2use, Data_key_ID_, buffer_CORDS))
            # Make sure input data is (lon, lat)! (CF netCDF will be lat, lon)!
            # e.g. (360, 180) for 1x1 vs. (180, 360)
            file_data = file_data.T
            # Try Radial Basis Function Interpolation / Kernel Smoothing
            try:
                # Now interpolate and extract data
                file_data_ = AC.interpolate_sparse_grid2value(
                    X_CORDS=file_lonc, Y_CORDS=file_latc, X=lon, Y=lat,
                    XYarray=file_data, buffer_CORDS=buffer_CORDS)
                # Flag value if interpolated
                flagged = 'Interpolated (buffer={})'.format(buffer_CORDS)
            except:
                # Try again but with twice the buffer_CORDS
                try:
                    debug_ = True
                    # Now interpolate and extract data
                    file_data_ = AC.interpolate_sparse_grid2value(
                        X_CORDS=file_lonc, Y_CORDS=file_latc, X=lon, Y=lat,
                        XYarray=file_data, buffer_CORDS=buffer_CORDS*2,
                        debug=debug_)
                    # Flag value if interpolated
                    flagged = 'Interpolated 2nd time(buffer={})'
                    flagged = flagged.format(buffer_CORDS*2)
                except:
                    flagged = 'FAILED INTERPOLATION!'
        else:
            flagged = 'False'
    if rtn_flag:
        return file_data_, flagged
    else:
        return file_data_


def get_WOA_Nitrate4indices(lat_idx=None, lon_idx=None, month=None,
                            var2use='n_an', verbose=True, debug=False):
    """
    Extract Wold ocean atlas (WOA) climatology value for nitrate

    Parameters
    -------
    lat_idx (list): indicies for latitude
    lon_idx (list): indicies for longitude
    month (int): month number (1-12)
    debug (boolean): print out debug information?
    var2use (str): var to extract from NetCDF

    Returns
    -------
    (array)

    NOTES
    ---
     - default var to use is n_an
     - n_mn:long_name = "Average of all unflagged interpolated values at each
     standard depth level for moles_concentration_of_nitrate_in_sea_water in
     each grid-square which contain at least one measurement."
     - n_an:long_name = "Objectively analyzed mean fields for
     moles_concentration_of_nitrate_in_sea_water at standard depth levels."
    """
    # lat=20; lon=-40; month=1; var2use='n_an'; debug=False
    # Set folder that files are in
    folder = get_file_locations('data_root') + '/WOA_2013/Nitrate_1x1/'
    # Select the correct file
    filename = 'woa13_all_n{:0>2}_01.nc'.format(month)
    # Fix depth = 0 for now...
    depth = 0
    # Open file
    with Dataset(folder+filename, 'r') as rootgrp:
        # Select values for indices
        # now extract data
        # array dims = (time lat lon depth) - (!!!incorrect in file!!!!)
        # lon== 1440, lat=720 so time, depth, lat, lon
        file_data = rootgrp[var2use][0, depth, ...]
        # Mask zero (land) values - already masked array
#        file_data = np.ma.array( file_data, mask=file_data <= 0 )
        # Extract for location(s)
        file_data_ = file_data[lat_idx, lon_idx]
    return file_data_


def get_WOA_Nitrate_4_loc(lat=None, lon=None, month=None, var2use='n_an',
                          buffer_CORDS=5, rtn_flag=True, Data_key_ID_=None,
                          verbose=True, debug=False):
    """
    Extract Wold ocean atlas (WOA) climatology value for nitrate

    Parameters
    -------
    lat (float): latitude degrees north
    lon (float): latitude degress east
    month (int): month number (1-12)
    Data_key_ID_ (str): ID for input data point
    debug (boolean): print out debug information?
    _fill_value (float): fill value in NetCDF
    buffer_CORDS (int): value (degrees lat/lon) added to loc define edges of
        rectangle used for interpolation.
    var2use (str): var to extract from NetCDF

    Returns
    -------
    (float), (str)

    NOTES
    ---
     - default var to use is n_an
     - n_mn:long_name = "Average of all unflagged interpolated values at each
     standard depth level for moles_concentration_of_nitrate_in_sea_water in
     each grid-square which contain at least one measurement."
     - n_an:long_name = "Objectively analyzed mean fields for
     moles_concentration_of_nitrate_in_sea_water at standard depth levels."
    """
    # lat=20; lon=-40; month=1; var2use='n_an'; debug=False
    # Set folder that files are in
    folder = get_file_locations('data_root') + '/WOA_2013/Nitrate_1x1/'
    # Select the correct file
    filename = 'woa13_all_n{:0>2}_01.nc'.format(month)
    # Open file
    with Dataset(folder+filename, 'r') as rootgrp:
        # Get index of cell mid point closest to obs lat
        #         half_grid_cell = (rootgrp['lat_bnds'][0][0]-rootgrp['lat_bnds'][0][1])/2
        #         file_latc = rootgrp['lat']+abs(half_grid_cell)
        #         lat_ind = AC.find_nearest_value( file_latc, lat )
        # Use lower-left coordinate system
        file_lat = rootgrp['lat']
        lat_ind = AC.find_nearest_value(file_lat, lat)
        # Get index of cell mid point closest to obs lat
#         half_grid_cell = (rootgrp['lon_bnds'][0][0]-rootgrp['lon_bnds'][0][1])/2
#         file_lonc = rootgrp['lon']+abs(half_grid_cell)
#         lon_ind = AC.find_nearest_value( file_lonc, lon )
        # Use lower-left coordinate system
        file_lon = rootgrp['lon']
        lon_ind = AC.find_nearest_value(file_lon, lon)
        # Fix depth = 0 for now...
        depth = 0
        # get indices of the data array for the provided lat and lon
        # now extract data
        # array dims = (time lat lon depth) - (!!!incorrect in file!!!!)
        # lon== 1440, lat=720 so time, depth, lat, lon
        if debug:
            print(rootgrp[var2use].shape,  lat_ind, lon_ind, depth)
        file_data = rootgrp[var2use][0, depth, ...]
        # Mask zero (land) values - already masked array
#        file_data = np.ma.array( file_data, mask=file_data <= 0 )
        # Extract for location
        file_data_ = file_data[lat_ind, lon_ind]
        # What is the data is masked for this location?
        if isinstance(file_data_, np.ma.core.MaskedConstant):
            ptr_str = 'interpolating to get {} for {} (buffer_CORDS={})'
            if verbose:
                print(ptr_str.format(var2use, Data_key_ID_, buffer_CORDS))
            # Make sure input data is (lon, lat)! (CF netCDF will be lat, lon)!
            # e.g. (360, 180) for 1x1 vs. (180, 360)
            file_data = file_data.T
            # Try Radial Basis Function Interpolation / Kernel Smoothing
            try:
                # Now interpolate and extract data
                file_data_ = AC.interpolate_sparse_grid2value(
                    X_CORDS=file_lonc, Y_CORDS=file_latc, X=lon, Y=lat,
                    XYarray=file_data, buffer_CORDS=buffer_CORDS)
                # Flag value if interpolated
                flagged = 'Interpolated (buffer={})'.format(buffer_CORDS)
            except:
                # Try again but with twice the buffer_CORDS
                try:
                    # Now interpolate and extract data
                    file_data_ = AC.interpolate_sparse_grid2value(
                        X_CORDS=file_lonc, Y_CORDS=file_latc, X=lon, Y=lat,
                        XYarray=file_data, buffer_CORDS=buffer_CORDS*2)
                    # Flag value if interpolated
                    flagged = 'Interpolated 2nd time(buffer={})'
                    flagged = flagged.format(buffer_CORDS*2)
                except:
                    flagged = 'FAILED INTERPOLATION!'
        else:
            flagged = 'False'
    if rtn_flag:
        return file_data_, flagged
    else:
        return file_data_


def get_WOA_Salinity4indices(lat_idx=None, lon_idx=None, month=None,
                             var2use='s_an', verbose=True, debug=False):
    """
    Extract Wold ocean atlas (WOA) climatology value for Salinity

    Parameters
    -------
    lat_idx (list): indicies for latitude
    lon_idx (list): indicies for longitude
    month (int): month number (1-12)
    debug (boolean): print out debug information?
    var2use (str): var to extract from NetCD

    Returns
    -------
    (array)
    """
    # Set folder that files are in
    folder = get_file_locations('data_root') + '/WOA_2013/Salinity_025x025/'
    # Select the correct file
    filename = 'woa13_decav_s{:0>2}_04v2.nc'.format(month)
    # Fix depth = 0 for now...
    depth = 0
    # Open file
    with Dataset(folder+filename, 'r') as rootgrp:
        # Extract data for indices
        # array dims = (time lat lon depth) - (!!!incorrect in file!!!!)
        # lon== 1440, lat=720 so time, depth, lat, lon
        file_data = rootgrp[var2use][0, depth, ...]
        # Extract for location(s)
        file_data_ = file_data[lat_idx, lon_idx]
    return file_data_


def get_WOA_Salinity_4_loc(lat=None, lon=None, month=None, var2use='s_an',
                           buffer_CORDS=5, rtn_flag=True, Data_key_ID_=None,
                           verbose=True, debug=False):
    """
    Extract Wold ocean atlas (WOA) climatology value for Salinity

    Parameters
    -------
    lat (float): latitude degrees north
    lon (float): latitude degress east
    month (int): month number (1-12)
    Data_key_ID_ (str): ID for input data point
    debug (boolean): print out debug information?
    _fill_value (float): fill value in NetCDF
    buffer_CORDS (int): value (degrees lat/lon) added to loc define edges of
        rectangle used for interpolation.
    var2use (str): var to extract from NetCDF

    Returns
    -------
    (float), (str)

    NOTES
    ---
     - default var to use is n_an
     - "Average of all unflagged interpolated values at each standard depth
     level for sea_water_salinity in each grid-square which contain at least
     one measurement."
     - s_an:long_name = "Objectively analyzed mean fields for
     sea_water_salinity at standard depth levels." ;
    """
#    debug=True
    # Set folder that files are in
    folder = get_file_locations('data_root') + '/WOA_2013/Salinity_025x025/'
    # Select the correct file
    filename = 'woa13_decav_s{:0>2}_04v2.nc'.format(month)
    # Open file
    with Dataset(folder+filename, 'r') as rootgrp:
        # Get index of cell mid point closest to obs lat
        #         half_grid_cell = (rootgrp['lat_bnds'][0][0]-rootgrp['lat_bnds'][0][1])/2
        #         file_latc = rootgrp['lat']+abs(half_grid_cell)
        #         lat_ind = AC.find_nearest_value( file_latc, lat )
        # Use lower-left coordinate system
        file_lat = rootgrp['lat']
        lat_ind = AC.find_nearest_value(file_lat, lat)
        # Get index of cell mid point closest to obs lat
#         half_grid_cell = (rootgrp['lon_bnds'][0][0]-rootgrp['lon_bnds'][0][1])/2
#         file_lonc = rootgrp['lon']+abs(half_grid_cell)
#         lon_ind = AC.find_nearest_value( file_lonc, lon )
        # Use lower-left coordinate system
        file_lon = rootgrp['lon']
        lon_ind = AC.find_nearest_value(file_lon, lon)
        # Fix depth = 0 for now...
        depth = 0
        # get indices of the data array for the provided lat and lon
        # now extract data
        # array dims = (time lat lon depth) - (!!!incorrect in file!!!!)
        # lon== 1440, lat=720 so time, depth, lat, lon
        if debug:
            print(rootgrp[var2use].shape,  lat_ind, lon_ind, depth)
        file_data = rootgrp[var2use][0, depth, ...]
        file_data_ = file_data[lat_ind, lon_ind]
        # What is the data is masked for this location?
        if isinstance(file_data_, np.ma.core.MaskedConstant):
            ptr_str = 'interpolating to get {} for {} (buffer_CORDS={})'
            if verbose:
                print(ptr_str.format(var2use, Data_key_ID_, buffer_CORDS))
            # Make sure input data is (lon, lat)! (CF netCDF will be lat, lon)!
            # e.g. (360, 180) for 1x1 vs. (180, 360)
            file_data = file_data.T
            # Try Radial Basis Function Interpolation / Kernel Smoothing
            try:
                # Now interpolate and extract data
                file_data_ = AC.interpolate_sparse_grid2value(
                    X_CORDS=file_lonc, Y_CORDS=file_latc, X=lon, Y=lat,
                    XYarray=file_data, buffer_CORDS=buffer_CORDS)
                # Flag value if interpolated
                flagged = 'Interpolated (buffer={})'.format(buffer_CORDS)
            except:
                # Try again but with twice the buffer_CORDS
                try:
                    # Now interpolate and extract data
                    file_data_ = AC.interpolate_sparse_grid2value(
                        X_CORDS=file_lonc, Y_CORDS=file_latc, X=lon, Y=lat,
                        XYarray=file_data, buffer_CORDS=buffer_CORDS*2)
                    # Flag value if interpolated
                    flagged = 'Interpolated 2nd time(buffer={})'
                    flagged = flagged.format(buffer_CORDS*2)
                except:
                    flagged = 'FAILED INTERPOLATION!'
        else:
            flagged = 'False'
    if rtn_flag:
        return file_data_, flagged
    else:
        return file_data_


def get_WOA_Silicate4indices(lat_idx=None, lon_idx=None, month=None,
                             var2use='i_an', verbose=True, debug=False):
    """
    Extract Wold ocean atlas (WOA) climatology value for Silicate

    Parameters
    -------
    lat (float): latitude degrees north
    lon (float): latitude degress east
    month (int): month number (1-12)
    Data_key_ID_ (str): ID for input data point
    debug (boolean): print out debug information?
    _fill_value (float): fill value in NetCDF
    buffer_CORDS (int): value (degrees lat/lon) added to loc define edges of
        rectangle used for interpolation.
    var2use (str): var to extract from NetCDF

    Returns
    -------
    (float), (str)

    NOTES
    ---
     - default var to use is n_an
     - i_an:long_name = "Objectively analyzed mean fields for
     moles_concentration_of_silicate_in_sea_water at standard depth levels." ;
     - i_mn:long_name = "Average of all unflagged interpolated values at
     each standard depth level for
     moles_concentration_of_silicate_in_sea_water in each grid-square which
     contain at least one measurement." ;
    """
    # Set folder that files are in
    folder = get_file_locations('data_root') + '/WOA_2013/Silicate_1x1/'
    # Select the correct file
    filename = 'woa13_all_i{:0>2}_01.nc'.format(month)
    # Fix depth = 0 for now...
    depth = 0
    # Open file
    with Dataset(folder+filename, 'r') as rootgrp:
        # Extract data for indices
        # array dims = (time lat lon depth) - (!!!incorrect in file!!!!)
        # lon== 1440, lat=720 so time, depth, lat, lon
        file_data = rootgrp[var2use][0, depth, ...]
        # Extract for location(s)
        file_data_ = file_data[lat_idx, lon_idx]
    return file_data_


def get_WOA_Silicate_4_loc(lat=None, lon=None, month=None, var2use='i_an',
                           buffer_CORDS=5, rtn_flag=True, Data_key_ID_=None,
                           verbose=True, debug=False):
    """
    Extract Wold ocean atlas (WOA) climatology value for Silicate

    Parameters
    -------
    lat (float): latitude degrees north
    lon (float): latitude degress east
    month (int): month number (1-12)
    Data_key_ID_ (str): ID for input data point
    debug (boolean): print out debug information?
    _fill_value (float): fill value in NetCDF
    buffer_CORDS (int): value (degrees lat/lon) added to loc define edges of
        rectangle used for interpolation.
    var2use (str): var to extract from NetCDF

    Returns
    -------
    (float), (str)

    NOTES
    ---
     - default var to use is n_an
     - i_an:long_name = "Objectively analyzed mean fields for
     moles_concentration_of_silicate_in_sea_water at standard depth levels." ;
     - i_mn:long_name = "Average of all unflagged interpolated values at
     each standard depth level for
     moles_concentration_of_silicate_in_sea_water in each grid-square which
     contain at least one measurement." ;
    """
    # Set folder that files are in
    folder = get_file_locations('data_root') + '/WOA_2013/Silicate_1x1/'
    # Select the correct file
    filename = 'woa13_all_i{:0>2}_01.nc'.format(month)
    # Open file
    with Dataset(folder+filename, 'r') as rootgrp:
        # Get index of cell mid point closest to obs lat
        #         half_grid_cell = (rootgrp['lat_bnds'][0][0]-rootgrp['lat_bnds'][0][1])/2
        #         file_latc = rootgrp['lat']+abs(half_grid_cell)
        #         lat_ind = AC.find_nearest_value( file_latc, lat )
        # Use lower-left coordinate system
        file_lat = rootgrp['lat']
        lat_ind = AC.find_nearest_value(file_lat, lat)
        # Get index of cell mid point closest to obs lat
#         half_grid_cell = (rootgrp['lon_bnds'][0][0]-rootgrp['lon_bnds'][0][1])/2
#         file_lonc = rootgrp['lon']+abs(half_grid_cell)
#         lon_ind = AC.find_nearest_value( file_lonc, lon )
        # Use lower-left coordinate system
        file_lon = rootgrp['lon']
        lon_ind = AC.find_nearest_value(file_lon, lon)
        # Fix depth = 0 for now...
        depth = 0
        # get indices of the data array for the provided lat and lon
        # now extract data
        # array dims = (time lat lon depth) - (!!!incorrect in file!!!!)
        # lon== 1440, lat=720 so time, depth, lat, lon
        if debug:
            print(rootgrp[var2use].shape,  lat_ind, lon_ind, depth)
        file_data = rootgrp[var2use][0, depth, ...]
        file_data_ = file_data[lat_ind, lon_ind]
        # What is the data is masked for this location?
        if isinstance(file_data_, np.ma.core.MaskedConstant):
            ptr_str = 'interpolating to get {} for {} (buffer_CORDS={})'
            if verbose:
                print(ptr_str.format(var2use, Data_key_ID_, buffer_CORDS))
            # Make sure input data is (lon, lat)! (CF netCDF will be lat, lon)!
            # e.g. (360, 180) for 1x1 vs. (180, 360)
            file_data = file_data.T
            # Try Radial Basis Function Interpolation / Kernel Smoothing
            try:
                # Now interpolate and extract data
                file_data_ = AC.interpolate_sparse_grid2value(
                    X_CORDS=file_lonc, Y_CORDS=file_latc, X=lon, Y=lat,
                    XYarray=file_data, buffer_CORDS=buffer_CORDS)
                # Flag value if interpolated
                flagged = 'Interpolated (buffer={})'.format(buffer_CORDS)
            except:
                # Try again but with twice the buffer_CORDS
                try:
                    # Now interpolate and extract data
                    file_data_ = AC.interpolate_sparse_grid2value(
                        X_CORDS=file_lonc, Y_CORDS=file_latc, X=lon, Y=lat,
                        XYarray=file_data, buffer_CORDS=buffer_CORDS*2)
                    # Flag value if interpolated
                    flagged = 'Interpolated 2nd time(buffer={})'
                    flagged = flagged.format(buffer_CORDS*2)
                except:
                    flagged = 'FAILED INTERPOLATION!'
        else:
            flagged = 'False'
    if rtn_flag:
        return file_data_, flagged
    else:
        return file_data_


def get_WOA_Phosphate4indices(lat_idx=None, lon_idx=None, month=None,
                              var2use='p_an', verbose=True, debug=False):
    """
    Extract Wold ocean atlas (WOA) climatology value for Salinity

    Parameters
    -------
    lat_idx (list): indicies for latitude
    lon_idx (list): indicies for longitude
    month (int): month number (1-12)
    debug (boolean): print out debug information?
    var2use (str): var to extract from NetCDF

    Returns
    -------
    (array)
    """
    # Set folder that files are in
    folder = get_file_locations('data_root') + '/WOA_2013/Phosphate_1x1/'
    # Select the correct file
    filename = 'woa13_all_p{:0>2}_01.nc'.format(month)
    # Fix depth = 0 for now...
    depth = 0
    # Open file
    with Dataset(folder+filename, 'r') as rootgrp:
        # Extract data for indices
        # array dims = (time lat lon depth) - (!!!incorrect in file!!!!)
        # lon== 1440, lat=720 so time, depth, lat, lon
        if debug:
            print(rootgrp[var2use].shape,  lat_ind, lon_ind, depth)
        file_data = rootgrp[var2use][0, depth, ...]
        # Extract for location(s)
        file_data_ = file_data[lat_idx, lon_idx]
    return file_data_


def get_WOA_Phosphate_4_loc(lat=None, lon=None, month=None, var2use='p_an',
                            buffer_CORDS=5, rtn_flag=True, Data_key_ID_=None,
                            verbose=True, debug=False):
    """
    Extract Wold ocean atlas (WOA) climatology value for phosphate

    Parameters
    -------
    lat (float): latitude degrees north
    lon (float): latitude degress east
    month (int): month number (1-12)
    Data_key_ID_ (str): ID for input data point
    debug (boolean): print out debug information?
    _fill_value (float): fill value in NetCDF
    buffer_CORDS (int): value (degrees lat/lon) added to loc define edges of
        rectangle used for interpolation.
    var2use (str): var to extract from NetCDF

    Returns
    -------
    (float), (str)

    NOTES
    ---
     - default var to use is n_an
     - p_an:long_name = "Objectively analyzed mean fields for
     moles_concentration_of_phosphate_in_sea_water at standard depth levels." ;
    - p_mn:long_name = "Average of all unflagged interpolated values at each
    standard depth level for moles_concentration_of_phosphate_in_sea_water
    in each grid-square which contain at least one measurement." ;
    """
    # Set folder that files are in
    folder = get_file_locations('data_root') + '/WOA_2013/Phosphate_1x1/'
    # Select the correct file
    filename = 'woa13_all_p{:0>2}_01.nc'.format(month)
    # Open file
    with Dataset(folder+filename, 'r') as rootgrp:
        # Get index of cell mid point closest to obs lat
        #        half_grid_cell = (rootgrp['lat_bnds'][0][0]-rootgrp['lat_bnds'][0][1])/2
        #        file_latc = rootgrp['lat']+abs(half_grid_cell)
        #        lat_ind = AC.find_nearest_value( file_latc, lat )
        # Use lower-left coordinate system
        file_lat = rootgrp['lat']
        lat_ind = AC.find_nearest_value(file_lat, lat)
        # Get index of cell mid point closest to obs lat
#        half_grid_cell = (rootgrp['lon_bnds'][0][0]-rootgrp['lon_bnds'][0][1])/2
#        file_lonc = rootgrp['lon']+abs(half_grid_cell)
#        lon_ind = AC.find_nearest_value( file_lonc, lon )
        # Use lower-left coordinate system
        file_lon = rootgrp['lon']
        lon_ind = AC.find_nearest_value(file_lon, lon)
        # Fix depth = 0 for now...
        depth = 0
        # get indices of the data array for the provided lat and lon
        # now extract data
        # array dims = (time lat lon depth) - (!!!incorrect in file!!!!)
        # lon== 1440, lat=720 so time, depth, lat, lon
        if debug:
            print(rootgrp[var2use].shape,  lat_ind, lon_ind, depth)
        file_data = rootgrp[var2use][0, depth, ...]
        file_data_ = file_data[lat_ind, lon_ind]
        # What is the data is masked for this location?
        if isinstance(file_data_, np.ma.core.MaskedConstant):
            ptr_str = 'interpolating to get {} for {} (buffer_CORDS={})'
            if verbose:
                print(ptr_str.format(var2use, Data_key_ID_, buffer_CORDS))
            # Make sure input data is (lon, lat)! (CF netCDF will be lat, lon)!
            # e.g. (360, 180) for 1x1 vs. (180, 360)
            file_data = file_data.T
            # Try Radial Basis Function Interpolation / Kernel Smoothing
            try:
                # Now interpolate and extract data
                file_data_ = AC.interpolate_sparse_grid2value(
                    X_CORDS=file_lonc, Y_CORDS=file_latc, X=lon, Y=lat,
                    XYarray=file_data, buffer_CORDS=buffer_CORDS)
                # Flag value if interpolated
                flagged = 'Interpolated (buffer={})'.format(buffer_CORDS)
            except:
                # Try again but with twice the buffer_CORDS
                try:
                    # Now interpolate and extract data
                    file_data_ = AC.interpolate_sparse_grid2value(
                        X_CORDS=file_lonc, Y_CORDS=file_latc, X=lon, Y=lat,
                        XYarray=file_data, buffer_CORDS=buffer_CORDS*2)
                    # Flag value if interpolated
                    flagged = 'Interpolated 2nd time(buffer={})'
                    flagged = flagged.format(buffer_CORDS*2)
                except:
                    flagged = 'FAILED INTERPOLATION!'
        else:
            flagged = 'False'
    if rtn_flag:
        return file_data_, flagged
    else:
        return file_data_


def get_WOA_Dissolved_O2_4indices(lat_idx=None, lon_idx=None, month=None,
                                  var2use='o_an', verbose=True, debug=False):
    """
    Extract Wold ocean atlas (WOA) climatology value for dissolved O2

    Parameters
    -------
    lat_idx (list): indicies for latitude
    lon_idx (list): indicies for longitude
    month (int): month number (1-12)
    debug (boolean): print out debug information?
    var2use (str): var to extract from NetCDF

    Returns
    -------
    (array)

    NOTES
    ---
     - default var to use is n_an
     - o_an:long_name = "Objectively analyzed mean fields for
     volume_fraction_of_oxygen_in_sea_water at standard depth levels." ;
     - o_mn:long_name = "Average of all unflagged interpolated values at each
    standard depth level for volume_fraction_of_oxygen_in_sea_water in each
    grid-square which contain at least one measurement." ;
    """
    # Set folder that files are in
    folder = get_file_locations('data_root') + '/WOA_2013/Dissolved_O2_1x1/'
    # Select the correct file
    filename = 'woa13_all_o{:0>2}_01.nc'.format(month)
    # Fix depth = 0 for now...
    depth = 0
    # Open file
    with Dataset(folder+filename, 'r') as rootgrp:
        # Extract data for indices
        # array dims = (time lat lon depth) - (!!!incorrect in file!!!!)
        # lon== 1440, lat=720 so time, depth, lat, lon
        file_data = rootgrp[var2use][0, depth, ...]
        # Extract for location(s)
        file_data_ = file_data[lat_idx, lon_idx]
    return file_data_


def get_WOA_Dissolved_O2_4_loc(lat=None, lon=None, month=None, var2use='o_an',
                               buffer_CORDS=5, rtn_flag=True, Data_key_ID_=None,
                               verbose=True, debug=False):
    """
    Extract Wold ocean atlas (WOA) climatology value for dissolved O2

    Parameters
    -------
    lat (float): latitude degrees north
    lon (float): latitude degress east
    month (int): month number (1-12)
    Data_key_ID_ (str): ID for input data point
    debug (boolean): print out debug information?
    _fill_value (float): fill value in NetCDF
    buffer_CORDS (int): value (degrees lat/lon) added to loc define edges of
        rectangle used for interpolation.
    var2use (str): var to extract from NetCDF

    Returns
    -------
    (float), (str)

    NOTES
    ---
     - default var to use is n_an
     - o_an:long_name = "Objectively analyzed mean fields for
     volume_fraction_of_oxygen_in_sea_water at standard depth levels." ;
     - o_mn:long_name = "Average of all unflagged interpolated values at each
    standard depth level for volume_fraction_of_oxygen_in_sea_water in each
    grid-square which contain at least one measurement." ;
    """
    # Set folder that files are in
    folder = get_file_locations('data_root') + '/WOA_2013/Dissolved_O2_1x1/'
    # Select the correct file
    filename = 'woa13_all_o{:0>2}_01.nc'.format(month)
    # Open file
    with Dataset(folder+filename, 'r') as rootgrp:
        # Get index of cell mid point closest to obs lat
        #         half_grid_cell = (rootgrp['lat_bnds'][0][0]-rootgrp['lat_bnds'][0][1])/2
        #         file_latc = rootgrp['lat']+abs(half_grid_cell)
        #         lat_ind = AC.find_nearest_value( file_latc, lat )
        # Use lower-left coordinate system
        file_lat = rootgrp['lat']
        lat_ind = AC.find_nearest_value(file_lat, lat)
        # Get index of cell mid point closest to obs lat
#         half_grid_cell = (rootgrp['lon_bnds'][0][0]-rootgrp['lon_bnds'][0][1])/2
#         file_lonc = rootgrp['lon']+abs(half_grid_cell)
#         lon_ind = AC.find_nearest_value( file_lonc, lon )
        # Use lower-left coordinate system
        file_lon = rootgrp['lon']
        lon_ind = AC.find_nearest_value(file_lon, lon)
        # Fix depth = 0 for now...
        depth = 0
        # get indices of the data array for the provided lat and lon
        # now extract data
        # array dims = (time lat lon depth) - (!!!incorrect in file!!!!)
        # lon== 1440, lat=720 so time, depth, lat, lon
        if debug:
            print(rootgrp[var2use].shape,  lat_ind, lon_ind, depth)
        file_data = rootgrp[var2use][0, depth, ...]
        file_data_ = file_data[lat_ind, lon_ind]
        # What is the data is masked for this location?
        if isinstance(file_data_, np.ma.core.MaskedConstant):
            ptr_str = 'interpolating to get {} for {} (buffer_CORDS={})'
            if verbose:
                print(ptr_str.format(var2use, Data_key_ID_, buffer_CORDS))
            # Make sure input data is (lon, lat)! (CF netCDF will be lat, lon)!
            # e.g. (360, 180) for 1x1 vs. (180, 360)
            file_data = file_data.T
            # Try Radial Basis Function Interpolation / Kernel Smoothing
            try:
                # Now interpolate and extract data
                file_data_ = AC.interpolate_sparse_grid2value(
                    X_CORDS=file_lonc, Y_CORDS=file_latc, X=lon, Y=lat,
                    XYarray=file_data, buffer_CORDS=buffer_CORDS)
                # Flag value if interpolated
                flagged = 'Interpolated (buffer={})'.format(buffer_CORDS)
            except:
                # Try again but with twice the buffer_CORDS
                try:
                    # Now interpolate and extract data
                    file_data_ = AC.interpolate_sparse_grid2value(
                        X_CORDS=file_lonc, Y_CORDS=file_latc, X=lon, Y=lat,
                        XYarray=file_data, buffer_CORDS=buffer_CORDS*2)
                    # Flag value if interpolated
                    flagged = 'Interpolated 2nd time(buffer={})'
                    flagged = flagged.format(buffer_CORDS*2)
                except:
                    flagged = 'FAILED INTERPOLATION!'
        else:
            flagged = 'False'
    if rtn_flag:
        return file_data_, flagged
    else:
        return file_data_
