"""

Generic analysis and processing of model data output/input in sparse2spatial

"""

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import geopandas
from rasterio import features
from affine import Affine
# import AC_tools (https://github.com/tsherwen/AC_tools.git)
import AC_tools as AC
# Internal loads within s2s
import sparse2spatial.utils as utils


def add_loc_ocean2df(df=None, LatVar='lat', LonVar='lon'):
    """
    Add the ocean of a location to dataframe

    Parameters
    -------
    df (pd.DataFrame): DataFrame of data
    LatVar (str): variable name in DataFrame for latitude
    LonVar (str): variable name in DataFrame for longitude

    Returns
    -------
    (pd.DataFrame)
    """
    from geopandas.tools import sjoin
    # Get the shapes for the ocean
    featurecla = 'ocean'
    group = AC.get_shapes4oceans(rtn_group=True, featurecla=featurecla)
    # Turn the dataframe into a geopandas dataframe
    gdf = geopandas.GeoDataFrame(
        df, geometry=geopandas.points_from_xy(df[LonVar], df[LatVar]))
    # Work out if any of the points are within the polys
    pointInPolys = sjoin(gdf, group, how='left')
    # Check how many were assigned to a region
    Nnew = float(pointInPolys['name'].dropna().shape[0])
    N = float(df.shape[0])
    if N != Nnew:
        pstr = 'WARNING: Only {:.2f}% assigned ({} of {})'
        print(pstr.format((Nnew/N)*100, int(Nnew), int(N)))
    # Add the ocean assignment
    df[featurecla] = pointInPolys['name'].values
    return df


def mk_NetCDF_of_global_oceans(df=None, LatVar='lat', LonVar='lon',
                               save2NetCDF=False):
    """
    Add the regional location of observations to dataframe

    Parameters
    -------
    df (pd.DataFrame): DataFrame of data
    LatVar (str): variable name in DataFrame for latitude
    LonVar (str): variable name in DataFrame for longitude

    Returns
    -------
    (pd.DataFrame)
    """
    # Get AC_tools location, then set example data folder location
    import os
    import xarray as xr
    import inspect
    filename = inspect.getframeinfo(inspect.currentframe()).filename
    path = os.path.dirname(os.path.abspath(filename))
    folder = path+'/data/LM/LANDMAP_LWI_ctm_0125x0125/'
    # Get coords from LWI 0.125x0.125 data and remove the time dimension
    ds = xr.open_dataset(folder+'ctm.nc')
    ds = ds.mean(dim='time')
    # Add a raster array for the oceans
    ds = AC.add_raster_of_oceans2ds(ds, test_plot=True, country=country)
    # save as a NetCDF?
    if save2NetCDF:
        ds.to_netcdf()
    else:
        return ds


def get_stats_on_spatial_predictions_4x5_2x25(ds=None, res='4x5', ex_str='',
                                              target='Iodide',
                                              use_annual_mean=True,
                                              filename=None,
                                              folder=None,
                                              just_return_df=False,
                                            var2template='Chance2014_STTxx2_I',
                                              ):
    """
    Evaluate the spatial predictions between models at a resolution of 4x5 or 2x2.5

    Parameters
    -------
    target (str): Name of the target variable (e.g. iodide)
    res (str): horizontal resolution of dataset (e.g. 4x5)
    var2template (str): variable to use a template for making new variables in ds
    use_annual_mean (bool): use the annual mean of the variable

    Returns
    -------

    Notes
    -----
    """
    if isinstance(ds, type(None)):
        # If filename or folder not given, then use defaults
        if isinstance(filename, type(None)):
            filename = 's2s_predicted_{}_{}.nc'.format(target, res)
        if isinstance(folder, type(None)):
            data_root = utils.get_file_locations('data_root')
            folder = '{}/{}/outputs/'.format(data_root, target)
        ds = xr.open_dataset(folder + filename)
    # variables to consider
    vars2plot = list(ds.data_vars)
    # add LWI and surface area to array
    ds = utils.add_LWI2array(ds=ds, var2template=var2template)
    IS_WATER = ds['IS_WATER'].mean(dim='time')
    # -- get general annual stats in a dataframe
    df = pd.DataFrame()
    for var_ in vars2plot:
        ds_tmp = ds[var_].copy()
        # take annual average
        if use_annual_mean:
            ds_tmp = ds_tmp.mean(dim='time')
        # mask to only consider (100%) water boxes
        arr = ds_tmp.values
        arr = arr[(IS_WATER == True)]
        # sve to dataframe
        df[var_] = pd.Series(arr.flatten()).describe()
    # Get area weighted mean
    vals = []
    for var_ in vars2plot:
        ds_tmp = ds[var_]
        # take annual average
        if use_annual_mean:
            ds_tmp = ds_tmp.mean(dim='time')
        # mask to only consider (100%) water boxes
        arr = np.ma.array(ds_tmp.values, mask=~(LWI == 0).T)
        # also mask s_area
        s_area_tmp = np.ma.array(s_area, mask=~(LWI == 0))
        # save value
        vals += [AC.get_2D_arr_weighted_by_X(arr, s_area=s_area_tmp.T)]
    # Add area weighted mean to df
    df = df.T
    df['mean (weighted)'] = vals
    df = df.T
    # Save or just return the values
    file_save = 's2s_annual_stats_global_ocean_{}{}.csv'.format(res, ex_str)
    if just_return_df:
        return df
    df.T.to_csv(file_save)


def get_stats_on_spatial_predictions_4x5_2x25_by_lat(res='4x5', ex_str='',
                                                     target='Iodide',
                                                     use_annual_mean=False,
                                                     filename=None,
                                                     folder=None, ds=None,
                                            var2template='Chance2014_STTxx2_I',
                                                     debug=False):
    """
    Evaluate the spatial predictions between models, binned by latitude

    Parameters
    -------
    target (str): Name of the target variable (e.g. iodide)
    res (str): horizontal resolution of dataset (e.g. 4x5)
    debug (bool): print out debugging output?
    var2template (str): variable to use a template for making new variables in ds
    use_annual_mean (bool): use the annual mean of the variable

    Returns
    -------
    (pd.DataFrame)
    """
    if isinstance(ds, type(None)):
        # If filename or folder not given, then use defaults
        if isinstance(filename, type(None)):
            filename = 's2s_predicted_{}_{}.nc'.format(target, res)
        if isinstance(folder, type(None)):
            data_root = utils.get_file_locations('data_root')
            folder = '{}/{}/outputs/'.format(data_root, target)
        ds = xr.open_dataset(folder + filename)
    # Variables to consider
    vars2analyse = list(ds.data_vars)
    # Add LWI to array
    ds = utils.add_LWI2array(ds=ds, var2template=var2template, res=res)
    # - Get general annual stats
    df = pd.DataFrame()
    # take annual average
    if use_annual_mean:
        ds_tmp = ds.mean(dim='time')
    else:
        ds_tmp = ds
    for var_ in vars2analyse:
        # Mask to only consider (100%) water boxes
        arr = ds_tmp[var_].values
        if debug:
            print(arr.shape, (ds_tmp['IS_WATER'] == False).shape)
        arr[(ds_tmp['IS_WATER'] == False).values] = np.NaN
        # Update values to include np.NaN
        ds_tmp[var_].values = arr
        # Setup series objects to hold stats
        s_mean = pd.Series()
        s_75 = pd.Series()
        s_50 = pd.Series()
        s_25 = pd.Series()
        # Loop by latasave to dataframe
        for lat_ in ds['lat'].values:
            vals = ds_tmp[var_].sel(lat=lat_).values
            stats_ = pd.Series(vals.flatten()).dropna().describe()
            # At poles all values will be the same (masked) value
#            if len( set(vals.flatten()) ) == 1:
#                pass
#            else:
            # save quartiles and mean
    #            try:
            s_mean[lat_] = stats_['mean']
            s_25[lat_] = stats_['25%']
            s_75[lat_] = stats_['75%']
            s_50[lat_] = stats_['50%']
    #            except KeyError:
    #                print( 'Values not considered for lat={}'.format( lat_ ) )
        # Save variables to DataFrame
        var_str = '{} - {}'
        stats_dict = {'mean': s_mean, '75%': s_75, '25%': s_25, 'median': s_50}
        for stat_ in stats_dict.keys():
            df[var_str.format(var_, stat_)] = stats_dict[stat_]
    return df


def get_spatial_predictions_0125x0125_by_lat(use_annual_mean=False, ds=None,
                                             target='Iodide',
                                             debug=False, res='0.125x0.125'):
    """
    Evaluate the spatial predictions between models

    Parameters
    -------
    target (str): Name of the target variable (e.g. iodide)
    res (str): horizontal resolution of dataset (e.g. 4x5)
    debug (bool): print out debugging output?
    var2template (str): variable to use a template for making new variables in ds
    use_annual_mean (bool): use the annual mean of the variable

    Returns
    -------
    (pd.DataFrame)

    Notes
    -----

    """
    # ----
    # get data
    if isinstance(ds, type(None)):
        filename = 's2s_predicted_{}_{}.nc'.format(target, res)
        folder = '/shared/earth_home/ts551/labbook/Python_progs/'
    #    ds = xr.open_dataset( folder + filename )
        ds = xr.open_dataset(filename)
    # variables to consider
    vars2analyse = list(ds.data_vars)
    # add LWI to ds
    vars2plot = list(ds.data_vars)
    # add LWI and surface area to array
    ds = utils.add_LWI2array(
        ds=ds, res=res, var2template='Chance2014_STTxx2_I')
    # ----
    df = pd.DataFrame()
    # -- get general annual stats
    # take annual average
    if use_annual_mean:
        ds_tmp = ds.mean(dim='time')
    else:
        ds_tmp = ds
    for var_ in vars2analyse:
        # mask to only consider (100%) water boxes
        arr = ds_tmp[var_].values
        if debug:
            print(arr.shape, (ds_tmp['IS_WATER'] == False).shape)
        arr[(ds_tmp['IS_WATER'] == False).values] = np.NaN
        # update values to include np.NaN
        ds_tmp[var_].values = arr
        # setup series objects to hold stats
        s_mean = pd.Series()
        s_75 = pd.Series()
        s_50 = pd.Series()
        s_25 = pd.Series()
        s_std = pd.Series()
        # loop by latasave to dataframe
        for lat_ in ds['lat'].values:
            vals = ds_tmp[var_].sel(lat=lat_).values
            stats_ = pd.Series(vals.flatten()).dropna().describe()
            # save quartiles and mean
            s_mean[lat_] = stats_['mean']
            s_25[lat_] = stats_['25%']
            s_75[lat_] = stats_['75%']
            s_50[lat_] = stats_['50%']
            s_std[lat_] = stats_['std']
        # Save variables to DataFrame
        var_str = '{} - {}'
        stats_dict = {
            'mean': s_mean, '75%': s_75, '25%': s_25, 'median': s_50,
            'std': s_std,
        }
        for stat_ in stats_dict.keys():
            df[var_str.format(var_, stat_)] = stats_dict[stat_]
    return df


def get_stats_on_spatial_predictions_0125x0125(ds=None,
                                               use_annual_mean=True,
                                               target='Iodide',
                                               ex_str='',
                                               vars2use=None,
                                               save2csv=True,
                                               filename2save=None,
                                               res='0.125x0.125',
                                            var2template='Chance2014_STTxx2_I',
                                               debug=False):
    """
    Evaluate the spatial predictions between models at 0.125x0.125

    Parameters
    -------
    target (str): Name of the target variable (e.g. iodide)
    debug (bool): print out debugging output?
    (above argument is a iodide specific option - remove this)
    just_return_df (bool): just return the data as dataframe
    ex_str (str): extra string to include in file name to save data
    use_annual_mean (bool): use the annual mean of the variable for statistics
    var2template (str): variable to use a template for making new variables in ds

    Returns
    -------
    (pd.DataFrame)
    """
    # Variables to consider
    if isinstance(vars2use, type(None)):
        vars2use = list(ds.data_vars)
    # Add LWI and surface area to array
    ds = utils.add_LWI2array(ds=ds, res=res, var2template=var2template)
    # Set a name for output to saved as
    if isinstance(filename2save, type(None)):
        FileStr = 's2s_annual_stats_global_ocean_{}{}'
        filename2save = FileStr.format(res, ex_str)
    # build an array with general statistics
    df = pd.DataFrame()
    # -- get general annual stats
    # Take annual average over time (if using annual mean)
    if use_annual_mean:
        ds_tmp = ds.mean(dim='time')
    for var_ in vars2use:
        # mask to only consider (100%) water boxes
        arr = ds_tmp[var_].values
        arr = arr[(ds_tmp['IS_WATER'] == True)]
        # save to dataframe
        df[var_] = pd.Series(arr.flatten()).describe()
    # Get area weighted mean too
    vals = []
    # Take annual average over time (if using annual mean) -
    # Q: why does this need to be done twice separately?
    if use_annual_mean:
        ds_tmp = ds.mean(dim='time')
    for var_ in vars2use:
        # Mask to only consider (100%) water boxes
        mask = ~(ds_tmp['IS_WATER'] == True)
        arr = np.ma.array(ds_tmp[var_].values, mask=mask)
        # Also mask surface area (s_area)
        s_area_tmp = np.ma.array(ds_tmp['AREA'].values, mask=mask)
        # Save value to list
        vals += [AC.get_2D_arr_weighted_by_X(arr, s_area=s_area_tmp)]
    # Add area weighted mean to df
    df = df.T
    df['mean (weighted)'] = vals
    df = df.T
    # save the values?
    if save2csv:
        df.T.to_csv(filename2save+'.csv')
    #  just return the dataframe of global stats
    return df


def add_ensemble_avg_std_to_dataset(res='0.125x0.125', RFR_dict=None,
                                    target='Iodide',
                                    stats=None, ds=None, topmodels=None,
                                    var2template='Chance2014_STTxx2_I',
                                    var2use4Ensemble='Ensemble_Monthly_mean',
                                    var2use4std='Ensemble_Monthly_std',
                                    save2NetCDF=True):
    """
    Add ensemble average and std to dataset

    Parameters
    -------
    target (str): Name of the target variable (e.g. iodide)
    var2use4Ensemble (str): variable name to use for ensemble prediction
    var2use4Std (str): variable name to use for ensemble prediction's std dev.
    var2template (str): variable to use a template to make new variables
    res (str): horizontal resolution of dataset (e.g. 4x5)
    topmodels (list): list of models to include in ensemble prediction
    save2NetCDF (bool): save the dataset as NetCDF file
    RFR_dict (dict): dictionary of core variables and data
    var2template (str): variable to use a template for making new variables in ds

    Returns
    -------
    (xr.Dataset)
    """
    # Get existing dataset from NetCDF if ds not provided
    filename = 's2s_predicted_{}_{}.nc'.format(target, res)
    if isinstance(ds, type(None)):
        data_root = utils.get_file_locations('data_root')
        folder = '{}/{}/'.format(data_root, target)
        ds = xr.open_dataset(folder + filename)
    # Just use top 10 models are included
    # ( with derivative variables )
    if isinstance(topmodels, type(None)):
        # extract the models...
        if isinstance(RFR_dict, type(None)):
            RFR_dict = build_or_get_models()
        # Get list of
        topmodels = get_top_models(
            RFR_dict=RFR_dict, vars2exclude=['DOC', 'Prod'])
    # Now get average concentrations and std dev. per month
    avg_ars = []
    std_ars = []
    for month in range(1, 13):
        ars = []
        for var in topmodels:
            ars += [ds[var].sel(time=(ds['time.month'] == month)).values]
        # Concatenate the models
        arr = np.concatenate(ars, axis=0)
        # Save the monthly average and standard deviation
        avg_ars += [np.ma.mean(arr, axis=0)]
        std_ars += [np.ma.std(arr, axis=0)]
    # Combine the arrays and then make the model variable
    # 1st Template an existing variable, then overwrite
    ds[var2use4Ensemble] = ds[var2template].copy()
    ds[var2use4Ensemble].values = np.stack(avg_ars)
    # And repeat for standard deviation
    ds[var2use4std] = ds[var2template].copy()
    ds[var2use4std].values = np.stack(std_ars)
    # Save the list of models used to make ensemble to array
    attrs = ds.attrs.copy()
    attrs['Ensemble_members ({})'.format(
        var2use4Ensemble)] = ', '.join(topmodels)
    ds.attrs = attrs
    # Save to NetCDF
    if save2NetCDF:
        ds.to_netcdf(filename)
    else:
        return ds
