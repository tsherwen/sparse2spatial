"""
Shared utility functions for sparse2 spatial
"""
import getpass
import platform
import numpy as np
import pandas as pd
from netCDF4 import Dataset


def mk_LWI_avg_array():
    """
    Make an array of average Land Water Ice (LWI) indices from NASA "nature run" output
    """
    import glob
    import xarray as xr
    folder = '/work/home/ts551/YARCC_TEMP_DIR_ON_EARTH0/data/NASA/LWI/'
    files = glob.glob(folder+'*ctm*nc')
    ds_l = [xr.open_dataset(i) for i in files]
    ds = xr.concat(ds_l)
    savename = 'nature_run_lev_72_res_0.125_spec_LWI_all_ctm.nc'
    ds.to_netcdf(savename)
    savename = 'nature_run_lev_72_res_0.125_spec_LWI_avg_ctm.nc'
    ds.mean(dim='time').to_netcdf(savename)
    #
    folder = '/shared/earthfs//NASA/nature_run/LWI/monthly/'
    files = glob.glob(folder+'*nc')
    dates = [(int(i[-14:-14+4]), int(i[-9:-9+2]), 1) for i in files]
    dates = [datetime.datetime(*i) for i in dates]
    ds.rename({'concat_dims': 'time'}, inplace=True)
    ds['time'] = dates
    ds.rename({'lwi': 'LWI'}, inplace=True)
    ds.to_netcdf('nature_run_lev_72_res_0.125_spec_LWI_monthly_ctm.nc')


def mk_da_of_predicted_values(model=None, modelname=None, res='4x5', target='Iodide',
                              dsA=None, testing_features=None):
    """
    Make a dataset of 3D predicted values from model
    """
    # Local variables
    target_name = target
    # Get feature values for resolution
    if isinstance(dsA, type(None)):
        data_root = get_file_locations('data_root')
        filename = 'Oi_prj_feature_variables_{}.nc'.format(res)
        dsA = xr.open_dataset(data_root + filename)
    # Take coordinate variables from dsA
    lat = dsA['lat'].values
    lon = dsA['lon'].values
    dates = dsA['time'].values
    # Loop and predict by month
    months = np.arange(1, 13)
    da_l = []
    for n_month, month in enumerate(months):
        # Select data for month
        ds = dsA.sel(time=(dsA['time.month'] == month))
        # Remove time (all values have only 1 time (confirm with asssert)
        assert len(ds.time) == 1, 'Only 1 time should be selected!'
        ds = ds.mean(dim='time')
        # Extract feature variables to 2D DataFrame
        df = pd.DataFrame()
        for fvar in testing_features:
            #
            df_tmp = pd.DataFrame(ds[fvar].values)
            df_tmp.columns = lon
            df_tmp.index = lat
            # construct Series by unstacking
            df[fvar] = df_tmp.unstack()
        # Now predict values with feature variables
        df[target] = model.predict(df[testing_features].values)
        # Now re-build into a 3D dataset
        df = df[target].unstack()
        # Sort lat to be -90 to 90
#        df = df[lat]
        # Extract array to be included in dataset
        arr = df.values.T[None, ...]
#        df = pd.DataFrame( df_tmp, index=[lat,lon] ).unstack()
        # Convert to Dataset
        date = dates[n_month]
        da_l += [
            xr.Dataset(
                data_vars={modelname: (['time', 'lat', 'lon', ], arr)},
                coords={'lat': lat, 'lon': lon, 'time': [date]})
        ]
    # Combine to a single dataArray
    ds = xr.concat(da_l, dim='time')
    return ds


def mk_NetCDF_of_surface_target_by_month4param(res='4x5', target='iodide',
                                               param='Chance2014'):
    """
    Make a NetCDF of (monthly) target fields from previous params
    """
    # res='4x5'; extr_str='tree_X_STRAT_JUST_TEMP_K_GEBCO_SALINTY'
    import xarray as xr
    from time import gmtime, strftime
    cal_dict = {
        'Macdonald2014': calc_iodide_MacDonald2014,
        'Chance2014': calc_iodide_chance2014_STTxx2_I,
    }
    # Get the model
    testing_features = ['WOA_TEMP']
    # Initialise a dictionary to store data
    ars_dict = {}
    months = np.arange(1, 13)
    for month in months:
        # get array of predictor for lats and lons (at res... )
        df_predictors = get_predict_lat_lon_array(res=res, month=month)
        # now make predictions for target ("y") from loaded predictors
        TEMP = df_predictors[testing_features].values
        target_predictions = cal_dict[param](TEMP)
        # Save values
        ars_dict[month] = mk_uniform_2D_array(df_predictors=df_predictors,
                                              target_name=[target], res=res,
                                              target_predictions=target_predictions)
    # Get coordinates
    lon, lat, alt = AC.get_latlonalt4res(res=res)
    # Make a dataset of arrays
    da_l = []
    for month in months:
        arr = ars_dict[month].T[None, ...]
        da_l += [xr.Dataset(
            data_vars={target: (['time', 'lat', 'lon', ], arr)},
            coords={'lat': lat, 'lon': lon, 'time': [month]})]
    # Concatenate to a single dataset
    ds = xr.concat(da_l, dim='time')
    # Update time ...
    ds = update_time_in_NetCDF2save(ds)
    # Add attributes for target variable and global variables
    ds = add_attrs2target_ds(ds, varname=target)
    # Add units etc for all of the dataset variables
    for var2use in ds.data_vars:
        ds = add_attrs2target_ds(ds, add_global_attrs=False,
                                 varname=var2use)
    # save to NetCDF
    filename = 'Oi_prj_{}_monthly_param_{}_{}.nc'.format(target, res, param)
    ds.to_netcdf(filename, unlimited_dims={'time': True})


def mk_NetCDF_of_surface_target_by_month(res='4x5', extr_str='', target='iodide'):
    """
    Make a NetCDF of predicted data fields by month
    """
    # res='4x5'; extr_str='tree_X_STRAT_JUST_TEMP_K_GEBCO_SALINTY'
    import xarray as xr
    from time import gmtime, strftime
    # Get the model
    model = get_current_model(extr_str=extr_str)
    testing_features = ['WOA_TEMP_K', 'WOA_Salinity', 'Depth_GEBCO']

    # Initialise a dictionary to store data
    ars_dict = {}
    months = np.arange(1, 13)
    for month in months:
        # get array of predictor for lats and lons (at res... )
        df_predictors = get_predict_lat_lon_array(res=res, month=month)
        # now make predictions for target ("y") from loaded predictors
        target_predictions = model.predict(df_predictors[testing_features])
        # Convert output vector to 2D lon/lat array
        model_name = "RandomForestRegressor '{}'"
        model_name = model_name.format('+'.join(testing_features))
        ars_dict[month] = mk_uniform_2D_array(df_predictors=df_predictors,
                                              target_name=[target], res=res,
                                              target_predictions=target_predictions)
    # Get coordinates
    lon, lat, alt = AC.get_latlonalt4res(res=res)
    # Make a dataset of arrays
    da_l = []
    for month in months:
        arr = ars_dict[month].T[None, ...]
        da_l += [xr.Dataset(
            data_vars={target: (['time', 'lat', 'lon', ], arr)},
            coords={'lat': lat, 'lon': lon, 'time': [month]})]
    # Concatenate to a single dataset
    ds = xr.concat(da_l, dim='time')
    # Update time ...
#    sdate = datetime.datetime(1970, 1, 1) # Unix time
#    da['time'] = [ AC.add_months( sdate, i-1) for i in months ]
    sdate = datetime.datetime(1985, 1, 1)  # Climate model tiem
    ds['time'] = [AC.add_months(sdate, i-1) for i in months]
    # Update to hours since X
    hours = [(AC.dt64_2_dt([i])[0] - sdate).days *
             24. for i in ds['time'].values]
    ds['time'] = hours
    # Now turn into a DataArray?
#    ds = da.to_dataset()
    attrs_dict = {'units': 'hours since 1985-01-01 00:00:00'}
    ds['time'].attrs = attrs_dict
    # Add attributes
    ds = add_attrs2target_ds(ds, varname=extr_str, convert_to_kg_m3=True)
    # Save to NetCDF
    filename = 'Oi_prj_{}_monthly_param_{}.nc'.format(target, res)
    ds.to_netcdf(filename, unlimited_dims={'time': True})


def add_units2ds(ds):
    """
    Add input ancillary units
    """
    unit_dict = {
        u'DOC': 'ADD THIS',
        u'DOCaccum': 'ADD THIS',
        u'Depth_GEBCO': 'ADD THIS',
        u'Prod': 'ADD THIS',
        u'SWrad': 'ADD THIS',
        u'SeaWIFs_ChlrA': "mg m$^{-3}$",
        u'WOA_Dissolved_O2': 'ADD THIS',
        u'WOA_MLDpd': 'ADD THIS',
        u'WOA_MLDpd_max': 'ADD THIS',
        u'WOA_MLDpd_sum': 'ADD THIS',
        u'WOA_MLDpt': 'ADD THIS',
        u'WOA_MLDpt_max': 'ADD THIS',
        u'WOA_MLDpt_sum': 'ADD THIS',
        u'WOA_MLDvd': 'ADD THIS',
        u'WOA_MLDvd_max': 'ADD THIS',
        u'WOA_MLDvd_sum': 'ADD THIS',
        u'WOA_Nitrate': "$\mu$M",
        u'WOA_Phosphate': 'ADD THIS',
        u'WOA_Salinity': 'PSU',
        u'WOA_Silicate': 'ADD THIS',
        u'WOA_TEMP': '$^{o}$C',
        u'WOA_TEMP_K': '$^{o}$K',
    }
    # Loop data variables and add units
    for var_ in ds.data_vars:
        updated_units = False
        attrs = ds[var_].attrs.copy()
        if ('units' not in attrs.keys()):
            attrs['units'] = unit_dict[var_]
            updated_units = True
        if updated_units:
            ds[var_].attrs = attrs
    return ds


def interpolate_array_with_GRIDDATA(arr_, da=None):
    """
    Interpolate an array with scipy's griddata function
    """
    import gc
    from time import gmtime, strftime
    import time
#    from matplotlib.mlab import griddata
    from scipy.interpolate import griddata
    # Print timings
    time_now = strftime("%c", gmtime())
    print('Started intpolating @ {}'.format(time_now))
    # Select grid of interest
    subX = da['lon'].values
    subY = da['lat'].values
    # construt into 2D DataFrame
    df = pd.DataFrame(arr_)
    df.index = subY
    df.columns = subX
    # get just points that are known
    df = df.unstack().dropna()
    df = df.reset_index(level=[0, 1])
    # set the locations and data fro non-nan points
    x = df['level_0'].values
    y = df['level_1'].values
    z = df[0].values
    # mesh grid to axes
    # define grid.
    xi = subX
    yi = subY
    Xi, Yi = np.meshgrid(subX, subY)
    # grid the data. (using matplotlib.mlab's  griddata)
    # detail here: https://matplotlib.org/api/mlab_api.html#matplotlib.mlab.griddata
#    zi = griddata(x, y, z, xi, yi, interp='linear')
    # grid the data. (using scipys's  griddata)
    zi = griddata(zip(x, y), z, (Xi, Yi), method='nearest')
    # Overwrite values that are NaNs with interpolated values
    nans = np.isnan(arr_)
    arr_[nans] = zi[nans]
    # Clean memory
    gc.collect()
    # Print timings
    time_now = strftime("%c", gmtime())
    print('finished intpolating @ {}'.format(time_now))
    # Return the array
    return arr_


def interpolate_array_with_RBF(arr_, subX=None, subY=None):
    import gc
    from time import gmtime, strftime
    import time
    # Print timings
    time_now = strftime("%c", gmtime())
    print('Started intpolating @ {}'.format(time_now))
    # mesh grid to axes
    rr, cc = np.meshgrid(subX, subY)
    # Fill masked values with nans
    M = np.ma.filled(arr_, fill_value=np.nan)
    # only consider non nan values as values to interpolate with
    vals = ~np.isnan(M)
    # interpolate
    f = interpolate.Rbf(rr[vals], cc[vals], M[vals], function='linear')
    # Extract interpolation...
    interpolated = f(rr, cc)
    # Overwrite values that are NaNs with interpolated values
    arr_[~vals] = interpolated[~vals]
    # Clean memory
    gc.collect()
    # Print timings
    time_now = strftime("%c", gmtime())
    print('finished intpolating @ {}'.format(time_now))
    # return the array
    return arr_


def make_2D_RDF_of_gridded_data(res='1x1', X_locs=None, Y_locs=None,
                                Z_data=None):
    """ Make a 2D interpolation using RadialBasisFunctions """
    import numpy as np
    from scipy.interpolate import Rbf
    import matplotlib.pyplot as plt
    # --- Process dataframe here for now
    X_locs = df['Longitude'].values
    Y_locs = df['Latitude'].values
    Z_data = df['Iodide'].values
    # --- Degrade resolution
    if res == '1x1':
        X_COORDS, Y_COORDS, NIU = AC.get_latlonalt4res(res=res)
    # --- Remove double ups in data for now...
    print([len(i) for i in (X_locs, Y_locs)])
    # Degrade to 1x1 resolution...
    X_locs = [int(i) for i in X_locs]
    Y_locs = [int(i) for i in Y_locs]
    # Make a dictionary to remove double ups...
    Z_dict = dict(list(zip(list(zip(X_locs, Y_locs)), Z_data)))
    # Unpack
    locs = sorted(Z_dict.keys())
    Z_data = [Z_dict[i] for i in locs]
    X_locs, Y_locs = list(zip(*locs))
    print([len(i) for i in (X_locs, Y_locs)])
    # ---  Setup meshgrid...
    XI, YI = np.meshgrid(X_COORDS, Y_COORDS)
    # --- interpolate onto this...
    # Creating the interpolation function and populating the output matrix value
    rbf = Rbf(X_locs, Y_locs, Z_data, function='inverse')
    ZI = rbf(XI, YI)
    # Plotting the result
    n = plt.normalize(0.0, 100.0)
    plt.subplot(1, 1, 1)
    plt.pcolor(XI, YI, ZI)
#    plt.scatter(X_locs, Y_locs, 100, Z_data)
    plt.scatter(X_locs, Y_locs, 100, Z_data)
    plt.title('RBF interpolation')
    plt.xlim(-180, 180)
    plt.ylim(-90, 90)
    plt.colorbar()


def mk_uniform_2D_array(df_predictors=None, target_predictions=None,
                        res='4x5', lon_var='Longitude', lat_var='Latitude',
                        target_name=None,  debug=False):
    """
    Make a uniform 2D array from df containing some lat and lon values
    """
    # recombine into a 2D array
    coord_vars = [df_predictors[i].values for i in (lat_var, lon_var)]
    df_target = pd.DataFrame(coord_vars + [target_predictions]).T
    df_target.columns = [lat_var, lon_var] + target_name

    # --- Use pandas to stack array - FAILED due to double ups... - why?
    # load an empty dataframe with (All!) lats and lons as columns
#     df_template = AC.get_2D_df_of_lon_lats_and_time()
#     # set values to nan
#     fill_value = -999999.999
#     df_template[target_name[0]] = fill_value
#     df_template = df_template[columns]
    # fill template with values
#    df_template.update(df_target)
#    df_target = pd.DataFrame( df_template[target_name[0]].values, \
#        index=[df_template['lon'], df_template['lat'] ] )

    # --- Manually build up 2D array instead
    # Setup zero array to fill with target data
    lons, lats, NIU = AC.get_latlonalt4res(res=res)
    arr = np.zeros((len(lons), len(lats)))
    # Loop months...
#    months =
#    for month_ in months
#    ars = []
    # Loop Lons
    for lon_ in sorted(set(df_target[lon_var].values))[::-1]:
        # Select values for lon_
        sub_df = df_target[df_target[lon_var] == lon_]
        # Get index for lon
        lon_ind = AC.get_gc_lon(lon_, res=res)
        if debug:
            print(lon_, sub_df.shape, sub_df)
        # Loop lats and Extract values
        for lat_ in sorted(sub_df[lat_var].values)[::-1]:
            # Get index for lon
            lat_ind = AC.get_gc_lat(lat_, res=res)
            # Select values for lon_
            val = sub_df[sub_df[lat_var] == lat_][target_name[0]].values[0]
            if debug:
                print(lon_, lat_, lon_ind, lat_ind, val)
            # Fill in value
            arr[lon_ind, lat_ind] = val
    return arr


def transform_from_latlon(lat, lon):
    """
    Tranform from latitude and longitude
    """
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    trans = Affine.translation(lon[0], lat[0])
    scale = Affine.scale(lon[1] - lon[0], lat[1] - lat[0])
    return trans * scale


def rasterize(shapes, coords, fill=np.nan, **kwargs):
    """
    Rasterize a list of (geometry, fill_value) tuples onto the given
    xray coordinates. This only works for 1d latitude and longitude
    arrays.
    """
    transform = transform_from_latlon(coords['lat'], coords['lon'])
    out_shape = (len(coords['lat']), len(coords['lon']))
    raster = features.rasterize(shapes, out_shape=out_shape,
                                fill=fill, transform=transform,
                                dtype=float, **kwargs)
    return xray.DataArray(raster, coords=coords, dims=('lat', 'lon'))




def update_time_in_NetCDF2save(ds, convert_time2dt=False):
    """
    Update time of monthly output to be in NetCDF saveable format
    """
    # Climate model time
    sdate = datetime.datetime(1985, 1, 1)
    # Convert / setup time dim?
    if convert_time2dt:
        months = np.arange(1, 13)
        ds['time'] = [AC.add_months(sdate, i-1) for i in months]
    # Update to hours since X
    hours = [(AC.dt64_2_dt([i])[0] - sdate).days *
             24. for i in ds['time'].values]
    ds['time'] = hours
    attrs_dict = {'units': 'hours since 1985-01-01 00:00:00'}
    ds['time'].attrs = attrs_dict
    return ds


def add_attrs2target_ds_global_and_iodide_param(ds):
    """
    Helper func to add both global and iodide parm attrs
    """
    # add param values
    for var2use in ds.data_vars:
        ds = add_attrs2target_ds(ds, add_global_attrs=False, varname=var2use)
    # Add global attributes
    ds = add_attrs2target_ds(ds, add_varname_attrs=False)
    return ds


def add_attrs2target_ds(ds, convert_to_kg_m3=False, attrs_dict={},
                        varname='Ensemble_Monthly_mean',
                        add_global_attrs=True, add_varname_attrs=True,
                        update_varnames_to_remove_spaces=False,
                        global_attrs_dict={},
                        convert2HEMCO_time=False ):
    """
    Update attributes for iodide dataset saved as NetCDF
    """
    # --- Coordinate and global values
    if add_varname_attrs:
        # convert the units?
        if convert_to_kg_m3:
            # get surface array
#            print('Update of units not implimented')
#            sys.exit()
            # Convert units from nM to kg/m3 (=> M => mass => /m3 => /kg)
            ds[varname] = ds[varname]/1E9 * 127 * 1E3 / 1E3
            # for variable
            attrs_dict['units'] = "kg/m3"
            attrs_dict['units_longname'] = "kg/m3"
        else:
            # for variable
            attrs_dict['units'] = "nM"
            attrs_dict['units_longname'] = "Nanomolar"
        # add COARDS variables
        attrs_dict['add_offset'] = int(0)
        attrs_dict['scale_factor'] = int(1)
        attrs_dict['missing_value'] = float(-1e-32)
        attrs_dict['_FillValue'] = float(-1e-32)
        ds[varname].attrs = attrs_dict

    # --- Update Name for use in external NetCDFs
    if update_varnames_to_remove_spaces:
        for var_ in ds.data_vars:
            if ' ' in var_:
                print( 'removing spaces from {}'.format( var_ ) )
                new_varname = var_.replace(' ', '_')
                # make new var as a copy of the old one
                ds[new_varname] = ds[var_].copy()
                # now remove the old var
                del ds[var_]
            else:
                pass

    # --- Coordinate and global values
    if add_global_attrs:
        # for lat...
        attrs_dict = ds['lat'].attrs
        attrs_dict['long_name'] = "latitude"
        attrs_dict['units'] = "degrees_north"
        attrs_dict["standard_name"] = "latitude"
        attrs_dict["axis"] = "Y"
        ds['lat'].attrs = attrs_dict
        # and lon...
        attrs_dict = ds['lon'].attrs
        attrs_dict['long_name'] = "longitude"
        attrs_dict['units'] = "degrees_east"
        attrs_dict["standard_name"] = "longitude"
        attrs_dict["axis"] = "X"
        ds['lon'].attrs = attrs_dict
        # and time
        attrs_dict = ds['time'].attrs
        attrs_dict["standard_name"] = "time"
        attrs_dict['long_name'] = attrs_dict["standard_name"]
        attrs_dict["axis"] = "T"
        if convert2HEMCO_time:
            attrs_dict['units'] = 'hours since 2000-01-01 00:00:00'
            attrs_dict['calendar'] = 'standard'
            # Assume a generic year
            REFdatetime = datetime.datetime( 2000, 1, 1 )
            dts = [ datetime.datetime( 2000, i, 1 ) for i in range(1,13) ]
            hours = [(i-REFdatetime).days*24. for i in dts]
#            times = [ AC.add_months(REFdatetime, int(i) ) for i in range(13) ]
            ds['time'].values = hours
        ds['time'].attrs = attrs_dict
        # Add details to the global attribute dictionary
        History_str = 'Last Modified on: {}'
        global_attrs_dict['History'] = History_str.format(strftime("%B %d %Y", gmtime()))
        global_attrs_dict['Conventions'] = "COARDS"
        global_attrs_dict['Main parameterisation variable'] = varname
        global_attrs_dict['format'] = 'NetCDF-4'
        ds.attrs = global_attrs_dict
    return ds




# ---------------------------------------------------------------------------
# ------------------------- Global helper functions -------------------------
# ---------------------------------------------------------------------------
def check_plots4plotting():
    """
    Do a test plot of the colour cycle being used for plotting
    """
    # Get colours
    CB_color_cycle = AC.get_CB_color_cycle()
    CB_color_cycle += ['darkgreen']
    # Do a quick plots for these
    x = np.arange(10)
    for n_color, color in enumerate(CB_color_cycle):
        plt.plot(x, x*n_color, color=color)


def add_LWI2array(ds=None,  res='4x5', var2template='Chance2014_STTxx2_I',
                  inc_booleans_and_area=True):
    """
    Add Land/Water/Ice (LWI) values to xr.DataArray/xr.Dataset
    """
    if res == '0.125x0.125':
        ds = add_LWI2ds_0125x0125(ds=ds, res=res, var2template=var2template,
                                  inc_booleans_and_area=inc_booleans_and_area)
    elif (res == '4x5') or (res == '2x2.5'):
        ds = add_LWI2ds_2x25_4x5(ds=ds, res=res, var2template=var2template,
                                 inc_booleans_and_area=inc_booleans_and_area)
    else:
        print
        sys.exit('res not setup in add_LWI2array')
    return ds


def add_LWI2ds_0125x0125(ds, var2template='Chance2014_STTxx2_I',
                         res='0.125x0.125', inc_booleans_and_area=True):
    """
    Add Land/Water/Ice (LWI) values to xr.DataArray
    """
#    folderLWI = '/shared/earthfs//NASA/nature_run/LWI/monthly/'
#    filenameLWI = 'nature_run_lev_72_res_0.125_spec_LWI_monthly_ctm.nc'
    folderLWI = get_file_locations('AC_tools')
    folderLWI += '/data/LM/TEMP_NASA_Nature_run/'
    filenameLWI = 'ctm.nc'
    LWI = xr.open_dataset(folderLWI+filenameLWI)
    # updates dates (to be Jan=>Dec)
    new_dates = [datetime.datetime(1970, i, 1) for i in LWI['time.month']]
    LWI.time.values = new_dates
    # Sort by new dates
    LWI = LWI.loc[{'time': sorted(LWI.coords['time'].values)}]
#    LWI = AC.get_land_map(res=res)[...,0]
    if inc_booleans_and_area:
        ds['IS_WATER'] = ds[var2template].copy()
        ds['IS_WATER'].values = (LWI['LWI'] == 0)
        # add is land
        ds['IS_LAND'] = ds['IS_WATER'].copy()
        ds['IS_LAND'].values = (LWI['LWI'] == 1)
        # get surface area
        s_area = AC.calc_surface_area_in_grid(res=res)  # m2 land map
        ds['AREA'] = ds[var2template].mean(dim='time')
        ds['AREA'].values = s_area.T
    else:
        ds['LWI'] = LWI['LWI']
        # Update attributes too
        attrs = ds['LWI'].attrs.copy()
        attrs['long_name'] = 'Land/Water/Ice index'
        attrs['Detail'] = 'A Land-Water-Ice mask. It is 1 over continental areas, 0 over open ocean, and 2 over seaice covered ocean.'
        attrs['add_offset'] = int(0)
        attrs['scale_factor'] = int(1)
        attrs['missing_value'] = float(-1e-32)
        attrs['_FillValue'] = float(-1e-32)
        attrs['units'] = 'unitless'
        ds['LWI'].attrs = attrs
    return ds


def add_LWI2ds_2x25_4x5(ds, var2template='Chance2014_STTxx2_I',
                        res='0.125x0.125', inc_booleans_and_area=True):
    """
    Add Land/Water/Ice (LWI) values to xr.DataArray

    Parameters
    -------
    ds (xr.Dataset), xarray dataset to add LWI to
    res (str), horizontal resolution of dataset (e.g. 4x5)
    var2template (str): variable to use a template for making LWI variable
    inc_booleans_and_area (boolean), include extra booleans and surface area

    Returns
    -------
    (xr.Dataset)
    """
    # add LWI to array
    LWI = AC.get_land_map(res=res)[..., 0]
    LWI = np.array([LWI.T]*12)
    print(LWI.shape,  ds[var2template].shape)
    if inc_booleans_and_area:
        ds['IS_WATER'] = ds[var2template].copy()
        ds['IS_WATER'].values = (LWI == 0)
        # add is land
        ds['IS_LAND'] = ds['IS_WATER']
        ds['IS_LAND'].values = (LWI == 1)
        # get surface area
        s_area = AC.get_surface_area(res)[..., 0]  # m2 land map
        ds['AREA'] = ds[var2template].mean(dim='time')
        ds['AREA'].values = s_area.T
    else:
        ds['LWI'] = LWI['LWI']
        # Update attributes too
        attrs = ds['LWI'].attrs.copy()
        attrs['long_name'] = 'Land/Water/Ice index'
        attrs['Detail'] = 'A Land-Water-Ice mask. It is 1 over continental areas, 0 over open ocean, and 2 over seaice covered ocean.'
        ds['LWI'].attrs = attrs
    return ds


def v10_ClBrI_TRA_XX_2_name(TRA_XX):
    """
    Convert version 3.0 GEOS-Chem output to actual name
    """
    d = {
        1: 'NO', 2: 'O3', 3: 'PAN', 4: 'CO', 5: 'ALK4', 6: 'ISOP', 7: 'HNO3', 8: 'H2O2', 9: 'ACET', 10: 'MEK', 11: 'ALD2', 12: 'RCHO', 13: 'MVK', 14: 'MACR', 15: 'PMN', 16: 'PPN', 17: 'R4N2', 18: 'PRPE', 19: 'C3H8', 20: 'CH2O', 21: 'C2H6', 22: 'N2O5', 23: 'HNO4', 24: 'MP', 25: 'DMS', 26: 'SO2', 27: 'SO4', 28: 'SO4s', 29: 'MSA', 30: 'NH3', 31: 'NH4', 32: 'NIT', 33: 'NITs', 34: 'BCPI', 35: 'OCPI', 36: 'BCPO', 37: 'OCPO', 38: 'DST1', 39: 'DST2', 40: 'DST3', 41: 'DST4', 42: 'SALA', 43: 'SALC', 44: 'Br2', 45: 'Br', 46: 'BrO', 47: 'HOBr', 48: 'HBr', 49: 'BrNO2', 50: 'BrNO3', 51: 'CHBr3', 52: 'CH2Br2', 53: 'CH3Br', 54: 'MPN', 55: 'ISOPN', 56: 'MOBA', 57: 'PROPNN', 58: 'HAC', 59: 'GLYC', 60: 'MMN', 61: 'RIP', 62: 'IEPOX', 63: 'MAP', 64: 'NO2', 65: 'NO3', 66: 'HNO2', 67: 'BrCl', 68: 'Cl2', 69: 'Cl', 70: 'ClO', 71: 'HOCl', 72: 'HCl', 73: 'ClNO2', 74: 'ClNO3', 75: 'ClOO', 76: 'OClO', 77: 'Cl2O2', 78: 'CH3Cl', 79: 'CH2Cl2', 80: 'CHCl3', 81: 'BrSALA', 82: 'BrSALC', 83: 'CH3IT', 84: 'CH2I2', 85: 'CH2ICl', 86: 'CH2IBr', 87: 'HOI', 88: 'I2', 89: 'IBr', 90: 'ICl', 91: 'I', 92: 'IO', 93: 'HI', 94: 'OIO', 95: 'INO', 96: 'IONO', 97: 'IONO2', 98: 'I2O2', 99: 'I2O3', 100: 'I2O4', 101: 'ISALA', 102: 'ISALC', 103: 'AERI'
    }
    return d[int(TRA_XX[4:])]


def calc_iodide_MacDonald2014(TEMP):
    """
    Temp. (C) to Macdonald2014 parameterised [iodide] in nmol/dm^-3 (nM)
    """
    # Parameterisation is Arrhenius expression
    # y= 1.46E6 * exp(-9134.0/TEMP(K)) * 1E9
    # NOTE: temp. is converted from degC to DegK
    return (1.46E6 * np.exp((-9134.0 / (TEMP+273.15))))*1E9


def calc_iodide_chance2014_STTxx2_I(TEMP):
    """
    Temp. (C) to Chance2014 parameterised [iodide] in nmol/dm^-3 (nM)
    """
    # Parameterisation is a linear regression
    # y= 0.225(x**2) + 19
    return (0.225*(TEMP**2)) + 19.0


def calc_iodide_chance2014_Multivariate(TEMP=None, MOD_LAT=None, NO3=None,
                                        sumMLDpt=None, salinity=None):
    """
    Take variable and returns multivariate parameterised iodide from Chance2014
    """
    iodide = (0.28*TEMP**2) + (1.7*MOD_LAT) + (0.9*NO3) -  \
        (0.020*sumMLDpt) + (7.0*salinity) - 309
    return iodide


def is_number(s):
    """
    check if input is a number (check via conversion to string)
    """
    try:
        float(str(s))
        return True
    except ValueError:
        return False


def get_file_locations(input_var, file_and_path='./sparse2spatial.rc'):
    """
    Dictionary store of data/file locations
    """
    # Get a dictionary of paths
    d = read_settings_rc_file2dict(file_and_path=file_and_path)
    # Try to add the user name and and platform to dictionary
    try:
        host = platform.node()
        d['host'] = host
        user = getpass.getuser()
        d['user'] = user
    except:
        print('Failed to add user and host to dictionary')
    return d[input_var]


def convert_fullname_to_shortname(input=None, rtn_dict=False, invert=False):
    """
    Convert short names to long names

    Parameters
    -------
    input (str), input string to use as key to return dictionary value
    invert (float), reverse the key/pair of the dictionary
    rtn_dict (boolean), return the entire dictionary.

    Returns
    -------
    (str)
    """
    name_dict = {
        u'DOC': u'DOC',
        u'DOCaccum': u'Accum. DOC',
        u'Depth_GEBCO': u'Depth',
        u'Prod': u'Productivity',
        u'SWrad': u'SWrad',
        u'SeaWIFs_ChlrA': u'ChlrA',
        u'WOA_Dissolved_O2': u'O2$_{(aq)}$',
        u'WOA_MLDpd': u'MLDpd',
        u'WOA_MLDpd_max': u'MLDpd(max)',
        u'WOA_MLDpd_sum': u'MLDpd(sum)',
        u'WOA_MLDpt': u'MLDpt',
        u'WOA_MLDpt_max': u'MLDpt(max)',
        u'WOA_MLDpt_sum': u'MLDpt(sum)',
        u'WOA_MLDvd': u'MLDvd',
        u'WOA_MLDvd_max': u'MLDvd(max)',
        u'WOA_MLDvd_sum': u'MLDvd(sum)',
        u'WOA_Nitrate': u'Nitrate',
        u'WOA_Phosphate': u'Phosphate',
        u'WOA_Salinity': u'Salinity',
        u'WOA_Silicate': u'Silicate',
        u'WOA_TEMP': u'Temperature',
        u'WOA_TEMP_K': u'Temperature',
    }
    # Invert the dictionary
    if invert:
        return {v: k for k, v in list(name_dict.items())}
        # return the
    if rtn_dict:
        return name_dict
    else:
        return name_dict[input]


def read_settings_rc_file2dict( file_and_path ):
    """
    Read the settings file (e.g. './sparse2spatial.rc')
    """
    # Setup dictionary to store lines that have been read-in
    d = {}
    # Loop lines in file and read lines
    with open( file_and_path, 'r') as file:
        for line in file:
            if line.startswith("#"):
                pass
            else:
                try:
                    key, value = line.split (' : ')
                    d[key.strip()] = value.strip()
                except:
                    print( 'failed to read line in *.rc file:{}'.format(line) )
    # Return the resultant dictionary
    return d


def get_outlier_value(df=None, var2use='Iodide', check_full_df_used=True):
    """
    Get the upper outlier value for a given variable in a DataFrame

    Parameters
    -------
    check_full_df_used (bool): check the entire iodide observation is used for calc.
    var2use (str): var to check from NetCDF
    df (pd.DataFrame): DataFrame to check variable ("var2use") within

    Returns
    -------
    (float)

    Notes
    -----
     - outliers are definded here as values greater than the 3rd quartile plus 1.5 times
       the interquartile range (Frigge et al., 1989).
     -  Citation(s):
    Frigge, M., Hoaglin, D. C., and Iglewicz, B.: Some implementations of the boxplot,
    American Statistician, https://doi.org/10.1080/00031305.1989.10475612, 1989.
    """
    # Check to make sure that the full observations are used to calc the outlier
    if check_full_df_used:
        folder = get_file_locations('data_root')
        filename = 'Iodide_data_above_20m.csv'
        dfA = pd.read_csv(folder+filename)
        dfA = dfA.loc[np.isfinite(dfA[var2use]), :]
        origN = dfA.shape[0]
        ptr_str = 'WARNING: outlier calc. not on orig values (#={} vs. {})'
        if df.shape[0] != origN:
            print(ptr_str.format(df.shape[0], origN))
            df = dfA[[var2use]].copy()
            print('Now using the original file - {}'.format(filename))
        # Now calculate the outlier
    IQR = df[var2use].describe()['75%'] - df['Iodide'].describe()['25%']
    OutlierDef = df[var2use].describe()['75%'] + (IQR*1.5)
    return OutlierDef


def get_hyperparameter_dict():
    """
    get default hyperparam settings to use
    """
    hyperparam_dict = {
        #    'n_estimators' : 100,
        'n_estimators': 500,
        #    'n_estimators' : 10,
        'oob_score': True,
        #    'oob_score' : False,
    }
    return hyperparam_dict


def check_or_mk_directory_struture():
    """
    Check all the required directories are present and make them if not.
    """
    pstr = 'TODO: Make function to check directory strcuture and add folders if not'
    pstr += '\n not present.'
    print(pstr)


def set_backup_month_if_unkonwn(lat=None, var2use='', main_var='',
                                Data_key_ID_=None, debug=True):
    """
    Some of the input data may not have a known month so use an arbitrary one

    Parameters
    -------
    lat (float): latitude degrees north
    Data_key_ID_ (str): ID for input data point
    var2use (str): var to extracted from NetCDF
    main_var (str): general variable (e.g. TEMP)

    Returns
    -------
    (float), (str)
    (or list of two sets of above variables if get_max_and_sum_of_values==True)

    Notes
    -----
     - a value of three months prior to summer solstice for NH and SH is assumed
    """
    # seasons  = 'DJF', 'MAM', 'JJA', 'SON'
    if lat > 0:  # if NH
        # if Lat assume mid of season as April (as June is summer solstice in the NH)
        # Choose 3 months before summer solstice (Northern Hemisphere)
        month_ = 3
    else:  # if SH
        # summer is from December to March and winter is from June to
        # September. September 22 or 23 is the vernal equinox and March
        # 20 or 21 is the autumnal equinox
        # Choose 3 months before summer solstice (Southern Hemisphere)
        month_ = 9
    if debug:
        warn_str = '!'*10
        warn_str += 'WARNING: Annual val unknown for '
        warn_str += '{}({})!! (use month:{})'.format(main_var, var2use, month_)
        warn_str += '(ID:{})'.format(Data_key_ID_)
        warn_str += '!'*10
    if debug:
        print(warn_str)
    return month_


def get_df_stats_MSE_RMSE(df=None, target='Iodide',
                          params=[], dataset_str='all', add_sklean_metrics=False):
    """
    Get stats (RSE/RMSE) on params. in DataFrame

    Parameters
    -------

    Returns
    -------

    Notes
    -----
    """
    mse = [(df[target]-df[param_])**2 for param_ in params]
    mse = [np.mean(i) for i in mse]
    MSE_varname = 'MSE ({})'.format(dataset_str)
    stats = pd.DataFrame(mse, index=params, columns=[MSE_varname])
    RMSE_varname = 'RMSE ({})'.format(dataset_str)
    stats[RMSE_varname] = np.sqrt(stats[MSE_varname])
    if add_sklean_metrics:
        stats = add_sklean_metrics2df(df=df, target=target, params=params,
                                      dataset_str=dataset_str, stats=stats)
    return stats


def add_sklean_metrics2df(df=None, stats=None, target='Iodide',
                          params=[], dataset_str='all',):
    """
    Also add other metrics from sklearn.metrics

    Parameters
    -------

    Returns
    -------

    Notes
    -----
    """
    from sklearn.metrics import r2_score
    from sklearn.metrics import explained_variance_score as EVS
    from sklearn.metrics import median_absolute_error as MAE

    # Add explained_variance_score
    EVS_varname = 'EVS ({})'.format(dataset_str)
    EVS = [EVS(df[target], df[param_]) for param_ in params]
    stats[EVS_varname] = EVS
    # Add r2_score
    R2_varname = 'R2 ({})'.format(dataset_str)
    R2 = [r2_score(df[target], df[param_]) for param_ in params]
    stats[R2_varname] = R2
    # Add Median Absolute Error
    MAE_varname = 'MAE ({})'.format(dataset_str)
    MAE = [MAE(df[target], df[param_]) for param_ in params]
    stats[MAE_varname] = MAE
    return stats


def extract4nearest_points_in_ds(lons=None, lats=None, months=None,
                                 var2extract='Ensemble_Monthly_mean',
                                 verbose=True, debug=False):
    """
    Extract requested variable for nearest point and time from NetCDF


    Parameters
    -------
    lons (np.array), list of Longitudes to use for spatial extraction
    lats (np.array), list of latitudes to use for spatial extraction
    months (np.array), list of months to use for temporal extraction
    var2extract (str), name of variable to extract data for
    rm_Skagerrak_data (boolean), remove the single data from the Skagerrak region

    Returns
    -------
    (xr.Dataset)
    """
    # Get data from NetCDF as a xarray dataset
    ds = get_predicted_values_as_ds()
    # Check that the same about of locations have been given for all months
    lens = [len(i) for i in (lons, lats, months)]
    assert len(set(lens)) == 1, 'All lists provided must be same length!'
    # Loop locations and extract
    extracted_vars = []
    for n_lon, lon_ in enumerate(lons):
        # Get lats and month too
        lat_ = lats[n_lon]
        month_ = months[n_lon]
        # Select for monnth
        ds_tmp = ds[var2extract].sel(time=(ds['time.month'] == month_))
        # Select nearest data
        vals = ds_tmp.sel(lat=lat_, lon=lon_, method='nearest')
        if debug:
            print(vals)
        extracted_vars += [vals.values[0]]
    return extracted_vars


def get_predicted_values_as_ds(rm_Skagerrak_data=False):
    """
    Get predicted values from saved NetCDF file
    """
    folder = get_file_locations('data_root')
    filename = 'Oi_prj_predicted_iodide_0.125x0.125{}.nc'
    if rm_Skagerrak_data:
        filename = filename.format('_No_Skagerrak')
    else:
        filename = filename.format('')
    ds = xr.open_dataset(folder + filename)
    return ds


def get_feature_variables_as_ds(res='4x5'):
    """
    Get feature variables from saved NetCDF file
    """
    filename = 'Oi_prj_feature_variables_{}.nc'.format(res)
    folder = get_file_locations('data_root')
    ds = xr.open_dataset(folder + filename)
    return ds
