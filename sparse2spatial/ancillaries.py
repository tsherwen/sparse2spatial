"""

Processing scripts for ancillary data to used as dependent variable for predition

"""



def interpolate_NaNs_in_feature_variables(ds=None, res='4x5',
                                          save2NetCDF=False):
    """ Interpolate the NaNs in 2D arrarys of feature variables """
    import gc
    from multiprocessing import Pool
    from time import gmtime, strftime
    import time
    from functools import partial
    # Local variables
    months = np.arange(1, 13)
    # get Dataset?
    filename = 'Oi_prj_feature_variables_{}.nc'.format(res)
    if isinstance(ds, type(None)):
        ds = xr.open_dataset(filename)
    for var in ds.data_vars:
        # Get DataArray and coordinates for variable
        da = ds[var]
        coords = [i for i in da.coords]
        # make sure all values are floats
        da = da.astype(np.float64)
        arr = np.ma.array(da.values)
        # If depth, make all values greater than 0 NaNs
        # (only depths < 0 are of interest)
        if var == 'Depth_GEBCO':
            arr[arr >= -1.] = np.NaN
#            arr.mask = [ arr >=-100. ]
#            arr.mask = [ arr >=0. ]
#            arr = np.array(arr)
        # If World ocean atlas, set 0 values as NaNs (to be interpolated later)
        # A handful of values are 0,  but not masked
        # (only MLD depths > 0 are of interest - )
        # ( also the -99 fill_value is not being removed, so mask this )
        if 'WOA_' in var:
            arr[arr == 0.] = np.NaN
            arr[arr == -99.9] = np.NaN
        # Check if there are NaNs in the array
        NaNs_presnt = arr[~np.isfinite(arr)].shape[0]
        print("Interpolating={} for '{}' at {}".format(NaNs_presnt, var, res))
        if NaNs_presnt > 0:
            # Split into list of 2D arrays (by time)
            if 'time' in coords:
                months_eq = ds['time.month'].values == months
                assert months_eq.all(), 'Months not in order!'
                ars = [arr[i, ...] for i in range(len(months))]
#                ars_dict = dict( zip( months, ars ) )
            else:
                ars = [arr]
#                ars_dict = dict( ('Annual', arr) )
            # convert ars_dict to list
#            ars = [ars_dict[i] for i in sorted( ars_dict.keys() ) ]
            # Select grid of interest
            subX = da['lon'].values
            subY = da['lat'].values
            # Define a function to interpolate arrays
            # MOVED TO OUTSIDE FUNCTION
            # Initialise pool to parrellise over
            p = Pool(12)
#            keys = list( sorted(ars_dict.keys()) )
            #
#            print( [(i.shape, i.mean(), i.max(), i.min()) for i in ars ] )
            print(ars[0][:5, :5])
            # Use RBF
#             ars = p.map( partial(interpolate_array_with_RBF, subX=subX,
#                subY=subY, ), ars )
            # Use interpolation of nearest on a grid
            ars = p.map(partial(interpolate_array_with_GRIDDATA, da=da), ars)
            # close the pool
            p.close()
            # Now overwrite the values in the array
            if 'time' in coords:
                #                da.values = np.ma.array( [ars_dict[i] in range(len(months)) ] )
                da.values = np.ma.array(ars)
            else:
                da.values = ars[0]
        # Update the DataSet
        ds[var] = da.copy()
        # Clean memory
        gc.collect()
#        del da
        # Then apply LWI mask again
        # ( ONLY FOR 100% boxes that are not islands... )
    # Now save the updated file
    if save2NetCDF:
        print('Saving interpolated NetCDF at {}'.format(res))
        ds.to_netcdf(filename.split('.nc')[0]+'_INTERP_NEAREST.nc')
    print('Interpolated variables at {}'.format(res))
    # return DataSet
    return ds



def add_derivitive_variables(ds=None):
    """ Add variables that are deirived from others """
    # Add temperature in Kelvin
    TEMP_var = 'WOA_TEMP'
    ds[TEMP_var+'_K'] = ds['WOA_TEMP'].copy() + 273.15
    # Add sum, mean for each MLD variable
    MLDvars = [i for i in ds.data_vars if 'MLD' in i]
    for MLDvar in MLDvars:
        ds[MLDvar+'_sum'] = ds[MLDvar].sum(dim='time')
        ds[MLDvar+'_max'] = ds[MLDvar].max(dim='time')
    return ds


def Convert_DOC_file_into_Standard_NetCDF():
    """ Make DOC file(s) from UC-SB CF compliant """
    import xarray as xr
    # - conver the surface DOC file into a monthly average file
    # Directory?
    file_dir = get_Oi_file_locations('DOC')
    # file str
    file_str = 'DOCmodelSR.nc'
    # Open dataset
    ds = xr.open_dataset(file_dir+file_str)
    # ---  Force use of coordinate variables in netCDF
    ds['latitude'] = ds['LAT'][0, 0, :].values
    ds['latitude'].attrs = ds['LAT'].attrs
    ds['longitude'] = ds['LON'][0, :, 0].values
    ds['longitude'] .attrs = ds['LON'].attrs
    # copy across depth variable and attributes
    ds['depth'] = ds['DEPTH'][:, 0, 0].values
    ds['depth'] .attrs = ds['DEPTH'].attrs
    # --- Rename dimensions
    dims_dict = {'latitude': 'lat', 'longitude': 'lon', 'depth': 'depth'}
    dims = dims_dict.values()
    ds.rename(dims_dict, inplace=True)
    # --- Only keep the variables of interest
    var2keep = [u'Area', u'Vol', u'DOCmdl_avg', u'DOCmdl_std', ]
    ds = ds.drop(labels=[i for i in ds.variables if i not in var2keep+dims])
    # --- Add history to attirubtes
    d = ds.attrs
    date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    hst_str = 'File structure/variables updated to CF by TMS ({}) on {}'
    d['History'] = hst_str.format('University of York', date)
    d['Originating author'] = 'SR - Saeed Roshan (saeed.roshan@geog.ucsb.edu)'
    d['Editting author'] = 'TMS - (tomas.sherwen@york.ac.uk)'
    d['Citation'] = 'doi.org/10.1038/s41467-017-02227-3'
    ds.attrs = d
    # --- Save the new NetCDF file
    newfile_str = file_str.split('.nc')[0]+'_TMS_EDIT.nc'
    ds.to_netcdf(file_dir + newfile_str)


def Convert_DOC_prod_file_into_Standard_NetCDF():
    """ Convert Saeed Roshan's file into CF compliant format """
    import xarray as xr
    # - conver the surface DOC file into a monthly average file
    # Directory?
    file_dir = get_Oi_file_locations('DOC')
    # file str
    file_str = 'DOC_Accum_rate_SR.nc'
    # Open dataset
    ds = xr.open_dataset(file_dir+file_str)
    # ---  Force use of coordinate variables in netCDF
    ds['latitude'] = ds['lat'][0, :].values
    ds['latitude'].attrs = ds['lat'].attrs
    ds['longitude'] = ds['lon'][:, 0].values
    ds['longitude'] .attrs = ds['lon'].attrs
    # --- Rename dimensions
    dims_dict = {'latitude': 'lat', 'longitude': 'lon'}
    # - Only keep the variables of interest
    var2keep = [u'DOCaccum_avg', u'DOCaccum_std', ]
    var2keep += dims_dict.keys()
    ds = ds.drop(labels=[i for i in ds.variables if i not in var2keep])
    ds.rename(dims_dict, inplace=True)
    # --- Add history to attirubtes
    d = ds.attrs
    date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    hst_str = 'File structure/variables updated to CF by TMS ({}) on {}'
    d['History'] = hst_str.format('University of York', date)
    d['Originating author'] = 'SR - Saeed Roshan (saeed.roshan@geog.ucsb.edu)'
    d['Editting author'] = 'TMS - (tomas.sherwen@york.ac.uk)'
    d['Citation'] = 'doi.org/10.1038/s41467-017-02227-3'
    ds.attrs = d
    # --- Save the new NetCDF file
    newfile_str = file_str.split('.nc')[0]+'_TMS_EDIT.nc'
    ds.to_netcdf(file_dir + newfile_str)


def mk_RAD_NetCDF_monthly():
    """ Resample NetCDF from daily to monthly """
    # Directory?
    file_dir = get_Oi_file_locations('GFDL')
    # File str
    file_str = 'ncar_rad.15JUNE2009.nc'
    ds = xr.open_dataset(folder + filename)
    # Resample to monthly
    ds = ds.resample(dim='TIME', freq='M')
    # Save as NetCDF
    newfile_str = file_str.split('.nc')[0]+'_TMS_EDIT.nc'
    ds.to_netcdf(file_dir+newfile_str)


def Convert_martins_productivity_file_into_a_NetCDF():
    """ Convert Martin's .csv file into a NetCDF file """
    import xarray as xr
    from time import gmtime, strftime
    # ---  Local vars
    folder = '/work/home/ts551/data/iodide/Martin_Wadley/'
    # which file to use?
    filename = 'productivity_behrenfeld_and_falkowski_1997_extrapolated.csv'
    # setup coordinates
    lon = np.arange(-180, 180, 1/6.)
    lat = np.arange(-90, 90, 1/6.)
    lat = np.append(lat, [90])
    # setup time
    varname = 'vgpm'
    months = np.arange(1, 13)
    # ---  Extract data
    df = pd.read_csv(folder+filename, header=None)
    print(df.shape)
    # ---  Extract data by month
    da_l = []
    for n in range(12):
        # Assume the data is in blocks by month
        arr = df.values[:, n*1081: (n+1)*1081].T[None, ...]
        # Assume the data is in blocks by month #  Not the case!
#        arr = df.values[:,n::12] # Not the case!
#        arr = arr.T[None,...] # Not the case!
        print(arr.shape)
        da_l += [xr.Dataset(
            data_vars={varname: (['time', 'lat', 'lon', ], arr)},
            coords={'lat': lat, 'lon': lon, 'time': [n]})]
    # concatenate
    ds = xr.concat(da_l, dim='time')
    # Update time ...
    sdate = datetime.datetime(1985, 1, 1)  # Climate model tiem
    ds['time'] = [AC.add_months(sdate, i-1) for i in months]
    # Update to hours since X
    hours = [(AC.dt64_2_dt([i])[0] - sdate).days *
             24. for i in ds['time'].values]
    ds['time'] = hours
    # add units
    attrs_dict = {'units': 'hours since 1985-01-01 00:00:00'}
    ds['time'].attrs = attrs_dict
    # --- Add attributes
    # for variable
    attrs_dict = {
        'long_name': "net primary production",
        'units': "mg C / m**2 / day",
    }
    ds[varname].attrs = attrs_dict
    # for lat...
    attrs_dict = {
        'long_name': "latitude",
        'units': "degrees_north",
        "standard_name": "latitude",
        "axis": "Y",
    }
    ds['lat'].attrs = attrs_dict
    # and lon...
    attrs_dict = {
        'long_name': "longitude",
        'units': "degrees_east",
        "standard_name": "longitude",
        "axis": "X",
    }
    ds['lon'].attrs = attrs_dict
    # Add extra details
    global_attribute_dictionary = {
        'Title': 'A parameterisation sea-surface iodide on a monthly basis',
        'Author': 'Tomas Sherwen (tomas.sherwen@york.ac.uk)',
        'Notes': "Data extracted from OCRA and extrapolated to poles by Martin Wadley. NetCDF contructed using xarray (xarray.pydata.org) by Tomas Sherwen. \n NOTES from oringal site (http://orca.science.oregonstate.edu/) from 'based on the standard vgpm algorithm. npp is based on the standard vgpm, using modis chl, sst4, and par as input; clouds have been filled in the input data using our own gap-filling software. For citation, please reference the original vgpm paper by Behrenfeld and Falkowski, 1997a as well as the Ocean Productivity site for the data.' ",
        'History': 'Last Modified on:' + strftime("%B %d %Y", gmtime()),
        'Conventions': "COARDS",
    }
    ds.attrs = global_attribute_dictionary
    # Save to NetCDF
    filename = 'productivity_behrenfeld_and_falkowski_1997_extrapolated.nc'
    ds.to_netcdf(filename, unlimited_dims={'time': True})



def set_backup_month_if_unkonwn(lat=None, var2use='', main_var='',
                                Data_key_ID_=None, debug=True):
    """
    Some of the input data doesn't have a known month, so assume
    three months prior to summer solstice for NH and SH.

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
    """
    # seasons  = 'DJF', 'MAM', 'JJA', 'SON'
    if lat > 0:  # if NH
        # if Lat assume mid of season as April (as June is summer solstice in the NH)
        # Choose 3 months before summer solstice (southern hemisphere)
        month_ = 3
    else:  # if SH
        # summer is from December to March and winter is from June to
        # September. September 22 or 23 is the vernal equinox and March
        # 20 or 21 is the autumnal equinox
        # Choose 3 months before summer solstice (southern hemisphere)
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


def process_MLD_csv2NetCDF(debug=False, _fill_value=-9999.9999E+10):
    """ Process NOAA WOA94 csv files to netCDF """
    # The MLD fields available are computed from climatological monthly mean
    # profiles of potential temperature and potential density based on three
    # different criteria: a temperature change from the ocean surface of 0.5
    # degree Celsius, a density change from the ocean surface of 0.125
    # (sigma units), and a variable density change from the ocean surface
    # corresponding to a temperature change of 0.5 degree Celsius. The MLD
    # based on the variable density criterion is designed to account for the
    # large variability of the coefficient of thermal expansion that
    # characterizes seawater.
    # Citation: Monterey, G. and Levitus, S., 1997: Seasonal Variability of
    # Mixed Layer Depth for the World Ocean. NOAA Atlas NESDIS 14, U.S.
    # Gov. Printing Office, Wash., D.C., 96 pp. 87 figs. (pdf, 13.0 MB).
    # variables for
    MLD_vars = ['pt', 'pd', 'vd']
    file_dir = get_Oi_file_locations('WOA_1994')
    # --- loop MLD variables.
    for var_ in MLD_vars:
        file_str = 'mld*{}*'.format(var_)
        files = sorted(glob.glob(file_dir+file_str))
        print(files)
        # loop files and extract data as an arrayu
        ars = []
        for file in files:
            # values are assume to have been outputed in a row major way
            # e.g. (lon, lat)
            # open
            with open(file, 'rb') as file_:
                # extract all values.
                lines = [i.split() for i in file_]
                # convert to floats (and masked values (e.g. "-") to NaN ),
                # the concatenate to "big" list
                big = []
                for n, line in enumerate(lines):
                    for value in line:
                        try:
                            value = float(value)
                        except ValueError:
                            value = np.NaN
                        big += [value]
            # now reshape
            ars += [np.ma.array(big).reshape((180, 360)).T]
            # Debug (?) by showing 2D grid
            if debug:
                plt.pcolor(np.arange(0, 360), np.arange(0, 180),  ars[0])
                plt.colorbar()
                plt.show()
#                sys.exit()
        # force to be in COARDS format? (e.g. lat, lon) instead of (lon, lat)
        ars = [i.T for i in ars]
        # fill nans with _fill_value,
        ars = [np.ma.filled(i, fill_value=_fill_value) for i in ars]
        # then convert to numpy array...
        ars = [np.array(i) for i in ars]
        print([type(i) for i in ars])
        # force dates
        dates = [datetime.datetime(1985, 1, i+1) for i in range(12)]
#        lons = np.arange(-180+0.5, 180+0.5,1)
        lons = np.arange(0+0.5, 360+0.5, 1)
        lats = np.arange(-90+0.5, 90+0.5, 1)
        res = '1x1'
        # save to NetCDF
        AC.save_2D_arrays_to_3DNetCDF(ars=ars, dates=dates, varname=var_,
                                      res=res,
                                      filename='WOA94_MLD_1x1_{}'.format(var_),
                                      lons=lons,
                                      lats=lats)

