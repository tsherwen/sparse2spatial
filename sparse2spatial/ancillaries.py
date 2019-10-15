"""

Processing scripts for ancillary data to used as dependent variable for predition

"""
import xarray as xr
import numpy as np
import pandas as pd
import xarray as xr
import gc
from multiprocessing import Pool
from time import gmtime, strftime
import time
import glob
from functools import partial
# import AC_tools (https://github.com/tsherwen/AC_tools.git)
import AC_tools as AC
# import from s2s
import sparse2spatial.utils as utils
import sparse2spatial.plotting as s2splotting


def interpolate_NaNs_in_feature_variables(ds=None, res='4x5',
                                          save2NetCDF=False, debug=False):
    """
    Interpolate the NaNs in 2D arrarys of feature variables

    Parameters
    -------
    ds (xr.Dataset): dataset object with variables to interpolate
    res (str): horizontal resolution (e.g. 4x5) of Dataset
    save2NetCDF (bool): save interpolated Dataset to as a NetCDF?
    debug (bool): print out debugging output?

    Returns
    -------
    (xr.Dataset)
    """
    # Local variables
    months = np.arange(1, 13)
    # Get Dataset?
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
        # If depth, set all values greater or equal to 1 to NaNs
        # (only depths < 0 are of interest)
        if var == 'Depth_GEBCO':
            arr[arr >= -1.] = np.NaN
        # If World ocean atlas, set 0 values as NaNs (to be interpolated later)
        # A handful of values are 0,  but not masked
        # (only MLD depths > 0 are of interest )
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
            else:
                ars = [arr]
            # Select grid of interest
            subX = da['lon'].values
            subY = da['lat'].values
            # Define a function to interpolate arrays
            # MOVED TO OUTSIDE FUNCTION
            # Initialise pool to parrellise over
            p = Pool(12)
            if debug:
                print(ars[0][:5, :5])
            # Use Radial basis functions (RBF) for interpolation (hashed out)
#             ars = p.map( partial(interpolate_array_with_RBF, subX=subX,
#                subY=subY, ), ars )
            # Use interpolation of nearest on a grid (default)
            ars = p.map(partial(interpolate_array_with_GRIDDATA, da=da), ars)
            # close the pool
            p.close()
            # Now overwrite the values in the array
            if 'time' in coords:
                da.values = np.ma.array(ars)
            else:
                da.values = ars[0]
        # Update the DataSet
        ds[var] = da.copy()
        # Clean memory
        gc.collect()
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
    """
    Add variables to dataset that are derived from others within the Dataset
    """
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
    """
    Make DOC file(s) from UC-SB CF compliant
    """
    # - convert the surface DOC file into a monthly average file
    # Directory?
    older = utils.get_file_locations('data_root') +'/DOC/'
    # Filename as a string
    file_str = 'DOCmodelSR.nc'
    # Open dataset
    ds = xr.open_dataset(folder+file_str)
    # - Force use of coordinate variables in netCDF
    ds['latitude'] = ds['LAT'][0, 0, :].values
    ds['latitude'].attrs = ds['LAT'].attrs
    ds['longitude'] = ds['LON'][0, :, 0].values
    ds['longitude'] .attrs = ds['LON'].attrs
    # Copy across depth variable and attributes
    ds['depth'] = ds['DEPTH'][:, 0, 0].values
    ds['depth'] .attrs = ds['DEPTH'].attrs
    # - Rename dimensions
    dims_dict = {'latitude': 'lat', 'longitude': 'lon', 'depth': 'depth'}
    dims = dims_dict.values()
    ds.rename(dims_dict, inplace=True)
    # - Only keep the variables of interest
    var2keep = [u'Area', u'Vol', u'DOCmdl_avg', u'DOCmdl_std', ]
    ds = ds.drop(labels=[i for i in ds.variables if i not in var2keep+dims])
    # - Add history to attirubtes
    d = ds.attrs
    date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    hst_str = 'File structure/variables updated to CF by TMS ({}) on {}'
    d['History'] = hst_str.format('University of York', date)
    d['Originating author'] = 'SR - Saeed Roshan (saeed.roshan@geog.ucsb.edu)'
    d['Editting author'] = 'TMS - (tomas.sherwen@york.ac.uk)'
    d['Citation'] = 'doi.org/10.1038/s41467-017-02227-3'
    ds.attrs = d
    # - Save the new NetCDF file
    newfile_str = file_str.split('.nc')[0]+'_TMS_EDIT.nc'
    ds.to_netcdf(folder + newfile_str)


def Convert_DOC_prod_file_into_Standard_NetCDF():
    """
    Convert Saeed Roshan's file into CF compliant format
    """
    # - convert the surface DOC file into a monthly average file
    # Directory?
    older = utils.get_file_locations('data_root') +'/DOC/'
    # Filename as a string
    file_str = 'DOC_Accum_rate_SR.nc'
    # Open dataset
    ds = xr.open_dataset(folder+file_str)
    # - Force use of coordinate variables in netCDF
    ds['latitude'] = ds['lat'][0, :].values
    ds['latitude'].attrs = ds['lat'].attrs
    ds['longitude'] = ds['lon'][:, 0].values
    ds['longitude'] .attrs = ds['lon'].attrs
    # - Rename dimensions
    dims_dict = {'latitude': 'lat', 'longitude': 'lon'}
    # - Only keep the variables of interest
    var2keep = [u'DOCaccum_avg', u'DOCaccum_std', ]
    var2keep += dims_dict.keys()
    ds = ds.drop(labels=[i for i in ds.variables if i not in var2keep])
    ds.rename(dims_dict, inplace=True)
    # - Add history to attirubtes
    d = ds.attrs
    date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    hst_str = 'File structure/variables updated to CF by TMS ({}) on {}'
    d['History'] = hst_str.format('University of York', date)
    d['Originating author'] = 'SR - Saeed Roshan (saeed.roshan@geog.ucsb.edu)'
    d['Editting author'] = 'TMS - (tomas.sherwen@york.ac.uk)'
    d['Citation'] = 'doi.org/10.1038/s41467-017-02227-3'
    ds.attrs = d
    # - Save the new NetCDF file
    newfile_str = file_str.split('.nc')[0]+'_TMS_EDIT.nc'
    ds.to_netcdf(folder + newfile_str)


def mk_RAD_NetCDF_monthly():
    """
    Resample shortwave radiation NetCDF from daily to monthly
    """
    # Directory?
    folder = utils.get_file_locations('data_root') +'/GFDL/'
    # Filename as a string
    file_str = 'ncar_rad.15JUNE2009.nc'
    ds = xr.open_dataset(folder + filename)
    # Resample to monthly
    ds = ds.resample(dim='TIME', freq='M')
    # Save as NetCDF
    newfile_str = file_str.split('.nc')[0]+'_TMS_EDIT.nc'
    ds.to_netcdf(folder+newfile_str)


def mk_NetCDF_from_productivity_data():
    """
    Convert productivity .csv file (Behrenfeld and Falkowski, 1997) into a NetCDF file
    """
    # Location of data (update to use public facing host)
    folder = utils.get_file_locations('data_root') + '/Productivity/'
    # Which file to use?
    filename = 'productivity_behrenfeld_and_falkowski_1997_extrapolated.csv'
    # Setup coordinates
    lon = np.arange(-180, 180, 1/6.)
    lat = np.arange(-90, 90, 1/6.)
    lat = np.append(lat, [90])
    # Setup time
    varname = 'vgpm'
    months = np.arange(1, 13)
    # Extract data
    df = pd.read_csv(folder+filename, header=None)
    print(df.shape)
    # Extract data by month
    da_l = []
    for n in range(12):
        # Assume the data is in blocks by longitude?
        arr = df.values[:, n*1081: (n+1)*1081].T[None, ...]
        print(arr.shape)
        da_l += [xr.Dataset(
            data_vars={varname: (['time', 'lat', 'lon', ], arr)},
            coords={'lat': lat, 'lon': lon, 'time': [n]})]
    # Concatenate to data xr.Dataset
    ds = xr.concat(da_l, dim='time')
    # Update time ...
    sdate = datetime.datetime(1985, 1, 1)  # Climate model tiem
    ds['time'] = [AC.add_months(sdate, i-1) for i in months]
    # Update to hours since X
    hours = [(AC.dt64_2_dt([i])[0] - sdate).days *
             24. for i in ds['time'].values]
    ds['time'] = hours
    # Add units
    attrs_dict = {'units': 'hours since 1985-01-01 00:00:00'}
    ds['time'].attrs = attrs_dict
    # Add attributes for variable
    attrs_dict = {
        'long_name': "net primary production",
        'units': "mg C / m**2 / day",
    }
    ds[varname].attrs = attrs_dict
    # For latitude...
    attrs_dict = {
        'long_name': "latitude",
        'units': "degrees_north",
        "standard_name": "latitude",
        "axis": "Y",
    }
    ds['lat'].attrs = attrs_dict
    # And longitude...
    attrs_dict = {
        'long_name': "longitude",
        'units': "degrees_east",
        "standard_name": "longitude",
        "axis": "X",
    }
    ds['lon'].attrs = attrs_dict
    # Add extra global attributes
    global_attribute_dictionary = {
        'Title': 'Sea-surface productivity (Behrenfeld and Falkowski, 1997)',
        'Author': 'Tomas Sherwen (tomas.sherwen@york.ac.uk)',
        'Notes': "Data extracted from OCRA and extrapolated to poles by Martin Wadley. NetCDF contructed using xarray (xarray.pydata.org) by Tomas Sherwen. \n NOTES from oringal site (http://orca.science.oregonstate.edu/) from 'based on the standard vgpm algorithm. npp is based on the standard vgpm, using modis chl, sst4, and par as input; clouds have been filled in the input data using our own gap-filling software. For citation, please reference the original vgpm paper by Behrenfeld and Falkowski, 1997a as well as the Ocean Productivity site for the data.' ",
        'History': 'Last Modified on:' + strftime("%B %d %Y", gmtime()),
        'Conventions': "COARDS",
    }
    ds.attrs = global_attribute_dictionary
    # Save to NetCDF
    filename = 'productivity_behrenfeld_and_falkowski_1997_extrapolated.nc'
    ds.to_netcdf(filename, unlimited_dims={'time': True})


def process_MLD_csv2NetCDF(debug=False, _fill_value=-9999.9999E+10):
    """
    Process NOAA WOA94 csv files into NetCDF files

    Parameters
    -------
    _fill_value (float): fill value to use for new NetCDF
    debug (bool): perform debugging and verbose printing?

    Returns
    -------
    (xr.Dataset)
    """
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
    folder = utils.get_file_locations('data_root') + '/WOA94/'
    # - Loop MLD variables
    for var_ in MLD_vars:
        file_str = 'mld*{}*'.format(var_)
        files = sorted(glob.glob(folder+file_str))
        print(files)
        # Loop files and extract data as an arrayu
        ars = []
        for file in files:
            # values are assume to have been outputed in a row major way
            # e.g. (lon, lat)
            # open
            with open(file, 'rb') as file_:
                # Extract all values
                lines = [i.split() for i in file_]
                # Convert to floats (and masked values (e.g. "-") to NaN ),
                # the concatenate to "big" list
                big = []
                for n, line in enumerate(lines):
                    for value in line:
                        try:
                            value = float(value)
                        except ValueError:
                            value = np.NaN
                        big += [value]
            # Now reshape
            ars += [np.ma.array(big).reshape((180, 360)).T]
            # Debug (?) by showing 2D grid
            if debug:
                plt.pcolor(np.arange(0, 360), np.arange(0, 180),  ars[0])
                plt.colorbar()
                plt.show()
        # Force to be in COARDS format? (e.g. lat, lon) instead of (lon, lat)
        ars = [i.T for i in ars]
        # Fill nans with _fill_value,
        ars = [np.ma.filled(i, fill_value=_fill_value) for i in ars]
        # Then convert to numpy array...
        ars = [np.array(i) for i in ars]
        print([type(i) for i in ars])
        # Force dates
        dates = [datetime.datetime(1985, 1, i+1) for i in range(12)]
        lons = np.arange(0+0.5, 360+0.5, 1)
        lats = np.arange(-90+0.5, 90+0.5, 1)
        res = '1x1'
        # Save to NetCDF
        AC.save_2D_arrays_to_3DNetCDF(ars=ars, dates=dates, varname=var_,
                                      res=res,
                                      filename='WOA94_MLD_1x1_{}'.format(var_),
                                      lons=lons,
                                      lats=lats)


def mk_PDF_of_annual_avg_spatial_ancillary_plots():
    """
    Make a PDF of annual avg. spatial values in ancillary NetCDF
    """
    # Get input folder
    folder = utils.get_file_locations('data_root') +'/data/'
    filename = 'Oi_prj_feature_variables_0.125x0.125.nc'
    ds = xr.open_dataset( folder+filename )
    # version
    extr_str = 'INPUT_VAR'

    # remove seaborn settings
    # - Not plot all
    # make sure seaborn settings are off
    import seaborn as sns
    sns.reset_orig()
    #
    vars2plot = [i for i in ds.data_vars]
    for var2plot in vars2plot:
        # Get annual average of the variable in the dataset
        try:
            ds2plot = ds[[var2plot]].mean(dim='time')
        except ValueError:
            ds2plot = ds[[var2plot]]
            print('WARNING: not averaging over time for {}'.format(var2plot))
        # Set a title for the plot
        title = "Annual average of '{}'".format(var2plot)
        # Now plot
        s2splotting.plot_spatial_data(ds=ds2plot, var2plot=var2plot, extr_str=extr_str,
                                      target=var2plot,
#            LatVar=LatVar, LonVar=LonVar, vmin=vmin, vmax=vmax,
                                       title=title)

        plt.close('all')
        del ds2plot


def download_data4spec(lev2use=72, spec='LWI', res='0.125',
                       file_prefix='nature_run', doys_list=None, verbose=True,
                       debug=False):
    """
    Download all data for a given species at a given resolution

    Parameters
    -------
    spec (str): variable to extract from archived data
    res (str): horizontal resolution of dataset (e.g. 4x5)
    file_prefix (str): file prefix to add to saved file
    debug (bool): print out debugging output?

    Returns
    -------
    (None)

    Notes
    -----
     - use level=71 for lowest level
     (NetCDF is ordered the oposite way, python 0-71. Xarray numbering makes
     this level=72)
     (or use dictionary through xarray)
    """
    # - local variables
    # Where is the remote data?
    root_url = 'https://opendap.nccs.nasa.gov/dods/OSSE/G5NR-Chem/Heracles/'
#    url_str = root_url+'12.5km/{}_deg/inst/inst1_3d_TRC{}_Nv'.format(res,spec)
    url_str = root_url+'12.5km/{}_deg/tavg/tavg1_2d_chm_Nx'.format(res)
    # Where should i save the data?
    save_dir = utils.get_file_locations('data_root') + '/NASA/LWI/'
    # - Open dataset via URL with xarray
    # Using xarray (issues found with NASA OpenDAP data model - via PyDAP)
    ds = xr.open_dataset(url_str)
    if verbose:
        print(ds, '\n\n\n')
    # Get list of (all) doys to extract (unless provided as argv.)
    if isinstance(doys_list, type(None)):
        doys_list = list(set(ds['time.dayofyear'].values))
    # Variable to extract?
    var_name = '{}'.format(spec.lower())
    # Just test a small extraction.
    # if debug:
    #    data = ds[var_name][:10, lev, :, :]
    # select level and download all data
    ds = ds[var_name][:, :, :]
    # Make sure time is the dimension not module
    time = ds.time
    # - loop days of year (doy)
    # Custom mask
    def is_dayofyear(doy):
        return (doy == doy_)
    # Loop doys
    for doy_ in doys_list[:4]:
        try:
            if verbose:
                print(doy_, spec)
            # Now select for month
            ds_tmp = ds.sel(time=is_dayofyear(ds['time.dayofyear']))
            # Save as NetCDF
            year_ = list(set(ds_tmp['time.year'].values))[0]
            # What is the filename?
            fstr = '{}_lev_{}_res_{}_spec_{}_{}_{:0>3}_ctm.nc'
            file2save = fstr.format(file_prefix, lev2use, res, spec, year_, str(doy_))
            # Now save downloaded data as a NetCDF locally...
            if verbose:
                print(save_dir+file2save)
            ds_tmp.to_netcdf(save_dir+file2save)
            # Remove from memory
            del ds_tmp
        except RuntimeError:
            err_str = 'TMS ERROR - FAIL for spec={} (doy={})'.format(
                spec, doy_)
            print(err_str)
