"""

Function for interpolating all variables onto high resolution base grid (~1km-12km)

"""
import numpy as np
import pandas as pd
import xarray as xr
import gc
import xesmf as xe
import sparse2spatial.utils as utils


def regrid_ds_field2G5NR_res(ds=None, res='0.125x0.125', target='Iodide',
                             filename2save=None, folder2save=None,
                             save2netCDF=False, vars2regrid=None):
    """
    Re-grid input a dataset's field to G5NR resolution (~12x12km; 0.125x0.125)

    Parameters
    -------
    dsA (xr.Dataset): data to regrid and save to NetCDFs
    save2netCDF (bool): save re-gridded dataset to NetCDF
    debug (bool): perform debugging and verbose printing?
    res (str): resolution to re-gridd to (e.g. G5NR or ~12x12km)

    Returns
    -------
    (xr.Dataset)
    """
    # Regrid all variables in dataset unless list specified (vars2regrid)
    if isinstance(vars2regrid, type(None)):
        vars2regrid = list(ds.data_vars)
    # Get grid for Nature run resolution
    lon, lat, NIU = AC.get_latlonalt4res(res=res)
    # Create a dataset to re-grid into
    ds_out = xr.Dataset({
        # 'time': ( ['time'], dsA['time'] ),
        'lat': (['lat'], lat),
        'lon': (['lon'], lon),
    })
    # Create a regidder (to be reused )
    regridder = xe.Regridder(ds, ds_out, 'bilinear', reuse_weights=True)
    # Loop and regrid variables
    ds_l = []
    for var2use in vars2regrid:
        # Create a dataset to re-grid into
        ds_out = xr.Dataset({
            # 'time': ( ['time'], dsA['time'] ),
            'lat': (['lat'], lat),
            'lon': (['lon'], lon),
        })
        # Get a DataArray
        dr = ds[var2use]
        # Build regridder
        dr_out = regridder(dr)
        # Important note: Extra dimensions must be on the left, i.e. (time, lev, lat, lon) is correct but (lat, lon, time, lev) would not work. Most data sets should have (lat, lon) on the right (being the fastest changing dimension in the memory). If not, use DataArray.transpose or numpy.transpose to preprocess the data.
        # Exactly the same as input?
        xr.testing.assert_identical(dr_out['time'], ds['time'])
        # Save variable
        ds_l += [dr_out]
    # Setup a new dataset object to hold the new values
    dsA = xr.Dataset()
    # Add variables into a new dataset
    for n, var2use in enumerate(vars2regrid):
        dsA[var2use] = ds_l[n]
        # transfer attributes
        dsA[var2use].attrs = ds[var2use].attrs.copy()
    # Add core attributes to coordinates and global attrs dictionary
    dsA = utils.add_get_core_attributes2ds(dsA)
    # Clean up
    regridder.clean_weight_file()
    # Save the file
    if save2netCDF:
        if isinstance(folder2save, type(None)):
            folder2save = './'
        if isinstance(filename2save, type(None)):
            filename = 's2s_regridded_{}_field_{}'.format(target, res)
            filename = AC.rm_spaces_and_chars_from_str(filename)
        dsA.to_netcdf(folder2save+filename2save+'.nc')
    else:
        return dsA
