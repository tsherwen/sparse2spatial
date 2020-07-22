#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Functions related to processing/analysis of process-based iodide field

Wadley, M.R., Stevens, D.P., Jickells, T.D., Hughes, C., Chance, R., Hepach, H., Tinel, L. and Carpenter, L.J., A Global Model for Iodine Speciation in the Upper Ocean. Global Biogeochemical Cycles, p.e2019GB006467.
"""


def get_Wadley_iodide_fields():
    """
    Retrieve the raw Wadley2020 iodide fields
    """
    # Where is the data located and what is its name
    data_root = utils.get_file_locations('data_root')
    folder = '/{}/../Oi/UEA/'.format(data_root)
    filename = 'iodide_from_model_PRESENT_DAY.nc'
    ds = xr.open_dataset(folder+filename)
    return ds


def interp_Wadley_iodide_fields():
    """
    Interpolate the fields from
    """
    from multiprocessing import Pool
    from functools import partial
    # import function for interpolation
    from sparse2spatial import interpolate_array_with_GRIDDATA
    # Where is the data located and what is its name
    ds = get_Wadley_iodide_fields()
    var2use = 'Present_Day_Iodide'
    # Update the ordering to be COORDS compliant
    ds = ds.transpose('lon', 'lat', 'time', 'lev', )
    # Select the surface (first level)
    ds = ds.sel(lev=0)
    # Select the data as a dataset
    da = ds[var2use].mean(dim='time')
    coords = [i for i in da.coords]
    # Loop by month and update the array
    da_l = []
    times2use = ds.time.values
    # Now process the fields as a batch of 12 jobs
    p = Pool(12)
    # Split data by time dimension
    ars = [ds[var2use].sel(time=i).values.T for i in times2use]
    # Call interpolation in parrellel
    # NOTE: must use python 2 virtual environment!
    ars = p.map(partial(interpolate_array_with_GRIDDATA, da=da), ars)
    # Update the interpolated variables
    ds = ds.transpose( 'time', 'lat', 'lon', )
    ds[var2use].values = np.ma.array(ars)
    # Save to netcdf
    NewFilename = 'iodide_from_model_PRESENT_DAY_interp.nc'
    ds.to_netcdf(folder+NewFilename)


def regrid_process_based_field_to_12x12km():
    """
    Regrid iodide process-based model output to 12x12
    """
    from sparse2spatial.ancillaries2grid import regrid_ds_field2G5NR_res
    # Load data
    data_root = utils.get_file_locations('data_root')
    folder = '/{}/../Oi/UEA/'.format(data_root)
    filename = 'iodide_from_model_PRESENT_DAY_interp.nc'
    ds = xr.open_dataset(folder+filename)
    del ds['lev']
    filename2save = 'iodide_from_model_PRESENT_DAY_interp_0.125x0.125'
    regrid_ds_field2G5NR_res(ds, folder2save=folder, save2netCDF=True,
                             filename2save=filename2save)


def convert_iodide2kg_m3():
	"""
	Convert Wadley2020 iodide field into units of kg/m3 and save
	"""
    # Convert units from nM to kg/m3 (=> M => mass => /m3 => /kg)
    FileName = 'iodide_from_model_PRESENT_DAY_interp_0.125x0.125.nc'
    data_root = utils.get_file_locations('data_root')
    folder = '/{}/../Oi/UEA/'.format(data_root)
    ds = xr.open_dataset(folder+FileName)
    NewVar = 'Present_Day_Iodide'
    species = 'I'
    ds[NewVar] = ds[NewVar]/1E9 * AC.species_mass(species) * 1E3 / 1E3
    # for variable
    attrs = ds[NewVar].attrs
    attrs['units'] = "kg/m3"
    attrs['units_longname'] = "kg({})/m3".format(species)
    attrs['Creator'] = 'Tomas Sherwen (tomas.sherwen@york.ac.uk)'
    attrs['Citation'] = "Wadley, M.R., Stevens, D.P., Jickells, T., Hughes, C., Chance, R., Hepach, H. and Carpenter, L.J., 2020. Modelling iodine in the ocean. https://www.essoar.org/doi/10.1002/essoar.10502078.1"
    ds[NewVar].attrs = attrs
    NewFilename = 'iodide_from_model_PRESENT_DAY_interp_0.125x0.125_kg_m3.nc'
    ds.to_netcdf(folder+NewFilename)


def convert_Wadley_iodide_fields_2NetCDF():
    """
    Retrieve the Wadley et al 2020 iodide fields
    """
    from scipy.io import loadmat
    # Where is the data located and what is its name
    data_root = utils.get_file_locations('data_root')
    folder = '/{}/../Oi/UEA/'.format(data_root)
    filename = 'iodide_from_model.mat'
    # Load Matlab object and extract data
    mat_obj = loadmat(folder+filename)
    var2use = 'iodide_from_model'
    data = mat_obj[var2use]
    # manually setup the coordinates
    lon = [i+-179.5 for i in np.arange(360)]
    lat = [i+-89.5 for i in np.arange(180)]
    lev = list(np.arange(3))
    time = [datetime.datetime(2000, i, 1) for i in np.arange(1,13)]
    # Manually construct into a xr.dataset
    da = xr.DataArray(data=data,
                      dims=['lon', 'lat', 'time', 'lev'],
                      coords=[lon, lat, time, lev],
                     )
    # Add some attributes
    attrs = da.lat.attrs
    attrs['units'] = 'Degrees North'
    attrs["axis"] = 'Y'
    attrs['long_name'] = "latitude",
    attrs["standard_name"] = "latitude"
    da.lat.attrs = attrs
    attrs = da.lon.attrs
    attrs['units'] = 'Degrees East'
    attrs["axis"] = 'X'
    attrs['long_name'] = "longitude",
    attrs["standard_name"] = "longitude"
    da.lon.attrs = attrs
    attrs = da.lev.attrs
    attrs['units'] = 'Ocean levels'
    description = 'The upper layer is the surface mixed layer value.\n The middle layer extends below this to the depth of the seasonal maximum mixed layer depth. \n  The bottom layer extends either to the ocean floor, or to a thickness of 500m, whichever is less. \n'
    attrs['description'] = description
    da.lev.attrs = attrs
    # Add to a new Dataset
    NewVar = 'Present_Day_Iodide'
    ds = xr.Dataset()
    ds[NewVar] =  da
    # Add additional attributes
    attrs = ds[NewVar].attrs
    attrs['Creator'] = 'Tomas Sherwen (tomas.sherwen@york.ac.uk)'
    attrs['Citation'] = "Wadley, M.R., Stevens, D.P., Jickells, T., Hughes, C., Chance, R., Hepach, H. and Carpenter, L.J., 2020. Modelling iodine in the ocean. https://www.essoar.org/doi/10.1002/essoar.10502078.1"
    ds[NewVar].attrs = attrs
    # Update the ordering to be COORDS compliant
    ds = ds.transpose('time', 'lev', 'lon', 'lat')
    # Save to NetCDF
    NewFilename = 'iodide_from_model_PRESENT_DAY.nc'
    ds.to_netcdf(folder+NewFilename)
    # Save annual average
    dsA = ds.mean(dim='time')
#    ds = ds.sel(lev=0)
    NewFilename = 'iodide_from_model_PRESENT_DAY_annual_avg.nc'
    dsA.to_netcdf(folder+NewFilename)

