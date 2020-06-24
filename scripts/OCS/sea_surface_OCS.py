
#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Module to hold processing/analysis functions for OCS work

Notes
----
ML = Machine Learning
target = the value aiming to be estimated or provided in training
feature = a induivual conpoinet of a predictor vector assigned to a target
( could be called an attribute )
predictor = vector assigned to a target value
"""

import numpy as np
import pandas as pd
# import AC_tools (https://github.com/tsherwen/AC_tools.git)
import AC_tools as AC
import matplotlib
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
# s2s imports
import sparse2spatial.RFRanalysis as RFRanalysis
import sparse2spatial.analysis as analysis
import sparse2spatial.RFRbuild as build
import sparse2spatial.utils as utils
import sparse2spatial.plotting as s2splotting
from sparse2spatial.ancillaries2grid import regrid_ds_field2G5NR_res
from sparse2spatial.utils import interpolate_array_with_GRIDDATA
from sparse2spatial.RFRbuild import mk_test_train_sets
from sparse2spatial.RFRbuild import build_or_get_models
from sparse2spatial.RFRbuild import get_top_models
#from sparse2spatial.RFRanalysis import get_stats_on_models
#from sparse2spatial.RFRanalysis import get_stats_on_multiple_global_predictions
# Local modules specific to OCS work
import observations as obs
# Temporarily included
import xesmf as xe
import xarray as xr
import datetime as datetime


def main():
    """
    Driver for module's man if run directly from command line. unhash
    functionalitliy to call.
    """
    interpolate_NaNs_in_Lennartz_fields()


def old_main():
    """
    """
    # - Set core local variables
    target = 'OCS'

    # - build models with the observations
    RFR_dict = build_or_get_models_OCS(rebuild=False, target=target)
    # Get stats ont these models
    stats = RFRanalysis.get_core_stats_on_current_models(RFR_dict=RFR_dict,
                                                      target=target,
                                                      verbose=True,
                                                      debug=True)
    # Get the top ten models
    topmodels = build.get_top_models(RFR_dict=RFR_dict, stats=stats,
                                     vars2exclude=['DOC', 'Prod'], n=10)

    # --- Predict values globally (only use 0.125)
    # extra strig for NetCDF save name
    xsave_str = '_TEST'
    # make NetCDF predictions from the main array
    save2NetCDF = True
    # Resolution to use? (full='0.125x0.125', test at lower e.g. '4x5')
    res = '0.125x0.125'
#    res = '4x5'
#    res='2x2.5'
#     build.mk_predictions_for_3D_features(None, res=res, RFR_dict=RFR_dict,
#                                          use_updated_predictor_NetCDF=False,
#                                          save2NetCDF=save2NetCDF, target=target,
#                                          models2compare=topmodels,
#                                          topmodels=topmodels,
#                                          xsave_str=xsave_str, add_ensemble2ds=True)


    # - Plot up the predicted field
    # get the predicted data as saved offline
    ds = utils.get_predicted_values_as_ds(target=target, )
    # annual average
    s2splotting.plot_up_annual_averages_of_prediction(target=target, ds=ds)
    # seasonally resolved average
    s2splotting.plot_up_seasonal_averages_of_prediction(target=target, ds=ds)
    # seasonally resolved average, with the same colourbar as the OCS values
    # From Lentt
    vmin, vmax = 0, 75
    version = 'ML_v0.0.0_limited_colourbar'
    s2splotting.plot_up_seasonal_averages_of_prediction(target=target, ds=ds)


    # --- Plot up the performance of the models
    # Get the main DataFrame for analysis of output
    df = RFR_dict['df']
    # Add the ensemble prediction
    df = add_ensemble_prediction2df(df=df, target=target)
    # Plot performance of models
    RFRanalysis.plt_stats_by_model(stats=stats, df=df, target=target )
    # Plot up also without derivative variables
    RFRanalysis.plt_stats_by_model_DERIV(stats=stats, df=df, target=target )

    # - Plot comparisons against observations
    # Plot up an orthogonal distance regression (ODR) plot
    ylim = (0, 80)
    xlim = (0, 80)
#    xlim, ylim =  None, None
    params =  ['RFR(Ensemble)']
    s2splotting.plot_ODR_window_plot(df=df, params=params, units='pM',
                                     target=target,
                                     ylim=ylim, xlim=xlim)

    # Plot up a PDF of concs and bias
    ylim = (0, 80)
    s2splotting.plot_up_PDF_of_obs_and_predictions_WINDOW(df=df, params=params,
                                                          units='pM',
                                                          target=target,
                                                          xlim=xlim)
    # plot out comparisons with observations by region
    LonVar = 'Longitude'
    LatVar = 'Latitude'
    s2splotting.plt_X_vs_Y_for_regions(df=df, target=target,
                                       LonVar=LonVar, LatVar=LatVar)


    # --- Save out the field in kg/m3 for use in models
    version = 'v0_0_0'
    folder = '/users/ts551/scratch/data/s2s/{}/outputs/'.format(target)
    filename = 'Oi_prj_predicted_{}_0.125x0.125_{}'.format(target, version)
    ds = xr.open_dataset( folder + filename+'.nc' )
    # Convert to kg/m3
    RMM = 60.08
    new_var = 'Ensemble_Monthly_mean_kg_m3'
    ds = utils.add_converted_field_pM_2_kg_m3(ds=ds,
                                              var2use='Ensemble_Monthly_mean',
                                              target=target, RMM=RMM,
                                              new_var=new_var)
    # Save with just the kg/m3 field to a NetCDF file
    ds = ds[[new_var]]
    ds = ds.rename(name_dict={new_var:'Ensemble_Monthly_mean'})
    ds.to_netcdf( folder + filename+'{}.nc'.format('_kg_m3') )


    # --- Do analysis and comparisons to existing predictions/fields
    # Plot the existing fields from Lennartz et al 2017
    plot_existing_fields_from_Lennartz_2017()
    # Do analysis on Lennartz et al 2017 fields
    do_analysis_on_existing_fields_from_Lennartz_2017()


    # --- Do analysis of online OCS fluxes
    # Quickly check the output
    quick_check_of_OCS_emissions()
    # Check the fluxes from Kettle


def check_Kettle1999_fluxes():
    """
    Check the emissions that result from the offline Kettle1999 NetCDF
    """
    # Get the Kettle fluxes
    dsK = get_OCS_fields_from_Kettle1999()
    vars2use =  [i for i in dsK.data_vars]
    # Add the AREA
#    AC.calc_surface_area_in_grid(res='1x1')
    EMEP_folder = '/mnt/lustre/groups/chem-acm-2018/earth0_data/GEOS//ExtData/'
    EMEP_folder += 'HEMCO//EMEP/v2015-03/'
    EMEP_filename = 'EMEP.geos.1x1.nc'
    dsE = xr.open_dataset(EMEP_folder + EMEP_filename)
    lon = dsE['lon'].values
    lat = dsE['lat'].values
    AREA = AC.calc_surface_area_in_grid(lon_e=lon, lat_e=lat, lon_c=lon,
                                        lat_c=lat )
    dsK['AREA'] = dsK['OCS_oc_dir'].copy().mean(dim='time')
    dsK['AREA'].values = AREA.T
    # loop by variable
    # setup a dictionary
    var_species_dict = dict(zip(vars2use, ['OCS']*len(vars2use) ) )

    # Convert the units
#    ds = AC.convert_HEMCO_ds2Gg_per_yr(ds2use, vars2convert=vars2use,
#                                       var_species_dict=var_species_dict)
    df = pd.DataFrame()
    for var in vars2use:
        # - Get the gross flux
        # Get the data
        arr = dsK[var].copy().values
        # only consider the positive fluxes
#        arr = np.where(arr<0, 0)
        arr[arr<0] = 0
        # convert to kg/s
#        arr = arr * dsK['AREA']
        arr = arr * dsK['AREA'].values[None, ...]
        # convert to kg/month
        arr = arr * 60 * 60 * 24 * 31
        # kg to Gg
        arr = arr / 1E6
        # Sum to get Gross
#        Gross = float(arr.sum().values)
        Gross = arr.sum()

        # - Get the net total
        # Get the data
        arr = dsK[var].copy()
        # convert to kg/s
        arr = arr * dsK['AREA']
        # convert to kg/month
        arr = arr * 60 * 60 * 24 * 31
        # kg to Gg
        arr = arr / 1E6
        # Sum to get Net
        Net = float(arr.sum().values)
        # Save to a dataframe
        s = pd.Series({'Net': Net, 'Gross': Gross })
        df[var] = s

#        print(var, arr.sum())


#    for var in dsK.data_vars:


        vals = dsK[var].copy() * dsK['AREA']



def interpolate_NaNs_in_Lennartz_fields(target='OCS'):
    """
    Interpolate NaNs in Lennartz datasets
    """
    import gc
    from multiprocessing import Pool
    from time import gmtime, strftime
    import time
    import glob
    from functools import partial


    # - Load data and
    data_root = utils.get_file_locations('data_root')
    folder = '{}/{}/inputs/'.format(data_root, target)

    # - Interpolate the monthly data
    process_monthly = False
    if process_monthly:
        # Open the monthly dataset as a database
        filenameM = 'ocs_diel_conc_0.125x0.125_v2_monthly.nc'
        dsM = xr.open_dataset( folder+ filenameM )
        # monthly data
        ds = dsM
        # Get DataArray and coordinates for variable
        var2use = 'cwocs'
        da = ds[var2use]
        coords = [i for i in da.coords]
        # Loop by month and update the array
        da_l = []
        times2use = ds.time.values
        #
        p = Pool(12)
        #
        ars = [ds[var2use].sel(time=i).values for i in times2use]
        ars = p.map(partial(interpolate_array_with_GRIDDATA, da=da), ars)
        p.close()
        da.values = np.ma.array(ars)
        # Update and save
        ds[var2use] = da.copy()
        # Clean memory
        gc.collect()
        # save the monthly data
        ds.to_netcdf(folder+filenameM.split('.nc')[0]+'_interp.nc' )

    # - Interpolate the diel data
    # Open the diel dataset as a database
    filenameD = 'ocs_diel_conc_0.125x0.125_v2_diel.nc'
    dsD = xr.open_dataset( folder+ filenameD )
    # Now process
    ds = dsD
    var2use = 'cwocs_diel'
    da = ds[var2use]
    print(ds[var2use].shape)
    coords = [i for i in da.coords]
    # Loop by month and update the array
    da_l = []
    times2use = ds.time.values
    # split up into chunks
    n_chunks = 12
    time_chunks = AC.chunks(times2use, n_chunks)
    ars_full = [ds[var2use].sel(time=i).values for i in times2use]
    ars_chunks = AC.chunks(ars_full, n_chunks)
    ars_new = []
    for n, ars in enumerate( ars_chunks ):
        pstr = 'Processing {} of {} ({:.1f} %)'
        times2use = time_chunks[n]
        pcent = (float(n)+1)/ float(len(ars_chunks)) *100
        print(pstr.format(n, len(ars_chunks), pcent))
        p = Pool(n_chunks)
        #
        ars = p.map(partial(interpolate_array_with_GRIDDATA, da=da), ars)
        ars_new += [ars.copy()]
        p.close()
    # Update and save
    ars_new = [item for sublist in ars_new for item in sublist]
    da.values = np.ma.array(ars_new)
    ds[var2use].values = da.copy()
    # Clean memory
    gc.collect()
    # save the monthly data
    ds.to_netcdf(folder+filenameD.split('.nc')[0]+'_interp.nc' )


def quick_check_of_OCS_emissions(target='OCS'):
    """
    Analyse the emissions of methyl iodide through HEMCO
    """
    #
    root = '/users/ts551/scratch/GC/rundirs/'
    file_str = 'geosfp_4x5_tropchem.v12.2.1.AQSA.{}'
    suffix = 'CH3I.ALL.test_other_sources.repeat.II.OCS/'
    run_dict = {
    # intial test runs
    'OCS_TEST' : root + file_str.format(suffix),
    }
    # use the run_dict from - obs.get_ground_surface_OCS_obs_DIRECT
    wds = run_dict # for debugging...
    target = 'OCS' # for testing
    #
    filename = 'HEMCO_diagnostics.201401010000.nc'
    # Get a dictionary of all the data
    dsDH = GetEmissionsFromHEMCONetCDFsAsDatasets(wds=run_dict)
    # - Analysis the totals
    # Extract the annual totals to a dataFrame
    df = pd.DataFrame()
    for run in dsDH.keys():
        print(run)
        # Get the values and variable names
        vars2use = [i for i in dsDH[run].data_vars if i != 'AREA']
        vars2use = list(sorted(vars2use))
        vals = [ dsDH[run][i].sum().values for i in vars2use ]
        # Save the summed values to the dataframe
        df[run] = pd.Series( dict(zip(vars2use, vals)) )
#        print( dsDH[run].sum() )
    # Print the DataFrame to screen
    print(df)



def do_analysis_on_existing_fields_from_Lennartz_2017():
    """
    Plot up the updated Lennartz2017 fields spatially and get general stats
    """
    # elephant
    # Folder of OCS data
    data_root = utils.get_file_locations('data_root')
    folder = '{}/{}/inputs/'.format(data_root, target)
    # Open the monthly dataset as a database
    filename = 'ocs_diel_conc_0.125x0.125_v2_monthly.nc'
    dsM = xr.open_dataset( folder+ filename )
    # Open the diel dataset as a database
    filename = 'ocs_diel_conc_0.125x0.125_v2_diel.nc'
    dsD = xr.open_dataset( folder+ filename )

    # - plot up spatial to sanity check the outputted values
    sns.reset_orig()
    # Monthly
    vmin, vmax = 0, 75
    version = 'Lennartz_2017_ltd_cbar_regrid_0.125x0.125_updated_2020_monthly'
    var2plot = 'cwocs'
    target = 'OCS'
    units = 'pM'
    var2plot_longname = 'Lennartz2017 (updated) monthly'
    s2splotting.plot_up_seasonal_averages_of_prediction(target=target,
#                                                        ds=ds,
                                                        ds=dsM,
                                                        version=version,
                                                        var2plot=var2plot,
                                         var2plot_longname=var2plot_longname,
                                                        vmin=vmin, vmax=vmax,
                                                        units=units)
    # Diel
    vmin, vmax = 0, 75
    version = 'Lennartz_2017_ltd_cbar_regrid_0.125x0.125_updated_2020_diel'
    var2plot = 'cwocs_diel'
    target = 'OCS'
    units = 'pM'
    var2plot_longname = 'Lennartz2017 (updated) diel'
    s2splotting.plot_up_seasonal_averages_of_prediction(target=target,
#                                                        ds=ds,
                                                        ds=dsD,
                                                        version=version,
                                         var2plot_longname=var2plot_longname,
                                                        var2plot=var2plot,
                                                        vmin=vmin, vmax=vmax,
                                                        units=units)




    # - Now get all the data together
    # - Set core local variables
    target = 'OCS'
    # retrive models with the observations
    RFR_dict = build_or_get_models_OCS(rebuild=False, target=target)
    df = RFR_dict['df']
    # Add the ML values to this
    df = add_ensemble_prediction2df(df=df, target=target)
    # Add the Lennetz values to this.
    LatVar = 'Latitude'
    LonVar = 'Longitude'
    MonthVar = 'Month'
    # extract the nearest values - from monthly Lennartz2020
    vals = utils.extract4nearest_points_in_ds(ds=dsM, lons=df[LonVar].values,
                                              lats=df[LatVar].values,
                                              months=df[MonthVar].values,
                                              var2extract='cwocs',)
    var2use = 'Lennartz2017_UP_monthly'
    df[var2use] = vals
    # now extra the diel values
    TimeVar = 'Date'
    HourVar = 'Hour'
    idx_dict = calculate_idx2extract_2D(df=df, ds=dsD,
                                        TimeVar=TimeVar, LatVar=LatVar,
                                        MonthVar=MonthVar, HourVar=HourVar,
                                        LonVar=LonVar)
    # Update to ds names
#    rename_dict = {'Latitude':'lat', 'Longitude':'lon', 'Date':'time'}
#    idx_dict = { rename_dict[k]: v for k, v in idx_dict.items() }
    # Loop and extract these points from database
    # TODO: this must be able to be done en masse!
    var2extract = 'cwocs_diel'
    var2use = 'Lennartz2017_UP_diel'
    idx2use = df.index.values
    vals = []
    for n, idx in enumerate( idx2use ):
        # get the indexes for a specific observation
        lat_idx = idx_dict[LatVar][n]
        lon_idx = idx_dict[LonVar][n]
        time_idx = idx_dict[TimeVar][n]
        if debug:
            print(lat_idx, lon_idx, time_idx)
        # retrive the datapoint
        ds_tmp = dsD.isel( lat=lat_idx, lon=lon_idx, time=time_idx )
        if debug:
            print( ds_tmp[var2extract].values )
        #
        vals += [float(ds_tmp[var2extract].values)]
    df[var2use] = vals

    # - Now plot up ODR and window comparisons

    # Plot up an orthogonal distance regression (ODR) plot
    ylim = (0, 80)
    xlim = (0, 80)
#    ylim = (0, 250)
#    xlim = (0, 250)
#    xlim, ylim =  None, None
    params =  [
    'RFR(Ensemble)', 'Lennartz2017_UP_monthly', 'Lennartz2017_UP_diel'
    ]
    # only points where there is data for both
    df = df.loc[df[params].dropna().index, :]
    # ODR
    s2splotting.plot_ODR_window_plot(df=df, params=params, units='pM',
                                     target=target,
                                     ylim=ylim, xlim=xlim)

    # Plot up a PDF of concs and bias
    ylim = (0, 80)
#    ylim = (0, 250)
    s2splotting.plot_up_PDF_of_obs_and_predictions_WINDOW(df=df, params=params,
                                                          units='pM',
                                                          target=target,
                                                          xlim=xlim)




def do_analysis_on_existing_fields_from_Lennartz_2017():
    """
    Plot up the existing Lennartz2017 fields spatially and get general stats
    """
    # Regrid the existing field to ~12x12km resolution
    # NOTE: Just run once, so hased out for now.
#    regrid_Lennartz2017_OCS_field()
    # open regrided field
    filename = 'OCS_concentration_0.125x0.125.nc'
    ds = xr.open_dataset( folder+ filename )
    # interpolate where there are values?
    # e.g. using the interpolate_array_with_GRIDDATA
    # plot up spatial to sanity check the outputted values
    vmin, vmax = 0, 75
    version = 'Lennartz_2017_limited_colourbar_regridded_0.125x0.125'
    var2plot = 'cwocs'
    target = 'OCS'
    units = 'pM'
    s2splotting.plot_up_seasonal_averages_of_prediction(target=target,
#                                                        ds=ds,
                                                        ds=ds_NEW,
                                                        version=version,
                                                        var2plot=var2plot,
                                                        vmin=vmin, vmax=vmax,
                                                        units=units)

    # Analysis on the reigridded existing field
    # Get the main dataframe of values
    df = RFR_dict['df']
    # Add the ML values to this
    df = add_ensemble_prediction2df(df=df, target=target)
    # Add the Lennetz values to this.
    LatVar = 'Latitude'
    LonVar = 'Longitude'
    MonthVar = 'Month'

    # extract the nearest values
    vals = utils.extract4nearest_points_in_ds(ds=ds, lons=df[LonVar].values,
                                              lats=df[LatVar].values,
                                              months=df[MonthVar].values,
                                              var2extract='cwocs',)
    var2use = 'Lennartz2017'
    df[var2use] = vals

    # - Plot up the performance

    # Plot performance of models
#    RFRanalysis.plt_stats_by_model(stats=stats, df=df, target=target )
    # Plot up also without derivative variables
#    RFRanalysis.plt_stats_by_model_DERIV(stats=stats, df=df, target=target )

    # - Plot comparisons against observations
    # Plot up an orthogonal distance regression (ODR) plot
    ylim = (0, 80)
    xlim = (0, 80)
#    xlim, ylim =  None, None
    params =  ['RFR(Ensemble)', 'Lennartz2017' ]
    s2splotting.plot_ODR_window_plot(df=df, params=params, units='pM',
                                     target=target,
                                     ylim=ylim, xlim=xlim)

    # Plot up a PDF of concs and bias
    ylim = (0, 80)
    s2splotting.plot_up_PDF_of_obs_and_predictions_WINDOW(df=df, params=params,
                                                          units='pM',
                                                          target=target,
                                                          xlim=xlim)


def calculate_idx2extract_2D(ds=None, df=None, LonVar='lon', LatVar='lat',
                             TimeVar='time',
#                             AltVar='hPa', dsAltVar='lev',
                             MonthVar='month', HourVar='hour',
                             dsLonVar='lon', dsLatVar='lat', dsTimeVar='time',
                             debug=False):
    """
    Calculate the indexes to extract of a dataset using a dataframe
    """
    # add a datetime column to the
    df[TimeVar] = pd.to_datetime( df[TimeVar].values )
    # Get arrays of the coordinate variables in the dataset
    ds_lat = ds[dsLatVar].values
    ds_lon = ds[dsLonVar].values
    ds_time = ds[dsTimeVar]
#    ds_month = ds[dsTimeVar+'.month'].values
    ds_hour = ds[dsTimeVar+'.hour'].values
    # Calculate the index individually by coordinate for lat and lon
    lat_idx = [AC.find_nearest(ds_lat, i) for i in df[LatVar].values]
    lon_idx = [AC.find_nearest(ds_lon, i) for i in df[LonVar].values]
    # Add a hour column if not present
    try:
        df[MonthVar]
    except:
#        df[MonthVar] = df[TimeVar+'.month'].values
        df[MonthVar] = [i.month for i in AC.dt64_2_dt( df['Date'].values ) ]
    # for time, first find correct month
    dfHours = np.array(list(set(ds_hour)))
#    nearest_hour = [ for i in df[HourVar].values]
    # map a check of two columns
    def closest_hr_and_month(month, hour, ds_time=ds_time):
        """
        select the time index closest to given month and hour
        """
        # First get nearest day
        ds_tmp = ds_time.sel(time=(ds_time['time.month'] == month) )
        # Then hour within that day
        nearest_hour_idx = AC.find_nearest(dfHours, hour)
        nearest_hour = dfHours[nearest_hour_idx]
        ds_tmp = ds_tmp.sel(time=(ds_tmp['time.hour'] == nearest_hour) )
        return int(AC.find_nearest(ds_time, ds_tmp.values))
    # Now apply the function to return indexes
    df['time_idx'] = df[[HourVar, MonthVar]].apply(lambda x:
                                                   closest_hr_and_month(
                                                   month=x[MonthVar],
                                                   hour=x[HourVar]),
                                                   axis=1
                                                   )
    time_idx = df['time_idx'].values
    del df['time_idx']
    # Return a dictionary of the values
    print([len(i) for i in (lat_idx, lon_idx, time_idx) ])
    return {LatVar:lat_idx, LonVar:lon_idx, TimeVar:time_idx}


def extract4nearest_points_in_ds_inc_hr(ds=None, lons=None, lats=None,
                                        months=None, hours=None,
                                        var2extract='Ensemble_Monthly_mean',
                                        select_within_time_dim=True,
                                        select_nearest_hour=True,
                                        target='Iodide', verbose=True,
                                        debug=False):
    """
    Extract requested variable for nearest point and time from NetCDF

    Parameters
    -------
    lons (np.array): list of Longitudes to use for spatial extraction
    lats (np.array): list of latitudes to use for spatial extraction
    months (np.array): list of months to use for temporal extraction
    var2extract (str): name of variable to extract data for
    select_within_time_dim (bool): select the nearest point in time?
    debug (bool): print out debugging output?

    Returns
    -------
    (xr.Dataset)
    """
    # Get data from NetCDF as a xarray dataset
    if isinstance(ds, type(None)):
        ds = get_predicted_values_as_ds(target=target)
    # Check that the same about of locations have been given for all months
    lens = [len(i) for i in (lons, lats, months)]
    assert len(set(lens)) == 1, 'All lists provided must be same length!'
    # Loop locations and extract
    extracted_vars = []
    for n_lon, lon_ in enumerate(lons):
        # Get lats and month too
        lat_ = lats[n_lon]
        month_ = months[n_lon]
        # Select for month
        ds_tmp = ds[var2extract]
        if select_within_time_dim:
            ds_tmp = ds_tmp.sel(time=(ds['time.month'] == month_))
            # If
            if select_nearest_hour:
                try:
                    hour_ = hours[n_lon]
                    if is_number(hour_):
                        ds_tmp = ds_tmp.sel(hourofday =
                                            (ds['hourofday'] == hour_)
                                            )
                    else:
                        p_str = "hour '{}' not a # (var #{} @ lat={}, lon={})"
                        print(p_str.format(hour_,n_lon, lat_, lon_ ))
                except:
                    pass

        # Select nearest data in space
        vals = ds_tmp.sel(lat=lat_, lon=lon_, method='nearest')
        if debug:
            pstr = '#={} ({:.2f}%) - vals:'
            print(pstr.format(n_lon, float(n_lon)/len(lons)*100), vals)
        extracted_vars += [float(vals.values)]
    return extracted_vars



def plot_existing_fields_from_Lennartz_2017():
    """
    Plot up the existing Lennartz2017 fields spatially and get general stats
    """
    # Local variables
    target = 'OCS'
    units = 'pM'
    version = 'Lennartz_2017'
    # Get the OCS files from Lennartz et al. 2017
    ds = get_OCS_fields_from_Lennartz_2017_as_ds()
    # plot these up as an annual average
    var2plot = 'cwocs'
    s2splotting.plot_up_annual_averages_of_prediction(target=target, ds=ds,
                                                      version=version,
                                                      var2plot=var2plot,
                                                      units=units)
    # plot these up as an annual average but with the same colourbar as Lennartz2017
    var2plot = 'cwocs'
    version = 'Lennartz_2017_limited_colourbar'
    title = 'Annual average OCS values from Lennartz2017'
#    s2splotting.
    plot_up_annual_averages_of_prediction(target=target, ds=ds,
                                                      version=version,
                                                      var2plot=var2plot,
                                                      vmin=vmin, vmax=vmax,
                                                      title=title,
                                                      units=units)


    # plot these up  - seasonally resolved average
    s2splotting.plot_up_seasonal_averages_of_prediction(target=target, ds=ds,
                                                        version=version,
                                                        var2plot=var2plot,
                                                        units=units)
    # plot up seasonally, but with the same colourbar as the ML prediction
    vmin, vmax = 0, 75
    version = 'Lennartz_2017_limited_colourbar'
    s2splotting.plot_up_seasonal_averages_of_prediction(target=target, ds=ds,
                                                        version=version,
                                                        var2plot=var2plot,
                                                        vmin=vmin, vmax=vmax,
                                                        units=units)
    # - also plot ML prediction with the same colourbar
    # get the predicted data as saved offline
    ds = utils.get_predicted_values_as_ds(target=target, )
    # seasonally resolved avg. - using same colourbar as the OCS values Lennartz2017
    vmin, vmax = 0, 75
    version = 'ML_v0.0.0_limited_colourbar'
    s2splotting.plot_up_seasonal_averages_of_prediction(target=target, ds=ds,
                                                       version=version,
#                                                      var2plot=var2plot,
                                                       vmin=vmin, vmax=vmax,
                                                       units=units)


def regrid_Lennartz2017_OCS_field():
    """
    Regrid Lennartz2017 data field to G5NR 0.125x0.125
    """
    # Set name and location for re-gridded file
    data_root = utils.get_file_locations('data_root')
    folder2save = '{}/{}/inputs/'.format(data_root, target)
    filename2save = 'ocs_diel_conc.nc_0.125x0.125'
    # Regrid the dataset
#    ds_NEW = regrid_ds_field2G5NR_res(ds, folder2save=folder2save,
#                                   filename2save=filename2save)
    # Save locally for now
#    ds_NEW.to_netcdf('TEST.nc')
    # Just process an save online using s2s functions
    regrid_ds_field2G5NR_res(ds, folder2save=folder2save, save2netCDF=True,
                             filename2save=filename2save)

    # - Also  regrid the updated fields.
    ds = get_monthly_OCS_fields_post_Lennartz_2017_as_ds()
    # monthly field
    filename2save = 'ocs_diel_conc_0.125x0.125_v2_monthly'
    regrid_ds_field2G5NR_res(ds, folder2save=folder2save, save2netCDF=True,
                             filename2save=filename2save)


    # diel
    ds = get_diel_OCS_fields_post_Lennartz_2017_as_ds()
    filename2save = 'ocs_diel_conc_0.125x0.125_v2_diel'
    regrid_ds_field2G5NR_res(ds, folder2save=folder2save, save2netCDF=True,
                             filename2save=filename2save)




def get_OCS_fields_from_Kettle1999():
    """
    Get a dataset of OCS fields from Kettle et al. 1997
    """
    folder = '/users/ts551/scratch/data/s2s/OCS/inputs/'
    filename = 'Kettle_for_GC.nc'
    ds = xr.open_dataset( folder + filename )
    return ds


def get_OCS_fields_from_Lennartz_2017_as_ds():
    """
    Get a dataset of OCS fields from Lennartz et al. 2017
    """
    # Location and name of file to load
    folder = '/users/ts551/scratch/data/s2s/OCS/inputs/'
    filename = 'ocs_concentration.nc'
    ds = xr.open_dataset( folder + filename )
    # Rename the coordinate fields to be consistent with other files used here
    LatVar = 'lat'
    LonVar = 'lon'
    name_dict = {'latitude': LatVar, 'longitude' : LonVar}
    ds = ds.rename(name_dict )
    # Make time an arbitrary year
    dt = [datetime.datetime(2001, i+1, 1) for i in np.arange(12)]
    ds.time.values = dt
    # For the ordering of the dimensions to be time, lat, lon
    ds = ds.transpose('time', 'lat', 'lon')
    # Update the Lon values to start at -180
    NewLon = ds[LonVar].values.copy() -180
#    var2use = 'cwocs'
    ds = ds.roll({LonVar: -64} )
    ds[LonVar].values = NewLon
    return ds


def get_monthly_OCS_fields_post_Lennartz_2017_as_ds():
    """
    Get a dataset of OCS fields updated in Jan 2020 (based on Lennartz2017)
    """
    # Location and name of file to load
    folder = '/users/ts551/scratch/data/s2s/OCS/inputs/'
    filename = 'ocs_diel_conc.nc'
    # Open and update
    ds = xr.open_dataset( folder + filename )
    LatVar = 'lat'
    LonVar =  'lon'
    # Update variables names
    ds = ds.rename( {
    'timeofday':'hourofday', 'latitude': LatVar, 'longitude':LonVar
    } )
    # Update the Lon values to start at -180
    NewLon = ds[LonVar].values.copy() -180
    ds = ds.roll({LonVar: -64} )
    ds[LonVar].values = NewLon
    # - Unify the time dimension
    vars2use = ['cwocs', 'em_ocs', 'area']
    ds = ds[vars2use]
    # Update the Time dimension
    dt = [datetime.datetime(2001, int(i), 15) for i in ds.month.values ]
    ds = ds.assign_coords({'time': dt})
    # Template existing dataset, then add new variables
    dsOLD = ds.copy()
    ds = ds[[vars2use[0]]].mean(dim='month').rename( {vars2use[0]: 'TEMPLATE'})
    # Add area to the dataset
    AREA_var = 'AREA'
    ds[AREA_var] = dsOLD[ 'area'].copy()
    # loop by variable and add to the dataset
    vars2use_exc_area = [i for i in vars2use if 'area' not in i.lower() ]
    for var in vars2use_exc_area:
        attrs = dsOLD[var].attrs.copy()
        #
        arr = dsOLD[var].values
        lons = dsOLD[LonVar]
        lats = dsOLD[LatVar]
        ds[var] = xr.DataArray(arr,
                               coords=[dt, lons, lats],
                               dims=['time', 'lon', 'lat']
                               )
        # restore attrs
        ds[var].attrs = attrs
    # Remove the template variable
    del ds['TEMPLATE']
    ds = ds.squeeze()
    # For the ordering of the dimensions to be time, lat, lon
    ds = ds.transpose('time', 'lat', 'lon')
    return ds


def get_diel_OCS_fields_post_Lennartz_2017_as_ds():
    """
    Get a dataset of OCS fields updated in Jan 2020 (based on Lennartz2017)
    """
    # Location and name of file to load
    folder = '/users/ts551/scratch/data/s2s/OCS/inputs/'
    filename = 'ocs_diel_conc.nc'
    # Open and update
    ds = xr.open_dataset( folder + filename )
    LatVar = 'lat'
    LonVar =  'lon'
    # Update variables names
    ds = ds.rename( {
    'timeofday':'hourofday', 'latitude': LatVar, 'longitude':LonVar
    } )
    # Update the Lon values to start at -180
    NewLon = ds[LonVar].values.copy() -180
    ds = ds.roll({LonVar: -64} )
    ds[LonVar].values = NewLon
    # - Unify the time dimension
    vars2use = ['cwocs_diel', 'em_ocs_diel', 'area']
    ds = ds[vars2use]
    # Update the Time dimension
    dt = []
    for month in ds.month.values:
        for hour in ds.hourofday.values:
            dt += [ datetime.datetime(2001, int(month), 15, int(hour)) ]
    ds = ds.assign_coords({'time': dt})
    # Template existing dataset, then add new variables
    dsOLD = ds.copy()
    ds = ds[[vars2use[0]]].mean(dim='month').rename( {vars2use[0]: 'TEMPLATE'})
    ds = ds.mean(dim='hourofday')
    # Add area to the dataset
    AREA_var = 'AREA'
    ds[AREA_var] = dsOLD[ 'area'].copy()
    # loop by variable and add to the dataset
    vars2use_exc_area = [i for i in vars2use if 'area' not in i.lower() ]
    for var in vars2use_exc_area:
        attrs = dsOLD[var].attrs.copy()
        # extract array ver date and time
        ds_l = []
        for month in dsOLD.month.values:
            for hour in dsOLD.hourofday.values:
                ds_l += [dsOLD[var].sel(hourofday=hour, month=month).values]

        # Now recombine into a single array and add to dataset
        arr = np.ma.stack(ds_l)
        lons = dsOLD[LonVar]
        lats = dsOLD[LatVar]
        ds[var] = xr.DataArray(arr,
                               coords=[dt, lons, lats],
                               dims=['time', 'lon', 'lat']
                               )
        # restore attrs
        ds[var].attrs = attrs
    # Remove the template variable
    del ds['TEMPLATE']
    ds = ds.squeeze()
    # For the ordering of the dimensions to be time, lat, lon
    ds = ds.transpose('time', 'lat', 'lon')
    return ds


def add_ensemble_prediction2df(df=None, LatVar='Latitude', LonVar='Longitude',
                               target='Iodide', version='_v0_0_0',
                               var='RFR(Ensemble)', MonthVar='Month',
                               var2extract='Ensemble_Monthly_mean'):
    """
    Wrapper function to add the ensemble prediction to a dataframe from NetCDF

    Parameters
    -------
    target (str): Name of the target variable (e.g. iodide)
    LatVar (str): variable name in DataFrame for latitude
    LonVar (str): variable name in DataFrame for longitude
    MonthVar (str): variable name in DataFrame for month
    version (str): Version number or string (present in NetCDF names etc)
    var2extract (str): variable to extract from the

    Returns
    -------
    (pd.DataFrame)
    """
    # Get the 3D prediction as a dataset
    ds = utils.get_predicted_values_as_ds(target=target, version=version)
    # extract the nearest values
    vals = utils.extract4nearest_points_in_ds(ds=ds, lons=df[LonVar].values,
                                              lats=df[LatVar].values,
                                              months=df[MonthVar].values,
                                              var2extract=var2extract)
    df[var] = vals
    return df


def explore_values_per_hour(df, target='OCS', dpi=320, debug=False):
    """
    Explore the concentrations of OCS on a hourly basis
    """
    import seaborn as sns
    sns.set(style="whitegrid")
    # - get the data
    df = obs.get_OCS_obs()
    N0 = float(df.shape[0])
    # drop the NaNs
    df.dropna()
    # - plot up the values by hour
    hrs = df['Hour'].value_counts(dropna=True)
    hrs = hrs.sort_index()
    N = float(hrs.sum())
    ax = hrs.plot.bar(x='hour of day', y='#', rot=0)
    title_str = '{} data that includes measured hour \n (N={}, {:.2f}% of all data)'
    plt.title( title_str.format(target, int(N), N/N0*100 ) )
    # Update the asthetics
    time_labels = hrs.index.values
    # make sure the values with leading zeros drop these
    index = [float(i) for i in time_labels]
    # Make the labels into strings of integers
    time_labels = [str(int(i)) for i in time_labels]
    if len(index) < 6:
        ax.set_xticks(index)
        ax.set_xticklabels(time_labels)
    else:
        ax.set_xticks(index[2::3])
        ax.set_xticklabels(time_labels[2::3])
    xticks = ax.get_xticks()
    if debug:
        print((xticks, ax.get_xticklabels()))
    # Save the plot
    plt.savefig( 's2s_obs_data_by_hour_{}'.format(target), dpi=dpi)
    plt.close('all')

    # - Plot the data X vs. Y for the obs vs. hour
    x_var, y_var = 'Hour', 'OCS'
    df_tmp = df[ [x_var, y_var] ]
    X = df_tmp[x_var].values
    Y = df_tmp[y_var].values
    # Drop NaNs

    # Plot all data
    fig = plt.figure(dpi=dpi, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(111)
    alpha = 0.5
    plt.scatter(X, Y, color='red', s=3, facecolor='none', alpha=alpha)
    # plot linear fit line


    # Now plot
#    AC.plt_df_X_vs_Y( df=df_tmp, x_var=x_var, y_var=y_var, save_plot=True )
    png_filename = 'X_vs_Y_{}_vs_{}'.format(x_var, y_var)
    png_filename = AC.rm_spaces_and_chars_from_str(png_filename)
    plt.savefig(png_filename, dpi=dpi)


    plt.close('all')


def build_or_get_models_OCS(rm_Skagerrak_data=True, target='OCS',
                               rm_LOD_filled_data=False,
                               rm_outliers=True,
                               rebuild=False):
    """
    Wrapper call to build_or_get_models for sea-surface OCS

    Parameters
    -------
    target (str): Name of the target variable (e.g. iodide)
    rm_outliers (bool): remove the outliers from the observational dataset
    rm_LOD_filled_data (bool): remove the limit of detection (LOD) filled values?
    rebuild (bool): rebuild the models or just read them from disc?

    Returns
    -------
    (pd.DataFrame)
    """
    # Get the dictionary  of model names and features (specific to OCS)
    model_feature_dict = utils.get_model_features_used_dict(rtn_dict=True)
    # Get the observational dataset prepared for ML pipeline
    df = get_dataset_processed4ML(target=target, rm_outliers=rm_outliers)
    # Now extract built models or build new models
    if rebuild:
        RFR_dict = build_or_get_models(save_model_to_disk=True,
                                       model_feature_dict=model_feature_dict,
                                       df=df, target=target,
                                       read_model_from_disk=False,
                                       delete_existing_model_files=True)
    else:
        RFR_dict = build_or_get_models(save_model_to_disk=False,
                                       model_feature_dict=model_feature_dict,
                                       df=df, target=target,
                                       read_model_from_disk=True,
                                       delete_existing_model_files=False)
    return RFR_dict


def get_dataset_processed4ML(restrict_data_max=False, target='OCS',
                             rm_Skagerrak_data=False, rm_outliers=True,
                             rm_LOD_filled_data=False):
    """
    Get dataset as a DataFrame with standard munging settings

    Parameters
    -------
    restrict_data_max (bool): restrict the obs. data to a maximum value?
    target (str): Name of the target variable (e.g. iodide)
    rm_outliers (bool): remove the outliers from the observational dataset
    rm_LOD_filled_data (bool): remove the limit of detection (LOD) filled values?

    Returns
    -------
    (pd.DataFrame)
    """
    from observations import add_extra_vars_rm_some_data
    from observations import get_processed_df_obs_mod
    # - Local variables
    features_used = None
    target = 'OCS'
    target_name = [target]
    # - The following settings are set to False as default
    # settings for incoming feature data
    restrict_min_salinity = False
    # Apply transforms to data?
    do_not_transform_feature_data = True
    # Just use the forest outcomes and do not optimise
    use_forest_without_optimising = True
    # - Get data as a DataFrame
    df = get_processed_df_obs_mod()  # NOTE this df contains values >400nM
    # Add extra vairables and remove some data.
    df = add_extra_vars_rm_some_data(df=df, target=target,
                                     restrict_data_max=restrict_data_max,
                                     restrict_min_salinity=restrict_min_salinity,
                                     rm_outliers=rm_outliers,
                                     )    # add
    # Re-index to a single contiguous index
    df['Original Index'] = df.index.copy()
    N1 = df.shape[0]
    df.index = np.arange(N1)
    print('WARNING: Reindexed to shape of DataFrame processed for ML ({})'.format(N1))

    # - Add test and training set assignment to columns
    # standard split vars?  (values=rand_20_80, rand_strat)
    ways2split_data = {
        'rn. 20%': (True, False),
        'strat. 20%': (False, True),
    }
    # Loop training/test split methods
    for key_ in ways2split_data.keys():
        # Get settings
        rand_20_80, rand_strat = ways2split_data[key_]
        # Copy a df for splitting
#        df_tmp = df['OCS'].copy()
        # Now split using existing function
        returned_vars = mk_test_train_sets(df=df.copy(),
                                           target=target,
                                           rand_20_80=rand_20_80,
                                           rand_strat=rand_strat,
                                           features_used=df.columns.tolist(),
                                           )
        train_set, test_set, test_set_targets = returned_vars
        # Now assign the values
        key_varname = 'Test set ({})'.format(key_)
        df[key_varname] = False
        df.loc[test_set.index, key_varname] = True
        df.loc[train_set.index, key_varname] = False
    return df


def GetEmissionsFromHEMCONetCDFsAsDatasets(wds=None, average_over_time=False):
    """
    Get the emissions from the HEMCO NetCDF files as a dictionary of datasets.
    """
    import AC_tools as AC
    # Look at emissions through HEMCO
    # Get data locations and run names as a dictionary
    if isinstance(wds, type(None)):
        wds = get_run_dict4EGU_runs()
    runs = list(wds.keys())
    # variables to extract
    vars2use = [
        #
        'EmisCH3I_ordonez', 'EmisCH3I_SEAFLUX', 'EmisCH3I_TOTAL',
        'EmisCH2I2_Ocean', 'EmisCH2ICl_Ocean', 'EmisCH2IBr_Ocean', 'EmisI2_Ocean',
        'EmisHOI_Ocean', 'EmisI2_Ocean_Total', 'EmisHOI_Ocean_Total',
        'EmisCH2Br2_Ocean', 'EmisCHBr3_Ocean',
        #
        'EmisCH3I_B02_RICE', 'EmisCH3I_B02_WETL', 'EmisCH3I_B02_BIOBURN',
        'EmisCH3I_B02_BIOFUEL',
        # Also get values for
#        'EmisACET_Ocean', 'EmisALD2_Ocean',
        'EmisDMS_Ocean',
        #
        'EmisCH2Br2_SEAFLUX', 'EmisCHBr3_SEAFLUX',
        'EmisOCS_SEAFLUX',
    ]
    # Make sure there are no double ups in the list
    vars2use = list(set(vars2use))
    # Loop and extract files
    dsDH = {}
    for run in runs:
        wd = wds[run]
        print(run, wd)
        dsDH[run] = AC.get_HEMCO_diags_as_ds(wd=wd)
    # Get actual species
    specs = [i.split('Emis')[-1].split('_')[0] for i in vars2use]
#     specs = [
#     'CH3I','CH3I', 'CH3I', 'CH2Br2', 'CHBr3', 'CH2I2', 'CH2ICl', 'CH2IBr',
#     'HOI', 'I2', 'ISOP'
#     ]
    var_species_dict = dict(zip(vars2use, specs))
    # Only include core species in dataset
    for key in dsDH.keys():
        ds = dsDH[key][ ['AREA'] ]
        # Try and add al of the requested variables too
        for var in vars2use:
            try:
                ds[var] = dsDH[ key ][ var ]
            except KeyError:
                print("WARNING: Skipped '{}' for '{}'".format(key, var) )
        dsDH[key] = ds

    # Convert to Gg
    for run in runs:
        dsREF = dsDH[run].copy()
        # Average over the time dimension?
        if average_over_time:
            ds = dsDH[run].copy().mean(dim='time')
            for var in dsREF.data_vars:
                ds[var].attrs = dsREF[var].attrs
            ds2use = ds
        else:
            ds2use = dsREF
        # Convert the units
        ds = AC.convert_HEMCO_ds2Gg_per_yr(ds2use, vars2convert=vars2use,
                                           var_species_dict=var_species_dict)
        # Update the dictionary
        dsDH[run] = ds
    return dsDH




if __name__ == "__main__":
    main()

