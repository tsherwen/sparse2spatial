
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
from sparse2spatial.RFRbuild import mk_test_train_sets
from sparse2spatial.RFRbuild import build_or_get_models
from sparse2spatial.RFRbuild import get_top_models
#from sparse2spatial.RFRanalysis import get_stats_on_models
#from sparse2spatial.RFRanalysis import get_stats_on_multiple_global_predictions
# Local modules specific to OCS work
import observations as obs
# Temporarily included
import xesmf as xe


def main():
    """
    Driver for module's man if run directly from command line. unhash
    functionalitliy to call.
    """
    # - Set core local variables
    target = 'OCS'

    # - build models with the observations
    RFR_dict = build_or_get_models_OCS(rebuild=False, target=target)
    # Get stats ont these models
    stats = RFRanalysis.get_core_stats_on_current_models(RFR_dict=RFR_dict,
                                                      target=target, verbose=True,
                                                      debug=True)
    # Get the top ten models
    topmodels = build.get_top_models(RFR_dict=RFR_dict, stats=stats,
                                     vars2exclude=['DOC', 'Prod'], n=10)

    # --- Predict values globally (only use 0.125)
    # extra strig for NetCDF save name
    xsave_str = '_TEST'
    # make NetCDF predictions from the main array
    save2NetCDF = True
    # resolution to use? (full='0.125x0.125', test at lower e.g. '4x5')
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
    # from Lentt
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
    s2splotting.plot_ODR_window_plot(df=df, params=params, units='pM', target=target,
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
    s2splotting.plt_X_vs_Y_for_regions(df=df, target=target, LonVar=LonVar, LatVar=LatVar)


    # --- Save out the field in kg/m3 for use in models
    version = 'v0_0_0'
    folder = '/users/ts551/scratch/data/s2s/{}/outputs/'.format(target)
    filename = 'Oi_prj_predicted_{}_0.125x0.125_{}'.format(target, version)
    ds = xr.open_dataset( folder + filename+'.nc' )
    # Convert to kg/m3
    RMM = 60.08
    new_var = 'Ensemble_Monthly_mean_kg_m3'
    ds = utils.add_converted_field_pM_2_kg_m3(ds=ds, var2use='Ensemble_Monthly_mean',
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
    AREA = AC.calc_surface_area_in_grid(lon_e=lon, lat_e=lat, lon_c=lon, lat_c=lat )
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
        vals



def quick_check_of_OCS_emissions(target='OCS'):
    """
    Analyse the emissions of methyl iodide through HEMCO
    """
    #
    root = '/users/ts551/scratch/GC/rundirs/'
    file_str = 'geosfp_4x5_tropchem.v12.2.1.AQSA.{}'
    run_dict = {
    # intial test runs
    'OCS_TEST' : root + file_str.format('CH3I.ALL.test_other_sources.repeat.II.OCS/',
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
    Plot up the existing Lennartz et al. 2017 fields spatially and get general stats
    """
    # regrid the existing field to ~12x12km resolution
    # NOTE: Just run once, so hased out for now.
#    regrid_Lennartz2017_OCS_feild()
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
    LonVar = 'Londitude'
    LatVar='Latitude'
    LonVar='Longitude'
    MonthVar='Month'

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
    s2splotting.plot_ODR_window_plot(df=df, params=params, units='pM', target=target,
                                     ylim=ylim, xlim=xlim)

    # Plot up a PDF of concs and bias
    ylim = (0, 80)
    s2splotting.plot_up_PDF_of_obs_and_predictions_WINDOW(df=df, params=params,
                                                          units='pM',
                                                          target=target,
                                                          xlim=xlim)


def plot_existing_fields_from_Lennartz_2017():
    """
    Plot up the existing Lennartz et al. 2017 fields spatially and get general stats
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


def regrid_Lennartz2017_OCS_feild():
    """
    Regrid Lennartz2017 data field to G5NR 0.125x0.125
    """
    # Set name and location for re-gridded file
    folder2save = get_file_locations('data_root')+'/{}/inputs/'.format(target)
    filename2save = 'OCS_concentration_0.125x0.125.nc'
    # regrid the dataset
#    ds_NEW = regrid_ds_field2G5NR_res(ds, folder2save=folder2save,
#                                   filename2save=filename2save)
    # Save locally for now
#    ds_NEW.to_netcdf('TEST.nc')
    # Just process an save online using s2s functions
    regrid_ds_field2G5NR_res(ds, folder2save=folder2save,save2netCDF=True,
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
    folder = '/users/ts551/scratch/data/s2s/OCS/inputs/'
    filename = 'ocs_concentration.nc'
    ds = xr.open_dataset( folder + filename )
    # rename the coordinate fields to be consistent with other files used here
    LatVar = 'lat'
    LonVar = 'lon'
    name_dict = {'latitude': LatVar, 'longitude' : LonVar}
    ds = ds.rename(name_dict )
    # Make time an arbitrary year
    dt = [datetime.datetime(2001, i+1, 1) for i in np.arange(12)]
    ds.time.values = dt
    # for the ordering of the dimensions to be time, lat, lon
    ds = ds.transpose('time', 'lat', 'lon')
    # Update the Lon values to start at -180
    NewLon = ds['lon'].values.copy() -180
    var2use = 'cwocs'
    ds = ds.roll({'lon': -64} )
    ds['lon'].values = NewLon
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

