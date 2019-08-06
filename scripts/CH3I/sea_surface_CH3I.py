#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Module to hold processing/analysis functions for CH3I workflow

Notes
----
ML = Machine Learning
target = the variable to be estimated and provided whilst training models
feature = a individual conpoinet of a predictor vector assigned to a target
( could be called an attribute )
"""

import numpy as np
import pandas as pd
import xarray as xr
import sparse2spatial as s2s
import sparse2spatial.utils as utils
#import sparse2spatial.ancillaries2grid_oversample as ancillaries2grid
#import sparse2spatial.archiving as archiving
from sparse2spatial.RFRbuild import mk_test_train_sets
import sparse2spatial.RFRbuild as build
import sparse2spatial.RFRanalysis as RFRanalysis
import sparse2spatial.analysis as analysis
import sparse2spatial.plotting as plotting
from sparse2spatial.RFRbuild import build_or_get_models

# Get CH3I specific functions
from observations import get_CH3I_obs


def main():
    """
    Driver for module's man if run directly from command line. unhash
    functionalitliy to call.
    """
    # - Set core local variables
    target = 'CH3I'

    # - Get the observations? (Not needed for core workflow as also held in RFR_dict)
    # (This processese the observations and only needs to be done once)
#    df = get_dataset_processed4ML(target=target, rm_outliers=rm_outliers)

    # - build models with the observations
    RFR_dict = build_or_get_models_CH3I(rebuild=False, target=target)
    # build the models (just run once!)
#    RFR_dict = build_or_get_models_CH3I(rebuild=True, target=target)

    # Get stats ont these models
    stats = RFRanalysis.get_core_stats_on_current_models(RFR_dict=RFR_dict,
                                                         target=target, verbose=True,
                                                         debug=True)

    # Get the top ten models
    topmodels = build.get_top_models(RFR_dict=RFR_dict, stats=stats,
                                     vars2exclude=['DOC', 'Prod'], n=10)

    # --- Predict values globally (only use 0.125)
    # Extra string for NetCDF save name
    xsave_str = '_INITIAL'
    # Save predictions to NetCDF
    save2NetCDF = True
    # Resolution to use? (full='0.125x0.125', test at lower e.g. '4x5')
#    res = '0.125x0.125'
    res = '4x5'
    build.mk_predictions_for_3D_features(None, res=res, RFR_dict=RFR_dict,
                                         save2NetCDF=save2NetCDF, target=target,
                                         models2compare=topmodels,
                                         topmodels=topmodels,
                                         xsave_str=xsave_str, add_ensemble2ds=True)

    # --- Plot up the performance of the models
    df = RFR_dict['df']
    # Plot performance of models
    analysis.plt_stats_by_model(stats=stats, df=df, target=target )
    # Plot up also without derivative variables
    analysis.plt_stats_by_model_DERIV(stats=stats, df=df, target=target )

    # ---- Explore the predicted concentrations
    # Get the data
    ds = utils.get_predicted_3D_values(target=target)
    # plot up an annual mean
    plotting.plot_up_annual_averages_of_prediction(ds=ds, target=target)
    # Plot up the values by season
    plotting.plot_up_seasonal_averages_of_prediction(ds=ds, target=target)

    # ---- Check analysis for original emissions from Bell et al.
    # Check budgets

    # plot



def check_budgets_from_Bell_emiss():
    """
    """
    # target species for analysis
    target = 'CH3I'
    # Root data directory
    root = '/mnt/lustre/groups/chem-acm-2018/earth0_data/'
    root += 'GEOS//ExtData/HEMCO/'
    # Folder of CH3I emissions
    folder = root + 'CH3I/v2014-07/'
    # - Get the wetland emissions
    filename = 'ch4_wetl.geos.4x5.nc'
    dsW = xr.open_dataset( folder + filename )
    var2plot = 'CH4_SRCE__CH4cl'
    # Plot up the annual average emissions
    ds2plot = dsW[[var2plot]].mean(dim='time')
    # Set a title for the plot
    extr_str = 'Wetland_emissions_CH3I'
    units = dsW[var2plot].units
    title = "Wetland emissions of {} ({})".format(target, units)
    plotting.plot_spatial_data(ds=ds2plot, var2plot=var2plot, extr_str=extr_str,
                               fillcontinents=False,
                               target=target, title=title, units=units)

    # - Get the rice emissions
    filename = 'ch4_rice.geos.4x5.nc'
    dsR = xr.open_dataset( folder + filename )
    var2plot = 'CH4_SRCE__CH4an'
    # Plot up the annual average emissions
    ds2plot = dsR[[var2plot]].mean(dim='time')
    # Set a title for the plot
    extr_str = 'Rice_emissions_CH3I'
    units = dsR[var2plot].units
    title = "Rice emissions of {} ({})".format(target, units)
    plotting.plot_spatial_data(ds=ds2plot, var2plot=var2plot, extr_str=extr_str,
                               fillcontinents=False,
                               target=target, title=title, units=units)


    # - Get the ocean concentrations
    filename = 'ocean_ch3i.geos.4x5.nc'
    dsO = xr.open_dataset( folder + filename )
    var2plot = 'CH3I_OCEAN'
    # Plot up the annual average concentrations
    ds2plot = dsO[[var2plot]].mean(dim='time')
    # Set a title for the plot
    extr_str = 'Ocean_concs_CH3I'
    units = dsO[var2plot].units
    title = "Ocean concentrations of {} ({})".format(target, units)
    plotting.plot_spatial_data(ds=ds2plot, var2plot=var2plot, extr_str=extr_str,
                                target=target, title=title, units=units)


    # - Get the ocean concentrations (assuming HEMCO conversion error)
    filename = 'ocean_ch3i.geos.4x5.nc'
    dsO = xr.open_dataset( folder + filename )
    var2plot = 'CH3I_OCEAN'
    # Plot up the annual average concentrations
    ds2plot = dsO[[var2plot]].mean(dim='time')
    #
    ds2plot[var2plot] = ds2plot[var2plot] / ( 141.9 * 1E-9 )
    # Set a title for the plot
    extr_str = 'Ocean_concs_CH3I_nM'
    units = 'nM'
    title = "Ocean concentrations of {} \n ".format(target)
    title += "({} - assuming HEMCO conversion)".format( 'nM')
    plotting.plot_spatial_data(ds=ds2plot, var2plot=var2plot, extr_str=extr_str,
                                target=target, title=title, units=units)

    # - Get the ocean concentrations
    filename = 'ocean_ch3i.geos.4x5.nc'
    dsO = xr.open_dataset( folder + filename )
    var2plot = 'CH3I_OCEAN'
    # Plot up the annual average concentrations
    ds2plot = dsO[[var2plot]].mean(dim='time')
    #
    ds2plot[var2plot] = ds2plot[var2plot] / ( 141.9 * 1E-9 ) *10
    # Set a title for the plot
    extr_str = 'Ocean_concs_CH3I_nM_Assume_error'
    units = 'nM'
    title = "Ocean concentrations of {} \n ".format(target)
    title += "({} - assuming error of x10)".format('nM')
    plotting.plot_spatial_data(ds=ds2plot, var2plot=var2plot, extr_str=extr_str,
                                target=target, title=title, units=units)

    # - plot up the new kg/m3 concs.
    folder2 = '/users/ts551/scratch/data/s2s/CH3I/outputs/'
    filename = 'Oi_prj_predicted_CH3I_0.125x0.125_kg_m3.nc'
    dsML = xr.open_dataset( folder2 + filename )
    var2plot = 'Ensemble_Monthly_mean_kg_m3'
    # Plot up the annual average concentrations
    ds2plot = dsML[[var2plot]].mean(dim='time')
    # Set a title for the plot
    extr_str = 'Ocean_concs_CH3I_ML'
    units = dsML[var2plot].units
    title = "ML ocean concentrations of {} ({})".format(target, units)
    plotting.plot_spatial_data(ds=ds2plot, var2plot=var2plot, extr_str=extr_str,
                                target=target, title=title, units=units)



def plot_seasaonl_model_vs_obs(dpi=320):
    """
    Plot seasonal comparisons of observations and models
    """
    # Get observational data
    dfs = get_ground_surface_CH3I_obs()
    sites = list(dfs.keys())
    # get model_data
    run_root = '/users/ts551/scratch/GC/rundirs/'
    run_dict = {
    'Ordonez2012': run_root+'geosfp_4x5_tropchem.v12.2.1.AQSA.GFAS/'
    }
    runs = list(run_dict.keys())

    # loop and extract by site
    res='4x5'
    model_data = {}
    for run in runs:
        # Get data location and filename
        wd = run_dict[run]
        filename = 'ctm.nc'
        # open the file
        ds = xr.open_dataset(wd+filename)
        # select the nearest location to a given site
        df = pd.DataFrame()
        for site in sites:
            print(site)
            # Get: lat, lon, alt (m)
            lat, lon, alt, = AC.get_loc(site)
            # Nearest lat
            ds_tmp = ds['IJ_AVG_S__'+'CH3I'].sel(latitude=lat, method='nearest')
            # Nearest lon
            ds_tmp = ds_tmp.sel(longitude=lon, method='nearest')
            # take the bottom-=most box
            ds_tmp = ds_tmp.sel(model_level_number=1)
            # convert to pptv
            ds_tmp = ds_tmp *1E3
            # colaspe to a series and save to dataframe
            df[site] = ds_tmp.to_pandas().copy()
            del ds_tmp
        # Save the data to a DataFrame
        model_data[run] = df.copy()
        del df

    # Now loop by site and plot up
    for site in sites:
        print(site)
#        fig = plt.figure(figsize=(16, 10))
        fig = plt.figure()
        #
#        ax = fig.add_subplot(*axn)
        ax = fig.add_subplot()
        # Get Obs. for site
        DateVar = 'Datetime'
        DataVar = 'CH3I'
        df = dfs[site][[DateVar, DataVar]].dropna()
        dates = df[DateVar].values
        data = df[DataVar].values
        print(dates.shape, data.shape)
        title = site
        # Plot up observations
        AC.BASIC_seasonal_plot(data=data, dates=dates, color='k', label='Obs.',
                               title=title)
        # plot up model data
        for run in runs:
            df = model_data[run]
            #
            dates = df.index.values
            data = df[site].values
            # Plot up observations
            AC.BASIC_seasonal_plot(data=data, dates=dates, color='red', label=run)


        #
        filename = 's2s_{}_seasonal_cycle_{}'.format( target, site)
        filename = AC.rm_spaces_and_chars_from_str(filename)
        plt.savefig(filename)


def quick_check_of_CH3I_emissions():
    """
    """
    #
    root = '/users/ts551/scratch/GC/rundirs/'
    file_str = 'geosfp_4x5_tropchem.v12.2.1.AQSA.GFAS.{}'
    run_dict = {
    'Ordonez2012' : root + file_str.format('CH3I.Ordonez2012/spin_up/'),
    'Bell2002' : root + file_str.format('CH3I/spin_up/'),
    'Bell2002x10' : root + file_str.format('CH3I.x10/spin_up/'),
    'ML'    : root + file_str.format('CH3I.ML/spin_up/'),
    }
    wds = run_dict # for debugging...
    #
    filename = 'HEMCO_diagnostics.201401010000.nc'
    # Get a dictionary of all the data
    dsDH = GetEmissionsFromHEMCONetCDFsAsDatasets(wds=run_dict)

    #
    for run in dsDH.keys():
        print(run)

        print( dsDH[run].sum() )


def check_units_of_Bell2002_files():
    """
    convert kg/m3 to ng/L
    """
    #
    pass




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
        # Testing vars from v5.0
#         'TOTAL_CH3I_L', 'TOTAL_SEA_CH3I', 'TOTAL_CH3I_a', 'TOTAL_CH2BR2', 'TOTAL_CHBR3',
#         'TOTAL_CH2I2', 'TOTAL_CH2ICL', 'TOTAL_CH2IBR', 'TOTAL_HOI', 'TOTAL_I2',
#         'TOTAL_ISOP'
        #
        'EmisCH3I_ordonez', 'EmisCH3I_SEAFLUX', 'EmisCH3I_TOTAL',
        'EmisCH2I2_Ocean', 'EmisCH2ICl_Ocean', 'EmisCH2IBr_Ocean', 'EmisI2_Ocean',
        'EmisHOI_Ocean', 'EmisI2_Ocean_Total', 'EmisHOI_Ocean_Total',
        'EmisCH2Br2_Ocean', 'EmisCHBr3_Ocean'
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
        dsDH[key] = dsDH[ key ][ ['AREA']+vars2use  ]

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


def plot_up_obs_data_by_year( target='CH3I',):
    """
    Plot up the amount of observational in a given year, by region
    """
    # Get the observational data
    df = get_CH3I_obs()
    # Add the ocean that the data is in
    df = analysis.add_loc_ocean2df(df=df,  LatVar='Latitude', LonVar='Longitude')
    # Plot up the observational data by year
    plot_up_df_data_by_yr( df=df, target=target )


def plot_up_df_data_by_yr(df=None, Datetime_var='datetime', TimeWindow=5,
                          start_from_last_obs=False, drop_bins_without_data=True,
                          target='Iodide', dpi=320):
    """
    Plot up # of obs. data binned by region against year
    """
    # Sort the dataframe by date
    df.sort_values( by=Datetime_var, inplace=True )
    # Get the minimum and maximum dates
    min_date = df[Datetime_var].min()
    max_date = df[Datetime_var].max()
    # How many years of data are there?
    yrs_of_data = (max_date-min_date).total_seconds()/60/60/24/365
    nbins = AC.myround(yrs_of_data/TimeWindow, base=1 )
    # Start from last observation or from last block of time
    sdate_block = AC.myround(max_date.year, 5)
    sdate_block =  datetime.datetime(sdate_block, 1, 1)
    # Make sure the dates used are datetimes
    min_date, max_date = pd.to_datetime( [min_date, max_date] ).values
    min_date, max_date = AC.dt64_2_dt( [min_date, max_date])
    # Calculate the numer of points for each bin by region
    dfs = {}
    for nbin in range(nbins+2):
        # Start from last observation or from last block of time?
        days2rm = int(nbin*365*TimeWindow)
        if start_from_last_obs:
            bin_start = AC.add_days( max_date, -int(days2rm+(365*TimeWindow)))
            bin_end = AC.add_days( max_date, -days2rm )
        else:
            bin_start = AC.add_days( sdate_block,-int(days2rm+(365*TimeWindow)))
            bin_end = AC.add_days( sdate_block, -days2rm )
        # Select the data within the observational dates
        bool1 = df[Datetime_var] > bin_start
        bool2 = df[Datetime_var] <= bin_end
        df_tmp = df.loc[bool1 & bool2, :]
        # Print the number of values in regions for bin
        if verbose:
            print(bin_start, bin_end, df_tmp.shape)
        # String to save data with
        if start_from_last_obs:
            bin_start_str = bin_start.strftime( '%Y/%m/%d')
            bin_end_str = bin_end.strftime( '%Y/%m/%d')
        else:
            bin_start_str = bin_start.strftime( '%Y')
            bin_end_str = bin_end.strftime( '%Y')
        str2use = '{}-{}'.format(bin_start_str, bin_end_str)
        # Sum up the number of values by region
        dfs[ str2use] = df_tmp['ocean'].value_counts(dropna=False)
    # Combine to single dataframe and sort by date
    dfA = pd.DataFrame( dfs )
    dfA = dfA[list(sorted(dfA.columns)) ]
    # Drop the years without any data
    if drop_bins_without_data:
        dfA = dfA.T.dropna(how='all').T
    # Update index names
    dfA = dfA.T
    dfA.columns
    rename_cols = {
    np.NaN : 'Other',  'INDIAN OCEAN': 'Indian Ocean', 'SOUTHERN OCEAN' : 'Southern Ocean'
    }
    dfA = dfA.rename(columns=rename_cols)
    dfA = dfA.T
    # Plot up as a stacked bar plot
    import seaborn as sns
    sns.set()
    dfA.T.plot(kind='bar', stacked=True)
    # Add title etc
    plt.ylabel( '# of observations')
    plt.title( '{} obs. data by region'.format(target))
    # Save plotted figure
    savename = 's2s_{}_data_by_year_region'.format(target)
    plt.savefig(savename, dpi=dpi, bbox_inches='tight', pad_inches=0.05)



def plt_X_vs_Y_for_regions(RFR_dict=None, df=None, params2plot=[], LatVar='lat',
                           LonVar='lon', target='CH3I',
                           obs_var='Obs.'):
    """
    Plot up the X vs. Y performance by region - using core s2s functions
    """
    # Get the dataframe of observations and predictions
    df = RFR_dict['df']
    # Add ensemble to the df
    LatVar = 'Latitude'
    LonVar = 'Longitude'
    ds = utils.get_predicted_values_as_ds(target=target)
    vals = utils.extract4nearest_points_in_ds(ds=ds, lons=df[LonVar].values,
                                              lats=df[LatVar].values,
                                              months=df['Month'].values,
                                              var2extract='Ensemble_Monthly_mean',)
    var = 'RFR(Ensemble)'
    df[var] = vals
    # Just withheld data?
    testset = 'Test set (strat. 20%)'
    df = df.loc[df[testset] == True, :]
    # Only consider the variables to be plotted
    obs_var = target
#    params2plot = [var,  'Chance2014_STTxx2_I', 'MacDonald2014_iodide',]
    params2plot = [var,  ]
    df = df[params2plot+[LonVar, LatVar, obs_var]]
    # Add ocean columns to dataframe
    df = AC.add_loc_ocean2df(df=df, LatVar=LatVar, LonVar=LonVar)
    # Split by regions
    regions = list(set(df['ocean'].dropna()))
    dfs = [df.loc[df['ocean']==i,:] for i in regions]
    dfs = dict(zip(regions,dfs))
    # Also get an open ocean dataset
    # TODO ...
    # Use an all data for now
    dfs['all'] = df.copy()
    regions += ['all']
    # loop and plot by region
    for region in regions:
        print(region)
        df = dfs[region]
        extr_str=region+' (withheld)'
#        extr_str=region
        # Now plot
        plotting.plt_X_vs_Y_for_obs_v_params(df=df, params2plot=params2plot,
                                             obs_var=obs_var,
                                             extr_str=extr_str)



def build_or_get_models_CH3I(target='CH3I',
                             rm_LOD_filled_data=False,
                             rm_outliers=True,
                             rebuild=False):
    """
    Wrapper call to build_or_get_models for sea-surface CH3I
    """
    # Get the dictionary  of model names and features (specific to iodide)
    model_feature_dict = utils.get_model_features_used_dict(rtn_dict=True)

    # Get the observational dataset prepared for ML pipeline
    df = get_dataset_processed4ML(target=target, rm_outliers=rm_outliers)

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


def get_dataset_processed4ML(restrict_data_max=False, target='CH3I',
                             rm_outliers=True,
                             rm_LOD_filled_data=False):
    """
    Get dataset as a DataFrame with standard munging settings


    Parameters
    -------
    restrict_data_max (bool): restrict the obs. data to a maximum value?

    Returns
    -------
    (pd.DataFrame)

    Notes
    -----

    """
    from observations import add_extra_vars_rm_some_data
    from observations import get_processed_df_obs_mod
    # - Local variables
    features_used = None
    target = 'CH3I'
    target_name = [target]
    # - The following settings are set to False as default
    # settings for incoming feature data
    restrict_min_salinity = False
#    use_median_value_for_chlor_when_NaN = False
#    add_modulus_of_lat = False
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
#        df_tmp = df['Iodide'].copy()
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


if __name__ == "__main__":
    main()
