#!/usr/bin/python
# -*- coding: utf-8 -*-
"""

This module contains various analysis for the Ocean iodide (Oi!) project

This includes presentation at conferences etc.

TODO: obsolete code from this module needs to be removed

"""
import numpy as np
import pandas as pd
import sparse2spatial as s2s
import sparse2spatial.utils as utils
import matplotlib
import matplotlib.pyplot as plt
# import AC_tools (https://github.com/tsherwen/AC_tools.git)
import AC_tools as AC

# Get iodide specific functions
import observations as obs

def main():
    """
    Run various misc. scripts linked to the "iodide in the ocean" project
    """
    pass

    # ---- ----- ----- ----- ----- ----- ----- ----- -----
    # ----- ----- Misc (associated iodide project tasks)
    # These include getting CTM (GEOS-Chem) output for Anoop/Sawalha/TropMet
    # --- Make planeflight files for cruise
#    mk_pf_files4Iodide_cruise()
#    mk_pf_files4Iodide_cruise(mk_column_output_files=True)

    # Test the input files for these cruises?
#    test_input_files4Iodide_cruise_with_plots()

    # Test output files for cruises
#    TEST_iodide_cruise_output()
#    TEST_AND_PROCESS_iodide_cruise_output()
#    TEST_AND_PROCESS_iodide_cruise_output(just_process_surface_data=False)

    # Get numbers for data paper (data descriptor paper)
#    get_numbers_for_data_paper()

    # Get Longhurst province labelled NetCDF for res
#    add_LonghurstProvince2NetCDF(res='4x5', ExStr='TEST_VI' )
#    add_LonghurstProvince2NetCDF(res='2x2.5', ExStr='TEST_V' )
#    add_LonghurstProvince2NetCDF(res='0.125x0.125', ExStr='TEST_VIII' )

    # Add Longhurst Province to a lower res NetCDF file
#    folder = './'
#     filename = 'Oi_prj_output_iodide_field_1x1_deg_0_5_centre.nc'
#    filename = 'Oi_prj_output_iodide_field_0_5x0_5_deg_centre.nc'
#    ds = xr.open_dataset(folder+filename)
#    add_LonghurstProvince2NetCDF(ds=ds, res='0.5x0.5', ExStr='TEST_VIII')

    # Process this to csv files for Indian' sea-surface paper





# ---------------------------------------------------------------------------
# ---------------- New plotting of iodine obs/external data -----------------
# ---------------------------------------------------------------------------
def explore_extracted_Arctic_Antarctic_obs_v0(dsA=None, res='0.125x0.125',
                                              dpi=320):
    """
    Analyse the gridded data for the Arctic and Antarctic
    """
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.style.use('ggplot')
    import seaborn as sns
    sns.set()
    # - Local variables
    # Get input variables
    if isinstance(dsA, type(None)):
        filename = 'Oi_prj_predicted_iodide_{}.nc'.format(res)
#        folder = '/shared/earth_home/ts551/labbook/Python_progs/'
        folder = '/shared/earth_home/ts551/data/iodide/'
        filename = 'Oi_prj_feature_variables_{}.nc'.format(res)
        dsA = xr.open_dataset(folder + filename)
    # Variables to consider
    vars2analyse = list(dsA.data_vars)
    # Add LWI to array - NOTE: 1 = water in Nature run LWI files !
    # ( The above comment is not correct! why is this written here? )
    folderLWI = utils.get_file_locations(
        'AC_tools')+'/data/LM/TEMP_NASA_Nature_run/'
    filenameLWI = 'ctm.nc'
    LWI = xr.open_dataset(folderLWI+filenameLWI)
    # Updates dates (to be Jan=>Dec)
    new_dates = [datetime.datetime(1970, i, 1) for i in LWI['time.month']]
    LWI.time.values = new_dates
    # Sort by new dates
    LWI = LWI.loc[{'time': sorted(LWI.coords['time'].values)}]
#    LWI = AC.get_LWI_map(res=res)[...,0]
    dsA['IS_WATER'] = dsA['WOA_TEMP'].copy()
    dsA['IS_WATER'].values = (LWI['LWI'] == 0)
    # Add is land
    dsA['IS_LAND'] = dsA['IS_WATER'].copy()
    dsA['IS_LAND'].values = (LWI['LWI'] == 1)
    # Get surface area
    s_area = AC.calc_surface_area_in_grid(res=res)  # m2 land map
    dsA['AREA'] = dsA['WOA_TEMP'].mean(dim='time')
    dsA['AREA'].values = s_area.T

    # - Select data of interest by variable for locations
    # Setup dicts to store the extracted values
    df65N, df65S, dfALL = {}, {}, {}
    # - setup booleans for the data
    # now loop and extract variablesl
    vars2use = [
        'WOA_Nitrate',
        #    'WOA_Salinity', 'WOA_Phosphate', 'WOA_TEMP_K', 'Depth_GEBCO',
    ]
    # Setup PDF
    savetitle = 'Oi_prj_explore_Arctic_Antarctic_ancillaries_space_PERTURBED'
    pdff = AC.plot2pdfmulti(title=savetitle, open=True, dpi=dpi)
    # Loop by dataset (region) and plots
    for var_ in vars2use:
        # Select the boolean for if water
        IS_WATER = dsA['IS_WATER'].values
        if IS_WATER.shape != dsA[var_].shape:
            # Special case for depth
            # Get value for >= 65
            ds_tmp = dsA.sel(lat=(dsA['lat'] >= 65))
            arr = np.ma.array(12*[ds_tmp[var_].values])
            arr = arr[ds_tmp['IS_WATER'].values]
            # Add to saved arrays
            df65N[var_] = arr
            del ds_tmp
            # Get value for >= 65
            ds_tmp = dsA.sel(lat=(dsA['lat'] <= -65))
            arr = np.ma.array(12*[ds_tmp[var_].values])
            arr = arr[ds_tmp['IS_WATER'].values]
            # Add to saved arrays
            df65S[var_] = arr
            del ds_tmp
            # Get value for all
            ds_tmp = dsA.copy()
            arr = np.ma.array(12*[ds_tmp[var_].values])
            arr = arr[ds_tmp['IS_WATER'].values]
            # Add to saved arrays
            dfALL[var_] = arr
            del ds_tmp
        else:
            # Get value for >= 65
            ds_tmp = dsA.sel(lat=(dsA['lat'] >= 65))
            arr = ds_tmp[var_].values[ds_tmp['IS_WATER'].values]
            # Add to saved arrays
            df65N[var_] = arr
            del ds_tmp
            # Get value for >= 65
            ds_tmp = dsA.sel(lat=(dsA['lat'] <= -65))
            arr = ds_tmp[var_].values[ds_tmp['IS_WATER'].values]
            # Add to saved arrays
            df65S[var_] = arr
            del ds_tmp
            # Get value for >= 65
            ds_tmp = dsA.copy()
            arr = ds_tmp[var_].values[ds_tmp['IS_WATER'].values]
            # Add to saved arrays
            dfALL[var_] = arr
            del ds_tmp

    # Setup a dictionary of regions to plot from
    dfs = {
        '>=65N': pd.DataFrame(df65N), '>=65S': pd.DataFrame(df65S),
        'Global': pd.DataFrame(dfALL),
    }

    # - plot up the PDF distribution of each of the variables.
    for var2use in vars2use:
        print(var2use)
        # Set a single axis to use.
        fig, ax = plt.subplots()
        for dataset in datasets:
            # Select the DataFrame
            df = dfs[dataset][var2use]
            # Get sample size
            N_ = df.shape[0]
            # Do a dist plot
            label = '{} (N={})'.format(dataset, N_)
            sns.distplot(df, ax=ax, label=label)
            # Make sure the values are correctly scaled
            ax.autoscale()
            # Plot up the perturbations too
            for perturb in perturb2use:
                perturb
        # Beautify
        title_str = "PDF of ancillary input for '{}'"
        fig.suptitle(title_str.format(var2use))
        plt.legend()
        # Save to PDF and close plot
        AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
        plt.close()
    # -Save entire pdf
    AC.plot2pdfmulti(pdff, savetitle, close=True, dpi=dpi)


def explore_extracted_Arctic_Antarctic_obs(dsA=None,
                                           res='0.125x0.125',
                                           dpi=320):
    """
    Analyse the input data for the Arctic and Antarctic
    """
    import matplotlib
#    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.style.use('ggplot')
    import seaborn as sns
    sns.set()
    # - Local variables
    # Get input variables
    if isinstance(dsA, type(None)):
        filename = 'Oi_prj_predicted_iodide_{}.nc'.format(res)
#        folder = '/shared/earth_home/ts551/labbook/Python_progs/'
        folder = '/shared/earth_home/ts551/data/iodide/'
        filename = 'Oi_prj_feature_variables_{}.nc'.format(res)
        dsA = xr.open_dataset(folder + filename)
    # Variables to consider
    vars2analyse = list(dsA.data_vars)
    # Add LWI to array - NOTE: 1 = water in Nature run LWI files !
    # ( The above comment is not correct! why is this written here? )
    folderLWI = utils.get_file_locations(
        'AC_tools')+'/data/LM/TEMP_NASA_Nature_run/'
    filenameLWI = 'ctm.nc'
    LWI = xr.open_dataset(folderLWI+filenameLWI)
    # Updates dates (to be Jan=>Dec)
    new_dates = [datetime.datetime(1970, i, 1) for i in LWI['time.month']]
    LWI.time.values = new_dates
    # Sort by new dates
    LWI = LWI.loc[{'time': sorted(LWI.coords['time'].values)}]
    dsA['IS_WATER'] = dsA['WOA_TEMP'].copy()
    dsA['IS_WATER'].values = (LWI['LWI'] == 0)
    # Add is land
    dsA['IS_LAND'] = dsA['IS_WATER'].copy()
    dsA['IS_LAND'].values = (LWI['LWI'] == 1)
    # Get surface area
    s_area = AC.calc_surface_area_in_grid(res=res)  # m2 land map
    dsA['AREA'] = dsA['WOA_TEMP'].mean(dim='time')
    dsA['AREA'].values = s_area.T
    # - Select data of interest by variable for locations
    # Setup dicts to store the extracted values
    df65N, df65S, dfALL = {}, {}, {}
    # - setup booleans for the data
    # now loop and extract variablesl
    vars2use = [
        'WOA_Nitrate', 'WOA_Salinity', 'WOA_Phosphate', 'WOA_TEMP_K',
        'Depth_GEBCO',
    ]
    for var_ in vars2use:
        # Select the boolean for if water
        IS_WATER = dsA['IS_WATER'].values
        if IS_WATER.shape != dsA[var_].shape:
            # Special case for depth
            # Get value for >= 65
            ds_tmp = dsA.sel(lat=(dsA['lat'] >= 65))
            arr = np.ma.array(12*[ds_tmp[var_].values])
            arr = arr[ds_tmp['IS_WATER'].values]
            # Add to saved arrays
            df65N[var_] = arr
            del ds_tmp
            # Get value for >= 65
            ds_tmp = dsA.sel(lat=(dsA['lat'] <= -65))
            arr = np.ma.array(12*[ds_tmp[var_].values])
            arr = arr[ds_tmp['IS_WATER'].values]
            # Add to saved arrays
            df65S[var_] = arr
            del ds_tmp
            # Get value for all
            ds_tmp = dsA.copy()
            arr = np.ma.array(12*[ds_tmp[var_].values])
            arr = arr[ds_tmp['IS_WATER'].values]
            # Add to saved arrays
            dfALL[var_] = arr
            del ds_tmp
        else:
            # Get value for >= 65
            ds_tmp = dsA.sel(lat=(dsA['lat'] >= 65))
            arr = ds_tmp[var_].values[ds_tmp['IS_WATER'].values]
            # Add to saved arrays
            df65N[var_] = arr
            del ds_tmp
            # Get value for >= 65
            ds_tmp = dsA.sel(lat=(dsA['lat'] <= -65))
            arr = ds_tmp[var_].values[ds_tmp['IS_WATER'].values]
            # Add to saved arrays
            df65S[var_] = arr
            del ds_tmp
            # Get value for >= 65
            ds_tmp = dsA.copy()
            arr = ds_tmp[var_].values[ds_tmp['IS_WATER'].values]
            # Add to saved arrays
            dfALL[var_] = arr
            del ds_tmp

    # Setup a dictionary of regions to plot from
    dfs = {
        '>=65N': pd.DataFrame(df65N), '>=65S': pd.DataFrame(df65S),
        'Global': pd.DataFrame(dfALL),
    }

    # - Loop regions and plot PDFs of variables of interest
#    vars2use = dfs[ dfs.keys()[0] ].columns
    # Set PDF
    savetitle = 'Oi_prj_explore_Arctic_Antarctic_ancillaries_space'
    pdff = AC.plot2pdfmulti(title=savetitle, open=True, dpi=dpi)
    # Loop by dataset (region) and plots
    datasets = sorted(dfs.keys())
    for dataset in datasets:
        # Select the DataFrame
        df = dfs[dataset][vars2use]
        # Get sample size
        N_ = df.shape[0]
        # Do a pair plot
        g = sns.pairplot(df)
        # Add a title
        plt.suptitle("Pairplot for '{}' (N={})".format(dataset, N_))
        # Adjust plots
        g.fig.subplots_adjust(top=0.925, left=0.085)
        # Save to PDF and close plot
        AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
        plt.close()

    # - Plot up the PDF distribution of each of the variables.
    for var2use in vars2use:
        print(var2use)
        # Set a single axis to use.
        fig, ax = plt.subplots()
        for dataset in datasets:
            # Select the DataFrame
            df = dfs[dataset][var2use]
            # Get sample size
            N_ = df.shape[0]
            # Do a dist plot
            label = '{} (N={})'.format(dataset, N_)
            sns.distplot(df, ax=ax, label=label)
            # Make sure the values are correctly scaled
            ax.autoscale()
        # Beautify
        title_str = "PDF of ancillary input for '{}'"
        fig.suptitle(title_str.format(var2use))
        plt.legend()
        # Save to PDF and close plot
        AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
        plt.close()

    # - Plot up the number of oceanic data points by lat for each lat
    # Plot up number of samples for South pole
    ds = dsA.sel(lat=(dsA['lat'] <= -65))
    var_ = 'WOA_Salinity'
    N = {}
    for lat in ds['lat'].values:
        ds_tmp = ds.sel(lat=lat)
        N[lat] = ds_tmp[var_].values[ds_tmp['IS_WATER'].values].shape[-1]
    N = pd.Series(N)
    N.plot()
    plt.ylabel('number of gridboxes in predictor array')
    plt.xlabel('Latitude $^{\circ}$N')
    plt.title('Number of gridboxes for Antarctic (<= -65N)')
    # Save to PDF and close plot
    AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
    plt.close()
    # Plot up number of samples for North pole
    ds = dsA.sel(lat=(dsA['lat'] >= 65))
    var_ = 'WOA_Salinity'
    N = {}
    for lat in ds['lat'].values:
        ds_tmp = ds.sel(lat=lat)
        N[lat] = ds_tmp[var_].values[ds_tmp['IS_WATER'].values].shape[-1]
    N = pd.Series(N)
    N.plot()
    plt.ylabel('number of gridboxes in predictor array')
    plt.xlabel('Latitude $^{\circ}$N')
    plt.title('Number of gridboxes')
    plt.title('Number of gridboxes for Arctic (>= 65N)')
    # Save to PDF and close plot
    AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
    plt.close()
    # Save entire pdf
    AC.plot2pdfmulti(pdff, savetitle, close=True, dpi=dpi)


def explore_obse_data_in_Arctic_param_space(RFR_dict=None,
                                            plt_up_locs4var_conds=False,
                                            testset='Test set (strat. 20%)',
                                            dpi=320):
    """
    Analysis the input observational data for the Arctic and Antarctic
    """
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.style.use('ggplot')
    import seaborn as sns
    sns.set()

    # - Local variables
    df = RFR_dict['df']
    # Set splits in data to look at
    dfs = {}
    # All data
    dfs['All data'] = df.copy()
    # Get all the data above 65 N
    dfs['>=65N'] = df.loc[df['Latitude'] >= 65, :]
    # Get all the data above 65 N and in the testset
    bool_ = dfs['>=65N'][testset] == False
    dfs['>=65N (training)'] = dfs['>=65N'].loc[bool_, :]
    # Get all the data below 65 S
    dfs['<=65S'] = df.loc[df['Latitude'] <= -65, :]
    # Get all the data above 65 N and in the testset
    bool_ = dfs['<=65S'][testset] == False
    dfs['<=65S (training)'] = dfs['<=65S'].loc[bool_, :]
    # - variables to explore?
    vars2use = [
        'WOA_Nitrate', 'WOA_Salinity', 'WOA_Phosphate', 'WOA_TEMP_K',
        'Depth_GEBCO',
    ]

    # - Loop regions and plot pairplots of variables of interest
    # Set PDF
    savetitle = 'Oi_prj_explore_Arctic_Antarctic_obs_space'
    pdff = AC.plot2pdfmulti(title=savetitle, open=True, dpi=dpi)
    # Loop by dataset (region) and plots
    datasets = sorted(dfs.keys())
    for dataset in datasets:
        # Select the DataFrame
        df = dfs[dataset]
        # Get sample size
        N_ = df.shape[0]
        # Do a pair plot
        g = sns.pairplot(df[vars2use])
        # Add a title
        plt.suptitle("Pairplot for '{}' (N={})".format(dataset, N_))
        # Adjust plots
        g.fig.subplots_adjust(top=0.925, left=0.085)
        # Save to PDF and close plot
        AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
        plt.close()

    # - Loop regions and plot PDFs of variables of interest
    # Loop by dataset (region) and plots
    import seaborn as sns
    sns.reset_orig()
    datasets = sorted(dfs.keys())
    for dataset in datasets:
        fig, ax = plt.subplots()
        # Select the DataFrame
        dfA = dfs[dataset]
        # Set title
        title = "Locations for '{}'".format(dataset)
        p_size = 50
        alpha = 1
        # Plot up Non coatal locs
        df = dfA.loc[dfA['Coastal'] == False, :]
        color = 'blue'
        label = 'Non-coastal (N={})'.format(int(df.shape[0]))
        m = AC.plot_lons_lats_spatial_on_map(title=title, f_size=15,
                                             lons=df['Longitude'].values,
                                             lats=df['Latitude'].values,
                                             label=label, fig=fig, ax=ax,
                                             color=color,
                                             return_axis=True)
        # Plot up coatal locs
        df = dfA.loc[dfA['Coastal'] == True, :]
        color = 'green'
        label = 'Coastal (N={})'.format(int(df.shape[0]))
        lons = df['Longitude'].values
        lats = df['Latitude'].values
        m.scatter(lons, lats, edgecolors=color, c=color, marker='o',
                  s=p_size, alpha=alpha, label=label)
        plt.legend()
        # Save to PDF and close plot
        AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
        plt.close()

    # - Loop regions and plot PDFs of variables of interest
    import matplotlib.pyplot as plt
    matplotlib.style.use('ggplot')
    import seaborn as sns
    sns.set()
    df = RFR_dict['df']
    dfs = {}
    # All data
    dfs['All data'] = df.copy()
    # Get all the data above 65 N
    dfs['>=65N'] = df.loc[df['Latitude'] >= 65, :]
    # Get all the data below 65 S
    dfs['<=65S'] = df.loc[df['Latitude'] <= -65, :]
    # - variables to explore?
    vars2use = [
        'WOA_Nitrate', 'WOA_Salinity', 'WOA_Phosphate', 'WOA_TEMP_K',
        'Depth_GEBCO',
    ]
    # Plot up the PDF distribution of each of the variables.
    datasets = sorted(dfs.keys())
    for var2use in vars2use:
        print(var2use)
        # Set a single axis to use.
        fig, ax = plt.subplots()
        for dataset in datasets:
            # Select the DataFrame
            df = dfs[dataset][var2use]
            # Get sample size
            N_ = df.shape[0]
            # Do a dist plot
            label = '{} (N={})'.format(dataset, N_)
            sns.distplot(df, ax=ax, label=label)
            # Make sure the values are correctly scaled
            ax.autoscale()
        # Beautify
        title_str = "PDF of ancillary input for '{}'"
        fig.suptitle(title_str.format(var2use))
        plt.legend()
        # Save to PDF and close plot
        AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
        plt.close()

    # - Loop regions and plot PDFs of variables of interest
    if plt_up_locs4var_conds:
        df = RFR_dict['df']
        dfs = {}
        # Nitrate greater of equal to
        var_ = 'Nitrate >=15'
        dfs[var_] = df.loc[df['WOA_Nitrate'] >= 15, :]
        # Nitrate greater of equal to
        var_ = 'Nitrate <=15'
        dfs[var_] = df.loc[df['WOA_Nitrate'] <= 15, :]
        # Nitrate greater of equal to
        var_ = 'Nitrate >=10'
        dfs[var_] = df.loc[df['WOA_Nitrate'] >= 10, :]
        # Nitrate greater of equal to
        var_ = 'Nitrate <=10'
        dfs[var_] = df.loc[df['WOA_Nitrate'] <= 10, :]
        # Nitrate greater of equal to
        var_ = 'Nitrate <=9'
        dfs[var_] = df.loc[df['WOA_Nitrate'] <= 9, :]
        # Nitrate greater of equal to
        var_ = 'Nitrate <=8'
        dfs[var_] = df.loc[df['WOA_Nitrate'] <= 8, :]
        # Nitrate greater of equal to
        var_ = 'Nitrate <=7'
        dfs[var_] = df.loc[df['WOA_Nitrate'] <= 7, :]
        # Nitrate greater of equal to
        var_ = 'Nitrate <=6'
        dfs[var_] = df.loc[df['WOA_Nitrate'] <= 6, :]
        # Nitrate greater of equal to
        var_ = 'Nitrate <=5'
        dfs[var_] = df.loc[df['WOA_Nitrate'] <= 5, :]
        # Nitrate greater of equal to
        var_ = 'Nitrate <=4'
        dfs[var_] = df.loc[df['WOA_Nitrate'] <= 4, :]
        # Nitrate greater of equal to
        var_ = 'Nitrate <=3'
        dfs[var_] = df.loc[df['WOA_Nitrate'] <= 3, :]
        # Nitrate greater of equal to
        var_ = 'Nitrate <=2'
        dfs[var_] = df.loc[df['WOA_Nitrate'] <= 2, :]
        # Nitrate greater of equal to
        var_ = 'Nitrate <=1'
        dfs[var_] = df.loc[df['WOA_Nitrate'] <= 1, :]
        # Loop by dataset (nitrate values) and plots
        import seaborn as sns
        sns.reset_orig()
        datasets = sorted(dfs.keys())
        for dataset in datasets:
            fig, ax = plt.subplots()
            # Select the DataFrame
            dfA = dfs[dataset]
            # Set title
            title = "Locations for '{}'".format(dataset)
            p_size = 50
            alpha = 1
            # Plot up Non coatal locs
            df = dfA.loc[dfA['Coastal'] == False, :]
            color = 'blue'
            label = 'Non-coastal (N={})'.format(int(df.shape[0]))
            m = AC.plot_lons_lats_spatial_on_map(title=title, f_size=15,
                                                 lons=df['Longitude'].values,
                                                 lats=df['Latitude'].values,
                                                 label=label, fig=fig, ax=ax,
                                                 color=color,
                                                 return_axis=True)
            # Plot up coatal locs
            df = dfA.loc[dfA['Coastal'] == True, :]
            color = 'green'
            label = 'Coastal (N={})'.format(int(df.shape[0]))
            lons = df['Longitude'].values
            lats = df['Latitude'].values
            m.scatter(lons, lats, edgecolors=color, c=color, marker='o',
                      s=p_size, alpha=alpha, label=label)
            plt.legend()
            # Save to PDF and close plot
            AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
            plt.close()
    # - Save entire pdf
    AC.plot2pdfmulti(pdff, savetitle, close=True, dpi=dpi)


def Driver2analyse_new_data_vs_existing_data():
    """
    Driver to plot up all options for old vs. new analysis plots
    """
    regions = 'all', 'coastal', 'noncoastal'
    for limit_to_400nM in True, False:
        for region in regions:
            analyse_new_data_vs_existing_data(region=region,
                                              limit_to_400nM=limit_to_400nM)


def analyse_new_data_vs_existing_data(limit_to_400nM=True, region='all'):
    """
    build a set of analysis plots exploring the difference between new and
    exisiting datasets
    """
    # - Get obs. data
    # Get data (inc. additions) and meta data
    df_meta = obs.get_iodide_obs_metadata()
    pro_df = obs.get_processed_df_obs_mod()
    # - Setup plotting
    # Misc. shared variables
    axlabel = '[I$^{-}_{aq}$] (nM)'
    # Setup PDf
    savetitle = 'Oi_prj_new_vs_existing_datasets'
    if limit_to_400nM:
        # Exclude v. high values (N=7 -  in final dataset)
        pro_df = pro_df.loc[pro_df['Iodide'] < 400.]
        savetitle += '_limited_to_400nM'
    if region == 'all':
        savetitle += '_all'
    elif region == 'coastal':
        pro_df = pro_df.loc[pro_df['Coastal'] == 1, :]
        savetitle += '_{}'.format(region)
    elif region == 'noncoastal':
        pro_df = pro_df.loc[pro_df['Coastal'] == 0, :]
        savetitle += '_{}'.format(region)
    else:
        sys.exit()
    pdff = AC.plot2pdfmulti(title=savetitle, open=True, dpi=dpi)
    # Colours to use?
    import seaborn as sns
    sns.set(color_codes=True)
    current_palette = sns.color_palette("colorblind")

    # - Plot up new data ( ~timeseries? )
    New_datasets = df_meta.loc[df_meta['In Chance2014?'] == 'N'].Data_Key
    var2plot = 'Iodide'
    for dataset in New_datasets:
        # Select new dataset
        tmp_df = pro_df.loc[pro_df['Data_Key'] == dataset]
        Cruise = tmp_df['Cruise'].values[0]
        # If dates present in DataFrame, update axis
        dates4cruise = pd.to_datetime(tmp_df['Date'].values)
        if len(set(dates4cruise)) == tmp_df.shape[0]:
            tmp_df.index = dates4cruise
            xlabel = 'Date'
        else:
            xlabel = 'Obs #'
        tmp_df[var2plot].plot()
        ax = plt.gca()
        plt.xlabel(xlabel)
        plt.ylabel(axlabel)
        title_str = "New {} data from '{}' ({})"
        plt.title(title_str.format(var2plot.lower(), Cruise, dataset))
        AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
        plt.close()

    # - Plot up new data ( PDF of iodide )
    var2plot = 'Iodide'
    for dataset in New_datasets:
        # Select new dataset
        tmp_df = pro_df.loc[pro_df['Data_Key'] == dataset]
        Cruise = tmp_df['Cruise'].values[0]

        # - Plot up PDF plots for the dataset
        # Plot whole dataset
        obs_arr = pro_df[var2plot].values
        ax = sns.distplot(obs_arr, axlabel=axlabel,
                          color='k', label='Whole dataset')
        # Plot just new data
        ax = sns.distplot(tmp_df[var2plot], axlabel=axlabel, label=Cruise,
                          color='red', ax=ax)
        # Force y axis extend to be correct
        ax.autoscale()
        # Beautify
        title = "PDF of '{}' {} data ({}) at obs. locations"
        plt.title(title.format(dataset, var2plot, axlabel))
        plt.legend()
        # Save to PDF and close plot
        AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
        plt.close()

    # - Plot up new data ( PDF of salinity )
    var2plot = u'WOA_Salinity'
    for dataset in New_datasets:
        # Select new dataset
        tmp_df = pro_df.loc[pro_df['Data_Key'] == dataset]
        Cruise = tmp_df['Cruise'].values[0]

        # - Plot up PDF plots for the dataset
        # Plot whole dataset
        obs_arr = pro_df[var2plot].values
        ax = sns.distplot(obs_arr, axlabel=axlabel,
                          color='k', label='Whole dataset')
        # Plot just new data
        ax = sns.distplot(tmp_df[var2plot], axlabel=axlabel, label=Cruise,
                          color='red', ax=ax)
        # Force y axis extend to be correct
        ax.autoscale()
        # Beautify
        title = "PDF of '{}' {} data ({}) at obs. locations"
        plt.title(title.format(dataset, var2plot, axlabel))
        plt.legend()
        # Save to PDF and close plot
        AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
        plt.close()

    # - Plot up new data ( PDF of temperature )
    var2plot = 'WOA_TEMP'
    for dataset in New_datasets:
        # Select new dataset
        tmp_df = pro_df.loc[pro_df['Data_Key'] == dataset]
        Cruise = tmp_df['Cruise'].values[0]

        # - Plot up PDF plots for the dataset
        # Plot whole dataset
        obs_arr = pro_df[var2plot].values
        ax = sns.distplot(obs_arr, axlabel=axlabel,
                          color='k', label='Whole dataset')
        # Plot just new data
        ax = sns.distplot(tmp_df[var2plot], axlabel=axlabel, label=Cruise,
                          color='red', ax=ax)
        # Force y axis extend to be correct
        ax.autoscale()
        # Beautify
        title = "PDF of '{}' {} data ({}) at obs. locations"
        plt.title(title.format(dataset, var2plot, axlabel))
        plt.legend()
        # Save to PDF and close plot
        AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
        plt.close()

    # - Plot up new data ( PDF of depth )
    var2plot = u'Depth_GEBCO'
    for dataset in New_datasets:
        # Select new dataset
        tmp_df = pro_df.loc[pro_df['Data_Key'] == dataset]
        Cruise = tmp_df['Cruise'].values[0]

        # - Plot up PDF plots for the dataset
        # Plot whole dataset
        obs_arr = pro_df[var2plot].values
        ax = sns.distplot(obs_arr, axlabel=axlabel,
                          color='k', label='Whole dataset')
        # Plot just new data
        ax = sns.distplot(tmp_df[var2plot], axlabel=axlabel, label=Cruise,
                          color='red', ax=ax)
        # Force y axis extend to be correct
        ax.autoscale()
        # Beautify
        title = "PDF of '{}' {} data ({}) at obs. locations"
        plt.title(title.format(dataset, var2plot, axlabel))
        plt.legend()
        # Save to PDF and close plot
        AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
        plt.close()

    # -- Save entire pdf
    AC.plot2pdfmulti(pdff, savetitle, close=True, dpi=dpi)


def get_diagnostic_plots_analysis4observations(inc_all_extract_vars=False,
                                               include_hexbin_plots=False,
                                               model_name='TEMP+DEPTH+SAL',
                                               show_plot=False, dpi=320):
    """
    Produce a PDF of comparisons of observations in dataset inventory
    """
    # - Setup plotting
    # Misc. shared variables
    axlabel = '[I$^{-}_{aq}$] (nM)'
    # Setup PDf
    savetitle = 'Oi_prj_obs_plots'
    if inc_all_extract_vars:
        savetitle += '_all_extract_vars'
        include_hexbin_plots = True
    pdff = AC.plot2pdfmulti(title=savetitle, open=True, dpi=dpi)
    # Colours to use?
    import seaborn as sns
    # - Get obs. data
    # Get data (inc. additions) and meta data
    df_meta = obs.get_iodide_obs_metadata()
    pro_df = obs.get_processed_df_obs_mod()
    LOCAL_model_name = 'RFR({})'.format(model_name)
    pro_df[LOCAL_model_name] = get_model_predictions4obs_point(pro_df,
                                                         model_name=model_name)

    # Exclude v. high values (N=4 -  in intial dataset)
    # Exclude v. high values (N=7 -  in final dataset)
    pro_df = pro_df.loc[pro_df['Iodide'] < 400.]

    # Add coastal flag to data
    coastal_flag = 'coastal_flagged'
    pro_df = get_coastal_flag(df=pro_df, coastal_flag=coastal_flag)
    non_coastal_df = pro_df.loc[pro_df['coastal_flagged'] == 0]
    dfs = {'Open-Ocean': non_coastal_df, 'All': pro_df}
    # TODO ... add test dataset in here
    # Get the point data for params...
    point_ars_dict = {}
    for key_ in dfs.keys():
        point_ars_dict[key_] = {
            'Obs.': dfs[key_]['Iodide'].values,
            'MacDonald et al (2014)': dfs[key_]['MacDonald2014_iodide'].values,
            'Chance et al (2014)':  dfs[key_][u'Chance2014_STTxx2_I'].values,
            'Chance et al (2014) - Mutivariate':  dfs[key_][
                u'Chance2014_Multivariate'
            ].values,
            LOCAL_model_name: dfs[key_][LOCAL_model_name],
        }

    point_ars_dict = point_ars_dict['Open-Ocean']
    parm_name_dict = {
        'MacDonald et al (2014)': 'MacDonald2014_iodide',
        'Chance et al (2014)': u'Chance2014_STTxx2_I',
        'Chance et al (2014) - Mutivariate': u'Chance2014_Multivariate',
        LOCAL_model_name: LOCAL_model_name,
    }
    point_data_names = sorted(point_ars_dict.keys())
    point_data_names.pop(point_data_names.index('Obs.'))
    param_names = point_data_names

    # Setup color dictionary
    current_palette = sns.color_palette("colorblind")
    colour_dict = dict(zip(param_names, current_palette[:len(param_names)]))
    colour_dict['Obs.'] = 'K'

    #  --- Plot up locations of old and new data
    import seaborn as sns
    sns.reset_orig()
    plot_up_data_locations_OLD_and_new(save_plot=False, show_plot=False)
    # Save to PDF and close plot
    AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
    plt.close()

    #  --- Plot up all params against coastal data
    import seaborn as sns
    sns.set(color_codes=True)
    sns.set_context("paper")
    xlabel = 'Obs.'
    # just non-coastal
    for param_name in sorted(parm_name_dict.keys()):
        Y = non_coastal_df[parm_name_dict[param_name]].values
        X = non_coastal_df['Iodide'].values
        title = 'Regression plot of Open-ocean [I$^{-}_{aq}$] (nM) \n'
        title = title + '{} vs {} parameterisation'.format(xlabel, param_name)
        ax = sns.regplot(x=X, y=Y)
#        get_hexbin_plot(x=X, y=Y, xlabel=None, ylabel=point_name, log=False,
#            title=None, add_ODR_trendline2plot=True)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(param_name)
        # Adjust X and Y range
        max_val = max(max(X), max(Y))
        smidgen = max_val * 0.05
        plt.xlim(0-smidgen, max_val+smidgen)
        plt.ylim(0-smidgen, max_val+smidgen)
        # Add 1:1
        one2one = np.arange(0, max_val*2)
        plt.plot(one2one, one2one, color='k', linestyle='--', alpha=0.75,
                 label='1:1')
        plt.legend()
        if show_plot:
            plt.show()
        AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
        plt.close()

    #  --- Plot up all params against all data
    import seaborn as sns
    sns.set(color_codes=True)
    sns.set_context("paper")
    xlabel = 'Obs.'
    X = point_ars_dict[xlabel]
    for param_name in point_data_names:
        Y = point_ars_dict[param_name]
        title = 'Regression plot of all [I$^{-}_{aq}$] (nM) \n'
        title = title + '{} vs {} parameterisation'.format(xlabel, param_name)
        ax = sns.regplot(x=X, y=Y)
#        get_hexbin_plot(x=X, y=Y, xlabel=None, ylabel=point_name, log=False,
#            title=None, add_ODR_trendline2plot=True)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(param_name)
        # Adjust X and Y range
        max_val = max(max(X), max(Y))
        smidgen = max_val * 0.05
        plt.xlim(0-smidgen, max_val+smidgen)
        plt.ylim(0-smidgen, max_val+smidgen)
        # Add 1:1
        one2one = np.arange(0, max_val*2)
        plt.plot(one2one, one2one, color='k', linestyle='--', alpha=0.75,
                 label='1:1')
        plt.legend()
        if show_plot:
            plt.show()
        AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
        plt.close()

    # ---- Plot up new data
    New_datasets = df_meta.loc[df_meta['In Chance2014?'] == 'N'].Data_Key
    var2plot = 'Iodide'
    for dataset in New_datasets:
        tmp_df = pro_df.loc[pro_df['Data_Key'] == dataset]
        Cruise = tmp_df['Cruise'].values[0]
        # If dates present in DataFrame, update axis
        dates4cruise = pd.to_datetime(tmp_df['Date'].values)
        if len(set(dates4cruise)) == tmp_df.shape[0]:
            tmp_df.index = dates4cruise
            xlabel = 'Date'
        else:
            xlabel = 'Obs #'
        tmp_df[var2plot].plot()
        ax = plt.gca()
#        ax.axhline(30, color='red', label='Chance et al 2014 coastal divide')
        plt.xlabel(xlabel)
        plt.ylabel(axlabel)
        title_str = "New {} data from '{}' ({})"
        plt.title(title_str.format(var2plot.lower(), Cruise, dataset))
#        plt.legend()
        AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
        plt.close()
    # Plot up Salinity
#     var2plot = 'WOA_Salinity'
#     for dataset in New_datasets:
#         tmp_df = pro_df.loc[ pro_df['Data_Key'] == dataset ]
#         tmp_df[var2plot].plot()
#         ax= plt.gca()
#         ax.axhline(30, color='red', label='Chance et al 2014 coastal divide')
#         plt.xlabel( 'Obs #')
#         plt.ylabel( 'PSU' )
#         plt.title( '{} during cruise from {}'.format( var2plot, dataset ) )
#         plt.legend()
#         AC.plot2pdfmulti( pdff, savetitle, dpi=dpi )
#         plt.close()

    # ---- Plot up key comparisons for coastal an non-coastal data
    for key_ in sorted(dfs.keys()):
        # --- Ln(Iodide) vs. T
        ylabel = 'ln(Iodide)'
        Y = dfs[key_][ylabel].values
        xlabel = 'WOA_TEMP'
        X = dfs[key_][xlabel].values
        # Plot up
        ax = sns.regplot(x=X, y=Y)
        # Beautify
        title = '{} vs {} ({} data)'.format(ylabel, xlabel, key_)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if show_plot:
            plt.show()
        AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
        plt.close()

        # --- Ln(Iodide) vs. 1/T
        ylabel = 'ln(Iodide)'
        Y = dfs[key_][ylabel].values
        xlabel = 'WOA_TEMP_K'
        X = 1 / dfs[key_][xlabel].values
        # Plot up
        ax = sns.regplot(x=X, y=Y)
        # Beautify
        title = '{} vs {} ({} data)'.format(ylabel, '1/'+xlabel, key_)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if show_plot:
            plt.show()
        AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
        plt.close()

        # --- Ln(Iodide) vs. 1/T
        ylabel = 'ln(Iodide)'
        Y = dfs[key_][ylabel].values
        xlabel = 'WOA_Salinity'
        X = dfs[key_][xlabel].values
        # Plot up
        ax = sns.regplot(x=X, y=Y)
        # Beautify
        title = '{} vs {} ({} data)'.format(ylabel, xlabel, key_)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if show_plot:
            plt.show()
        AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
        plt.close()

        # ---
    if inc_all_extract_vars:
        for key_ in sorted(dfs.keys()):
            # List extract vraiables
            extracted_vars = [
                u'WOA_TEMP', u'WOA_Nitrate', u'WOA_Salinity', u'WOA_Dissolved_O2', u'WOA_Phosphate', u'WOA_Silicate', u'Depth_GEBCO', u'SeaWIFs_ChlrA', u'WOA_MLDpt', u'WOA_MLDpt_max', u'WOA_MLDpt_sum', u'WOA_MLDpd', u'WOA_MLDpd_max', u'WOA_MLDpd_sum', u'WOA_MLDvd', u'WOA_MLDvd_max', u'WOA_MLDvd_sum', u'DOC', u'DOCaccum', u'Prod', u'SWrad'
            ]
            # Loop extraced variables and plot
            for var_ in extracted_vars:
                ylabel = var_
                xlabel = 'Iodide'
                tmp_df = dfs[key_][[xlabel, ylabel]]
                # Kludge to remove '--' from MLD columns
                for col in tmp_df.columns:
                    bool_ = [i == '--' for i in tmp_df[col].values]
                    tmp_df.loc[bool_, :] = np.NaN
                    if tmp_df[col].dtype == 'O':
                        tmp_df[col] = pd.to_numeric(tmp_df[col].values,
                                                    errors='coerce')

                print(var_, tmp_df.min(), tmp_df.max())
#                X = dfs[key_][xlabel].values
                # Plot up                ax = sns.regplot(x=xlabel, y=ylabel, data=tmp_df )
                # Beautify
                title = '{} vs {} ({} data)'.format(ylabel, xlabel, key_)
                plt.title(title)
                plt.xlabel(xlabel)
                plt.ylabel(ylabel)
                AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
                if show_plot:
                    plt.show()
                plt.close()

    # --- Plot up Just observations and predicted values from models as PDF
    import seaborn as sns
    sns.set(color_codes=True)
    sns.set_context("paper")
    # Plot 1st model...
    point_name = 'Obs.'
    arr = point_ars_dict[point_name]
    ax = sns.distplot(arr, axlabel=axlabel, label=point_name,
                      color=colour_dict[point_name])
    # Add MacDonald, Chance...
    for point_name in point_data_names:
        arr = point_ars_dict[point_name]
        ax = sns.distplot(arr, axlabel=axlabel, label=point_name,
                          color=colour_dict[point_name])
    # Force y axis extend to be correct
    ax.autoscale()
    # Beautify
    plt.title('PDF of predicted iodide ({}) at obs. points'.format(axlabel))
    plt.legend()
    # Save to PDF and close plot
    AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
    plt.close()

    # --- Plot up Just observations and predicted values from models as CDF
    import seaborn as sns
    sns.set(color_codes=True)
    sns.set_context("paper")
    # Plot 1st model...
    point_name = 'Obs.'
    arr = point_ars_dict[point_name]
    ax = sns.distplot(arr, axlabel=axlabel, label=point_name,
                      color=colour_dict[point_name],
                      hist_kws=dict(cumulative=True),
                      kde_kws=dict(cumulative=True))
    # Add MacDonald, Chance...
    for point_name in point_data_names:
        arr = point_ars_dict[point_name]
        ax = sns.distplot(arr, axlabel=axlabel, label=point_name,
                          color=colour_dict[point_name],
                          hist_kws=dict(cumulative=True),
                          kde_kws=dict(cumulative=True))
    # Force y axis extend to be correct
    ax.autoscale()
    # Beautify
    plt.title('CDF of predicted iodide ({}) at obs. points'.format(axlabel))
    plt.legend()
    # Save to PDF and close plot
    AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
    plt.close()

    # --- Plot up parameterisations as regression
#     import seaborn as sns; sns.set(color_codes=True)
#     sns.set_context("paper")
#     xlabel = 'Obs.'
#     X = point_ars_dict[xlabel]
#     for point_name in point_data_names:
#         title = 'Regression plot of [I$^{-}_{aq}$] (nM) '
#         title = title + '{} vs {} parameterisation'.format(xlabel, point_name )
#         Y = point_ars_dict[point_name]
#         ax = sns.regplot(x=X, y=Y )
# #        get_hexbin_plot(x=X, y=Y, xlabel=None, ylabel=point_name, log=False,
# #            title=None, add_ODR_trendline2plot=True)
#         plt.title(title)
#         plt.xlabel(xlabel)
#         plt.ylabel(point_name)
#         # Save to PDF and close plot
#         AC.plot2pdfmulti( pdff, savetitle, dpi=dpi )
#         plt.close()

    # --- Plot up parameterisations as hexbin plot
    if include_hexbin_plots:
        xlabel = 'Obs.'
        X = point_ars_dict[xlabel]
        for point_name in point_data_names:
            title = 'Hexbin of [I$^{-}_{aq}$] (nM) \n'
            title = title + '{} vs {} parameterisation'.format(xlabel,
                                                               point_name)
            Y = point_ars_dict[point_name]
            get_hexbin_plot(x=X, y=Y, xlabel=None, ylabel=point_name,
                            log=False, title=title,
                            add_ODR_trendline2plot=True)
    #        plt.show()
            # Save to PDF and close plot
            AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
            plt.close()

    # -- Save entire pdf
    AC.plot2pdfmulti(pdff, savetitle, close=True, dpi=dpi)


def plt_predicted_iodide_vs_obs_Q1_Q3(dpi=320, show_plot=False,
                                      limit_to_400nM=False, inc_iodide=False):
    """
    Plot predicted iodide on a latitudinal basis

    NOTES
     - the is the just obs. location equivilent of the plot produced to show
        predict values for all global locations
        (Oi_prj_global_predicted_vals_vs_lat)
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(color_codes=True)
    sns.set_context("paper")

    # Get data
    folder = utils.get_file_locations('data_root')
    f = 'Iodine_obs_WOA.csv'
    df = pd.read_csv(folder+f, encoding='utf-8')

    # Local variables
    # Sub select variables of interest.
    params2plot = [
        'Chance2014_STTxx2_I',  'MacDonald2014_iodide',
    ]
    # Set names to overwrite variables with
    rename_titles = {u'Chance2014_STTxx2_I': 'Chance et al. (2014)',
                     u'MacDonald2014_iodide': 'MacDonald et al. (2014)',
                     'RFR(Ensemble)': 'RFR(Ensemble)',
                     'Iodide': 'Obs.',
                     #                     u'Chance2014_Multivariate': 'Chance et al. (2014) (Multi)',
                     }
    # Filename to save values
    filename = 'Oi_prj_global_predicted_vals_vs_lat_only_obs_locs'
    # Include iodide observations too?
    if inc_iodide:
        params2plot += ['Iodide']
        filename += '_inc_iodide'
    CB_color_cycle = AC.get_CB_color_cycle()
    color_d = dict(zip(params2plot, CB_color_cycle))
    #
    if limit_to_400nM:
        df = df.loc[df['Iodide'] < 400, :]
        filename += '_limited_400nM'
    # - Process data
    # Add binned mean
#    bins  = np.arange(-70, 70, 10 )
    bins = np.arange(-80, 90, 10)
#    groups = df.groupby( np.digitize(df[u'Latitude'], bins) )
    groups = df.groupby(pd.cut(df['Latitude'], bins))
    # Take means of groups
#    groups_avg = groups.mean()
    groups_des = groups.describe().unstack()

    # - setup plotting
    fig, ax = plt.subplots(dpi=dpi)
    # - Plot up
    X = groups_des['Latitude']['mean'].values  # groups_des.index
#    X =bins
    print(groups_des)
    # Plot groups
    for var_ in params2plot:
        # Get quartiles
        Q1 = groups_des[var_]['25%'].values
        Q3 = groups_des[var_]['75%'].values
        # Add median
        ax.plot(X, groups_des[var_]['50%'].values,
                color=color_d[var_], label=rename_titles[var_])
        # Add shading for Q1/Q3
        ax.fill_between(X, Q1, Q3, alpha=0.2, color=color_d[var_])

    # - Plot observations
    # Highlight coastal obs
    tmp_df = df.loc[df['Coastal'] == True, :]
    X = tmp_df['Latitude'].values
    Y = tmp_df['Iodide'].values
    plt.scatter(X, Y, color='k', marker='D', facecolor='none', s=3,
                label='Coastal obs.')
    # non-coastal obs
    tmp_df = df.loc[df['Coastal'] == False, :]
    X = tmp_df['Latitude'].values
    Y = tmp_df['Iodide'].values
    plt.scatter(X, Y, color='k', marker='D', facecolor='k', s=3,
                label='Non-coastal obs.')
    # - Beautify
    # Add legend
    plt.legend()
    # Limit plotted y axis extent
    plt.ylim(-20, 420)
    plt.ylabel('[I$^{-}_{aq}$] (nM)')
    plt.xlabel('Latitude ($^{\\rm o}$N)')
    plt.savefig(filename, dpi=dpi)
    if show_plot:
        plt.show()
    plt.close()


def plot_up_data_locations_OLD_and_new_CARTOPY(save_plot=True, show_plot=False,
                                       extension='eps', dpi=720):
    """
    Plot up old and new data on map
    """
    import seaborn as sns
    sns.reset_orig()
    # - Setup plot
#    figsize = (11, 5)
    figsize = (11*2, 5*2)
    fig = plt.figure(figsize=figsize, dpi=dpi)
#    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig, ax = None, None
    p_size = 15
    alpha = 0.5
    window = True
    axis_titles = False
    # - Get all observational data
    df, md_df = obs.get_iodide_obs()
    # Seperate into new and old data
    ChanceStr = 'In Chance2014?'
    df[ChanceStr] = None
    for ds in list(set(md_df['Data_Key'])):
        bool = df['Data_Key'] == ds
        IsChance = md_df.loc[md_df['Data_Key'] == ds,  ChanceStr].values[0]
        df.loc[bool, ChanceStr] = IsChance

    new_metadata_df = md_df.loc[
        md_df['In Chance2014?'] == 'N'
    ]
    new_Data_Keys = new_metadata_df['Data_Key'].values
    bool = df['Data_Key'].isin(new_Data_Keys)
    # old data
    df1 = df.loc[~bool]
    # new data
    df2 = df.loc[bool]
    # ---  add existing data
    # Get existing data... (Chance et al 2014 )
#    folder = utils.get_file_locations('data_root')
#    f = 'Iodine_obs_WOA.csv'
#    df1 = pd.read_csv(folderf, encoding='utf-8' )
    # Select lons and lats
    lats1 = df1['Latitude'].values
    lons1 = df1['Longitude'].values
    # Plot up and return basemap axis
    label = 'Chance et al. (2014) (N={})'.format(
        df1['Iodide'].dropna().shape[0])
    ax = plot_lons_lats_spatial_on_map_CARTOPY(lons=lons1, lats=lats1,
                                         fig=fig, ax=ax, color='blue',
                                         label=label,
                                         alpha=alpha, dpi=dpi,
#                                         window=window, axis_titles=axis_titles,
#                                         return_axis=True,
#                                         add_detailed_map=True,
                                         add_background_image=False,
                                         add_gridlines=False,
                                         s=p_size)

    # - Add in new data following Chance2014?
    # This is ~ 5 samples from the Atlantic (and some from Indian ocean?)
    # ... get this at a later date...

    # - Add in SOE-9 data
#    f = 'Iodine_climatology_ISOE9.xlsx'
#    df2 = pd.read_excel(folder'/Liselotte_data/'+f, skiprows=1 )
    # Data from SOE-9
    lats2 = df2['Latitude'].values
    lons2 = df2['Longitude'].values
    color = 'red'
    label = 'Additional data (N={})'
    label = label.format(df2['Iodide'].dropna().shape[0])
    ax.scatter(lons2, lats2, edgecolors=color, c=color, marker='o',
              s=p_size, alpha=alpha, label=label, zorder=1000)
    # - Save out / show
    leg = plt.legend(fancybox=True, loc='upper right', prop={'size': 6})
    leg.get_frame().set_alpha(0.95)
    if save_plot:
        savename = 'Oi_prj_Obs_locations.{}'.format(extension)
        plt.savefig(savename, bbox_inches='tight', dpi=dpi)
    if show_plot:
        plt.show()


def map_plot_of_locations_of_obs():
    """
    Plot up locations of observations of data to double check
    """
    import matplotlib.pyplot as plt

    # - Settings
    plot_all_as_one_plot = True
    show = True

    # - Get data
    folder = utils.get_file_locations('data_root')
    f = 'Iodine_obs_WOA.csv'
    df = pd.read_csv(folder+f, encoding='utf-8')

    # only consider non-coastal  locations
    print(df.shape)
#    df = df[ df['Coastal'] == 1.0  ] # Select coastal locations
#    df = df[ df['Coastal'] == 0.0  ]  # Select non coastal locations
    # only consider locations with salinity > 30
    df = df[df['Salinity'] > 30.0]  # Select coastal locations
    print(df.shape)
    # Get coordinate values
    all_lats = df['Latitude'].values
    all_lons = df['Longitude'].values
    # Get sub lists of unique identifiers for datasets
    datasets = list(set(df['Data_Key']))
    n_datasets = len(datasets)
    # - Setup plot
    #
    f_size = 10
    marker = 'o'
    p_size = 75
    dpi = 600
    c_list = AC.color_list(int(n_datasets*1.25))
    print(c_list, len(c_list))
    # Plot up white background
    arr = np.zeros((72, 46))
    vmin, vmax = 0, 0

    # - just plot up all sites to test
    if plot_all_as_one_plot:
        # Setup a blank basemap plot
        fig = plt.figure(figsize=(12, 6), dpi=dpi,
                         facecolor='w', edgecolor='k')
        ax1 = fig.add_subplot(111)
        plt, m = AC.map_plot(arr.T, return_m=True, cmap=plt.cm.binary,
                             f_size=f_size*2,
                             fixcb=[
                                 vmin, vmax], ax=ax1, no_cb=True,
                                 resolution='c',
                             ylabel=True, xlabel=True)
        # Scatter plot of points.
        m.scatter(all_lons, all_lats, edgecolors=c_list[1], c=c_list[1],
                  marker=marker, s=p_size, alpha=1,)
        # Save and show?
        plt.savefig('Iodide_dataset_locations.png', dpi=dpi, transparent=True)
        if show:
            plt.show()

    else:
        chunksize = 5
        chunked_list = AC.chunks(datasets, chunksize)
        counter = 0
        for n_chunk_, chunk_ in enumerate(chunked_list):
            # Setup a blank basemap plot
            fig = plt.figure(figsize=(12, 6), dpi=dpi, facecolor='w',
                             edgecolor='k')
            ax1 = fig.add_subplot(111)
            plt, m = AC.map_plot(arr.T, return_m=True, cmap=plt.cm.binary,
                                 f_size=f_size*2,
                                 fixcb=[vmin, vmax], ax=ax1,
                                 no_cb=True, resolution='c',
                                 ylabel=True, xlabel=True)

            # Loop all datasets
            for n_dataset_, dataset_ in enumerate(chunk_):
                print(n_chunk_, counter, dataset_, c_list[counter])
                #
                df_sub = df[df['Data_Key'] == dataset_]
                lats = df_sub['Latitude'].values
                lons = df_sub['Longitude'].values
                # Plot up and save.
                color = c_list[n_chunk_::chunksize][n_dataset_]
                m.scatter(lons, lats, edgecolors=color, c=color,
                          marker=marker, s=p_size, alpha=.5, label=dataset_)
                # Add one to counter
                counter += 1

            plt.legend()
            # Save chunk...
            plt.savefig('Iodide_datasets_{}.png'.format(n_chunk_), dpi=dpi,
                        transparent=True)
            if show:
                plt.show()


def plot_up_parameterisations(df=None, save2pdf=True, show=False):
    """
    Plot up parameterisations
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    # Consider both Chance and MacDonald parameterisations
    params = [i for i in df.columns if ('Mac' in i)]
    params += [i for i in df.columns if ('Chance' in i)]

    # Get details of parameterisations
#    filename='Chance_2014_Table2_PROCESSED_17_04_19.csv'
    filename = 'Chance_2014_Table2_PROCESSED.csv'
    folder = utils.get_file_locations('data_root')
    param_df = pd.read_csv(folder+filename)

    # only consider non-coastal locations?
    print(df.shape)
#    df = df[ df['Coastal'] == 1.0  ] # Select coastal locations
#    df = df[ df['Coastal'] == 0.0  ]  # Select non coastal locations
    # only consider locations with salinity > 30
    df = df[df['Salinity'] > 30.0]  # Select coastal locations
    print(df.shape)
#    df = df[ df['Iodide'] < 300 ]

    # Setup pdf
    if save2pdf:
        dpi = 320
        savetitle = 'Chance2014_params_vs_recomputed_params'
        pdff = AC.plot2pdfmulti(title=savetitle, open=True, dpi=dpi)

    # - Loop parameterisations
#    for param in params[:2]:  # Only loop two if debugging
    for param in params:

        # Get meta data for parameter
        sub_df = param_df[param_df['TMS ID'] == param]
        # Setup a new figure
        fig = plt.figure()

        # Extract Iodide and param data...
        # Take logs of data?
        iodide_var = 'Iodide'
        try:
            print(sub_df['ln(iodide)'].values[0])
            if sub_df['ln(iodide)'].values[0] == 'yes':
                iodide_var = 'ln(Iodide)'
                print('Using log values for ', param)
            else:
                print('Not using log values for ', param)
        except:
            print('FAILED to try and use log data for ', param)
        X = df[iodide_var].values
        # And parameter data?
        Y = df[param].values
        # Remove nans...
        tmp_df = pd.DataFrame(np.array([X, Y]).T, columns=['X', 'Y'])
        print(tmp_df.shape)
        tmp_df = tmp_df.dropna()
        print(tmp_df.shape)
        X = tmp_df['X'].values
        Y = tmp_df['Y'].values
        # PLOT UP as X vs. Y scatter...
        title = '{} ({})'.format(param, sub_df['Independent  variable'].values)
        ax = mk_X_Y_scatter_plot_param_vs_iodide(X=X, Y=Y, title=title,
                                                 iodide_var=iodide_var)
        # Add Chance2014's R^2 to plot...
        try:
            R2 = str(sub_df['R2'].values[0])
            c = str(sub_df['c'].values[0])
            m = str(sub_df['m'].values[0])
            eqn = 'y={}x+{}'.format(m, c)

            print(R2, c, m, eqn)

            alt_text = 'Chance et al (2014) R$^2$'+':{} ({})'.format(R2, eqn)
            ax.annotate(alt_text, xy=(0.5, 0.90), textcoords='axes fraction',
                        fontsize=10)
        except:
            print('FAILED to get Chance et al values for', param)
#        plt.text( 0.75, 0.8, alt_text, ha='center', va='center')

        # Show/save?
        if save2pdf:
            # Save out figure
            AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
        if show:
            plt.show()
        del fig
    #  save entire pdf
    if save2pdf:
        AC.plot2pdfmulti(pdff, savetitle, close=True, dpi=dpi)
    plt.close("all")


def mk_X_Y_scatter_plot_param_vs_iodide(X=None, Y=None, iodide_var=None,
                                        title=None):
    """
    Plots up a X vs. Y plot for a parameterisation of iodine (Y) against obs iodide (X)
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    # Plot up
    plt.scatter(X, Y, marker='+', alpha=0.5)
    plt.title(title)
    plt.ylabel('Param. [Iodide], nM')
    plt.xlabel('Obs. [{}], nM'.format(iodide_var))
    # Add a trendline
    ax = plt.gca()
    AC.Trendline(ax, X=X, Y=Y, color='green')
    # Adjust x and y axis limits
    round_max_X = AC.myround(max(X), 50, round_up=True)
    round_max_Y = AC.myround(max(Y), 50, round_up=True)
    if iodide_var == 'ln(Iodide)':
        round_max_X = AC.myround(max(X), 5, round_up=True)
        round_max_Y = AC.myround(max(Y), 5, round_up=True)
    plt.xlim(-(round_max_X/40), round_max_X)
    plt.ylim(-(round_max_Y/40), round_max_Y)
    # Add an N value to plot
    alt_text = '(N={})'.format(len(X))
    ax.annotate(alt_text, xy=(0.8, 0.10),
                textcoords='axes fraction', fontsize=10)
    return ax


def compare_obs_ancillaries_with_extracted_values_WINDOW(dpi=320, df=None):
    """
    Plot up a window plot of the observed vs. climatological ancillaries
    """
    import seaborn as sns
    sns.set(color_codes=True)
    current_palette = sns.color_palette("colorblind")
    sns.set_style("darkgrid")
    sns.set_context("paper", font_scale=0.75)
    # Get the observational data
    if isinstance(df, type(None)):
        df = obs.get_processed_df_obs_mod()  # NOTE this df contains values >400nM
    # -  Map observational variables to their shared extracted variables
    all_vars = df.columns.tolist()
    # Dictionary
    obs_var_dict = {
        # Temperature
        'WOA_TEMP':  'Temperature',
        # Chlorophyll-a
        'SeaWIFs_ChlrA': 'Chl-a',
        # Nitrate
        'WOA_Nitrate': 'Nitrate',
        # Salinity
        'WOA_Salinity': 'Salinity'
        # There is also 'Nitrite' and 'Ammonium'
    }
    # Units dict?
    units_dict = {
        'SeaWIFs_ChlrA': "mg m$^{-3}$",  # Chance et al uses micro g/L
        'WOA_Salinity': 'PSU',  # https://en.wikipedia.org/wiki/Salinity
        'WOA_Nitrate':  "$\mu$M",
        'WOA_TEMP': '$^{o}$C',
    }
    # Colors to use
    CB_color_cycle = AC.get_CB_color_cycle()
    # Set the order the dict keys are accessed
    vars_sorted = list(sorted(obs_var_dict.keys()))[::-1]

    #  setup plot
    fig = plt.figure(dpi=dpi, figsize=(5, 7.35))
    # - 1st plot Salinity ( all and >30 PSU )
    # - All above
    var2plot = 'WOA_Salinity'
    plot_n = 1
    color = CB_color_cycle[0]
    # Make a new axis
    ax = fig.add_subplot(3, 2, plot_n, aspect='equal')
    # Get the data
    df_tmp = df[[obs_var_dict[var2plot], var2plot]].dropna()
    N_ = int(df_tmp[[var2plot]].shape[0])
    MSE_ = np.mean((df_tmp[obs_var_dict[var2plot]] - df_tmp[var2plot])**2)
    RMSE_ = np.sqrt(MSE_)
    print(N_, MSE_, RMSE_)
    X = df_tmp[obs_var_dict[var2plot]].values
    Y = df_tmp[var2plot].values
    # Plot up the data as a scatter
    ax.scatter(X, Y, edgecolors=color, facecolors='none', s=5)
    # Label Y axis
    if plot_n in np.arange(1, 6)[::2]:
        ax.set_ylabel('Extracted')
    # Title the plots
    title = 'Salinity (all, {})'.format(units_dict[var2plot])
    ax.text(0.5, 1.05, title, horizontalalignment='center',
            verticalalignment='center', transform=ax.transAxes)
    # Add N value
    stats_str = 'N={} \nRMSE={:.3g}'.format(N_, RMSE_)
    ax.text(0.05, 0.9, stats_str, horizontalalignment='left',
            verticalalignment='center', transform=ax.transAxes)
    # Add a 1:1 line
    ax_max = df_tmp.max().max()
    ax_max = AC.myround(ax_max, 5, round_up=True) * 1.05
    ax_min = df_tmp.min().min()
    ax_min = ax_min - (ax_max*0.05)
    x_121 = np.arange(ax_min, ax_max*1.5)
    ax.plot(x_121, x_121, alpha=0.5, color='k', ls='--')
    # Add ODR line
    xvalues, Y_ODR = AC.get_linear_ODR(x=X, y=Y, xvalues=x_121,
                                       return_model=False, maxit=10000)
    ax.plot(xvalues, Y_ODR, color=color, ls='--')
    # Force axis extents
    ax.set_aspect('equal')
    ax.set_xlim(ax_min, ax_max)
    ax.set_ylim(ax_min, ax_max)
    ax.set_aspect('equal')

    # - All above
    var2plot = 'WOA_Salinity'
    plot_n = 2
    color = CB_color_cycle[0]
    # Make a new axis
    ax = fig.add_subplot(3, 2, plot_n, aspect='equal')
    # Get the data
    df_tmp = df[[obs_var_dict[var2plot], var2plot]].dropna()
    # Select only data greater that 30 PSU
    df_tmp = df_tmp.loc[df_tmp[obs_var_dict[var2plot]] >= 30, :]
    N_ = int(df_tmp[[var2plot]].shape[0])
    MSE_ = np.mean((df_tmp[obs_var_dict[var2plot]] - df_tmp[var2plot])**2)
    RMSE_ = np.sqrt(MSE_)
    print(N_, MSE_, RMSE_)
    X = df_tmp[obs_var_dict[var2plot]].values
    Y = df_tmp[var2plot].values
    # Plot up
    ax.scatter(X, Y, edgecolors=color, facecolors='none', s=5)
    # Label Y axis
    if plot_n in np.arange(1, 6)[::2]:
        ax.set_ylabel('Extracted')
    # Title the plots
    title = 'Salinity ($\geq$ 30, PSU)'.format(units_dict[var2plot])
    ax.text(0.5, 1.05, title, horizontalalignment='center',
            verticalalignment='center', transform=ax.transAxes)
    # Add N value
    stats_str = 'N={} \nRMSE={:.3g}'.format(N_, RMSE_)
    ax.text(0.05, 0.9, stats_str, horizontalalignment='left',
            verticalalignment='center', transform=ax.transAxes)
    # Add a 1:1 line
    ax_max = df_tmp.max().max()
    ax_max = AC.myround(ax_max, 1, round_up=True) * 1.05
    ax_min = 29
    x_121 = np.arange(ax_min, ax_max*1.5)
    ax.plot(x_121, x_121, alpha=0.5, color='k', ls='--')
    # Add ODR line
    xvalues, Y_ODR = AC.get_linear_ODR(x=X, y=Y, xvalues=x_121,
                                       return_model=False, maxit=10000)
    ax.plot(xvalues, Y_ODR, color=color, ls='--')
    # Force axis extents
    ax.set_aspect('equal')
    ax.set_xlim(ax_min, ax_max)
    ax.set_ylim(ax_min, ax_max)
    ax.set_aspect('equal')

    # ---  Loop and plot
    for n_var2plot, var2plot in enumerate(['WOA_TEMP', 'WOA_Nitrate', ]):
        plot_n = 2 + 1 + n_var2plot
        color = CB_color_cycle[plot_n]
        # Make a new axis
        ax = fig.add_subplot(3, 2, plot_n, aspect='equal')
        # Get the data
        df_tmp = df[[obs_var_dict[var2plot], var2plot]].dropna()
        N_ = int(df_tmp[[var2plot]].shape[0])
        MSE_ = np.mean((df_tmp[obs_var_dict[var2plot]] - df_tmp[var2plot])**2)
        RMSE_ = np.sqrt(MSE_)
        print(N_, MSE_, RMSE_)
        X = df_tmp[obs_var_dict[var2plot]].values
        Y = df_tmp[var2plot].values
        # Plot up
        ax.scatter(X, Y, edgecolors=color, facecolors='none', s=5)
        # Label Y axis
        if plot_n in np.arange(1, 6)[::2]:
            ax.set_ylabel('Extracted')
        # Title the plots
        title = '{} ({})'.format(obs_var_dict[var2plot], units_dict[var2plot])
        ax.text(0.5, 1.05, title, horizontalalignment='center',
                verticalalignment='center', transform=ax.transAxes)
        # Add N value
        stats_str = 'N={} \nRMSE={:.3g}'.format(N_, RMSE_)
        ax.text(0.05, 0.9, stats_str, horizontalalignment='left',
                verticalalignment='center', transform=ax.transAxes)
        # Add a 1:1 line
        ax_max = df_tmp.max().max()
        ax_max = AC.myround(ax_max, 5, round_up=True) * 1.05
        ax_min = df_tmp.min().min()
        ax_min = ax_min - (ax_max*0.05)
        x_121 = np.arange(ax_min, ax_max*1.5)
        ax.plot(x_121, x_121, alpha=0.5, color='k', ls='--')
        # Add a line for orthogonal distance regression (ODR)
        xvalues, Y_ODR = AC.get_linear_ODR(x=X, y=Y, xvalues=x_121,
                                           return_model=False, maxit=10000)
        ax.plot(xvalues, Y_ODR, color=color, ls='--')
        # Force axis extents
        ax.set_aspect('equal')
        ax.set_xlim(ax_min, ax_max)
        ax.set_ylim(ax_min, ax_max)
        ax.set_aspect('equal')

    # ---  1st plot Salinity ( all and >30 PSU )
    # - All above
    var2plot = 'SeaWIFs_ChlrA'
    plot_n = 5
    color = CB_color_cycle[5]
    # Make a new axis
    ax = fig.add_subplot(3, 2, plot_n, aspect='equal')
    # Get the data
    df_tmp = df[[obs_var_dict[var2plot], var2plot]].dropna()
    N_ = int(df_tmp[[var2plot]].shape[0])
    MSE_ = np.mean((df_tmp[obs_var_dict[var2plot]] - df_tmp[var2plot])**2)
    RMSE_ = np.sqrt(MSE_)
    print(N_, MSE_, RMSE_)
    X = df_tmp[obs_var_dict[var2plot]].values
    Y = df_tmp[var2plot].values
    # Plot up
    ax.scatter(X, Y, edgecolors=color, facecolors='none', s=5)
    # Label Y axis
    if plot_n in np.arange(1, 6)[::2]:
        ax.set_ylabel('Extracted')
    ax.set_xlabel('Observed')
    # Title the plots
    title = 'ChlrA (all, {})'.format(units_dict[var2plot])
    ax.text(0.5, 1.05, title, horizontalalignment='center',
            verticalalignment='center', transform=ax.transAxes)
    # Add N value
    stats_str = 'N={} \nRMSE={:.3g}'.format(N_, RMSE_)
    ax.text(0.05, 0.9, stats_str, horizontalalignment='left',
            verticalalignment='center', transform=ax.transAxes)
    # Add a 1:1 line
    ax_max = df_tmp.max().max()
    ax_max = AC.myround(ax_max, 5, round_up=True) * 1.05
    ax_min = df_tmp.min().min()
    ax_min = ax_min - (ax_max*0.05)
    x_121 = np.arange(ax_min, ax_max*1.5)
    ax.plot(x_121, x_121, alpha=0.5, color='k', ls='--')
    # Add ODR line
    xvalues, Y_ODR = AC.get_linear_ODR(x=X, y=Y, xvalues=x_121,
                                       return_model=False, maxit=10000)
    ax.plot(xvalues, Y_ODR, color=color, ls='--')
    # Force axis extents
    ax.set_aspect('equal')
    ax.set_xlim(ax_min, ax_max)
    ax.set_ylim(ax_min, ax_max)
    ax.set_aspect('equal')

    # - All above
    var2plot = 'SeaWIFs_ChlrA'
    plot_n = 6
    color = CB_color_cycle[5]
    # Make a new axis
    ax = fig.add_subplot(3, 2, plot_n, aspect='equal')
    # Get the data
    df_tmp = df[[obs_var_dict[var2plot], var2plot]].dropna()
    # Select only data greater that 30 PSU
    df_tmp = df_tmp.loc[df_tmp[obs_var_dict[var2plot]] <= 5, :]
    N_ = int(df_tmp[[var2plot]].shape[0])
    MSE_ = np.mean((df_tmp[obs_var_dict[var2plot]] - df_tmp[var2plot])**2)
    RMSE_ = np.sqrt(MSE_)
    print(N_, MSE_, RMSE_)
    X = df_tmp[obs_var_dict[var2plot]].values
    Y = df_tmp[var2plot].values
    # Plot up
    ax.scatter(X, Y, edgecolors=color, facecolors='none', s=5)
    # Label Y axis
    if plot_n in np.arange(1, 6)[::2]:
        ax.set_ylabel('Extracted')
    ax.set_xlabel('Observed')
    # Title the plots
    units = units_dict[var2plot]
    title = 'ChlrA ($\leq$5 {})'.format(units)
    ax.text(0.5, 1.05, title, horizontalalignment='center',
            verticalalignment='center', transform=ax.transAxes)
    # Add N value
    stats_str = 'N={} \nRMSE={:.3g}'.format(N_, RMSE_)
    ax.text(0.05, 0.9, stats_str, horizontalalignment='left',
            verticalalignment='center', transform=ax.transAxes)
    # Add a 1:1 line
    ax_max = df_tmp.max().max()
    ax_max = AC.myround(ax_max, 1, round_up=True) * 1.05
    ax_min = df_tmp.min().min()
    ax_min = ax_min - (ax_max*0.05)
    x_121 = np.arange(ax_min, ax_max*1.5)
    ax.plot(x_121, x_121, alpha=0.5, color='k', ls='--')
    # Add ODR line
    xvalues, Y_ODR = AC.get_linear_ODR(x=X, y=Y, xvalues=x_121,
                                       return_model=False, maxit=10000)
    ax.plot(xvalues, Y_ODR, color=color, ls='--')
    # Force axis extents
    ax.set_aspect('equal')
    ax.set_xlim(ax_min, ax_max)
    ax.set_ylim(ax_min, ax_max)
    ax.set_aspect('equal')

    # -- adjust figure and save
    # Adjust plot
    left = 0.075
    right = 0.975
    wspace = 0.05
    hspace = 0.175
    top = 0.95
    bottom = 0.075
    fig.subplots_adjust(left=left, bottom=bottom, right=right, top=top,
                        wspace=wspace, hspace=hspace)
    # Save
    filename = 'Oi_prj_Chance2014_Obs_params_vs_NEW_extracted_params_WINDOW'
    plt.savefig(filename, dpi=dpi)


def compare_obs_ancillaries_with_extracted_values(df=None, save2pdf=True,
                                                  show=False, dpi=320):
    """
    Some species in the dataframe have observed as well as climatology values.
    For these species, plot up X/Y and latitudinal comparisons
    """
    import seaborn as sns
    sns.set(color_codes=True)
    current_palette = sns.color_palette("colorblind")
    sns.set_style("darkgrid")
    # Get the observational data
    if isinstance(df, type(None)):
        df = obs.get_processed_df_obs_mod()  # NOTE this df contains values >400nM
    # - Map observational variables to their shared extracted variables
    all_vars = df.columns.tolist()
    # Dictionary
    obs_var_dict = {
        # Temperature
        'WOA_TEMP':  'Temperature',
        # Chlorophyll-a
        'SeaWIFs_ChlrA': 'Chl-a',
        # Nitrate
        'WOA_Nitrate': 'Nitrate',
        # Salinity
        'WOA_Salinity': 'Salinity'
        # There is also 'Nitrite' and 'Ammonium'
    }
    # Dict of units for variables
    units_dict = {
        'SeaWIFs_ChlrA': "mg m$^{-3}$",  # Chance et al uses micro g/L
        'WOA_Salinity': 'PSU',  # https://en.wikipedia.org/wiki/Salinity
        'WOA_Nitrate':  "$\mu$M",
        'WOA_TEMP': '$^{o}$C',
    }
    # Sort dataframe by latitude
#    df = df.sort_values('Latitude', axis=0, ascending=True)

    # Set the order the dict keys are accessed
    vars_sorted = list(sorted(obs_var_dict.keys()))[::-1]

    # Setup pdf
    if save2pdf:
        savetitle = 'Oi_prj_Chance2014_Obs_params_vs_NEW_extracted_params'
        pdff = AC.plot2pdfmulti(title=savetitle, open=True, dpi=dpi)

        # - Get variables and confirm which datasets are being used for plot
        dfs = {}
    for key_ in vars_sorted:
        print(obs_var_dict[key_], key_)
        # Drop nans...
        index2use = df[[obs_var_dict[key_], key_]].dropna().index
        dfs[key_] = df.loc[index2use, :]
        # Check which datasets are being used
    ptr_str = 'For variable: {} (#={})- using: {} \n'
    for key_ in vars_sorted:
        datasets = list(set(dfs[key_]['Data_Key']))
        dataset_str = ', '.join(datasets)
        print(ptr_str.format(key_, len(datasets), dataset_str))

    # - Loop variables and plot as a scatter plot...
    for key_ in vars_sorted:
        print(obs_var_dict[key_], key_)
        # new figure
        fig = plt.figure()
        # Drop nans...
        df_tmp = df[[obs_var_dict[key_], key_]].dropna()
        N_ = int(df_tmp[[key_]].shape[0])
        print(N_)
        # Plot up
        sns.regplot(x=obs_var_dict[key_], y=key_, data=df_tmp)
        # Add title
        plt.title('X-Y plot of {} (N={})'.format(obs_var_dict[key_], N_))
        plt.ylabel('Extracted ({}, {})'.format(key_, units_dict[key_]))
        plt.xlabel('Obs. ({}, {})'.format(
            obs_var_dict[key_], units_dict[key_]))
        # Save out figure &/or show?
        if save2pdf:
            AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
        if show:
            plt.show()
        plt.close()

    # - Loop variables and plot verus lat (with difference)
    for key_ in vars_sorted:
        print(obs_var_dict[key_], key_)
        # New figure
        fig = plt.figure()
        # Drop nans...
        df_tmp = df[[obs_var_dict[key_], key_, 'Latitude']].dropna()
        N_ = int(df_tmp[[key_]].shape[0])
        print(N_)
        # Get data to analyse
        obs = df_tmp[obs_var_dict[key_]].values
        climate = df_tmp[key_].values
        X = df_tmp['Latitude'].values
        # Plot up
        plt.scatter(X, obs, label=obs_var_dict[key_], color='red',
                    marker="o")
        plt.scatter(X, climate, label=key_, color='blue',
                    marker="o")
        plt.scatter(X, climate-obs, label='diff', color='green',
                    marker="o")
        # Athesetics of plot?
        plt.legend()
        plt.xlim(-90, 90)
        plt.ylabel('{} ({})'.format(obs_var_dict[key_], units_dict[key_]))
        plt.xlabel('Latitude ($^{o}$N)')
        plt.title('{} (N={}) vs. latitude'.format(obs_var_dict[key_], N_))
        # Save out figure &/or show?
        if save2pdf:
            AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
        if show:
            plt.show()
        plt.close()
    # Save entire pdf
    if save2pdf:
        AC.plot2pdfmulti(pdff, savetitle, close=True, dpi=dpi)


def plot_up_lat_STT_var(restrict_data_max=True, restrict_min_salinity=True):
    """
    Plot up a "pretty" plot of STT vs Lat, with scatter sizes and color by var.
    """
    # - Get data as a DataFrame
    df = obs.get_processed_df_obs_mod()
    if restrict_data_max:
        #        df = df[ df['Iodide']< 450. ]
        df = df[df['Iodide'] < 400.]  # Updated to use 400 nM as upper value
    if restrict_min_salinity:
        df = df[df['WOA_Salinity'] > 30.]

    # Add modulus
    df["Latitude (Modulus)"] = np.sqrt(df["Latitude"].copy()**2)

    # - Local vars
    X_varname = "Latitude (Modulus)"
    Y_varname = "WOA_TEMP"
    S_varname = 'Iodide'
    S_label = S_varname
    C_varname = S_varname
    # - plot
    fig, ax = plt.subplots(facecolor='w', edgecolor='w')
    df.plot(kind="scatter", x=X_varname, y=Y_varname, alpha=0.4,
            s=df[S_varname], label=S_label, figsize=(10, 7),
            c=S_varname, cmap=plt.get_cmap("jet"), colorbar=True,
            sharex=False, ax=ax, fig=fig)
    plt.show()


def plot_up_lat_varI_varII(restrict_data_max=True, restrict_min_salinity=True):
    """
    Plot up a "pretty" plot of STT vs Lat, with scatter sizes and color by var.
    """
    # - Get data as a DataFrame
    df = obs.get_processed_df_obs_mod()
    if restrict_data_max:
        #        df = df[ df['Iodide']< 450. ]
        df = df[df['Iodide'] < 400.]  # Updated to use 400 nM as upper value
    if restrict_min_salinity:
        df = df[df['WOA_Salinity'] > 30.]

    df["Latitude (Modulus)"] = np.sqrt(df["Latitude"].copy()**2)
    # - Local variables
    # override? (unhashed)
    varI = 'Iodide'
    varII = "WOA_TEMP"
    # name local vars
    X_varname = "Latitude (Modulus)"
    Y_varname = varI
    S_varname = varII
    S_label = S_varname
    C_varname = S_varname
    # - plot up
    fig, ax = plt.subplots(facecolor='w', edgecolor='w')
    df.plot(kind="scatter", x=X_varname, y=Y_varname, alpha=0.4,
            s=df[S_varname], label=S_label, figsize=(10, 7),
            c=S_varname, cmap=plt.get_cmap("jet"), colorbar=True,
            sharex=False, ax=ax, fig=fig)
    plt.ylim(-5, 500)
    plt.show()


def plot_chance_param(df=None, X_var='Temperature', Y_var='Iodide',
                      data_str='(Obs.) data'):
    """
    Plot up chance et al (2014) param vs. data in DataFrame
    """
    # Only include finite data points for temp
    # ( NOTE: down to 1/3 of data of obs. data?! )
    df = df[np.isfinite(df[X_var])]
    # Add a variable for C**2 fit
    Xvar2plot = X_var+'($^{2}$)'
    df[Xvar2plot] = df[X_var].loc[:].values**2
    # Plot up data and param.
    fig, ax = plt.subplots(facecolor='w', edgecolor='w')
    # Plot up
    df.plot(kind='scatter', x=Xvar2plot, y=Y_var, ax=ax)
    # Add a line of best fit reported param.
    actual_data = df[Xvar2plot].values
    test_data = np.linspace(AC.myround(actual_data.min()),
                            AC.myround(actual_data.max()), 20)
    m = 0.225
    c = 19.0
    plt.plot(test_data, ((test_data*m)+c), color='green', ls='--',
             label='Chance et al (2014) param.')
    # Limit axis to data
    plt.xlim(-50, AC.myround(df[Xvar2plot].values.max(), 1000))
    plt.ylim(-20, AC.myround(df[Y_var].values.max(), 50, round_up=True))
    # Add title and axis labels
    N = actual_data.shape[0]
    title = 'Linear param vs. {} (N={})'.format(data_str, N)
    plt.title(title)
    plt.xlabel(X_var + ' ($^{o}$C$^{2}$)')
    plt.ylabel(Y_var + ' (nM)')
    plt.legend(loc='upper left')
    # And show/save
    tmp_str = data_str.replace(" ", '_').replace("(", "_").replace(")", "_")
    savetitle = 'Chance_param_vs_{}.png'.format(tmp_str)
    plt.savefig(savetitle)
    plt.show()


def plot_macdonald_param(df=None, X_var='Temperature', Y_var='Iodide',
                         data_str='(Obs.) data'):
    """
    Plot up MacDonald et al (2014) param vs. data in DataFrame
    """
    # Only include finite data points for temp
    # ( NOTE: down to 1/3 of data of obs. data?! )
    df = df[np.isfinite(df[X_var])]
    # Add a variable for
    Xvar2plot = '1/'+X_var
    df[Xvar2plot] = 1. / (df[X_var].loc[:].values+273.15)
    Y_var2plot = 'ln({})'.format(Y_var)
    df[Y_var2plot] = np.log(df[Y_var].values)
    # Plot up data and param.
    fig, ax = plt.subplots(facecolor='w', edgecolor='w')
    df.plot(kind='scatter', x=Xvar2plot, y=Y_var2plot, ax=ax)
    # Add a line of best fit reported param.
    # (run some numbers through this equation... )
    actual_data = df[X_var].values + 273.15
    test_data = np.linspace(actual_data.min(), actual_data.max(), 20)
    test_data_Y = 1.46E6*(np.exp((-9134./test_data))) * 1E9
    plt.plot(1./test_data, np.log(test_data_Y),
             color='green', ls='--', label='MacDonald et al (2014) param.')
    # Limit axis to data
    plt.xlim(df[Xvar2plot].values.min()-0.000025,
             df[Xvar2plot].values.max()+0.000025)
    plt.ylim(0, 7)
    # Add title and axis labels
    N = actual_data.shape[0]
    title = 'Arrhenius param vs. {} (N={})'.format(data_str, N)
    plt.title(title)
    plt.xlabel(Xvar2plot + ' ($^{o}$K)')
    plt.ylabel(Y_var2plot + ' (nM)')
    plt.legend(loc='lower left')
    # And show/save
    tmp_str = data_str.replace(" ", '_').replace("(", "_").replace(")", "_")
    savetitle = 'MacDonald_parameterisation_vs_{}.png'.format(tmp_str)
    plt.savefig(savetitle)
    plt.show()


def plot_current_parameterisations():
    """
    Plot up a comparison of Chance et al 2014 and MacDonald et al 2014 params.
    """
    # - Get obs and processed data
    # Get raw obs
    raw_df = get_core_Chance2014_obs()
    # Don't consider iodide values above 30
    raw_df = raw_df[raw_df['Iodide'] > 30.]

    # - get processed obs.
    pro_df = obs.get_processed_df_obs_mod()
    restrict_data_max, restrict_min_salinity = True, True
    if restrict_data_max:
        #        pro_df = pro_df[ pro_df['Iodide'] < 450. ] # Used for July Oi! mtg.
        # restrict below 400 (per. com. RJC)
        pro_df = pro_df[pro_df['Iodide'] < 400.]
    if restrict_min_salinity:
        pro_df = pro_df[pro_df['WOA_Salinity'] > 30.]


    # - Plots with raw obs.
    # Plot up "linear" fit of iodide and temperature. (Chance et al 2014)
    # Plot up Chance
#    plot_chance_param(df=raw_df.copy())

    # Plot up "Arrhenius" fit of iodide and temperature. ( MacDonald et al 2014)
    plot_macdonald_param(df=raw_df.copy())

    # - Plots with extract Vars.
    # Plot up "linear" fit of iodide and temperature. (Chance et al 2014)
#    plot_chance_param(df=pro_df.copy(), data_str='Extracted data',
#        X_var='WOA_TEMP')

    # Plot up "Arrhenius" fit of iodide and temperature. ( MacDonald et al 2014)
    plot_macdonald_param(df=pro_df.copy(), data_str='Extracted data',
                         X_var='WOA_TEMP')


# ---------------------------------------------------------------------------
# ---------------- Misc. Support for iodide project ------------------------
# ---------------------------------------------------------------------------
def get_updated_budget4perspective_paper():
    """
    Get updated budget for iodine perspective paper
    """
    #
    sdate = datetime.datetime(2017, 1, 1)
    edate = datetime.datetime(2017, 2, 28)
#    edate = datetime.datetime(2017, 1, 31)
    dates2use = pd.date_range(sdate, edate, freq='1D')
#    dates2use = None
    RunRoot = '/users/ts551/scratch/GC/rundirs/'
    RunName = 'merra2_4x5_standard.v12.9.1.BASE.Oi.MacDonald2014.extra_tags'
    folder = '{}/{}/OutputDir/'.format(RunRoot, RunName)
    # Get the met state object
    # for FracOfTimeInTrop
    dsS = AC.get_StateMet_ds(wd=folder, dates2use=dates2use)

    # - Calculate ozone deposition
    # Over oceans
    vars2use = ['AREA', 'Met_LWI', 'FracOfTimeInTrop', 'Met_AD']
    dsS = dsS[vars2use].squeeze()
    # Save the subset to disk, then return reload just that
    savename = 'TEMP_budget_{}.nc'.format('StateMet')
    dsS = AC.save_ds2disk_then_reload(ds=dsS, savename=savename,
                                      folder='~/tmp/')
    ocean_bool = (dsS['Met_LWI'] == 0).values

    # - Calculate iodine deposition
    # Manually set Iy dry dep species
    prefix = 'DryDep_'
    Iy_DryDep = [
    'ISALC', 'ISALA', 'IONO2', 'IONO', 'ICl', 'IBr', 'I2O4', 'I2O3',
    'I2O2', 'I2', 'HOI', 'HI', 'AERI'
    ]
    vars2use = ['{}{}'.format(prefix, i) for i in Iy_DryDep+['O3']]
    # Get deposition data
    dsD = AC.get_DryDep_ds(wd=folder, dates2use=dates2use)
    dsD = dsD[vars2use].squeeze()
    # Save the subset to disk, then return reload just that
    savename = 'TEMP_budget_{}.nc'.format('DryDep')
    dsD = AC.save_ds2disk_then_reload(ds=dsD, savename=savename,
                                      folder='~/tmp/')
    # Convert units from molec cm-2 s-1 to kg/s-1 (same as wet dep diags)
    # remove area (*cm2)
    dsD = dsD * dsS['AREA'] * 1E4
    # convert from molecules to kg
    AVG = AC.constants('AVG')
    for var in dsD.data_vars:
#    for var in list(dsD.data_vars)[:-1]:
        print(var)
        values = dsD[var].values
        spec = var.split(prefix)[-1]
        RMM = AC.species_mass(spec)
        # Convert molecules to moles
        values = values / AVG
        # convert to kg mass
        values = (values * RMM)/ 1E3
        dsD[var].values = values
    # Calculate ozone deposition
    spec = 'O3'
    var2use = '{}{}'.format(prefix, spec)
    # Convert seconds to per month
    months = list(dsD['time.month'].values.flatten())
    years = list(dsD['time.year'].values.flatten())
    month2sec = AC.secs_in_month(years=years, months=months)
    n_months_in_ds =  len(dsD['time'].values.flatten())
#    n_months_in_ds =  len(dsD['time'].values.flatten())
#    n_months_in_ds =  len(dsD['time'].values.flatten())
    dsD[var2use] = dsD[var2use].values * month2sec[:, None, None]
    total = dsD[var2use] *(12/n_months_in_ds) /1E12 *1E3
    total = float(total.sum().values.flatten())
    ocean_dep = total
    units = 'Tg/yr'
    Pstr = "Dry deposition for '{}': {} {}"
    print(Pstr.format(spec, total, units))

    # - Calculate tropospheric burdens
    #
    dsSC = AC.GetSpeciesConcDataset(wd=folder, dates2use=dates2use)

    # Now extract species
    core_burden_specs = ['O3', 'CO', 'NO', 'NO2']
    iodine_specs = [
    'I2', 'HOI', 'IO', 'I', 'HI', 'OIO', 'INO', 'IONO', 'IONO2', 'I2O2',
    'I2O4', 'I2O3', 'CH3I', 'CH2I2', 'CH2ICl', 'CH2IBr', 'ICl', 'IBr', 'AERI',
    'ISALA', 'ISALC'
    ]
    specs2use = core_burden_specs+iodine_specs
#    specs2use = core_burden_specs
    prefix = 'SpeciesConc_'
    vars2use = [prefix+i for i in specs2use]
    use_time_in_trop = True
    rm_trop =  True
    # Average burden over time
#    ds = dsD[run]#.mean(dim='time', keep_attrs=True)
    S = AC.get_Gg_trop_burden(dsSC, vars2use=vars2use, StateMet=dsS,
                              use_time_in_trop=use_time_in_trop,
                              rm_trop=rm_trop,
                              avg_over_time=True,
                              sum_spatially=True,
                              )
    # Update iodine species to be in unit Gg(I)
    ref_spec = 'I'
    for spec in iodine_specs:
        var2use = '{}{}'.format(prefix, spec )
#        ref_spec = get_ref_spec(spec)
        factor = get_conversion_factor_kgX2kgREF(spec=spec,
                                                    ref_spec=ref_spec)
        print(spec, factor)
#        S[var2use] = val/species_mass(spec)*species_mass(ref_spec)
        S[var2use]  = S[var2use]*factor
    # Upate varnames
    varnames = ['{} burden ({})'.format(i, mass_unit) for i in specs2use]
    S = S.rename(index=dict(zip(list(S.index.values), varnames)))

    # - Calculate fluxes from gas-phase to aerosol
    #





def explore_diferences_for_Skagerak():
    """
    Explore how the Skagerak data differs from the dataset as a whole
    """
    # -  Get the observations and model output
    folder = utils.get_file_locations('data_root')
    filename = 'Iodine_obs_WOA_v8_5_1_ENSEMBLE_csv__avg_nSkag_nOutliers.csv'
    dfA = pd.read_csv(folder+filename, encoding='utf-8')
    # - Local variables
    diffvar = 'Salinity diff'
    ds_str = 'Truesdale_2003_I'
    obs_var_dict = {
        # Temperature
        'WOA_TEMP':  'Temperature',
        # Chlorophyll-a
        'SeaWIFs_ChlrA': 'Chl-a',
        # Nitrate
        'WOA_Nitrate': 'Nitrate',
        # Salinity
        'WOA_Salinity': 'Salinity'
        # There is also 'Nitrite' and 'Ammonium'
    }
    # - Analysis / updates to DataFrames
    dfA[diffvar] = dfA['WOA_Salinity'].values - dfA['diffvar'].values
    # - Get just the Skagerak dataset
    df = dfA.loc[dfA['Data_Key'] == ds_str]
    prt_str = 'The general stats on the Skagerak dataset ({}) are: '
    print(prt_str.format(ds_str))
    # General stats on the iodide numbers
    stats = df['Iodide'].describe()
    for idx in stats.index.tolist():
        vals = stats[stats.index == idx].values[0]
        print('{:<10}: {:<10}'.format(idx, vals))
    # - stats on the in-situ data
    print('\n')
    prt_str = 'The stats on the Skagerak ({}) in-situ ancillary obs. are: '
    print(prt_str.format(ds_str))
    # which in-situ variables are there
    vals = df[obs_var_dict.values()].count()
    prt_str = "for in-situ variable '{:<15}' there are N={} values"
    for idx in vals.index.tolist():
        vals2prt = vals[vals.index == idx].values[0]
        print(prt_str.format(idx, vals2prt))


def check_numbers4old_chance_and_new_chance():
    """
    Do checks on which datasets have changed between versions
    """
    # - Get all observational data
    NIU, md_df = obs.get_iodide_obs()
    folder = '/work/home/ts551/data/iodide/'
    filename = 'Iodide_data_above_20m_v8_5_1.csv'
    df = pd.read_csv(folder+filename)
    df = df[np.isfinite(df['Iodide'])]  # remove NaNs
    verOrig = 'v8.5.1'
    NOrig = df.shape[0]
    # Add the is chance flag to the dataset
    ChanceStr = 'In Chance2014?'
    df[ChanceStr] = None
    for ds in list(set(md_df['Data_Key'])):
        bool = df['Data_Key'] == ds
        IsChance = md_df.loc[md_df['Data_Key'] == ds,  ChanceStr].values[0]
        df.loc[bool, ChanceStr] = IsChance
    # Where are the new iodide data points
    newLODds = set(df.loc[df['ErrorFlag'] == 7]['Data_Key'])
    prt_str = 'The new datasets from ErrorFlag 7 are in: {}'
    print(prt_str.format(' , '.join(newLODds)))
    # Versions with a different number of iodide values
    filename = 'Iodide_data_above_20m_v8_2.csv'
    df2 = pd.read_csv(folder + filename)
    df2 = convert_old_Data_Key_names2new(df2)  # Use data descriptor names
    df2 = df2[np.isfinite(df2['Iodide'])]  # remove NaNs
    ver = '8.2'
    prt_str = 'Version {} of the data  - N={} (vs {} N={})'
    print(prt_str.format(ver, df2.shape[0], verOrig, NOrig))
    # Do analysis by dataset
    prt_str = "DS: '{}' (Chance2014={}) has changed by {} to {} ({} vs. {})"
    for ds in list(set(md_df['Data_Key'])):
        N0 = df.loc[df['Data_Key'] == ds, :].shape[0]
        N1 = df2.loc[df2['Data_Key'] == ds, :].shape[0]
        IsChance = list(set(df.loc[df['Data_Key'] == ds, ChanceStr]))[0]
        if N0 != N1:
            print(prt_str.format(ds, IsChance, N0-N1, N0, verOrig, ver))


def get_numbers_for_data_paper():
    """
    Get various numbers/analysis for data descriptor paper
    """
    # - Get the full iodide sea-surface dataset
    filename = 'Iodide_data_above_20m.csv'
    folder = utils.get_file_locations('s2s_root')+'/Iodide/inputs/'
    df = pd.read_csv(folder + filename, encoding='utf-8')
    # Exclude non finite data points.
    df = df.loc[np.isfinite(df['Iodide']), :]
    # Save the full data set as .csv for use in Data Descriptor paper
    cols2use = [
        u'Data_Key', u'Data_Key_ID', 'Latitude', u'Longitude',
        #	u'\xce\xb4Iodide',
        'Year',
        #	u'Month (Orig.)',  # This is RAW data, therefore Month is observation one
        u'Month',
        'Day',
        'Iodide', u'δIodide',
        'ErrorFlag', 'Method', 'Coastal',  u'LocatorFlag',
    ]
    df = df[cols2use]
    # Map references to final .csv from metadata
    md_df = obs.get_iodide_obs_metadata()
    col2use = u'Reference'
    Data_keys = set(df['Data_Key'].values)
    for Data_key in Data_keys:
        # Get ref for dataset from metadata
        bool_ = md_df[u'Data_Key'] == Data_key
        REF = md_df.loc[bool_, :][col2use].values[0].strip()
        # Add to main data array
        bool_ = df[u'Data_Key'] == Data_key
        df.loc[bool_, col2use] = REF
    # Round up the iodide values
    df['Iodide'] = df['Iodide'].round(1).values
    df[u'δIodide'] = df[u'δIodide'].round(1).values
    df[u'Longitude'] = df[u'Longitude'].round(6).values
    df[u'Latitude'] = df[u'Latitude'].round(6).values
    # Now lock in values by settings to strings.
    df[cols2use] = df[cols2use].astype(str)
    # Save the resultant file out
    filename = 'Oi_prj_Iodide_obs_surface4DataDescriptorPaper.csv'
    df.to_csv(filename, encoding='utf-8')
    # Get number of samples of iodide per dataset
    md_df = obs.get_iodide_obs_metadata()
    md_df.index = md_df['Data_Key']
    s = pd.Series()
    Data_Keys = md_df['Data_Key']
    for Data_Key in Data_Keys:
        df_tmp = df.loc[df['Data_Key'] == Data_Key]
        s[Data_Key] = df_tmp.shape[0]
    md_df['n'] = s
    md_df.index = np.arange(md_df.shape[0])
    md_df.to_csv('Oi_prj_metadata_with_n.csv', encoding='utf-8')
    # Check sum for assignment?
    prt_str = '# Assigned values ({}) should equal original DataFrame size:{}'
    print(prt_str.format(md_df['n'].sum(), str(df.shape[0])))
    # Get number of samples of iodide per obs. technique
    Methods = set(df['Method'])
    s_ds = pd.Series()
    s_n = pd.Series()
    for Method in Methods:
        df_tmp = df.loc[df['Method'] == Method]
        s_n[Method] = df_tmp.shape[0]
        s_ds[Method] = len(set(df_tmp['Data_Key']))
    # Combine and save
    dfS = pd.DataFrame()
    dfS['N'] = s_n
    dfS['datasets'] = s_ds
    dfS.index.name = 'Method'
    # Reset index
    index2use = [str(i) for i in sorted(pd.to_numeric(dfS.index))]
    dfS = dfS.reindex(index2use)
    dfS.to_csv('Oi_prj_num_in_Methods.csv', encoding='utf-8')
    # Check sum on assignment of methods
    prt_str = '# Assigned methods ({}) should equal original DataFrame size:{}'
    print(prt_str.format(dfS['N'].sum(), str(df.shape[0])))
    prt_str = '# Assigned datasets ({}) should equal # datasets: {}'
    print(prt_str.format(dfS['datasets'].sum(), len(set(df['Data_Key']))))
    # Check which methods are assign to each dataset
    dfD = pd.DataFrame(index=sorted(set(df['Method'].values)))
    S = []
    for Data_Key in Data_Keys:
        df_tmp = df.loc[df['Data_Key'] == Data_Key]
        methods_ = set(df_tmp['Method'].values)
        dfD[Data_Key] = pd.Series(dict(zip(methods_, len(methods_)*[True])))
    # Do any datasets have more than one method?
    print('These datasets have more than one method: ')
    print(dfD.sum(axis=0)[dfD.sum(axis=0) > 1])


def mk_PDF_plot_for_Data_descriptor_paper():
    """
    Make a PDF plot for the data descriptor paper
    """
    import seaborn as sns
    sns.set(color_codes=True)
    # Get the data
    df = obs.get_processed_df_obs_mod()  # NOTE this df contains values >400nM
#	df = df.loc[df['Iodide'] <400, : ]
    # Split data into all, Coastal and Non-Coastal
    dfs = {}
    dfs['All'] = df.copy()
    dfs['Coastal'] = df.loc[df['Coastal'] == 1, :]
    dfs['Non-coastal'] = df.loc[df['Coastal'] != 1, :]
    # If hist=True, use a count instead of density
    hist = False
    # Loop and plot
    axlabel = '[I$^{-}_{aq}$] (nM)'
    fig, ax = plt.subplots()
    vars2plot = dfs.keys()
    for key in vars2plot:
        sns.distplot(dfs[key]['Iodide'].values, ax=ax,
                     axlabel=axlabel, label=key, hist=hist)
        # Force y axis extend to be correct
        ax.autoscale()
    # Add a legend
    plt.legend()
    # Add a label for the Y axis
    plt.ylabel('Density')
    # Save plot
    if hist:
        savename = 'Oi_prj_Data_descriptor_PDF'
    else:
        savename = 'Oi_prj_Data_descriptor_PDF_just_Kernal'
    plt.savefig(savename+'.png', dpi=dpi)


def mk_pf_files4Iodide_cruise(dfs=None, test_input_files=False,
                              mk_column_output_files=False, num_tracers=103):
    """
    Make planeflight input files for iodide cruises
    """
    # Get locations for cruises as
    if isinstance(dfs, type(None)):
        dfs = get_iodide_cruise_data_from_Anoop_txt_files()
    # Test the input files?
    if test_input_files:
        test_input_files4Iodide_cruise_with_plots(dfs=dfs)
    # Make planeflight files for DataFrames of cruises data (outputting columns values)
    if mk_column_output_files:
        #    slist = ['O3', 'IO', 'BrO', 'CH2O']
        slist = ['TRA_002', 'TRA_046', 'TRA_092', 'TRA_020', 'GLYX']
        met_vars = [
            'GMAO_ABSH', 'GMAO_PSFC', 'GMAO_SURF', 'GMAO_TEMP', 'GMAO_UWND',
            'GMAO_VWND',
        ]
        slist = slist + met_vars
        for key_ in dfs.keys():
            print(key_, dfs[key_].shape)
            df = dfs[key_].dropna()
            print(df.shape)
            # Add TYPE flag
            df['TYPE'] = 'IDC'
            # Grid box level centers [hPa]
            alts_HPa = AC.gchemgrid('c_hPa_geos5_r')
            # Loop and add in column values
            dfs_all = []
            for n_alt, hPa_ in enumerate(alts_HPa):
                print(hPa_, n_alt)
                df_ = df.copy()
                df_['PRESS'] = hPa_
                dfs_all += [df_]
            df = pd.concat(dfs_all)
            # Make sure rows are in date order
            df.sort_values(['datetime', 'PRESS'], ascending=True, inplace=True)
            # now output files
            AC.prt_PlaneFlight_files(df=df, slist=slist)
    # Make planeflight files for DataFrames of cruises data
    # (outputting surface values)
    else:
        met_vars = [
            'GMAO_ABSH', 'GMAO_PSFC', 'GMAO_SURF', 'GMAO_TEMP', 'GMAO_UWND',
            'GMAO_VWND',
        ]
        assert isinstance(num_tracers, int), 'num_tracers must be an integer'
        slist = ['TRA_{:0>3}'.format(i) for i in np.arange(1, num_tracers+1)]
        species = ['OH', 'HO2', 'GLYX']
        slist = slist + species + met_vars
        for key_ in dfs.keys():
            print(key_)
            df = dfs[key_].dropna()
            # Add TYPE flag
            df['TYPE'] = 'IDS'
            #
            df['PRESS'] = 1013.0
            # now output files
            AC.prt_PlaneFlight_files(df=df, slist=slist)


def test_input_files4Iodide_cruise_with_plots(dfs=None, show=False):
    """"
    Plot up maps of iodide cruise routes
    """
    # Get locations for cruises as
    if isinstance(dfs, type(None)):
        dfs = get_iodide_cruise_data_from_Anoop_txt_files()

    # - Test input files
    # File to save?
    savetitle = 'GC_pf_input_iodide_cruises'
    dpi = 320
    pdff = AC.plot2pdfmulti(title=savetitle, open=True, dpi=dpi)
    vars2test = ['LON', 'LAT']
    for key_ in dfs.keys():
        df = dfs[key_]
        for var_ in vars2test:
            # -- Plot X vs Y plot
            df_tmp = df[['datetime', var_]]
            # Calc NaNs
            VAR_dropped_N = int(df_tmp.shape[0])
            df_tmp = df_tmp.dropna()
            VAR_N_data = int(df_tmp.shape[0])
            VAR_dropped_N = VAR_dropped_N-VAR_N_data
            # Plot
            df_tmp.plot(x='datetime', y=var_)
            #
            title = "Timeseries of '{}' for '{}'".format(var_, key_)
            title += ' (ALL N={}, exc. {} NaNs)'.format(VAR_N_data,
                                                        VAR_dropped_N)
            plt.title(title)
            # Save / show
            file2save_str = 'Iodide_input_file_{}_check_{}.png'.format(
                key_, var_)
            plt.savefig(file2save_str)
            if show:
                plt.show()
            print(df_tmp[var_].describe())
            AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
        # --  Plot up cruise track as map
        del df_tmp
        df_tmp = df.dropna()
        lons = df_tmp['LON'].values
        lats = df_tmp['LAT'].values
        title = "Cruise track for '{}'".format(key_)
        print('!'*100, 'plotting map for: ', key_)
        AC.plot_lons_lats_spatial_on_map(lons=lons, lats=lats, title=title)
        plt.ylim(AC.myround(lats.min()-20, 10, ),
                 AC.myround(lats.max()+20, 10, round_up=True))
        plt.xlim(AC.myround(lons.min()-20, 10, ),
                 AC.myround(lons.max()+20, 10, round_up=True))
        if show:
            plt.show()
        AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
    # Save entire pdf
    AC.plot2pdfmulti(pdff, savetitle, close=True, dpi=dpi)


def get_iodide_cruise_data_from_Anoop_txt_files(verbose=False):
    """
    Get observational data and locations from Anoop's txt files
    """
    # - Local variables
    folder = utils.get_file_locations('data_root')
    folder += 'LOCS_Inamdar_Mahajan_cruise_x3/'
    cruise_files = {
        # 1 8th Southern Ocean Expedition (SOE-8), possibly on the RV Sagar Nidhi
        #    'Iodide1': 'cruise1_2014.xlsx',
        'SOE-8': 'cruise1_2014.xlsx',
        # 2 2nd International Indian Ocean Expedition (<-2),
        # Possibly one of several cruises in this program
        # (IIOE-1 was decades ago). On board RV Sagar Nidhi.

        #    'Iodide2': 'cruise2_2015.xlsx',
        'IIOE-1': 'cruise2_2015.xlsx',
        # 3 9th Southern Ocean Expedition (SOE-9), cruise Liselotte Tinel took samples on
        # Ship RV Agulhas.
        #    'Iodide3': 'cruise3_2016.xlsx',
        'SOE-9': 'cruise3_2016.xlsx',
    }
    # - Extract data
    dfs = {}
    for cruise_name in cruise_files.keys():
        print('Extracting: ', cruise_name, cruise_files[cruise_name])
    #	cruise_name = cruise_files.keys()[0]
        df = pd.read_excel(folder+cruise_files[cruise_name])
        names_dict = {
            'Date': 'date', 'UTC': 'date', 'time (UTC)': 'time', 'lat': 'LAT',
             'lon': 'LON'
        }
        if verbose:
            print(df.head())
        df.rename(columns=names_dict, inplace=True)
        if verbose:
            print(df.head())
        # Convert dates to datetime
    #    def _convert_datetime(x):
    #        return (270-atan2(x['date'],x['GMAO_UWND'])*180/pi)%360
    #    df['datetime'] = df.apply( f, axis=1)
        df['datetime'] = df['date'].astype(str)+' '+df['time'].astype(str)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.index = df['datetime'].values
        if verbose:
            print(df.head())
        dfs[cruise_name] = df[['datetime', 'LON', 'LAT']]
    return dfs


def TEST_AND_PROCESS_iodide_cruise_output(just_process_surface_data=False):
    """
    Process, plot (test values), then save planeflight values to csv
    """
    # Local variables
    wd = '/scratch/ts551/GC/v10-01_HAL/'
    files_dict = {
        'SOE-8': wd+'run.ClBr.Iodide2015.SOE-8',
        'IIOE-1': wd+'run.ClBr.Iodide2016.IIOE-1',
        'SOE-9': wd+'run.ClBr.Iodide2017.SOE-9',
    }
    # Test surface output
    if just_process_surface_data:
        extra_str = 'surface'
        dfs = {}
        for key_ in files_dict.keys():
            wd = files_dict[key_]+'/plane_flight_logs_{}/'.format(extra_str)
            df = process_planeflight_files(wd=wd)
            dfs[key_] = df
            get_test_plots_surface_pf_output(df=df,
                                             name='{} ({})'.format(key_,
                                             extra_str))
        # Save the output as .csv
        for key_ in dfs.keys():
            savetitle = 'GC_planeflight_compiled_output_for_{}_{}.csv'
            savetitle = savetitle.format(key_, extra_str)
            savetitle = AC.rm_spaces_and_chars_from_str(savetitle)
            dfs[key_].to_csv(savetitle)

    # - Process the output files for column values
    else:
        specs = ['O3', 'BrO', 'IO', 'CH2O']
        extra_str = 'column'
        dfs = {}
        file_str = 'GC_planeflight_compiled_output_for_{}_{}_II.csv'
        for key_ in files_dict.keys():
            #        for key_ in ['IIOE-1']:
            print(key_)
            pf_wd = files_dict[key_]+'/plane_flight_logs_{}/'.format(extra_str)
            df = process_planeflight_files(wd=pf_wd)
            # now process to column values
            df = process_planeflight_column_files(wd=files_dict[key_], df=df)
            dfs[key_] = df
            # Save the output as .csv
            savetitle = file_str.format(key_, extra_str)
            df['datetime'] = df.index
            df.to_csv(AC.rm_spaces_and_chars_from_str(savetitle))
        #  Test plots?
        for key_ in files_dict.keys():
            savetitle = file_str.format(key_, extra_str)
            df = pd.read_csv(AC.rm_spaces_and_chars_from_str(savetitle))
            df.index = pd.to_datetime(df['datetime'])
            name = '{} ({})'.format(key_, extra_str)
            units = 'molec cm$^{-2}$'
            get_test_plots_surface_pf_output(df=df,
                                             name=name,
                                             specs=specs, units=units,
                                             scale=1)


def process_planeflight_column_files(wd=None, df=None, res='4x5', debug=False):
    """
    Process column of v/v values into single values for total column
    """
    # wd=files_dict[key_]; df = dfs[ key_ ]; res='4x5'
    specs = ['O3', u'BrO', u'IO', u'CH2O', u'GLYX']
    timestamps = list(sorted(set(df.index)))
    timestamps_with_duplicates = []
    RMM_air = AC.constants('RMM_air')
    AVG = AC.constants('AVG')
    specs = ['O3', 'BrO', 'IO', 'CH2O']
    # Get lon lat array of time in troposphere
    TPS = AC.get_GC_output(wd=wd+'/', vars=['TIME_TPS__TIMETROP'],
                           trop_limit=True)
    # Convert this to boolean (<1 == not strat)
    TPS[TPS != 1] = 9999.9
    TPS[TPS == 1] = False
    TPS[TPS == 9999.9] = True
    # And dates
    CTM_DATES = AC.get_gc_datetime(wd=wd+'/')
    CTM_months = np.array([i.month for i in CTM_DATES])
# a   EPOCH = datetime.datetime(1970,1,1)
#    CTM_EPOCH = np.array([ (i.month-EPOCH).total_seconds() for i in CTM_DATES ])
    # Also get grid of surface area ( m^2 ) and convert to cm2
    S_AREA = AC.get_surface_area(res=res) * 10000
    A_M = AC.get_GC_output(wd, vars=['BXHGHT_S__AD'],  trop_limit=True,
                           dtype=np.float64)
#    VOL = AC.get_volume_np( wd=wd, res=res, s_area=S_AREA[...,None])

    big_data_l = []
    dates = []
#    for ts in timestamps[::1000]:   # Test processing on first 1000 points
    n_timestamps = len(timestamps)
    for n_ts, ts in enumerate(timestamps):
        print('progress= {:.3f} %'.format((float(n_ts) / n_timestamps)*100.))
        tmp_df = df.loc[df.index == ts]
        if debug:
            print(ts, tmp_df.shape)
        # List of pressures (one set = 47 )
        PRESS_ = tmp_df['PRESS'].values
        # Special condition for where there is more than column set
        # For a timestamp
#        assert( len(PRESS) == 47 )
        if len(PRESS_) != 47:
            timestamps_with_duplicates += [ts]
            prt_str = 'WARNING: DOUBLE UP IN  TIMESTEP:{} ({}, shape={})'
            print(prt_str.format(ts,  len(PRESS_), tmp_df.shape))
            print('Just using 1st 47 values')
            tmp_df = tmp_df[0:47]
            dates += [ts]
        else:
            dates += [ts]
        # Now reverse data (as outputted from highest to lowest)
        tmp_df = tmp_df.loc[::-1]
        # Select everyother value?
        # Lon select locations
        LAT_ = tmp_df['LAT'].values
        LON_ = tmp_df['LON'].values
        # Check there is only one lat and lon
        assert len(set(LAT_)) == 1
        assert len(set(LON_)) == 1
        # - Select 3D vars from ctm.nc file
        # Get LON, LAT index of box
        LON_ind = AC.get_gc_lon(LON_[0], res=res)
        LAT_ind = AC.get_gc_lat(LAT_[0], res=res)
#        time_ind = AC.find_nearest( CTM_EPOCH, (ts-EPOCH).total_seconds() )
        time_ind = AC.find_nearest(CTM_months, ts.month)
        # Tropspause height? ('TIME_TPS__TIMETROP)
        TPS_ = TPS[LON_ind, LAT_ind, :, time_ind]
        # Select surface area of grid box
        S_AREA_ = S_AREA[LON_ind, LAT_ind, 0]
        # Comput column by spec
        A_M_ = A_M[LON_ind, LAT_ind, :, time_ind]
        # Number of molecules per grid box
        MOLECS_ = (((A_M_*1E3) / RMM_air) * AVG)
        # Extract for species
        data_l = []
        for spec in specs:
            # Get species in v/v
            data_ = tmp_df[spec].values
            # Mask for troposphere
            data_ = np.ma.array(data_[:38], mask=TPS_)
            # Get number of molecules
            data_ = (data_ * MOLECS_).sum()
            # Convert to molecs/cm2
            data_ = data_ / S_AREA_
            # Store data
            data_l += [data_]
        # Save location
        data_l += [LON_[0], LAT_[0]]
        # Save data for all specs
        big_data_l += [data_l]

    # Convert to DataFrame.
    df_col = pd.DataFrame(big_data_l)
    df_col.index = dates  # Timestamps[::1000]
    df_col.columns = specs + ['LON', 'LAT']
    print(df_col.shape)
    return df_col


def process_planeflight_files(wd=None):
    """
    Process planeflight files to pd.DataFrame
    """
    import glob
    import seaborn as sns
    sns.set_context("paper", font_scale=0.75)
    # Get planeflight data
    files = glob.glob(wd+'plane.log.*')
    print(wd, len(files), files[0])
    names, POINTS = AC.get_pf_headers(files[0])
    dfs = [AC.pf_csv2pandas(file=i, vars=names) for i in files]
    df = pd.concat(dfs)
    # Rename axis
    TRA_XXs = [i for i in df.columns if ('TRA_' in i)]
    TRA_dict = dict(
        zip(TRA_XXs, [v10_ClBrI_TRA_XX_2_name(i) for i in TRA_XXs]))
    df.rename(columns=TRA_dict, inplace=True)
    return df


def get_test_plots_surface_pf_output(wd=None, name='Planeflight',
                                     df=None, specs=None, units=None, scale=1,
                                     show_plot=False):
    """
    Test model output at surface for Indian ship cruises
    """
    import seaborn as sns
    sns.set(color_codes=True)
    # Get data
    if isinstance(df, type(None)):
        df = process_planeflight_files(wd=wd, name=name)
    # Now add summary plots
    dpi = 320
    savetitle = 'GC_planeflight_summary_plots_for_{}_V'.format(name)
    savetitle = AC.rm_spaces_and_chars_from_str(savetitle)
    pdff = AC.plot2pdfmulti(title=savetitle, open=True, dpi=dpi, no_dstr=True)
    # Locations outputted for?
    title = 'Locations of {} output'.format(name)
    fig, ax = plt.subplots()
    AC.plot_lons_lats_spatial_on_map(title=title, f_size=15,
                                     lons=df['LON'].values,
                                     lats=df['LAT'].values,
                                     fig=fig, ax=ax)
    AC.plot2pdfmulti(pdff, savetitle, dpi=dpi, no_dstr=True)
    if show_plot:
        plt.show()
    # Timeseries of key species
    if isinstance(specs, type(None)):
        key_spec = ['O3', 'NO', 'NO2', 'OH', 'HO2', 'IO', 'BrO']
        extras = ['SO4', 'DMS', 'CH2O', ]
        species = ['OH', 'HO2', 'GLYX']
        specs = key_spec + extras + species
        specs += ['LON', 'LAT']
        met = ['GMAO_ABSH', 'GMAO_PSFC', 'GMAO_SURF', 'GMAO_TEMP',
               'GMAO_UWND', 'GMAO_VWND']
        specs += met
    print(specs)
    for spec in specs:
        fig, ax = plt.subplots()
        if isinstance(units, type(None)):
            units, scale = AC.tra_unit(spec, scale=True)
        try:
            spec_LaTeX = AC.latex_spec_name(spec)
        except:
            spec_LaTeX = spec
        print(spec, units, spec_LaTeX, scale)
        dates = pd.to_datetime(df.index).values
        plt.plot(dates, df[spec].values*scale)
        plt.ylabel('{} ({})'.format(spec, units))
        title_str = "Timeseries of modelled '{}' during {}"
        plt.title(title_str.format(spec_LaTeX, name))
        plt.xticks(rotation=45)
        plt.subplots_adjust(bottom=0.15)
        AC.plot2pdfmulti(pdff, savetitle, dpi=dpi, no_dstr=True)
        if show_plot:
            plt.show()
        plt.close()
    # Save entire pdf
    AC.plot2pdfmulti(pdff, savetitle, close=True, dpi=dpi, no_dstr=True)


def mk_data_files4Indian_seasurface_paper(res='0.125x0.125'):
    """
    Make data files for the Indian ocean surface iodide paper
    """
    AreasOfInterest = {
        'SubT_NA': ('NASW', 'NATR', 'NASE', ),
        'SubT_SA': ('SATL',),
        'SubT_NP': (u'NPSW', 'NPTG'),
        'SubT_SP': ('SPSG',),
        'SubT_SI': ('ISSG',),
    }
    AreasOfInterest_Names = AreasOfInterest.copy()
    # Get dictionaries of province numbers and names
    num2prov = LonghurstProvinceFileNum2Province(
        None, invert=True, rtn_dict=True)
    MRnum2prov = MarineRegionsOrg_LonghurstProvinceFileNum2Province(
        None, invert=True, rtn_dict=True)
    Rnum2prov = RosieLonghurstProvinceFileNum2Province(
        None, invert=True, rtn_dict=True)
    # Convert regions to the LP numbers
    PrtStr = "{} = Requested province: {} - R's #={}, MIT(GitHub) #={}, LH(2010) #={}"
    for key_ in AreasOfInterest.keys():
        for a_ in AreasOfInterest[key_]:
            print(PrtStr.format(
                key_, a_, Rnum2prov[a_], num2prov[a_], MRnum2prov[a_]))
        nums = [MRnum2prov[i] for i in AreasOfInterest[key_]]
        AreasOfInterest[key_] = nums

    # - Get data all together
    Filename = 'Oi_prj_predicted_iodide_0.125x0.125_No_Skagerrak_WITH_Provinces.nc'
#	folder = '/work/home/ts551/data/iodide/'
    folder = './'
    ds = xr.open_dataset(folder + Filename)
    params = ['Chance2014_STTxx2_I',
              'MacDonald2014_iodide',  'Ensemble_Monthly_mean']
    vars2use = params + ['LonghurstProvince']
    ds = ds[vars2use]
    # Also add the features of interest
    Filename = 'Oi_prj_feature_variables_0.125x0.125_WITH_Provinces.nc'
    ds2 = xr.open_dataset(folder + Filename)
    vars2add = ['WOA_MLDpt', 'WOA_Nitrate', 'WOA_TEMP', 'WOA_Salinity']
    for var in vars2add:
        ds[var] = ds2[var]
    # Add axis X/Y assignment
    attrs = ds['lat'].attrs
    attrs["axis"] = 'Y'
    ds['lat'].attrs = attrs
    attrs = ds['lon'].attrs
    attrs["axis"] = 'X'
    ds['lon'].attrs = attrs

    # - Now extract the data and check the locations being extracted
    # Make files with the data of interest.
    file_str = 'Oi_OS_Longhurst_provinces_{}_{}_{}.{}'
    for key_ in AreasOfInterest.keys():
        nums = AreasOfInterest[key_]
        ds_tmp = ds.where(np.isin(ds.LonghurstProvince.values, nums))
        # - Plot a diagnostic figure
        fig, ax = plt.subplots()
        ds_tmp['LonghurstProvince'].mean(dim='time').plot(ax=ax)
        # get names and numbers of assigned areas
        Names = AreasOfInterest_Names[key_]
        nums = [str(i) for i in AreasOfInterest[key_]]
        # Add a title
        nums = [str(i) for i in nums]
        title = "For '{}' ({}), \n plotting #(s)={}"
        title = title.format(key_, ', '.join(Names), ', '.join(nums))
        plt.title(title)
        # Save to png
        png_filename = file_str.format(key_, '', res,  'png')
        plt.savefig(png_filename, dpi=dpi)
        plt.close()
        # - What is the area extent of the data
        var2use = 'WOA_Nitrate'
        ds_lat = ds_tmp[var2use].dropna(dim='lat', how='all')
        min_lat = ds_lat['lat'].min() - 2
        max_lat = ds_lat['lat'].max() + 2
        ds_lon = ds_tmp[var2use].dropna(dim='lon', how='all')
        min_lon = ds_lon['lon'].min() - 2
        max_lon = ds_lon['lon'].max() + 2
        # - Now save by species
        vars2save = [i for i in ds_tmp.data_vars if i != 'LonghurstProvince']
        for var_ in vars2save:
            print(var_)
            da = ds_tmp[var_]
            # Select the minimum area for the areas
            da = da.sel(lat=(da.lat >= min_lat))
            da = da.sel(lat=(da.lat < max_lat))
            if key_ in ('SubT_NP' 'SubT_SP'):
                print('just limiting lat for: {}'.format(key_))
            else:
                da = da.sel(lon=(da.lon >= min_lon))
                da = da.sel(lon=(da.lon < max_lon))
            # Save the data to NetCDF.
            filename = file_str.format(key_, var_, res, '')
            filename = AC.rm_spaces_and_chars_from_str(filename)
            da.to_netcdf(filename+'.nc')


# ---------------------------------------------------------------------------
# --------------- Functions for Atmospheric impacts work  -------------------
# ---------------------------------------------------------------------------

def Check_sensitivity_of_HOI_I2_param2WS():
    """
    Check the sensitivity of the Carpenter2013 parameterisation to wind speed
    """
    import seaborn as sns
    sns.set(color_codes=True)
    sns.set_context("paper", font_scale=1.75)
    import matplotlib.pyplot as plt
    # Core calculation for HOI emission
    def calc_HOI_flux_eqn_20(I=None, O3=None, WS=None, ):
        """ Eqn 20 from Carpenter et al 2013 """
        return O3 * ((4.15E5 * (np.sqrt(I) / WS)) -
                     (20.6 / WS) - (2.36E4 * np.sqrt(I)))
    # Slightly simpler calculation for HOI emission
    def calc_HOI_flux_eqn_21(I=None, O3=None, WS=None, ):
        """ Eqn 21 from Carpenter et al 2013 """
        return O3 * np.sqrt(I) * ((3.56E5/WS) - 2.16E4)
    # Plot up values for windspeed
    WS_l = np.arange(5, 40, 0.1)
    # - plot up
    # Eqn 20
    Y = [calc_HOI_flux_eqn_20(I=100E-9, O3=20, WS=i) for i in WS_l]
    plt.plot(WS_l, Y, label='Eqn 20')
    # Eqn 21
    Y = [calc_HOI_flux_eqn_21(I=100E-9, O3=20, WS=i) for i in WS_l]
    plt.plot(WS_l, Y, label='Eqn 21')
    # Update aesthetics of plot and save
    plt.title('Flu HOI vs. wind speed')
    plt.ylabel('HOI flux, nmol m$^{-2}$ d$^{-1}$')
    plt.xlabel('Wind speed (ms)')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
