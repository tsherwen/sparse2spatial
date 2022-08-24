"""
Calculations iodine emissions with updated iodide field
"""

import numpy as np
import pandas as pd
import xarray as xr
import sparse2spatial.utils as utils

# import AC_tools (https://github.com/tsherwen/AC_tools.git)
import AC_tools as AC


def Do_analysis_and_mk_plots_for_EGU19_poster():
    """
    Driver function for analysis and plotting for EGU poster
    """
    # - Get data
    # Data locations and names as a dictionary
    wds = get_run_dict4EGU_runs()
    runs = list(sorted(wds.keys()))
    # Get emissions
    dsDH = GetEmissionsFromHEMCONetCDFsAsDatasets(wds=wds)
    # Process the datasets?
#    a = [ AC.get_O3_burden( wd=wds[i] ) for i in runs ]
    # Get datasets objects from directories and in a dictionary
    dsD = {}
    for run in runs:
        ds = xr.open_dataset(wds[run]+'ctm.nc')
        dsD[run] = ds

    # - Do analysis
    # Get summary emission stats
    Check_global_statistics_on_emissions(dsDH=dsDH)
    # Look at differences in surface concentration.
    extra_str = 'EGU_runs_surface_Iy_stats_'
    df = evalulate_burdens_and_surface_conc(run_dict=wds, extra_str=extra_str)
    # Get general statistics about the emissions vs. Macdoanld et al 2014
    REF1 = 'Macdonald2014'
    extra_str = 'EGU_runs_general_stats_vs_{}_'.format(REF1)
    df = AC.get_general_stats4run_dict_as_df(run_dict=wds, REF1=REF1,
                                             extra_str=extra_str)
    # Get general statistics about the emissions vs. Macdoanld et al 2014
    REF1 = 'Chance2014'
    extra_str = 'EGU_runs_general_stats_vs_{}_'.format(REF1)
    df = AC.get_general_stats4run_dict_as_df(run_dict=wds, REF1=REF1,
                                             extra_str=extra_str)
    # Get general statistics about the emissions vs. Macdoanld et al 2014
    REF1 = 'ML_Iodide'
    extra_str = 'EGU_runs_general_stats_vs_{}_'.format(REF1)
    df = AC.get_general_stats4run_dict_as_df(run_dict=wds, REF1=REF1,
                                             extra_str=extra_str)
    # Get general statistics about the emissions vs. Macdoanld et al 2014
    REF1 = 'No_HOI_I2'
    extra_str = 'EGU_runs_general_stats_vs_{}_'.format(REF1)
    df = AC.get_general_stats4run_dict_as_df(run_dict=wds, REF1=REF1,
                                             extra_str=extra_str)

    # - Get spatial plots
    # Plot up emissions
    plot_up_surface_emissions(dsDH=dsDH)

    # - Do diferences plots
    # - Look at the HOI/I2 surface values and IO.
    # Species to look at?
    specs = ['O3', 'NO2', 'IO', 'HOI', 'I2']
    #  Chance vs. ML_iodide
    AC.plot_up_surface_changes_between2runs(ds_dict=dsD, BASE='Chance2014',
                                            NEW='ML_Iodide', specs=specs,
                                            update_PyGChem_format2COARDS=True)
    #  Macdonald vs. ML_iodide
    AC.plot_up_surface_changes_between2runs(ds_dict=dsD, BASE='Macdonald2014',
                                            NEW='ML_Iodide', specs=specs,
                                            update_PyGChem_format2COARDS=True)
    #  Macdonald vs. Chance
    AC.plot_up_surface_changes_between2runs(ds_dict=dsD, BASE='Macdonald2014',
                                            NEW='Chance2014', specs=specs,
                                            update_PyGChem_format2COARDS=True)
    #  Macdonald vs. No_HOI_I2
    AC.plot_up_surface_changes_between2runs(ds_dict=dsD, BASE='Macdonald2014',
                                            NEW='No_HOI_I2', specs=specs,
                                            update_PyGChem_format2COARDS=True)
    #  ML_iodide vs. No_HOI_I2
    AC.plot_up_surface_changes_between2runs(ds_dict=dsD, BASE='No_HOI_I2',
                                            NEW='ML_Iodide', specs=specs,
                                            update_PyGChem_format2COARDS=True)

#    ds_dict=dsD.copy(); BASE='Macdonald2014'; NEW='ML_Iodide'

    # - Get production figures.
    # Surface ozone figure - made in powerpoint for now...

    # Plot up emissions for EGU presentation
    BASE = 'ML_Iodide'
    DIFF1 = 'Chance2014'
    DIFF2 = 'Macdonald2014'
    plot_up_EGU_fig05_emiss_change(ds_dict=dsD, BASE=BASE, DIFF1=DIFF1,
                                   DIFF2=DIFF2,
                                   update_PyGChem_format2COARDS=True)


def plot_up_EGU_fig05_emiss_change(ds_dict=None, levs=[1], specs=[],
                                   BASE='', DIFF1='',  DIFF2='',
                                   prefix='IJ_AVG_S__',
                                   update_PyGChem_format2COARDS=False):
    """
    Plot up the change in emissions for EGU poster
    """
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt
    # Species to plot
    vars2use = [prefix+i for i in specs]
    unit = None
    PDFfilenameStr = 'Oi_surface_change_{}_vs_{}_lev_{:0>2}'
    # Set datasets to use and  Just include the variables to plot in the dataset
    title1 = BASE
    title2 = DIFF1
    title2 = DIFF2
    ds1 = ds_dict[BASE][vars2use].copy()
    ds2 = ds_dict[DIFF1][vars2use].copy()
    ds2 = ds_dict[DIFF2][vars2use].copy()
    # Average over time
    print(ds1, ds2, ds3)
    ds1 = ds1.mean(dim='time')
    ds2 = ds2.mean(dim='time')
    ds3 = ds3.mean(dim='time')
    # Remove vestigial coordinates.
    # (e.g. the time_0 coord... what is this?)
    vars2drop = ['time_0']
    dsL = [ds1, ds2, ds3]
    for var2drop in vars2drop:
        for n, ds in enumerate(dsL):
            CoordVars = [i for i in ds.coords]
            if var2drop in CoordVars:
                ds = ds.drop(var2drop)
                dsL[n] = ds
    ds1, ds2, ds3 = dsL
    # Update dimension names
    if update_PyGChem_format2COARDS:
        ds1 = Convert_PyGChem_Iris_DataSet2COARDS_NetCDF(ds=ds1)
        ds2 = Convert_PyGChem_Iris_DataSet2COARDS_NetCDF(ds=ds2)
        ds3 = Convert_PyGChem_Iris_DataSet2COARDS_NetCDF(ds=ds3)

    # Setup plot
    # Plot up map with mask present
    fig = plt.figure(figsize=(10, 6))
    vmin = -100
    vmax = 100
    # Add initial plot
    axn = [1, 1, 1]
    ax = fig.add_subplot(*axn, projection=ccrs.Robinson(), aspect='auto')
    ax.plot.imshow(x='lon', y='lat', ax=ax,
                   vmin=vmin, vmax=vmax,
                   transform=ccrs.PlateCarree())
    plt.title(savename)
    plt.savefig(savename+'.png')
    plt.close()


def evalulate_burdens_and_surface_conc(run_dict=None, extra_str='', REF1=None,
                                       REF2=None, REF_wd=None, res='4x5',
                                       trop_limit=True,
                                       save2csv=True, prefix='GC_',
                                       run_names=None,
                                       debug=False):
    """
    Check general statistics on the CTM model runs
    """
    # Extract names and locations of data
    if isinstance(run_dict, type(None)):
        run_dict = get_run_dict4EGU_runs()
    if isinstance(run_names, type(None)):
        run_names = sorted(run_dict.keys())
    wds = [run_dict[i] for i in run_names]
    # Mass unit scaling
    mass_scale = 1E3
    mass_unit = 'Tg'
    # V/v scaling?
    ppbv_unit = 'ppbv'
    ppbv_scale = 1E9
    pptv_unit = 'pptv'
    pptv_scale = 1E12
    # Get shared variables from a single model run
    if isinstance(REF_wd, type(None)):
        REF_wd = wds[0]
    # Get time in the troposphere diagnostic
    t_p = AC.get_GC_output(wd=REF_wd, vars=[u'TIME_TPS__TIMETROP'],
                           trop_limit=True)
    # Temperature
    K = AC.get_GC_output(wd=REF_wd, vars=[u'DAO_3D_S__TMPU'], trop_limit=True)
    # Airmass
    a_m = AC.get_air_mass_np(wd=REF_wd, trop_limit=True)
    # Surface area?
    s_area = AC.get_surface_area(res)[..., 0]  # m2 land map

    # ----
    # - Now build analysis in pd.DataFrame
    #
    # - Tropospheric burdens?
    # Get tropospheric burden for run
    varname = 'O3 burden ({})'.format(mass_unit)
    ars = [AC.get_O3_burden(i, t_p=t_p).sum() for i in wds]
    df = pd.DataFrame(ars, columns=[varname], index=run_names)

    # Get NO2 burden
    NO2_varname = 'NO2 burden ({})'.format(mass_unit)
    ars = [AC.get_trop_burden(spec='NO2', t_p=t_p, wd=i, all_data=False).sum()
           for i in wds]
    # Convert to N equivalent
    ars = [i/AC.species_mass('NO2')*AC.species_mass('N') for i in ars]
    df[NO2_varname] = ars

    # Get NO burden
    NO_varname = 'NO burden ({})'.format(mass_unit)
    ars = [AC.get_trop_burden(spec='NO', t_p=t_p, wd=i, all_data=False).sum()
           for i in wds]
    # Convert to N equivalent
    ars = [i/AC.species_mass('NO')*AC.species_mass('N') for i in ars]
    df[NO_varname] = ars

    # Combine NO and NO2 to get NOx burden
    NOx_varname = 'NOx burden ({})'.format(mass_unit)
    df[NOx_varname] = df[NO2_varname] + df[NO_varname]

    # Get HOI burden
    varname = 'HOI burden ({})'.format(mass_unit)
    ars = [AC.get_trop_burden(spec='HOI', t_p=t_p, wd=i, all_data=False).sum()
           for i in wds]
    # Convert to I equivalent
    ars = [i/AC.species_mass('HOI')*AC.species_mass('I') for i in ars]
    df[varname] = ars

    # Get I2 burden
    varname = 'I2 burden ({})'.format(mass_unit)
    ars = [AC.get_trop_burden(spec='I2', t_p=t_p, wd=i, all_data=False).sum()
           for i in wds]
    # Convert to I equivalent
    ars = [i/AC.species_mass('I2')*AC.species_mass('I') for i in ars]
    df[varname] = ars

    # Get I2 burden
    varname = 'IO burden ({})'.format(mass_unit)
    ars = [AC.get_trop_burden(spec='IO', t_p=t_p, wd=i, all_data=False).sum()
           for i in wds]
    # Convert to I equivalent
    ars = [i/AC.species_mass('IO')*AC.species_mass('I') for i in ars]
    df[varname] = ars

    # Scale units
    for col_ in df.columns:
        if 'Tg' in col_:
            df.loc[:, col_] = df.loc[:, col_].values/mass_scale

    # - Surface concentrations?
    # Surface ozone
    O3_sur_varname = 'O3 surface ({})'.format(ppbv_unit)
    ars = [AC.get_avg_surface_conc_of_X(spec='O3', wd=i, s_area=s_area)
           for i in wds]
    df[O3_sur_varname] = ars

    # Surface NOx
    NO_sur_varname = 'NO surface ({})'.format(ppbv_unit)
    ars = [AC.get_avg_surface_conc_of_X(spec='NO', wd=i, s_area=s_area)
           for i in wds]
    df[NO_sur_varname] = ars
    NO2_sur_varname = 'NO2 surface ({})'.format(ppbv_unit)
    ars = [AC.get_avg_surface_conc_of_X(spec='NO2', wd=i, s_area=s_area)
           for i in wds]
    df[NO2_sur_varname] = ars
    NOx_sur_varname = 'NOx surface ({})'.format(ppbv_unit)
    df[NOx_sur_varname] = df[NO2_sur_varname] + df[NO_sur_varname]

    # Surface HOI
    HOI_sur_varname = 'HOI surface ({})'.format(pptv_unit)
    ars = [AC.get_avg_surface_conc_of_X(spec='HOI', wd=i, s_area=s_area)
           for i in wds]
    df[HOI_sur_varname] = ars

    # Surface I2
    I2_sur_varname = 'I2 surface ({})'.format(pptv_unit)
    ars = [AC.get_avg_surface_conc_of_X(spec='I2', wd=i, s_area=s_area)
           for i in wds]
    df[I2_sur_varname] = ars

    # Surface I2
    I2_sur_varname = 'IO surface ({})'.format(pptv_unit)
    ars = [AC.get_avg_surface_conc_of_X(spec='IO', wd=i, s_area=s_area)
           for i in wds]
    df[I2_sur_varname] = ars

    # - Scale units
    for col_ in df.columns:
        if 'ppbv' in col_:
            df.loc[:, col_] = df.loc[:, col_].values*ppbv_scale
        if 'pptv' in col_:
            df.loc[:, col_] = df.loc[:, col_].values*pptv_scale

    # - Processing and save?
    # Calculate % change from base case for each variable
    if not isinstance(REF1, type(None)):
        for col_ in df.columns:
            pcent_var = col_+' (% vs. {})'.format(REF1)
            df[pcent_var] = (df[col_]-df[col_][REF1]) / df[col_][REF1] * 100
    if not isinstance(REF2, type(None)):
        for col_ in df.columns:
            pcent_var = col_+' (% vs. {})'.format(REF2)
            df[pcent_var] = (df[col_]-df[col_][REF2]) / df[col_][REF2] * 100

    # Re-order columns
    df = df.reindex_axis(sorted(df.columns), axis=1)
    # Reorder index
    df = df.T.reindex_axis(sorted(df.T.columns), axis=1).T
    # Now round the numbers
    df = df.round(3)
    # Save csv to disk
    csv_filename = '{}_summary_statistics{}.csv'.format(prefix, extra_str)
    df.to_csv(csv_filename)
    # return the DataFrame too
    return df


def compare_emissions(wd_dict=None, inorg_emiss=None, specs=None):
    """
    Compare emissions between runs with different parameterisations

    Parameters
    -------
    wd_dict (dict): dictionary of names (keys) and locations of model runs
    inorg_emiss (dict): dictionary of inorganic iodine emissions for runs

    Returns
    -------
    (pd.DataFrame)

    """
    # Get emission runs that test output
    if isinstance(wd_dict, type(None)):
        wd_dict = get_emissions_testing_runs()
    params = sorted(wd_dict.keys())

    # Get ozone burdens
    O3Burdens = [AC.get_O3_burden(wd_dict[i]) for i in params]
    O3Burdens = [i.sum()/1E3 for i in O3Burdens]
    # Compile date into dataframe
    df = pd.DataFrame(O3Burdens, index=params, columns=['O3 bud.'])

    # Get emissions
    if isinstance(inorg_emiss, type(None)):
        inorg_emiss, specs = get_inorg_emissions_for_params(wd_dict=wd_dict)
    # Sum emissions
    for param in params:
        inorg_emiss[param] = [i.sum() for i in inorg_emiss[param]]
    # Convert to DatFrame and combine
    inorg_emiss_names = [i+' emiss.' for i in specs]
    df2 = pd.DataFrame(inorg_emiss, index=inorg_emiss_names)
    df = pd.concat([df, df2.T], axis=1)
    # Add total inorganic flux? (Hasghed out for now )
#    df['Inorg emiss']  = df[inorg_emiss_names].sum(axis=1)

    # Now do calculations to get change and difference between runs
    # calculate % change in values between runs
    df = df.T
    #
    param = 'RFR(offline)'
    refs = 'Chance2014', 'MacDonald2014'
    # Loop and calculate percentages
    for ref in refs:
        col_name = '({}% vs. {})'.format(param, ref)
        df[col_name] = (df[param] - df[ref])/df[ref]*100
    df = df.T
    return df


def get_emissions_testing_runs():
    """
    Get dictionary of emission model run locations
    """
#    folder = get_file_locations('earth0_home_dir')
    folder = ''
    folder += '/data/all_model_simulations/iodine_runs/iGEOSChem_4.0_v10/'
    # Locations of model runs with different iodide fields
    RFR_dir = 'run.XS.UPa.FP.EU.BC.II.FP.2014.NEW_OFFLINE_IODIDE.several_months/'
    Chance_dir = '/run.XS.UPa.FP.EU.BC.II.FP.2014.re_run4HEMCO_diag/'
    MacDonald_dir = 'run.XS.UPa.FP.EU.BC.II.FP.2014.Chance_iodide/'
    extr_dir = '/'
#    extr_dir = '/spin_up/'
#    extr_dir = '/test_dates/'
    wd_dict = {
        'Chance2014': folder + MacDonald_dir + extr_dir,
        'MacDonald2014': folder + Chance_dir,
        'RFR(offline)': folder + RFR_dir + extr_dir,
    }
    return wd_dict


def get_inorg_emissions_for_params(wd_dict=None, res='4x5'):
    """
    Get inorganic emissions for the difference parameterisations
    """
    from A_PD_hal_paper_analysis_figures.halogen_family_emission_printer import get_species_emiss_Tg_per_yr
    specs = ['HOI', 'I2']
    # Retrieve the surface area for a given resolution
    s_area = AC.get_surface_area(res=res)
    # calc emissions!
    inorg_emiss = {}
    for param in wd_dict.keys():
        print(param)
        wd = wd_dict[param]
        months = AC.get_gc_months(wd=wd)
        years = AC.get_gc_years(wd=wd)
        # Get emissions
        ars = get_species_emiss_Tg_per_yr(wd=wd, specs=specs, ref_spec='I',
                                          s_area=s_area, years=years,
                                          months=months)
        # Add sums
        ars += [ars[0]+ars[1]]
        inorg_emiss[param] = ars
    return inorg_emiss, specs+['Inorg']


def plot_up_surface_emissions(dsDH=None, runs=None, show_plot=False,
                              wds=None, dpi=320):
    """
    Plot up emissions using HEMCO NetCDF files
    """
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt
    # names of runs to plot up?
    if isinstance(wds, type(None)):
        wds = get_run_dict4EGU_runs()
    if isinstance(runs, type(None)):
        runs = list(wds.keys())

    # - Add aggregated values to ds
    OrgVars = [
        'EmisCH2IBr_Ocean', 'EmisCH2ICl_Ocean', 'EmisCH2I2_Ocean', 'EmisCH3I_Ocean',
    ]
    InOrgVars = ['EmisI2_Ocean', 'EmisHOI_Ocean', ]
    vars2use = OrgVars + InOrgVars
    # Aggregate variables to use?
    TotalVar = 'I_Total'
    InOrgVar = 'Inorg_Total'
    OrgVar = 'Org_Total'
    # Setup the colourbar to use
    Divergent_cmap = plt.get_cmap('RdBu_r')
    cmap = AC.get_colormap(np.arange(10))
    # loop my run and add values
    for run in runs:
        # which dataset to use?
        print(run)
        ds = dsDH[run]
        # Add Inorg and org subtotals to array
        ds = add_Inorg_and_Org_totals2array(ds=ds)
        # Calculate totals
        # template off the first species
        ds[TotalVar] = dsDH[run][vars2use[0]].copy()
        # Sum values to this
        arr = ds[TotalVar].values
        for var_ in vars2use[1:]:
            print(var_)
            arr = arr + dsDH[run][var_].values
        ds[TotalVar].values = arr
        attrs = ds[TotalVar].attrs
        attrs['long_name'] = TotalVar
        ds[TotalVar].attrs = attrs

    # Setup PDF to save plot to
    savetitle = 'Oi_prj_emissions_diff_plots_EGU_runs'
    dpi = 320
    pdff = AC.plot2pdfmulti(title=savetitle, open=True, dpi=dpi)

    # - Plot up emissions spatial distribution of total emissions
    for run in runs:
        print(run)
        # dataset to plot
        ds = dsDH[run][[TotalVar]]
        # use annual sum of emissions
        ds = ds.sum(dim='time')
        # - Loop and plot species
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree(), aspect='auto')
        ds[TotalVar].plot.imshow(x='lon', y='lat',
                                 ax=ax,
                                 cmap=cmap,
                                 transform=ccrs.PlateCarree())
        # Add a title to the plot to the plot
        PtrStr = "Total iodine emissions (Gg I) in '{}'"
        PtrStr += "\n(max={:.1f}, min={:.1f}, sum={:.1f})"
        sum_ = float(ds[TotalVar].sum().values)
        max_ = float(ds[TotalVar].max().values)
        min_ = float(ds[TotalVar].min().values)
        plt.title(PtrStr.format(run, max_, min_, sum_))
        # Beautify the plot
        ax.coastlines()
        ax.set_global()
        # Save to PDF and close plot
        AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
        if show_plot:
            plt.show()
        plt.close()

    # - Plot up emissions spatial distribution of inorg emissions
    runs2plot = [i for i in runs if (i != 'No_HOI_I2')]
    for run in runs2plot:
        print(run)
        # dataset to plot
        ds = dsDH[run][[InOrgVar]]
        # use annual sum of emissions
        ds = ds.sum(dim='time')
        # - Loop and plot species
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree(), aspect='auto')
        ds[InOrgVar].plot.imshow(x='lon', y='lat',
                                 ax=ax,
                                 cmap=cmap,
                                 transform=ccrs.PlateCarree())
        # Add a title to the plot
        PtrStr = "Total Inorganic iodine emissions (Gg I) in '{}'"
        PtrStr += "\n(max={:.1f}, min={:.1f}, sum={:.1f})"
        sum_ = float(ds[InOrgVar].sum().values)
        max_ = float(ds[InOrgVar].max().values)
        min_ = float(ds[InOrgVar].min().values)
        plt.title(PtrStr.format(run, max_, min_, sum_))
        # Beautify the plot
        ax.coastlines()
        ax.set_global()
        # Save to PDF and close plot
        AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
        if show_plot:
            plt.show()
        plt.close()

    # - Plot up emissions spatial distribution inorg emissions (% of total)
    runs2plot = [i for i in runs if (i != 'No_HOI_I2')]
    for run in runs2plot:
        print(run)
        # dataset to plot
        ds = dsDH[run][[InOrgVar, TotalVar]]
        # use annual sum of emissions
        ds = ds.sum(dim='time')
        # Calculate the difference (perecent)
        DIFFvar = 'Inorg/Total'
        ds[DIFFvar] = ds[InOrgVar].copy()
        ds[DIFFvar].values = ds[InOrgVar].values/ds[TotalVar].values*100
        # Loop and plot species
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree(), aspect='auto')
        ds[DIFFvar].plot.imshow(x='lon', y='lat',
                                ax=ax,
                                cmap=cmap,
                                transform=ccrs.PlateCarree())
        # Add a title to the plot
        PtrStr = "Total Inorganic iodine emissions (% of total) in '{}' \n"
        PtrStr += '(max={:.1f}, min={:.1f})'
        max_ = float(ds[DIFFvar].max().values)
        min_ = float(ds[DIFFvar].min().values)
        plt.title(PtrStr.format(run, max_, min_))
        # Beautify the plot
        ax.coastlines()
        ax.set_global()
        # Save to PDF and close plot
        AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
        if show_plot:
            plt.show()
        plt.close()

    # -  plot up emissions as a % of REF (Chance2014)
    REF = 'Chance2014'
#    runs2plot = [i for i in runs if (i != REF)]
#    runs2plot = [i for i in runs if (i != 'No_HOI_I2')]
    runs2plot = ['ML_Iodide']
    for run in runs2plot:
        print(run)
        # dataset to plot (use annual sum of emissions)
        ds = dsDH[run][[InOrgVar]].sum(dim='time')
        dsREF = dsDH[REF][[InOrgVar]].sum(dim='time')
        #
        DIFFvar = 'Inorg/Inorg({})'.format(REF)
        ds[DIFFvar] = ds[InOrgVar].copy()
        ds[DIFFvar].values = ds[InOrgVar].values/dsREF[InOrgVar].values*100
        # - Loop and plot species
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree(), aspect='auto')
        ds[DIFFvar].plot.imshow(x='lon', y='lat',
                                #                               vmin=1, vmax=5,
                                vmin=0, vmax=200,
                                ax=ax,
                                cmap=cmap,
                                transform=ccrs.PlateCarree())

        # Add a title to the plot
        PtrStr = "Total Inorganic iodine emissions in '{}'\n as % of {}"
        PtrStr += '(max={:.1f}, min={:.1f})'
        max_ = float(ds[DIFFvar].max().values)
        min_ = float(ds[DIFFvar].min().values)
        plt.title(PtrStr.format(run, REF, max_, min_))
        # Beautify the plot
        ax.coastlines()
        ax.set_global()
        # Save to PDF and close plot
        AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
        if show_plot:
            plt.show()
        plt.close()

    # -  plot up emissions as a % of REF (Macdonald2014)
    REF = 'Macdonald2014'
#    runs2plot = [i for i in runs if (i != REF)]
#    runs2plot = [i for i in runs if (i != 'No_HOI_I2')]
    runs2plot = ['ML_Iodide']
    for run in runs2plot:
        print(run)
        # dataset to plot (use annual sum of emissions)
        ds = dsDH[run][[InOrgVar]].sum(dim='time')
        dsREF = dsDH[REF][[InOrgVar]].sum(dim='time')
        #
        DIFFvar = 'Inorg/Inorg({})'.format(REF)
        ds[DIFFvar] = ds[InOrgVar].copy()
        ds[DIFFvar].values = ds[InOrgVar].values/dsREF[InOrgVar].values*100
        # - Loop and plot species
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree(), aspect='auto')
        ds[DIFFvar].plot.imshow(x='lon', y='lat',
                                vmin=0, vmax=200,
                                ax=ax,
                                cmap=cmap,
                                transform=ccrs.PlateCarree())

        # Add a title to the plot
        PtrStr = "Total Inorganic iodine emissions in '{}'\n as % of {}"
        PtrStr += '(max={:.1f}, min={:.1f})'
        max_ = float(ds[DIFFvar].max().values)
        min_ = float(ds[DIFFvar].min().values)
        plt.title(PtrStr.format(run, REF, max_, min_))
        # Beautify the plot
        ax.coastlines()
        ax.set_global()
        # Save to PDF and close plot
        AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
        if show_plot:
            plt.show()
        plt.close()

    # -  plot up emissions as a % of REF (Chance2014)
    REF = 'Chance2014'
#    runs2plot = [i for i in runs if (i != REF)]
#    runs2plot = [i for i in runs if (i != 'No_HOI_I2')]
    runs2plot = ['ML_Iodide']
    for run in runs2plot:
        print(run)
        # dataset to plot (use annual sum of emissions)
        ds = dsDH[run][[InOrgVar]].sum(dim='time')
        dsREF = dsDH[REF][[InOrgVar]].sum(dim='time')
        #
        DIFFvar = 'Inorg/Inorg({})'.format(REF)
        ds[DIFFvar] = ds[InOrgVar].copy()
        ds[DIFFvar].values = ds[InOrgVar].values-dsREF[InOrgVar].values
        ds[DIFFvar].values = ds[DIFFvar].values / dsREF[InOrgVar].values*100
        # - Loop and plot species
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree(), aspect='auto')
        ds[DIFFvar].plot.imshow(x='lon', y='lat',
                                #                               vmin=1, vmax=5,
                                vmin=-100, vmax=100,
                                ax=ax,
                                #                                       cmap=cmap,
                                cmap=Divergent_cmap,
                                transform=ccrs.PlateCarree())

        # Add a title to the plot
        PtrStr = "Total Inorganic iodine emissions in '{}'\n as % of {}"
        PtrStr += '(max={:.1f}, min={:.1f})'
        max_ = float(ds[DIFFvar].max().values)
        min_ = float(ds[DIFFvar].min().values)
        plt.title(PtrStr.format(run, REF, max_, min_))
        # Beautify the plot
        ax.coastlines()
        ax.set_global()
        # Save to PDF and close plot
        AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
        if show_plot:
            plt.show()
        plt.close()

    # -  plot up emissions as a % of REF (Macdonald2014)
    REF = 'Macdonald2014'
#    runs2plot = [i for i in runs if (i != REF)]
#    runs2plot = [i for i in runs if (i != 'No_HOI_I2')]
    runs2plot = ['ML_Iodide']
    for run in runs2plot:
        print(run)
        # dataset to plot (use annual sum of emissions)
        ds = dsDH[run][[InOrgVar]].sum(dim='time')
        dsREF = dsDH[REF][[InOrgVar]].sum(dim='time')
        #
        DIFFvar = 'Inorg/Inorg({})'.format(REF)
        ds[DIFFvar] = ds[InOrgVar].copy()
        ds[DIFFvar].values = ds[InOrgVar].values-dsREF[InOrgVar].values
        ds[DIFFvar].values = ds[DIFFvar].values / dsREF[InOrgVar].values*100
        # - Loop and plot species
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree(), aspect='auto')
        ds[DIFFvar].plot.imshow(x='lon', y='lat',
                                vmin=-100, vmax=100,
                                ax=ax,
                                cmap=Divergent_cmap,
                                transform=ccrs.PlateCarree())

        # Add a title to the plot
        PtrStr = "Total Inorganic iodine emissions in '{}'\n as % of {}"
        PtrStr += '(max={:.1f}, min={:.1f})'
        max_ = float(ds[DIFFvar].max().values)
        min_ = float(ds[DIFFvar].min().values)
        plt.title(PtrStr.format(run, REF, max_, min_))
        # Beautify the plot
        ax.coastlines()
        ax.set_global()
        # Save to PDF and close plot
        AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
        if show_plot:
            plt.show()
        plt.close()

    # -- Save entire pdf
    AC.plot2pdfmulti(pdff, savetitle, close=True, dpi=dpi)


def get_run_dict4EGU_runs():
    """
    Return locations of data to use for analysis/plotting for EGU presentation
    """
    RunRoot = '/users/ts551/scratch/GC/rundirs/'
#    wds = glob.glob( RunRoot+ '*Oi*' )
#    runs = [i.split('Oi.')[-1] for i in wds]
    wds = {
        # spin up period
        #    'Chance2014': RunRoot+'/geosfp_4x5_tropchem.v12.2.1.Oi.Chance2014.O3/spin_up/',
        #    'No_HOI_I2': RunRoot+'/geosfp_4x5_tropchem.v12.2.1.Oi.No_HOI_I2/spin_up/',
        #    'Macdonald2014': RunRoot+'/geosfp_4x5_tropchem.v12.2.1.Oi.Macdonald2014.O3/spin_up/',
        #    'ML_Iodide': RunRoot+'/geosfp_4x5_tropchem.v12.2.1.Oi.ML_Iodide.O3/spin_up/'
        # analysis period
        #    'Chance2014': RunRoot+'/geosfp_4x5_tropchem.v12.2.1.Oi.Chance2014.O3/',
        #    'No_HOI_I2': RunRoot+'/geosfp_4x5_tropchem.v12.2.1.Oi.No_HOI_I2/',
        #    'Macdonald2014': RunRoot+'/geosfp_4x5_tropchem.v12.2.1.Oi.Macdonald2014.O3/',
        #    'ML_Iodide': RunRoot+'/geosfp_4x5_tropchem.v12.2.1.Oi.ML_Iodide.O3/'
        # analysis period with SSBr
        'Chance2014': RunRoot+'/geosfp_4x5_tropchem.v12.2.1.Oi.Chance2014.O3.SSBr/',
        'No_HOI_I2': RunRoot+'/geosfp_4x5_tropchem.v12.2.1.Oi.No_HOI_I2.O3.SSBr/',
        'Macdonald2014': RunRoot+'/geosfp_4x5_tropchem.v12.2.1.Oi.Macdonald2014.O3.SSBr/',
        'ML_Iodide': RunRoot+'/geosfp_4x5_tropchem.v12.2.1.Oi.ML_Iodide.O3.SSBr/'
    }
    return wds


def Check_global_statistics_on_emissions(dsDH=None, verbose=True, debug=False):
    """
    Get summary analysis on the updated iodide field
    """
    # - Files locations to use
    if isinstance(dsDH, type(None)):
        dsDH = GetEmissionsFromHEMCONetCDFsAsDatasets()
    # Set runs to use
    runs = ['Chance2014', 'No_HOI_I2', 'Macdonald2014', 'ML_Iodide']
    # vars to use
    InOrgVar = 'Inorg_Total'
    OrgVar = 'Org_Total'
    vars2use = [
        'EmisCH2IBr_Ocean', 'EmisCH2ICl_Ocean', 'EmisCH2I2_Ocean',
        'EmisCH3I_Ocean', 'EmisI2_Ocean', 'EmisHOI_Ocean',
    ]
    vars2useALL = vars2use + [InOrgVar, OrgVar]
    # - compile data into a pd.DataFrame
    df = pd.DataFrame()
    for run in runs:
        # Get the dataset to use
        ds = dsDH[run].copy()
        # Add inorg and org emissions
        ds = add_Inorg_and_Org_totals2array(ds=ds)
        # Sum data in to global values
        s = ds[vars2useALL].sum(dim='lat').sum(dim='lon').to_dataframe().sum()
        df[run] = s
        if debug:
            print(run, dsDH[run][vars2useALL].sum())
    # Add totals and print summary
    total = df.T[vars2use].T.sum().copy()
    df = df.T
    df['Total'] = total
    # in Tg units
    if verbose:
        print('-------- Global Gg (I) emission budgets ')
        print(df.T)

    # In units of % change of the surface values
    # vs. Macdonald
    dfP = df.T.copy()
    REF = 'Macdonald2014'
    cols = list(dfP.columns)
    cols.pop(cols.index(REF))
    for col in cols + [REF]:
        pcent = (dfP[col] - dfP[REF])/dfP[REF] * 100
        if debug:
            print(col, pcent)
        dfP[col] = pcent.values
    if verbose:
        print('-------- Vs. {} in % terms'.format(REF))
        print(dfP)
    # vs. Chance
    dfP = df.T.copy()
    REF = 'Chance2014'
    cols = list(dfP.columns)
    cols.pop(cols.index(REF))
    for col in cols + [REF]:
        pcent = (dfP[col] - dfP[REF])/dfP[REF] * 100
        if debug:
            print(col, pcent)
        dfP[col] = pcent.values
    if verbose:
        print('-------- Vs. {} in % terms'.format(REF))
        print(dfP)
