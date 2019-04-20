"""
"""

def compare_emissions(wd_dict=None, inorg_emiss=None, specs=None):
    """ Compare emissions between runs with different parameterisations """
    # Get emission runs that test output
    if isinstance(wd_dict, type(None)):
        wd_dict = get_emissions_testing_runs()
    params = sorted(wd_dict.keys())
    #

    # --- Get O3 burdens
    O3Burdens = [AC.get_O3_burden(wd_dict[i]) for i in params]
    O3Burdens = [i.sum()/1E3 for i in O3Burdens]
    #
    df = pd.DataFrame(O3Burdens, index=params, columns=['O3 bud.'])

    # ---- Get emissions
    if isinstance(inorg_emiss, type(None)):
        inorg_emiss, specs = get_inorg_emissions_for_params(wd_dict=wd_dict)
    # Sum emissions
    for param in params:
        inorg_emiss[param] = [i.sum() for i in inorg_emiss[param]]
    # Convert to DatFrame and combine
    inorg_emiss_names = [i+' emiss.' for i in specs]
    df2 = pd.DataFrame(inorg_emiss, index=inorg_emiss_names)
    df = pd.concat([df, df2.T], axis=1)
    # Add total inorganic flux
#    df['Inorg emiss']  = df[inorg_emiss_names].sum(axis=1)

    # ---- Now do calculations
    # calculate % change in values between runs
    df = df.T
    #
    param = 'RFR(offline)'
    refs = 'Chance2014', 'MacDonald2014'
#    comp_cols = df.columns.tolist()
#    comp_cols.pop( comp_cols.index(ref) )
    for ref in refs:
        col_name = '({}% vs. {})'.format(param, ref)
        df[col_name] = (df[param] - df[ref])/df[ref]*100
    df = df.T
    return df


def get_emissions_testing_runs():
    """ """
    folder = get_Oi_file_locations('earth0_home_dir')
    folder += '/data/all_model_simulations/iodine_runs/iGEOSChem_4.0_v10/'

    #
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
    """ Get inorganic emissions for the difference parameterisations """
    from A_PD_hal_paper_analysis_figures.halogen_family_emission_printer import get_species_emiss_Tg_per_yr
    specs = ['HOI', 'I2']
    # Surface area?
    s_area = AC.get_surface_area(res=res)
    # calc emissions!
    inorg_emiss = {}
    for param in wd_dict.keys():
        print(param)
        wd = wd_dict[param]
        months = AC.get_gc_months(wd=wd)
        years = AC.get_gc_years(wd=wd)
        # get emissions
        ars = get_species_emiss_Tg_per_yr(wd=wd, specs=specs, ref_spec='I',
                                          s_area=s_area, years=years, months=months)

        # Add sums
        ars += [ars[0]+ars[1]]
        inorg_emiss[param] = ars

    return inorg_emiss, specs+['Inorg']




def get_inorganic_iodide_emissions():
    """ Get emissions for """
    # Location of run data
    wd = get_Oi_file_locations('earth0_home_dir')
    wd += '/data/all_model_simulations/iodine_runs/iGEOSChem_4.0_v10/'
    #
    print('WARNING: Neither simulation used NEI2011 emissions!!')
    runs = {
        'MacDonald et al (2014)': wd + '/run.XS.UPa.FP.EU.BC.II.FP.2014/',
        'Chance et al (2014)': wd + 'run.XS.UPa.FP.EU.BC.II.FP.2014.Chance_iodide/',
    }
    # What are the O3 burdens for these runs?
    O3_burdens = dict([(i, AC.get_O3_burden(runs[i])) for i in runs.keys()])





def add_Inorg_and_Org_totals2array( ds, InOrgVar='Inorg_Total', OrgVar='Org_Total' ):
    """
    Add inorg. and org. sub totals to ds
    """
    # Add aggregated values to ds
    OrgVars = [
    'EmisCH2IBr_Ocean', 'EmisCH2ICl_Ocean', 'EmisCH2I2_Ocean','EmisCH3I_Ocean',
    ]
    InOrgVars = [ 'EmisI2_Ocean', 'EmisHOI_Ocean',]
    # - Inorganic
    # template off the first species
    ds[InOrgVar] = ds[InOrgVars[0]].copy()
    # sum values to this
    arr = ds[InOrgVar].values
    for var_ in InOrgVars[1:]:
        print( var_ )
        arr = arr + ds[var_].values
    ds[InOrgVar].values = arr
    attrs = ds[InOrgVar].attrs
    attrs['long_name'] = InOrgVar
    ds[InOrgVar].attrs = attrs
    # - Organic
    # template off the first species
    ds[OrgVar] = ds[OrgVars[0]].copy()
    # sum values to this
    arr = ds[OrgVar].values
    for var_ in OrgVars[1:]:
        print( var_ )
        arr = arr + ds[var_].values
    ds[OrgVar].values = arr
    attrs = ds[OrgVar].attrs
    attrs['long_name'] = OrgVar
    ds[OrgVar].attrs = attrs
    return ds


def plot_up_surface_emissions( dsDH=None, runs=None, show_plot=False,
                                wds=None, dpi=320 ):
    """
    Plot up emissions using HEMCO NetCDF files
    """
#    import cartopy.crs as ccrs
#    import matplotlib.pyplot as plt

    # names of runs to plot up?
    if isinstance( wds, type(None) ):
        wds = get_run_dict4EGU_runs()
    if isinstance( runs, type(None) ):
        runs = list(wds.keys())

    # - Add aggregated values to ds
    OrgVars = [
    'EmisCH2IBr_Ocean', 'EmisCH2ICl_Ocean', 'EmisCH2I2_Ocean','EmisCH3I_Ocean',
    ]
    InOrgVars = [ 'EmisI2_Ocean', 'EmisHOI_Ocean',]
    vars2use = OrgVars + InOrgVars
    # Aggregate variables to use?
    TotalVar = 'I_Total'
    InOrgVar = 'Inorg_Total'
    OrgVar = 'Org_Total'
    #
    Divergent_cmap = plt.get_cmap('RdBu_r')
    cmap = AC.get_colormap( np.arange(10) )
    # loop my run and add values
    for run in runs:
        # which dataset to use?
        print( run )
        ds = dsDH[run]
        # - Add Inorg and org subtotals to array
        ds = add_Inorg_and_Org_totals2array( ds=ds )
        # - Total
        # template off the first species
        ds[TotalVar] = dsDH[run][vars2use[0]].copy()
        # sum values to this
        arr = ds[TotalVar].values
        for var_ in vars2use[1:]:
            print( var_ )
            arr = arr + dsDH[run][var_].values
        ds[TotalVar].values = arr
        attrs = ds[TotalVar].attrs
        attrs['long_name'] = TotalVar
        ds[TotalVar].attrs = attrs
    # just return t data he


    #  - Setup PDF
    savetitle = 'Oi_prj_emissions_diff_plots_EGU_runs'
    dpi = 320
    pdff = AC.plot2pdfmulti(title=savetitle, open=True, dpi=dpi)

    # - Plot up emissions spatial distribution of total emissions
    for run in runs:
        print( run )
        # dataset to plot
        ds = dsDH[run][[TotalVar]]
        # use annual sum of emissions
        ds = ds.sum(dim='time')
        # - Loop and plot species
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree(), aspect='auto')
        ds[TotalVar].plot.imshow(x='lon', y='lat',
        #                               vmin=1, vmax=5,
                                       ax=ax,
                                       cmap=cmap,
                                       transform=ccrs.PlateCarree())

        # add a title
        PtrStr = "Total iodine emissions (Gg I) in '{}'"
        PtrStr += "\n(max={:.1f}, min={:.1f}, sum={:.1f})"
        sum_ = float(ds[TotalVar].sum().values)
        max_ = float(ds[TotalVar].max().values)
        min_ = float(ds[TotalVar].min().values)
        plt.title( PtrStr.format( run, max_, min_, sum_ ) )

        # beautify
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
        print( run )
        # dataset to plot
        ds = dsDH[run][[InOrgVar]]
        # use annual sum of emissions
        ds = ds.sum(dim='time')
        # - Loop and plot species
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree(), aspect='auto')
        ds[InOrgVar].plot.imshow(x='lon', y='lat',
        #                               vmin=1, vmax=5,
                                       ax=ax,
                                       cmap=cmap,
                                       transform=ccrs.PlateCarree())
        # add a title
        PtrStr = "Total Inorganic iodine emissions (Gg I) in '{}'"
        PtrStr += "\n(max={:.1f}, min={:.1f}, sum={:.1f})"
        sum_ = float(ds[InOrgVar].sum().values)
        max_ = float(ds[InOrgVar].max().values)
        min_ = float(ds[InOrgVar].min().values)
        plt.title( PtrStr.format( run, max_, min_, sum_ ) )
        # beautify
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
        print( run )
        # dataset to plot
        ds = dsDH[run][[InOrgVar,TotalVar ]]
        # use annual sum of emissions
        ds = ds.sum(dim='time')
        #
        DIFFvar = 'Inorg/Total'
        ds[DIFFvar] = ds[InOrgVar].copy()
        ds[DIFFvar].values = ds[InOrgVar].values/ds[TotalVar].values*100
        # - Loop and plot species
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree(), aspect='auto')
        ds[DIFFvar].plot.imshow(x='lon', y='lat',
        #                               vmin=-100, vmax=100,
                                       ax=ax,
                                       cmap=cmap,
                                       transform=ccrs.PlateCarree())

        # add a title
        PtrStr = "Total Inorganic iodine emissions (% of total) in '{}' \n"
        PtrStr += '(max={:.1f}, min={:.1f})'
        max_ = float(ds[DIFFvar].max().values)
        min_ = float(ds[DIFFvar].min().values)
        plt.title( PtrStr.format( run, max_, min_ ) )
        # beautify
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
        print( run )
        # dataset to plot (use annual sum of emissions)
        ds = dsDH[run][ [InOrgVar] ].sum(dim='time')
        dsREF = dsDH[REF][ [InOrgVar] ].sum(dim='time')
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

        # Add a title
        PtrStr = "Total Inorganic iodine emissions in '{}'\n as % of {}"
        PtrStr += '(max={:.1f}, min={:.1f})'
        max_ = float(ds[DIFFvar].max().values)
        min_ = float(ds[DIFFvar].min().values)
        plt.title( PtrStr.format( run, REF, max_, min_ ) )
        # beautify
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
        print( run )
        # dataset to plot (use annual sum of emissions)
        ds = dsDH[run][ [InOrgVar] ].sum(dim='time')
        dsREF = dsDH[REF][ [InOrgVar] ].sum(dim='time')
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
#                                       cmap=Divergent_cmap,
                                       cmap=cmap,
                                       transform=ccrs.PlateCarree())

        # Add a title
        PtrStr = "Total Inorganic iodine emissions in '{}'\n as % of {}"
        PtrStr += '(max={:.1f}, min={:.1f})'
        max_ = float(ds[DIFFvar].max().values)
        min_ = float(ds[DIFFvar].min().values)
        plt.title( PtrStr.format( run, REF, max_, min_ ) )
        # beautify
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
        print( run )
        # dataset to plot (use annual sum of emissions)
        ds = dsDH[run][ [InOrgVar] ].sum(dim='time')
        dsREF = dsDH[REF][ [InOrgVar] ].sum(dim='time')
        #
        DIFFvar = 'Inorg/Inorg({})'.format(REF)
        ds[DIFFvar] = ds[InOrgVar].copy()
        ds[DIFFvar].values = ds[InOrgVar].values-dsREF[InOrgVar].values
        ds[DIFFvar].values = ds[DIFFvar].values/ dsREF[InOrgVar].values*100
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

        # Add a title
        PtrStr = "Total Inorganic iodine emissions in '{}'\n as % of {}"
        PtrStr += '(max={:.1f}, min={:.1f})'
        max_ = float(ds[DIFFvar].max().values)
        min_ = float(ds[DIFFvar].min().values)
        plt.title( PtrStr.format( run, REF, max_, min_ ) )
        # beautify
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
        print( run )
        # dataset to plot (use annual sum of emissions)
        ds = dsDH[run][ [InOrgVar] ].sum(dim='time')
        dsREF = dsDH[REF][ [InOrgVar] ].sum(dim='time')
        #
        DIFFvar = 'Inorg/Inorg({})'.format(REF)
        ds[DIFFvar] = ds[InOrgVar].copy()
        ds[DIFFvar].values = ds[InOrgVar].values-dsREF[InOrgVar].values
        ds[DIFFvar].values = ds[DIFFvar].values/ dsREF[InOrgVar].values*100
        # - Loop and plot species
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection=ccrs.PlateCarree(), aspect='auto')
        ds[DIFFvar].plot.imshow(x='lon', y='lat',
                                       vmin=-100, vmax=100,
                                       ax=ax,
                                       cmap=Divergent_cmap,
#                                       cmap=cmap,
                                       transform=ccrs.PlateCarree())

        # Add a title
        PtrStr = "Total Inorganic iodine emissions in '{}'\n as % of {}"
        PtrStr += '(max={:.1f}, min={:.1f})'
        max_ = float(ds[DIFFvar].max().values)
        min_ = float(ds[DIFFvar].min().values)
        plt.title( PtrStr.format( run, REF, max_, min_ ) )
        # beautify
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
    """ Return locations of data to use for analysis/plotting """
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


def GetEmissionsFromHEMCONetCDFsAsDatasets(wds=None):
    """
    Get the emissions from the HEMCO NetCDF files as a dictionary of datasets.
    """
    # - Look at emissions through HEMCO
    # Get data locations and run names as a dictionary
    if isinstance( wds, type(None) ):
        wds = get_run_dict4EGU_runs()
    runs = list(wds.keys())
    #
#    vars2use = [i for i in dsDH[run].data_vars if 'I' in i ]
    vars2use = [
    'EmisCH2IBr_Ocean', 'EmisCH2ICl_Ocean', 'EmisCH2I2_Ocean',
    'EmisCH3I_Ocean', 'EmisI2_Ocean', 'EmisHOI_Ocean',
    ]
    # Loop and extract files
    dsDH ={}
    for run in runs:
        wd = wds[ run ]
        print( run, wd )
        dsDH[run] = AC.GetHEMCODiagnostics_AsDataset( wd=wd )
    # Get actual species
    specs = [i.split('Emis')[-1].split('_')[0] for i in vars2use ]
    var_species_dict = dict( zip(vars2use, specs ) )
    # convert to Gg
    for run in runs:
        ds = dsDH[run]
        ds = AC.Convert_HEMCO_ds2Gg_per_yr( ds, vars2convert=vars2use,
                                        var_species_dict=var_species_dict )
        dsDH[run] = ds
    return dsDH


def Check_global_statistics_on_emissions(dsDH=None, verbose=True, debug=False):
    """
    Get summary analysis on the updated iodide field
    """
    # - Files locations to use
    if isinstance( dsDH, type(None) ):
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
    # - compile data.
    df = pd.DataFrame()
    for run in runs:
        # Get the dataset to use
        ds = dsDH[run].copy()
        # add inorg and org emissions
        ds = add_Inorg_and_Org_totals2array( ds=ds )
        # sum data in to global values
        s = ds[vars2useALL].sum(dim='lat').sum(dim='lon').to_dataframe().sum()
        df[run] = s
        if debug:
            print( run, dsDH[run][vars2useALL].sum() )
    # - Add totals and print summary
    total = df.T[vars2use].T.sum().copy()
    df = df.T
    df['Total'] = total

    # in Tg units
    if verbose:
        print('-------- Global Gg (I) emission budgets ' )
        print( df.T )
    #
    # - in units of % change of the surface values
    # vs. Macdonald
    dfP = df.T.copy()
    REF = 'Macdonald2014'
    cols = list(dfP.columns)
    cols.pop(cols.index(REF))
    for col in cols+ [REF]:
        pcent = (dfP[col] - dfP[REF])/dfP[REF] *100
        if debug:
            print( col, pcent )
        dfP[col] =  pcent.values
    if verbose:
        print('-------- Vs. {} in % terms'.format(REF) )
        print( dfP )
    # vs. Chance
    dfP = df.T.copy()
    REF = 'Chance2014'
    cols = list(dfP.columns)
    cols.pop(cols.index(REF))
    for col in cols+ [REF]:
        pcent = (dfP[col] - dfP[REF])/dfP[REF] *100
        if debug:
            print( col, pcent )
        dfP[col] =  pcent.values
    if verbose:
        print('-------- Vs. {} in % terms'.format(REF) )
        print( dfP )
