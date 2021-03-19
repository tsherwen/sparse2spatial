"""
Analysis of emissions of updates ocean iodine fields
"""

import numpy as np
import pandas as pd
import xarray as xr
import sparse2spatial.utils as utils
import sparse2spatial.plotting as plotting
import seaborn as sns

# import AC_tools (https://github.com/tsherwen/AC_tools.git)
import AC_tools as AC


def do_analysis_on_iodine_emission_options(dpi=320, context="paper"):
    """
    Perform analysis on HOI+I2 emissions with different I- fields
    """
    sns.set_context(context)
    sns.set_palette( 'colorblind' )
    # Get dictionary of model runs
    d = Get_GEOSChem_run_dict(version='v12.9.1', RunSet='emission_options')
    # Update directories to include the NetCDF output folder
    keys2use = d.keys()
    dNetCDF = {}
    for key in keys2use:
        dNetCDF[key] = d[key]+'/OutputDir/'
    # Now retrieve HEMCO files
    dates2use = None
    dsD = get_HEMCO_diags_as_ds_dict4runs(wds=dNetCDF, dates2use=dates2use)
    # Check that a full year of output has been found
    # There wasn't so some runs are being repeated

    # Calculate annual stats
    for key in keys2use:
        dsD[key] = add_Inorg_and_Org_totals2array(dsD[key])
    df = pd.DataFrame()
    for key in keys2use:
        print(key)
        ds = dsD[key].sum(dim=('lat', 'lon'))
        S = pd.Series(dtype=np.float)
        for var in ds.data_vars:
            S[var] = np.float(ds[var].values)
        df[key] = S
    df = df.drop('AREA')
    # Add total
    df = df.T
    df['Total'] = df[['Inorg_Total', 'Org_Total']].sum(axis=1)
    df = df.T
    # Save total
    df.round(0).to_csv('PDI_iodine_emissions_options_annual_totals.csv')

    # - Convert to % difference and save values
    REF = 'MacDonald2014'
    cols2use =  [i for i in df.columns if i != REF]
    for col in cols2use:
        print(col)
        df.loc[:,col] = (df.loc[:,col]-df.loc[:,REF])/df.loc[:,REF]*100
    savename = 'PDI_iodine_emissions_options_annual_totals_percent_vs_{}.csv'
    df.round(0).to_csv(savename.format(REF))

    # - Plot up a summary emission plot for HOI, I2, CH3I, CH2IX
    run2plot = 'Sherwen2019x0.5'
    ds2plot = dsD[run2plot].copy()
    #
    data_vars = [i for i in list(ds2plot.data_vars) if 'Emis' in i]
    vars2plot = [i.split('Emis')[-1].split('_O')[0] for i in data_vars ]
    ds2plot = ds2plot.rename(name_dict=dict(zip(data_vars,vars2plot)) )
    # Combine CH2IX
    ds2plot['CH2IX'] = ds2plot['CH2IBr'].copy()
    ds2plot['CH2IX'] += ds2plot['CH2ICl'].copy()
    ds2plot['CH2IX'] += ds2plot['CH2I2'].copy()
    # Now plot up the core emissions
    vars2plot = [ 'I2', 'HOI', 'CH3I', 'CH2IX' ]
    fig = plt.figure(figsize=(9, 5), dpi=dpi)
    projection = ccrs.Robinson()
    units = 'Gg yr$^{-1}$'
    # Loop by season
    for _n, _var in enumerate(vars2plot):
        # Select data for month
        _ds2plot = ds2plot[[_var]]
        # update long name
        attrs = _ds2plot[_var].attrs
#        attrs['long_name'] = '{} ({})'.format(_var, units)
        attrs['long_name'] = _var
        _ds2plot[_var].attrs = attrs
        # Setup the axis
        axn = (2, 2, _n+1)
        ax = fig.add_subplot(*axn, projection=projection, aspect='auto')
        # Now plot
        plotting.plot_spatial_data(ds=ds2plot, var2plot=_var,
                                   ax=ax, fig=fig,
                                   target=_var, title=_var,
                                   rm_colourbar=False,
                                   save_plot=False)
        # Capture the image from the axes
        im = ax.images[0]

    # Save or show plot
    filename = 'PDI_emissions_{}.png'.format(run2plot)
    plt.savefig(filename, dpi=dpi)
    plt.close('all')

    # - Plot up spatial change in emissions
    matplotlib.rc_file_defaults()
#    dsD = get_HEMCO_diags_as_ds_dict4runs(wds=dNetCDF, dates2use=dates2use,
#                                          convert2Gg_yr=False)
    keys2use = [i for i in list(dsD.keys()) if i != REF]
    savetitle = 'PDI_iodine_emissions_options_diff'
    pdff = AC.plot2pdfmulti(title=savetitle, open=True, dpi=dpi)
    # limit colour-bar
    vmin = -100
    vmax = 100
    cmap = AC.get_colormap(arr=np.array([vmin, vmax]))
    kwargs = {
    'vmin': vmin,
    'vmax': vmax,
    'cmap': cmap,
    'extend': 'both',
    }
    var2use = 'Inorg_Total'
    NewVarStr = '{}-REF-{}-pcent'
    title_str = '$\Delta$ I$_2$+HOI emission (%) - {} vs {}'
    for key in keys2use:
        NewVar = NewVarStr.format(key, REF)
        ds2plot = xr.Dataset()
        ds2plot[NewVar] = dsD[key][var2use].copy()
        REF_var = dsD[REF][var2use].values
        ds2plot[NewVar].values = (dsD[key][var2use].values-REF_var)/REF_var*100
        attrs = ds2plot[NewVar].attrs
        attrs['units'] = '%'
        title =  title_str.format(key, REF)
        AC.quick_map_plot(ds2plot, var2plot=NewVar, title=title, **kwargs)
        plt.title(title)
        # Save out figure &/or show?
        AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
        plt.close()

    # higher restriction on colour-bar
    kwargs['vmax'] = -200
    kwargs['vmin'] = 200
    title_str = '$\Delta$ I$_2$+HOI emission (%) - {} vs {}'
    for key in keys2use:
        NewVar = NewVarStr.format(key, REF)
        ds2plot = xr.Dataset()
        ds2plot[NewVar] = dsD[key][var2use].copy()
        REF_var = dsD[REF][var2use].values
        ds2plot[NewVar].values = (dsD[key][var2use].values-REF_var)/REF_var*100
        title =  title_str.format(key, REF)
        AC.quick_map_plot(ds2plot, var2plot=NewVar, title=title, **kwargs)
        plt.title(title)
        # Save out figure &/or show?
        AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
        plt.close()


    # higher restriction on colour-bar
    kwargs['vmax'] = -300
    kwargs['vmin'] = 300
    title_str = '$\Delta$ I$_2$+HOI emission (%) - {} vs {}'
    for key in keys2use:
        NewVar = NewVarStr.format(key, REF)
        ds2plot = xr.Dataset()
        ds2plot[NewVar] = dsD[key][var2use].copy()
        REF_var = dsD[REF][var2use].values
        ds2plot[NewVar].values = (dsD[key][var2use].values-REF_var)/REF_var*100
        title =  title_str.format(key, REF)
        AC.quick_map_plot(ds2plot, var2plot=NewVar, title=title, **kwargs)
        plt.title(title)
        # Save out figure &/or show?
        AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
        plt.close()

    # higher restriction on colour-bar
    kwargs['vmax'] = -400
    kwargs['vmin'] = 400
    title_str = '$\Delta$ I$_2$+HOI emission (%) - {} vs {}'
    for key in keys2use:
        NewVar = NewVarStr.format(key, REF)
        ds2plot = xr.Dataset()
        ds2plot[NewVar] = dsD[key][var2use].copy()
        REF_var = dsD[REF][var2use].values
        ds2plot[NewVar].values = (dsD[key][var2use].values-REF_var)/REF_var*100
        title =  title_str.format(key, REF)
        AC.quick_map_plot(ds2plot, var2plot=NewVar, title=title, **kwargs)
        plt.title(title)
        # Save out figure &/or show?
        AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
        plt.close()

    # Save the entire PDF
    AC.plot2pdfmulti(pdff, savetitle, close=True, dpi=dpi)


def do_analysis_on_emission4scalar_runs(dpi=320, context="paper"):
    """
    Do analysis on emissions from iodide fields with different scalings
    """
    sns.set_context(context)
    sns.set_palette( 'colorblind' )
    # Get dictionary of model runs
    d = Get_GEOSChem_run_dict(version='v12.9.1', RunSet='scalar_runs')
    # Set number, and which, days to use
    sdate = datetime.datetime(2016, 1, 1, 0, 0)
#    edate = datetime.datetime(2016, 1, 7, 0, 0)  # 6 days
    edate = datetime.datetime(2016, 1, 4, 0, 0)   # 3 days
    num_days = (edate-sdate).days
    dates2use = pd.date_range(sdate, edate, freq='24H')
#    dates2use, num_days = [dates2use[0]], 1 # Just use first day
    # Update directories to include the NetCDF output folder
    keys2use = d.keys()
    dNetCDF = {}
    for key in keys2use:
        dNetCDF[key] = d[key]+'/OutputDir/'
    # Now retrieve HEMCO files
    dsD = get_HEMCO_diags_as_ds_dict4runs(wds=dNetCDF, dates2use=dates2use)
    # Include totals for organic and inorganic iodine emission
    for key in keys2use:
        dsD[key] = add_Inorg_and_Org_totals2array(dsD[key])
    # - Now plot up global totals
    df = pd.DataFrame()
    for key in keys2use:
        print(key)
        ds = dsD[key].sum(dim=('lat', 'lon'))
        S = pd.Series(dtype=np.float)
        for var in ds.data_vars:
            S[var] = np.float(ds[var].values)
        df[key] = S
    # Plot up the change in organic and inorganic flux
    IodideField = 'Sherwen2019'
    InOrgVar = 'Inorg_Total'
    OrgVar = 'Org_Total'
    TotalVar = 'Iodine_Total'
    CurrentColumns = df.columns
    NeWColumns = [i.replace(IodideField,'') for i in CurrentColumns ]
    NeWColumns = [float(i.replace('x','')) for i in NeWColumns ]
    NeWColumns = [(i-1.)*100. for i in NeWColumns]
    NeWColumns = [ AC.myround(i, base=1) for i in NeWColumns ]
    df = df.rename(columns=dict(zip(CurrentColumns,NeWColumns)) )
    df = df.T
    df.loc[:,TotalVar] = df.loc[:,OrgVar] + df.loc[:,InOrgVar]
    vars2plot = [InOrgVar, OrgVar, TotalVar, 'HOI', 'I2']
    # Update columns to just be species names
    EmisVars = [i for i in df.columns if ('Emis' in i)]
    NewNames = [i.split('_')[0].split('Emis')[-1]  for i in EmisVars ]
    df = df.rename(columns=dict( zip(EmisVars, NewNames)))
    # Save the data too
    xVar =  '$\Delta$ in global iodide field (%)'
    df.index.name = xVar
    df_ACTUAL = df.copy()
    filename = 's2s_stats_on_iodine_flux_vs_iodide_field_delta_{}day.csv'
    df.to_csv( filename.format(num_days) )
    # Now plot...
    fig, ax = plt.subplots(dpi=dpi)
    df[vars2plot].plot()
    units = 'Gg yr$^{-1}$'
    title_str = "$\Delta$ emissions ({}) with scaling of '{}' iodide field"
    plt.title(title_str.format(units, IodideField))
    plt.xlabel('$\Delta$ in global iodide field (%)')
    plt.ylabel('$\Delta$ iodide flux ({})'.format(units))
    plt.legend()
    plt.tight_layout()
    filename = 's2s_iodide_scalar_emissions_{}day'.format(num_days)
    plt.savefig(filename, dpi=dpi)
    AC.close_plot()

    # Plot the same values as a percent
    units = '%'
    REF = 0
    vars2plot = [InOrgVar, OrgVar, TotalVar, 'HOI', 'I2']
    for col in list(df.columns):
        vals = (df.loc[:,col] - df.loc[REF,col])/df.loc[REF,col]*100
        df.loc[:,col] = vals
    # Save the data
    filename = 's2s_stats_on_iodine_flux_vs_iodide_field_delta_{}day_pcent.csv'
    df.to_csv( filename.format(num_days) )
    # Now plot...
    fig, ax = plt.subplots(dpi=dpi)
    df[vars2plot].plot()
    title_str = "$\Delta$ emissions ({}) with scaling of '{}' iodide field"
    plt.title(title_str.format(units, IodideField))
    plt.xlabel('$\Delta$ in global iodide field (%)')
    plt.ylabel('$\Delta$ iodide flux ({})'.format(units))
    fig.legend(loc=7)
    plt.tight_layout()
    filename = 's2s_iodide_scalar_emissions_{}day_pcent'.format(num_days)
    plt.savefig(filename, dpi=dpi)
    AC.close_plot()

    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn import datasets, linear_model
    from sklearn.metrics import mean_squared_error, r2_score
    # Create linear regression object
    regr = linear_model.LinearRegression()
    # Train the model using the entire datapoints
    X_train = df.index.values.flatten()[:,np.newaxis]
    y_train = df[InOrgVar].values.flatten()
    regr.fit(X_train, y_train)
#    y_test = np.arange(y_train.min(), y_train.max(), 0.1)[:, np.newaxis]
    y_pred = regr.predict( X[:,np.newaxis] )
    # Calculate the % change in emissions for specific iodide field change
    vals2test = [5.6, 5.7, 5.8]
    regr.predict( np.array(vals2test)[:, np.newaxis] )
    # The coefficients
    print('Coefficients: \n', regr.coef_)
    # The mean squared error
    print('Mean squared error: %.2f'
          % mean_squared_error(y_train, y_pred))
    # The coefficient of determination: 1 is perfect prediction
    print('Coefficient of determination: %.2f'
          % r2_score(y_train, y_pred))

    # Also plot this specifically for Lucy's perspectives paper
    # Now plot...
    sns.set_style("dark")
    fig, ax = plt.subplots(dpi=dpi)
    X = df.index.values
    Y = df[InOrgVar].values
#    plt.scatter( X, Y, lw=7.5 ) # plot the actual values too? - NO
    # plot a line of best fit
    plt.plot( X, y_pred, lw=7.5 ) # just plot the least squares fit
    title_str = "Change in emissions ({}) with scaling of sea-surface \niodide field from {}"
    plt.title(title_str.format(units, 'Sherwen et al (2019)'))
    plt.xlabel('Change in global sea-surface iodide field (%)')
    plt.ylabel('Change in inorganic iodine ({}) emission ({})'.format('HOI+I$_{2}$', units))
#    fig.legend(loc=7) # Set legend to be outside of plot
    # Add dashed lines through zero
    bottom, top = plt.gca().get_ylim()
    left, right = plt.gca().get_xlim()
    yrange = np.arange(bottom, top, 0.01)
    xrange = np.arange(left, right, 0.01)
    plt.plot(xrange, [0.]*len(xrange), ls='--', color='grey', zorder=0,
             alpha=0.5)
    plt.plot([0.]*len(yrange), yrange, ls='--', color='grey', zorder=0,
             alpha=0.5)
    # Add dashed lines showing 1:1? - NO
    plt.tight_layout()
    filename = 's2s_iodide_scalar_emissions_{}day_pcent_formated'
    plt.savefig(filename.format(num_days), dpi=dpi)
    AC.close_plot()


def add_Inorg_and_Org_totals2array(ds, InOrgVar='Inorg_Total',
                                   OrgVar='Org_Total'):
    """
    Add inorganic and organic sub totals to dataset
    """
    # Add aggregated values to ds
    OrgVars = [
        'EmisCH2IBr_Ocean', 'EmisCH2ICl_Ocean', 'EmisCH2I2_Ocean',
        'EmisCH3I_Ocean',
    ]
    InOrgVars = ['EmisI2_Ocean', 'EmisHOI_Ocean', ]
    # - Inorganic
    # Template off the first species
    ds[InOrgVar] = ds[InOrgVars[0]].copy()
    # Sum values to this
    arr = ds[InOrgVar].values
    for var_ in InOrgVars[1:]:
        print(var_)
        arr = arr + ds[var_].values
    ds[InOrgVar].values = arr
    attrs = ds[InOrgVar].attrs
    attrs['long_name'] = InOrgVar
    ds[InOrgVar].attrs = attrs
    # - Organic
    # Template off the first species
    ds[OrgVar] = ds[OrgVars[0]].copy()
    # Sum values to this
    arr = ds[OrgVar].values
    for var_ in OrgVars[1:]:
        print(var_)
        arr = arr + ds[var_].values
    ds[OrgVar].values = arr
    attrs = ds[OrgVar].attrs
    attrs['long_name'] = OrgVar
    ds[OrgVar].attrs = attrs
    return ds


def get_HEMCO_diags_as_ds_dict4runs(wds=None, dates2use=None,
                                    convert2Gg_yr=True):
    """
    Get the emissions from the HEMCO NetCDF files as a dictionary of datasets.
    """
    # Look at emissions through HEMCO
    # Get data locations and run names as a dictionary
    if isinstance(wds, type(None)):
        wds = get_run_dict4EGU_runs()
    runs = list(wds.keys())
    # Define halogen emission variables to consider
#    vars2use = [i for i in dsDH[run].data_vars if 'I' in i ]
    vars2use = [
        'EmisCH2IBr_Ocean', 'EmisCH2ICl_Ocean', 'EmisCH2I2_Ocean',
        'EmisCH3I_Ocean', 'EmisI2_Ocean', 'EmisHOI_Ocean',
    ]
    # Loop and extract files
    dsDH = {}
    for run in runs:
        wd = wds[run]
        print(run, wd)
        ds = AC.get_HEMCO_diags_as_ds(wd=wd, dates2use=dates2use)
        # Only consider variables of interest (and AREA)
        ds = ds[ vars2use + ['AREA'] ]
        # Average over time
        ds = ds.mean(dim='time', keep_attrs=True)
        dsDH[run] = ds
    # Get actual species
    specs = [i.split('Emis')[-1].split('_')[0] for i in vars2use]
    var_spec_dict = dict(zip(vars2use, specs))
    # Convert to Gg
    if convert2Gg_yr:
        for run in runs:
            ds = dsDH[run]
            ds = AC.convert_HEMCO_ds2Gg_per_yr(ds, vars2convert=vars2use,
                                               var_species_dict=var_spec_dict)
            dsDH[run] = ds
    return dsDH


def Get_GEOSChem_run_dict( version='v12.9.1', RunSet='scalar_runs'):
    """
    retrieve a dictionary of runs directories for GEOS-chem model output
    """
    run_root = utils.get_file_locations('run_root')
    # Select the runs for a given version (and optional set of runs)
    if version=='v12.9.1':
        if RunSet == 'scalar_runs':
            # These are the runs used for Carpenter et al 2020
            RunStr = 'merra2_4x5_standard.v12.9.1.BASE.Oi.{}{}'
            IodideField = 'Sherwen2019'
#            scalars = 'x0.90', 'x0.95', 'x1.00', 'x1.05', 'x1.10', 'x1.25'
            scalars = 'x0.90', 'x0.95', 'x1.0', 'x1.05', 'x1.1', 'x1.25'
            FilenameScalars = [i.replace('.', '_') for i in scalars]
            Runs = [RunStr.format(IodideField, i) for i in FilenameScalars]
            FullRoots = ['{}/{}/'.format(run_root,i) for i in Runs]
            FullNames = ['{}{}'.format(IodideField,i) for i in scalars]
            #
            d = dict( zip(FullNames, FullRoots) )
        elif RunSet == 'emission_options':
            # These are the runs used for Sherwen et al 2020
            RunStr = 'merra2_4x5_standard.v12.9.1.BASE.Oi.{}'
            options = ['Chance2014', 'Sherwen2019', ]
            options += ['Wadley2020','Hughes2020', 'MacDonald2014']
            d = {}
            PathStr = '/{}/{}.AllOutput/'
            for option in options:
                d[option] = PathStr.format(run_root, RunStr.format(option))
#            suffix = 'MacDonald2014.tagged/'
#            d['MacDonald2014'] = run_root+RunStr.format(suffix)
            suffix ='Sherwen2019.scaledx50/'
            d['Sherwen2019x0.5'] = run_root+RunStr.format(suffix)
        else:
            pass
    return d
