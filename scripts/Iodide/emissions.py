"""
Analysis of emissions of updates ocean iodine fields
"""

import numpy as np
import pandas as pd
import xarray as xr
import sparse2spatial.utils as utils

# import AC_tools (https://github.com/tsherwen/AC_tools.git)
import AC_tools as AC


def get_stats_on_scalar_emission_runs(dpi=320):
    """
    """
    # Get dictionary of model runs
    d = Get_GEOSChem_run_dict(version='v12.9.1', RunSet='scalar_runs')
    #
    sdate = datetime.datetime(2016, 1, 1, 0, 0)
    edate = datetime.datetime(2016, 1, 7, 0, 0)
    dates2use = pd.date_range(sdate, edate, freq='24H')
    #
    keys2use = d.keys()
    dNetCDF = {}
    for key in keys2use:
        dNetCDF[key] = d[key]+'/OutputDir/'
    # Now retrieve HEMCO files
    dsD = get_HEMCO_diags_as_ds_LOCAL(wds=dNetCDF, dates2use=dates2use)
    #
    for key in keys2use:
        dsD[key] = add_Inorg_and_Org_totals2array(dsD[key])

    # -  Now plot up global totals
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
    # Now plot...
    fig, ax = plt.subplots(dpi=dpi)
    df[vars2plot].plot()
    units = 'Gg yr$^{-1}$'
    title_str = "$\Delta$ emissions ({}) with scaling of '{}' iodide field"
    plt.title(title_str.format(units, IodideField))
    plt.xlabel('% change in global iodide field')
    plt.ylabel('$\Delta$ iodide flux ({})'.format(units))
    plt.legend()
    plt.tight_layout()
    filename = 's2s_iodide_scalar_emissions'
    plt.savefig(filename, dpi=dpi)
    AC.close_plot()

    # plot the same values as a percent
    units = '%'
    REF = 0
    vars2plot = [InOrgVar, OrgVar, TotalVar, 'HOI', 'I2']
    for var in vars2plot:
        vals = (df.loc[:,var] - df.loc[REF,var])/df.loc[REF,var]*100
        df.loc[:,var] = vals
    # Now plot...
    fig, ax = plt.subplots(dpi=dpi)
    df[vars2plot].plot()
    title_str = "$\Delta$ emissions ({}) with scaling of '{}' iodide field"
    plt.title(title_str.format(units, IodideField))
    plt.xlabel('% change in global iodide field')
    plt.ylabel('$\Delta$ iodide flux ({})'.format(units))
    fig.legend(loc=7)
    plt.tight_layout()
    filename = 's2s_iodide_scalar_emissions_pcent'
    plt.savefig(filename, dpi=dpi)
    AC.close_plot()


def get_HEMCO_files4run(folder):
    """
    """
    ds = AC.get_HEMCO_diags_as_ds(wd=folder)

    # Dates to use?
    # First 3 days

    #

    return ds


def Get_GEOSChem_run_dict( version='v12.9.1', RunSet='scalar_runs'):
    """
    retrieve a dictionary of runs directories for GEOS-chem model output
    """
    #
    run_root = utils.get_file_locations('run_root')
    # Select the runs for a given version (and optional set of runs)
    if version=='v12.9.1':
        if RunSet == 'scalar_runs':
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
        else:
            pass

    return d


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
    # template off the first species
    ds[InOrgVar] = ds[InOrgVars[0]].copy()
    # sum values to this
    arr = ds[InOrgVar].values
    for var_ in InOrgVars[1:]:
        print(var_)
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
        print(var_)
        arr = arr + ds[var_].values
    ds[OrgVar].values = arr
    attrs = ds[OrgVar].attrs
    attrs['long_name'] = OrgVar
    ds[OrgVar].attrs = attrs
    return ds


def get_HEMCO_diags_as_ds_LOCAL(wds=None, dates2use=None):
    """
    Get the emissions from the HEMCO NetCDF files as a dictionary of datasets.
    """
    # Look at emissions through HEMCO
    # Get data locations and run names as a dictionary
    if isinstance(wds, type(None)):
        wds = get_run_dict4EGU_runs()
    runs = list(wds.keys())
    #
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
    var_species_dict = dict(zip(vars2use, specs))
    # Convert to Gg
    for run in runs:
        ds = dsDH[run]
#        ds = AC.Convert_HEMCO_ds2Gg_per_yr(ds, vars2convert=vars2use,
#                                           var_species_dict=var_species_dict)
        # Import failing - use local function for now
        ds = convert_HEMCO_ds2Gg_per_yr_LOCAL(ds, vars2convert=vars2use,
                                           var_species_dict=var_species_dict)
        dsDH[run] = ds
    return dsDH



def convert_HEMCO_ds2Gg_per_yr_LOCAL(ds, vars2convert=None,
                                     var_species_dict=None,
                                     output_freq='End', verbose=False,
                                     debug=False):
    """
    Convert emissions in HEMCO dataset to mass/unit time

    vars2convert (list), NetCDF vairable names to convert
    var_species_dict (dict), dictionary to map variables names to chemical species
    output_freq (str), output frequency dataset made from HEMCO NetCDF file output

    """
    # Get chemical species for each variable name
    var_species = {}
    for var in vars2convert:
        try:
            var_species[var] = var_species_dict[var]
        except:
            #            if verbose:
            PrtStr = "WARNING - using variable name '{}' as chemical species!"
            print(PrtStr.format(var))
            var_species[var] = var
    # Print assumption about end units.
    if output_freq == 'End':
        print("WARNING - Assuming Output frequnecy ('End') is monthly")

    # Get equivalent unit for chemical species (e.g. I, Br, Cl, N, et c)
    ref_specs = {}
    for var in vars2convert:
        try:
            ref_specs[var] = AC.get_ref_spec(var_species[var])
        except KeyError:
            print("WARNING: Using '{}' as reference species for '{}'".format(var, var))
    # Loop dataset by variable
    for var_n, var_ in enumerate(vars2convert):
        if debug:
            print('{:<2} {} '.format(var_n, var_))
        # Extract the variable array
        try:
            arr = ds[var_].values
        except KeyError:
            print("WARNING: skipping variable '({})' as not in dataset".format(var_))
            continue

        # --- Adjust units to be in kg/gridbox
        # remove area units
        if ds[var_].units == 'kg/m2/':
            arr = arr * ds['AREA']
        elif ds[var_].units == 'kg/m2/s':
            # remove area units
            arr = arr * ds['AREA']
            # now remove seconds
            convert_unaveraged_time = False
            if convert_unaveraged_time:
                if output_freq == 'Hourly':
                    arr = arr*60.*60.
                elif output_freq == 'Daily':
                    arr = arr*60.*60.*24.*(365.)
                elif output_freq == 'Weekly':
                    arr = arr*60.*60.*24.*(365./52.)
                elif (output_freq == 'Monthly') or (output_freq == 'End'):
                    arr = arr*60.*60.*24.*(365./12.)
                else:
                    print('WARNING: ({}) output convert. unknown'.format(
                        output_freq))
                    sys.exit()
            else:
                arr = arr*60.*60.*24.*365.
        elif ds[var_].units == 'kg':
            pass  # units are already in kg .
        else:
            print('WARNING: unit convert. ({}) unknown'.format(ds[var_].units))
            sys.exit()
        # --- convert to Gg species
        # get spec name for output variable
        spec = var_species[var_]
        # Get equivalent unit for species (e.g. I, Br, Cl, N, et c)
        ref_spec = ref_specs[var_]
        # get stoichiometry of ref_spec in species
        stioch = AC.spec_stoich(spec, ref_spec=ref_spec)
        RMM_spec = AC.species_mass(spec)
        RMM_ref_spec = AC.species_mass(ref_spec)
        # update values in array
        arr = arr / RMM_spec * RMM_ref_spec * stioch
        # (from kg=>g (*1E3) to g=>Gg (/1E9))
        arr = arr*1E3 / 1E9
        if set(ref_specs) == 1:
            units = '(Gg {})'.format(ref_spec)
        else:
            units = '(Gg X)'
        if debug:
            print(arr.shape)
        # reassign arrary
        ds[var_].values = arr
        # Update units too
        attrs = ds[var_].attrs
        attrs['units'] = units
        ds[var_].attrs = attrs
    return ds

