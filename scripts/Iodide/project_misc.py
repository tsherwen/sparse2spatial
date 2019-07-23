"""

This module contains analysis done for the Ocean iodide (Oi!) project

This includes presentation at conferences etc...

"""

import numpy as np
import pandas as pd
import sparse2spatial as s2s
import sparse2spatial.utils as utils

# import AC_tools (https://github.com/tsherwen/AC_tools.git)
import AC_tools as AC

# Get iodide specific functions
import observations as obs

def main():
    """
    Run various misc. scripted tasks linked to the "iodide in the ocean" project
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

    # process this to csv files for Indian' sea-surface paper


# ---------------------------------------------------------------------------
# ---------- Functions to produce output for Iodide obs. paper -------------
# ---------------------------------------------------------------------------
def get_PDF_of_iodide_exploring_data_rootset(show_plot=False,
                                             ext_str=None):
    """ Get PDF of plots exploring the iodide dataset """
    import seaborn as sns
    sns.set(color_codes=True)
    # Get the data
    df = obs.get_processed_df_obs_mod()  # NOTE this df contains values >400nM
    #
    if ext_str == 'Open_ocean':
        # Kludge data
        #        Kludge_tinel_data=True
        #         if Kludge_tinel_data:
        #             new_Data = [ 'He_2014', 'He_2013']
        #             new_Data += ['Chance_2018_'+i for i in 'I', 'II', 'III']
        #             df.loc[ df['Data_Key'].isin(new_Data), 'Coastal'] = False
        # only take data flagged open ocean
        df = df.loc[df[u'Coastal'] == 0.0, :]
    elif ext_str == 'Coastal':
        df = df.loc[df[u'Coastal'] == 1.0, :]
    elif ext_str == 'all':
        print('Using entire dataset')
    else:
        print('Need to set region of data to explore - currently', ext_str)
        sys.exit()

    # setup PDF
    savetitle = 'Oi_prj_data_root_exploration_{}'.format(ext_str)
    dpi = 320
    pdff = AC.plot2pdfmulti(title=savetitle, open=True, dpi=dpi)
    # colours to use?
#    current_palette = sns.color_palette()
    current_palette = sns.color_palette("colorblind")

    # --- --- --- --- --- --- --- ---
    # ---- Add in extra varibles
    # iodide / iodate
    I_div_IO3_var = 'I$^{-}$/IO$_{3}^{-}$ (ratio)'
    df[I_div_IO3_var] = df['Iodide'] / df['Iodate']
    # total iodide
    I_plus_IO3 = 'I$^{-}$+IO$_{3}^{-}$'
    df[I_plus_IO3] = df['Iodide'] + df['Iodate']

    # --- Add ocean basin to dataframe
    area_var = 'Region'
    df[area_var] = None
    # setup a dummy column

    # --- --- --- --- --- --- --- ---
    # --- Plot dataset locations
    sns.reset_orig()
    # Get lats, lons and size of dataset
    lats = df['Latitude'].values
    lons = df['Longitude'].values
    N_size = df.shape[0]
    if ext_str == 'Open_ocean':
        title = 'Iodide data (Open Ocean) explored in PDF (N={})'
    else:
        title = 'Iodide data (all) explored in this PDF (N={})'
    # plot up
    AC.plot_lons_lats_spatial_on_map(lats=lats, lons=lons,
                                     title=title.format(N_size),
                                     split_title_if_too_long=False,
                                     f_size=10)
    # Save to PDF and close plot
    AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
    if show_plot:
        plt.show()
    plt.close()

    # --- --- --- --- --- --- --- ---
    # --- iodide to iodide ratio
    import seaborn as sns
    sns.set(color_codes=True)
    current_palette = sns.color_palette("colorblind")

    # plot up with no limits
    df.plot(kind='scatter', y=I_div_IO3_var, x='Latitude')
    # beautify
    plt.title(I_div_IO3_var + ' ({}, y axis unlimited)'.format(ext_str))
    # Save to PDF and close plot
    AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
    if show_plot:
        plt.show()
    plt.close()

    # plot up with limits at 3
    ylimits = 1.5, 0.75, 0.5,
    for ylimit in ylimits:
        df.plot(kind='scatter', y=I_div_IO3_var, x='Latitude')
        # beautify
        title = ' ({}, y axis limit: {})'.format(ext_str, ylimit)
        plt.title(I_div_IO3_var + title)
        plt.ylim(-0.05, ylimit)
        # Save to PDF and close plot
        AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
        if show_plot:
            plt.show()
        plt.close()

    # --- --- --- --- --- --- --- ---
    # TODO - update to use  proper definitions
    # for southern ocean use the files below
    # for  rest https://www.nodc.noaa.gov/woce/woce_v3/wocedata_1/woce-uot/summary/bound.htm
    #

    # --- iodide to iodide ratio ( split by region )
    # Between 120E and -80E its Pacific
    upper_val = 120
    lower_val = -80
    unit = '$^{o}$E'
    bool_1 = df[u'Longitude'] >= upper_val
    bool_2 = df[u'Longitude'] < lower_val
    bool = (np.column_stack((bool_2, bool_1)).any(axis=1))
    varname = 'Pacific Ocean ({} to {}{})'.format(upper_val, lower_val, unit)
    df.loc[bool, area_var] = varname

    # Between -80E and 30E its Atlantic
    upper_val = -80
    lower_val = 30
    unit = '$^{o}$E'
    bool_1 = df[u'Longitude'] >= upper_val
    bool_2 = df[u'Longitude'] < lower_val
    bool = (np.column_stack((bool_2, bool_1)).all(axis=1))
    varname = 'Atlantic Ocean ({} to {}{})'.format(lower_val, upper_val, unit)
    df.loc[bool, area_var] = varname

    # Between 30E and 120E its Indian
    upper_val = 30
    lower_val = 120
    unit = '$^{o}$E'
    bool_1 = df[u'Longitude'] >= upper_val
    bool_2 = df[u'Longitude'] < lower_val
    bool = (np.column_stack((bool_2, bool_1)).all(axis=1))
    varname = 'Indian Ocean ({} to {}{})'.format(lower_val, upper_val, unit)
    df.loc[bool, area_var] = varname

    # if latitude below 60S, overwrite to be Southern ocean
    varname = 'Southern Ocean'
    df.loc[df['Latitude'] < -60, area_var] = varname

    # --- --- --- --- --- --- --- ---
    # ---  locations of data
    sns.reset_orig()
    # loop regions
    for var_ in list(set(df[area_var].tolist())):
        # select data for area
        df_tmp = df[df[area_var] == var_]
        # locations ?
        lons = df_tmp[u'Longitude'].tolist()
        lats = df_tmp[u'Latitude'].tolist()
        # Now plot
        AC.plot_lons_lats_spatial_on_map(lons=lons, lats=lats)
 #           fig=fig, ax=ax , color='blue', label=label, alpha=alpha,
#            window=window, axis_titles=axis_titles, return_axis=True,
#            p_size=p_size)
        plt.title('{} ({})'.format(var_, ext_str))
        if show_plot:
            plt.show()

        # Save to PDF and close plot
        AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
        plt.close()

    # --- --- --- --- --- --- --- ---
    # --- iodide to iodide ratio
    import seaborn as sns
    sns.set(color_codes=True)
    current_palette = sns.color_palette("colorblind")

    # loop regions
    for var_ in list(set(df[area_var].tolist())):
        # select data for area
        df_tmp = df[df[area_var] == var_]
        # plot up with no limits
        df_tmp.plot(kind='scatter', y=I_div_IO3_var, x='Latitude')
        # beautify
        plt.title(I_div_IO3_var + ' ({}, y axis unlimited)'.format(var_))
        # Save to PDF and close plot
        AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
        if show_plot:
            plt.show()
        plt.close()

        # plot up with limits at 3
        ylimits = 1.5, 0.75, 0.5
        for ylimit in ylimits:
            df_tmp.plot(kind='scatter', y=I_div_IO3_var, x='Latitude')
            # beautify
            title = ' ({}, y axis limit: {})'.format(var_, ylimit)
            plt.title(I_div_IO3_var + title)
            plt.ylim(-0.05, ylimit)
            # Save to PDF and close plot
            AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
            if show_plot:
                plt.show()
            plt.close()

    # --- --- --- --- --- --- --- ---
    # --- iodide + iodide
    import seaborn as sns
    sns.set(color_codes=True)
    current_palette = sns.color_palette("colorblind")

    # loop regions
    for var_ in list(set(df[area_var].tolist())):
        # select data for area
        df_tmp = df[df[area_var] == var_]
        # plot up with no limits
        df_tmp.plot(kind='scatter', y=I_plus_IO3, x='Latitude')
        # beautify
        plt.title(I_plus_IO3 + ' ({}, y axis unlimited)'.format(var_))
        # Save to PDF and close plot
        AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
        if show_plot:
            plt.show()
        plt.close()

        # plot up with limits at 3
#         ylimits = 1.5, 0.75, 0.5
#         for ylimit in ylimits:
#             df.plot(kind='scatter', y=I_plus_IO3, x='Latitude' )
#             # beautify
#             title= ' ({}, y axis limited to {})'.format(var_, ylimit)
#             plt.title( I_plus_IO3 + title )
#             plt.ylim(-0.05, ylimit )
#             # Save to PDF and close plot
#             AC.plot2pdfmulti( pdff, savetitle, dpi=dpi )
#             if show_plot: plt.show()
#             plt.close()

        # plot up with limits on y
        ylimits = [100, 600]
#        for ylimit in ylimits:
        df_tmp.plot(kind='scatter', y=I_plus_IO3, x='Latitude')
        # beautify
        title = ' ({}, y axis={}-{})'.format(var_, ylimits[0], ylimits[1])
        plt.title(I_plus_IO3 + title)
        plt.ylim(ylimits[0], ylimits[1])
        # Save to PDF and close plot
        AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
        if show_plot:
            plt.show()
        plt.close()

    # -- Save entire pdf
    AC.plot2pdfmulti(pdff, savetitle, close=True, dpi=dpi)


# ---------------------------------------------------------------------------
# ---------- Funcs. to process iodine obs/external data --------------------
# ---------------------------------------------------------------------------
def check_points_for_cruises(target='Iodide', verbose=False, debug=False):
    """
    Check the cruise points for the new data (Tinel, He, etc...)
    """
    # Get the observational data
    df = obs.get_processed_df_obs_mod()  # NOTE this df contains values >400nM
    # And the metadata
    metadata_df = obs.get_iodide_obs_metadata()
    # Only consider new datasets
    new_cruises = metadata_df[metadata_df['In Chance2014?'] == 'N']
    df = df[df['Data_Key'].isin(new_cruises['Data_Key'].tolist())]
    # Strings to format printing
    ptr_str_I = '- '*5 + 'Cruise: {:<20}'
    ptr_str_II = '(Source: {:<20}, Location: {:<15}, N: {}, N(Iodide): {})'
    # Print by cruise
    for data_key in set(df['Data_Key']):
        df_m_tmp = metadata_df[metadata_df['Data_Key'] == data_key]
        df_tmp = df[df['Data_Key'] == data_key]
        # Extract metadata
        Cruise = df_m_tmp['Cruise'].values[0]
        Source = df_m_tmp['Source'].values[0]
        Location = df_m_tmp['Location'].values[0]
        #
        N = df_tmp.shape[0]
        N_I = df_tmp[target].dropna().shape[0]
        print(ptr_str_I.format(Cruise))
        print(ptr_str_II.format(Source, Location, N, N_I))
    # Points for all cruises
    N = df.shape[0]
    N_I = df[target].dropna().shape[0]
    print(ptr_str_I.format('ALL new data'))
    print(ptr_str_II.format('', '', N, N_I))


def plot_threshold_plus_SD_spatially(var=None, value=None, std=None, res='4x5',
                                     fillcontinents=True, show_plot=False,
                                     dpi=320, save2png=True,
                                     verbose=True, debug=False):
    """
    Plot up the spatial extent of a input variable value + Std. Dev.
    """
    # - Local variables
    # Get the core input variables
    data_root = utils.get_file_locations('data_root')
    filename = 'Oi_prj_feature_variables_{}.nc'.format(res)
    ds = xr.open_dataset(data_root + filename)
    # make sure the dataset has units
    ds = add_units2ds(ds)
    # Use appropriate plotting settings for resolution
    if res == '0.125x0.125':
        centre = True
    else:
        centre = False
    # Get data
    arr = ds[var].mean(dim='time').values
    # colour in values above and below threshold (works)
    arr[arr >= value] = 1
    arr[arr >= value-std] = 0.5
    arr[(arr != 1) & (arr != 0.5)] = 0.01
    # Get units from dataset
    units = ds[var].units
    # Plot up
    title_str = "'{}' ({}) threshold Value ({}) +  \n Standard deviation ({})"
    title = title_str.format(var, units, value, std)
    if var == 'WOA_TEMP_K':
        title += ' (in degC={}, std={})'.format(value-273.15, std)
    # Plot using AC_tools
    AC.plot_spatial_figure(arr,
                           #        extend=extend,
                           #        fixcb=fixcb, nticks=nticks, \
                           res=res, show=False, title=title, \
                           fillcontinents=fillcontinents, centre=centre, units=units,
                           #        f_size=f_size,
                           no_cb=False)
    # Use a tight layout
    plt.tight_layout()
    # Now save or show
    if show_plot:
        plt.show()
    savetitle = 'Oi_prj_threshold_std_4_var_{}_{}'.format(var, res)
    if save2png:
        plt.savefig(savetitle+'.png', dpi=dpi)
    plt.close()


# ---------------------------------------------------------------------------
# -------------- Reproduction of Chance et al (2014) figures ----------------
# ---------------------------------------------------------------------------
def plot_up_iodide_vs_latitude(show_plot=True):
    """
    Reproduce Fig. 3 in Chance et al (2014)

    Notes
    ----
     - figure captions:
     Variation of sea-surface iodide concentration with latitude for entire
     data set (open diamonds) and open ocean data only (filled diamonds).
    For clarity, one exceptionally high coastal iodide value (700 nM, 58.25N)
    has been omitted.
    """
    # -  Get data
    df = get_core_Chance2014_obs()
    # Select data of interest
    # ( later add a color selection based on coastal values here? )
    vars = ['Iodide', 'Latitude']
    print(df)
    # and select coastal/open ocean
    df_coastal = df[df['Coastal'] == True][vars]
    df_open_ocean = df[~(df['Coastal'] == True)][vars]
    # - Now plot Obs.
    # plot coastal
    ax = df_coastal.plot(kind='scatter', x='Latitude', y='Iodide', marker='D',
                         color='blue', alpha=0.1,
                         #        markerfacecolor="None", **kwds )
                         )
    # plot open ocean
    ax = df_open_ocean.plot(kind='scatter', x='Latitude', y='Iodide',
                            marker='D', color='blue', alpha=0.5, ax=ax,
                            #        markerfacecolor="None", **kwds )
                            )
    # Update aesthetics of plot
    plt.ylabel('[Iodide], nM')
    plt.xlabel('Latitude, $^{o}$N')
    plt.ylim(-5, 500)
    plt.xlim(-80, 80)
    # save or show?
    if show_plot:
        plt.show()
    plt.close()


def plot_up_ln_iodide_vs_Nitrate(show_plot=True):
    """
    Reproduc Fig. 11 in Chance et al (2014)

    Original caption:

    Ln[iodide] concentration plotted against observed ( ) and
    climatological ( ) nitrate concentration obtained from the World
    Ocean Atlas as described in the text for all data (A) and nitrate
    concentrations below 2 mM (B) and above 2 mM (C). Dashed lines in B
    and C show the relationships between iodide and nitrate adapted from
    Campos et al.41 by Ganzeveld et al.27
    """
    #  - location of data to plot
    df = obs.get_processed_df_obs_mod()
    # take log of iodide
    df['Iodide'] = np.log(df['Iodide'].values)
    # - Plot up all nitrate concentrations
    df.plot(kind='scatter', x='Nitrate', y='Iodide', marker='D',
            color='k')  # ,
    plt.ylabel('LN[Iodide], nM')
    plt.xlabel('LN[Nitrate], mM')
    if show_plot:
        plt.show()
    plt.close()
    # - Plot up all nitrate concentrations below 2 mM
    df_tmp = df[df['Nitrate'] < 2]
    df_tmp.plot(kind='scatter', x='Nitrate', y='Iodide', marker='D',
                color='k')  # ,
    plt.ylabel('LN[Iodide], nM')
    plt.xlabel('LN[Nitrate], mM')
    if show_plot:
        plt.show()
    plt.close()
    # - Plot up all nitrate concentrations above 2 mM
    df_tmp = df[df['Nitrate'] > 2]
    df_tmp.plot(kind='scatter', x='Nitrate', y='Iodide', marker='D',
                color='k'),
    plt.ylabel('LN[Iodide], nM')
    plt.xlabel('LN[Nitrate], mM')
    if show_plot:
        plt.show()
    plt.close()


def plot_up_ln_iodide_vs_SST(show_plot=True):
    """
    Reproduc Fig. 8 in Chance et al (2014)

    Original caption:

    Ln[iodide] concentration plotted against observed sea surface
    temperature ( ) and climatological sea surface temperature ( ) values
    obtained from the World Ocean Atlas as described in the text.
    """
    # - location of data to plot
    folder = utils.get_file_locations('data_root')
    f = 'Iodine_obs_WOA.csv'
    df = pd.read_csv(folder+f, encoding='utf-8')
    # take log of iodide
    df['Iodide'] = np.log(df['Iodide'].values)
    # - Plot up all nitrate concentrations
    df.plot(kind='scatter', x='Temperature', y='Iodide', marker='D',
            color='k')
    plt.ylabel('LN[Iodide], nM')
    plt.xlabel('Sea surface temperature (SST), $^{o}$C')
    if show_plot:
        plt.show()
    plt.close()


def plot_up_ln_iodide_vs_salinity(show_plot=True):
    """
    Reproduc Fig. 8 in Chance et al (2014)

    Original caption:

    Ln[iodide] concentration plotted against observed salinity ( , ) and
    climatological salinity ( ) values obtained from the World Ocean Atlas as
    described in the text for: (A) all data; (B) samples with salinity greater
    than 30, shown in shaded area in (A). Note samples with salinity less than
    30 have been excluded from further analysis and are not shown in Fig. 8–11.
    """
    # - location of data to plot
    folder = utils.get_file_locations('data_root')
    f = 'Iodine_obs_WOA.csv'
    df = pd.read_csv(folder+f, encoding='utf-8')
    # Just select non-coastal data
#    df = df[ ~(df['Coastal']==True) ]
    # take log of iodide
    df['Iodide'] = np.log(df['Iodide'].values)
    # - Plot up all nitrate concentrations
    df.plot(kind='scatter', x='Salinity', y='Iodide', marker='D', color='k')
    plt.ylabel('LN[Iodide], nM')
    plt.xlabel('Salinity')
    plt.xlim(-2, AC.myround(max(df['Salinity']), 10, round_up=True))
    if show_plot:
        plt.show()
    plt.close()
    # - Plot up all nitrate concentrations
    df_tmp = df[df['Salinity'] < 30]
    df_tmp.plot(kind='scatter', x='Salinity',
                y='Iodide', marker='D', color='k')
    plt.ylabel('LN[Iodide], nM')
    plt.xlabel('Salinity')
    plt.xlim(-2, AC.myround(max(df['Salinity']), 10, round_up=True))
    if show_plot:
        plt.show()
    plt.close()
    # - Plot up all nitrate concentrations
    df_tmp = df[df['Salinity'] > 30]
    df_tmp.plot(kind='scatter', x='Salinity',
                y='Iodide', marker='D', color='k')
    plt.ylabel('LN[Iodide], nM')
    plt.xlabel('Salinity')
    plt.xlim(29, AC.myround(max(df['Salinity']), 10, round_up=True))
    if show_plot:
        plt.show()
    plt.close()


def plot_pair_grid(df=None, vars_list=None):
    """
    Make a basic pair plot to test the data
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    from itertools import cycle
    # make a kde plot
    def make_kde(*args, **kwargs):
        sns.kdeplot(*args, cmap=next(make_kde.cmap_cycle), **kwargs)
    # define colormap to cycle
    make_kde.cmap_cycle = cycle(('Blues_r', 'Greens_r', 'Reds_r', 'Purples_r'))
    # Plot a pair plot
    pg = sns.PairGrid(data, vars=vars_list)


# ---------------------------------------------------------------------------
# ---------------- New plotting of iodine obs/external data -----------------
# ---------------------------------------------------------------------------
def explore_extracted_data_in_Oi_prj_explore_Arctic_Antarctic_obs(dsA=None,
                                                                  res='0.125x0.125',
                                                                  dpi=320):
    """
    Analyse the gridded data for the Arctic and Antarctic
    """
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.style.use('ggplot')
    import seaborn as sns
    sns.set()
    # - local variables
    # Get input variables
    if isinstance(dsA, type(None)):
        filename = 'Oi_prj_predicted_iodide_{}.nc'.format(res)
#        folder = '/shared/earth_home/ts551/labbook/Python_progs/'
        folder = '/shared/earth_home/ts551/data/iodide/'
        filename = 'Oi_prj_feature_variables_{}.nc'.format(res)
        dsA = xr.open_dataset(folder + filename)
#        ds = xr.open_dataset( filename )
    # variables to consider
    vars2analyse = list(dsA.data_vars)
    # Add LWI to array - NOTE: 1 = water in Nature run LWI files !
    # ( The above comment is not correct! why is this written here? )
    folderLWI = utils.get_file_locations(
        'AC_tools')+'/data/LM/TEMP_NASA_Nature_run/'
    filenameLWI = 'ctm.nc'
    LWI = xr.open_dataset(folderLWI+filenameLWI)
    # updates dates (to be Jan=>Dec)
    new_dates = [datetime.datetime(1970, i, 1) for i in LWI['time.month']]
    LWI.time.values = new_dates
    # Sort by new dates
    LWI = LWI.loc[{'time': sorted(LWI.coords['time'].values)}]
#    LWI = AC.get_LWI_map(res=res)[...,0]
    dsA['IS_WATER'] = dsA['WOA_TEMP'].copy()
    dsA['IS_WATER'].values = (LWI['LWI'] == 0)
    # add is land
    dsA['IS_LAND'] = dsA['IS_WATER'].copy()
    dsA['IS_LAND'].values = (LWI['LWI'] == 1)
    # get surface area
    s_area = AC.calc_surface_area_in_grid(res=res)  # m2 land map
    dsA['AREA'] = dsA['WOA_TEMP'].mean(dim='time')
    dsA['AREA'].values = s_area.T

    # - Select data of interest by variable for locations
    # setup dicts to store the extracted values
    df65N, df65S, dfALL = {}, {}, {}
    # - setup booleans for the data
    # now loop and extract variablesl
    vars2use = [
        'WOA_Nitrate',
        #    'WOA_Salinity', 'WOA_Phosphate', 'WOA_TEMP_K', 'Depth_GEBCO',
    ]
    # setup PDF
    savetitle = 'Oi_prj_explore_Arctic_Antarctic_ancillaries_space_PERTURBED'
    pdff = AC.plot2pdfmulti(title=savetitle, open=True, dpi=dpi)
    # Loop by dataset (region) and plots
    for var_ in vars2use:
        # select the boolean for if water
        IS_WATER = dsA['IS_WATER'].values
        if IS_WATER.shape != dsA[var_].shape:
            # special case for depth
            # get value for >= 65
            ds_tmp = dsA.sel(lat=(dsA['lat'] >= 65))
            arr = np.ma.array(12*[ds_tmp[var_].values])
            arr = arr[ds_tmp['IS_WATER'].values]
            # add to saved arrays
            df65N[var_] = arr
            del ds_tmp
            # get value for >= 65
            ds_tmp = dsA.sel(lat=(dsA['lat'] <= -65))
            arr = np.ma.array(12*[ds_tmp[var_].values])
            arr = arr[ds_tmp['IS_WATER'].values]
            # add to saved arrays
            df65S[var_] = arr
            del ds_tmp
            # get value for all
            ds_tmp = dsA.copy()
            arr = np.ma.array(12*[ds_tmp[var_].values])
            arr = arr[ds_tmp['IS_WATER'].values]
            # add to saved arrays
            dfALL[var_] = arr
            del ds_tmp
        else:
            # get value for >= 65
            ds_tmp = dsA.sel(lat=(dsA['lat'] >= 65))
            arr = ds_tmp[var_].values[ds_tmp['IS_WATER'].values]
            # add to saved arrays
            df65N[var_] = arr
            del ds_tmp
            # get value for >= 65
            ds_tmp = dsA.sel(lat=(dsA['lat'] <= -65))
            arr = ds_tmp[var_].values[ds_tmp['IS_WATER'].values]
            # add to saved arrays
            df65S[var_] = arr
            del ds_tmp
            # get value for >= 65
            ds_tmp = dsA.copy()
            arr = ds_tmp[var_].values[ds_tmp['IS_WATER'].values]
            # add to saved arrays
            dfALL[var_] = arr
            del ds_tmp

    # setup a dictionary of regions to plot from
    dfs = {
        '>=65N': pd.DataFrame(df65N), '>=65S': pd.DataFrame(df65S),
        'Global': pd.DataFrame(dfALL),
    }

    # - plot up the PDF distribution of each of the variables.
    for var2use in vars2use:
        print(var2use)
        # set a single axis to use.
        fig, ax = plt.subplots()
        for dataset in datasets:
            # select the DataFrame
            df = dfs[dataset][var2use]
            # Get sample size
            N_ = df.shape[0]
            # do a dist plot
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


def explore_extracted_data_in_Oi_prj_explore_Arctic_Antarctic_obs(dsA=None,
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
    # - local variables
    # get input variables
    if isinstance(dsA, type(None)):
        filename = 'Oi_prj_predicted_iodide_{}.nc'.format(res)
#        folder = '/shared/earth_home/ts551/labbook/Python_progs/'
        folder = '/shared/earth_home/ts551/data/iodide/'
        filename = 'Oi_prj_feature_variables_{}.nc'.format(res)
        dsA = xr.open_dataset(folder + filename)
#        ds = xr.open_dataset( filename )
    # variables to consider
    vars2analyse = list(dsA.data_vars)
    # add LWI to array - NOTE: 1 = water in Nature run LWI files !
    # ( The above comment is not correct! why is this written here? )
    folderLWI = utils.get_file_locations(
        'AC_tools')+'/data/LM/TEMP_NASA_Nature_run/'
    filenameLWI = 'ctm.nc'
    LWI = xr.open_dataset(folderLWI+filenameLWI)
    # updates dates (to be Jan=>Dec)
    new_dates = [datetime.datetime(1970, i, 1) for i in LWI['time.month']]
    LWI.time.values = new_dates
    # Sort by new dates
    LWI = LWI.loc[{'time': sorted(LWI.coords['time'].values)}]
#    LWI = AC.get_LWI_map(res=res)[...,0]
    dsA['IS_WATER'] = dsA['WOA_TEMP'].copy()
    dsA['IS_WATER'].values = (LWI['LWI'] == 0)
    # add is land
    dsA['IS_LAND'] = dsA['IS_WATER'].copy()
    dsA['IS_LAND'].values = (LWI['LWI'] == 1)
    # get surface area
    s_area = AC.calc_surface_area_in_grid(res=res)  # m2 land map
    dsA['AREA'] = dsA['WOA_TEMP'].mean(dim='time')
    dsA['AREA'].values = s_area.T
    # - Select data of interest by variable for locations
    # setup dicts to store the extracted values
    df65N, df65S, dfALL = {}, {}, {}
    # - setup booleans for the data
    # now loop and extract variablesl
    vars2use = [
        'WOA_Nitrate', 'WOA_Salinity', 'WOA_Phosphate', 'WOA_TEMP_K', 'Depth_GEBCO',
    ]
    for var_ in vars2use:
        # select the boolean for if water
        IS_WATER = dsA['IS_WATER'].values
        if IS_WATER.shape != dsA[var_].shape:
            # special case for depth
            # get value for >= 65
            ds_tmp = dsA.sel(lat=(dsA['lat'] >= 65))
            arr = np.ma.array(12*[ds_tmp[var_].values])
            arr = arr[ds_tmp['IS_WATER'].values]
            # add to saved arrays
            df65N[var_] = arr
            del ds_tmp
            # get value for >= 65
            ds_tmp = dsA.sel(lat=(dsA['lat'] <= -65))
            arr = np.ma.array(12*[ds_tmp[var_].values])
            arr = arr[ds_tmp['IS_WATER'].values]
            # add to saved arrays
            df65S[var_] = arr
            del ds_tmp
            # get value for all
            ds_tmp = dsA.copy()
            arr = np.ma.array(12*[ds_tmp[var_].values])
            arr = arr[ds_tmp['IS_WATER'].values]
            # add to saved arrays
            dfALL[var_] = arr
            del ds_tmp
        else:
            # get value for >= 65
            ds_tmp = dsA.sel(lat=(dsA['lat'] >= 65))
            arr = ds_tmp[var_].values[ds_tmp['IS_WATER'].values]
            # add to saved arrays
            df65N[var_] = arr
            del ds_tmp
            # get value for >= 65
            ds_tmp = dsA.sel(lat=(dsA['lat'] <= -65))
            arr = ds_tmp[var_].values[ds_tmp['IS_WATER'].values]
            # add to saved arrays
            df65S[var_] = arr
            del ds_tmp
            # get value for >= 65
            ds_tmp = dsA.copy()
            arr = ds_tmp[var_].values[ds_tmp['IS_WATER'].values]
            # add to saved arrays
            dfALL[var_] = arr
            del ds_tmp

    # setup a dictionary of regions to plot from
    dfs = {
        '>=65N': pd.DataFrame(df65N), '>=65S': pd.DataFrame(df65S),
        'Global': pd.DataFrame(dfALL),
    }

    # - Loop regions and plot PDFs of variables of interest
#    vars2use = dfs[ dfs.keys()[0] ].columns
    # set PDF
    savetitle = 'Oi_prj_explore_Arctic_Antarctic_ancillaries_space'
    pdff = AC.plot2pdfmulti(title=savetitle, open=True, dpi=dpi)
    # Loop by dataset (region) and plots
    datasets = sorted(dfs.keys())
    for dataset in datasets:
        # select the DataFrame
        df = dfs[dataset][vars2use]
        # Get sample size
        N_ = df.shape[0]
        # do a pair plot
        g = sns.pairplot(df)
        # Add a title
        plt.suptitle("Pairplot for '{}' (N={})".format(dataset, N_))
        # adjust plots
        g.fig.subplots_adjust(top=0.925, left=0.085)
        # Save to PDF and close plot
        AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
        plt.close()

    # - Plot up the PDF distribution of each of the variables.
    for var2use in vars2use:
        print(var2use)
        # set a single axis to use.
        fig, ax = plt.subplots()
        for dataset in datasets:
            # select the DataFrame
            df = dfs[dataset][var2use]
            # Get sample size
            N_ = df.shape[0]
            # do a dist plot
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


def explore_observational_data_in_Arctic_parameter_space(RFR_dict=None,
                                                         plt_up_locs4var_conds=False,
                                                         testset='Test set (strat. 20%)',
                                                         dpi=320):
    """
    Analysis the input observational data for the Arctic and Antarctic
    """
    import matplotlib
#    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.style.use('ggplot')
    import seaborn as sns
    sns.set()

    # - local variables
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
        'WOA_Nitrate', 'WOA_Salinity', 'WOA_Phosphate', 'WOA_TEMP_K', 'Depth_GEBCO',
    ]

    # - Loop regions and plot pairplots of variables of interest
    # set PDF
    savetitle = 'Oi_prj_explore_Arctic_Antarctic_obs_space'
    pdff = AC.plot2pdfmulti(title=savetitle, open=True, dpi=dpi)
    # Loop by dataset (region) and plots
    datasets = sorted(dfs.keys())
    for dataset in datasets:
        # select the DataFrame
        df = dfs[dataset]
        # Get sample size
        N_ = df.shape[0]
        # do a pair plot
        g = sns.pairplot(df[vars2use])
        # Add a title
        plt.suptitle("Pairplot for '{}' (N={})".format(dataset, N_))
        # adjust plots
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
        # select the DataFrame
        dfA = dfs[dataset]
        # Set title
        title = "Locations for '{}'".format(dataset)
        p_size = 50
        alpha = 1
        # plot up Non coatal locs
        df = dfA.loc[dfA['Coastal'] == False, :]
        color = 'blue'
        label = 'Non-coastal (N={})'.format(int(df.shape[0]))
        m = AC.plot_lons_lats_spatial_on_map(title=title, f_size=15,
                                             lons=df['Longitude'].values,
                                             lats=df['Latitude'].values,
                                             label=label, fig=fig, ax=ax, color=color,
                                             return_axis=True)
        # Plot up coatal locs
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
        'WOA_Nitrate', 'WOA_Salinity', 'WOA_Phosphate', 'WOA_TEMP_K', 'Depth_GEBCO',
    ]
    # plot up the PDF distribution of each of the variables.
    datasets = sorted(dfs.keys())
    for var2use in vars2use:
        print(var2use)
        # set a single axis to use.
        fig, ax = plt.subplots()
        for dataset in datasets:
            # select the DataFrame
            df = dfs[dataset][var2use]
            # Get sample size
            N_ = df.shape[0]
            # do a dist plot
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
            # select the DataFrame
            dfA = dfs[dataset]
            # Set title
            title = "Locations for '{}'".format(dataset)
            p_size = 50
            alpha = 1
            # plot up Non coatal locs
            df = dfA.loc[dfA['Coastal'] == False, :]
            color = 'blue'
            label = 'Non-coastal (N={})'.format(int(df.shape[0]))
            m = AC.plot_lons_lats_spatial_on_map(title=title, f_size=15,
                                                 lons=df['Longitude'].values,
                                                 lats=df['Latitude'].values,
                                                 label=label, fig=fig, ax=ax, color=color,
                                                 return_axis=True)
            # plot up coatal locs
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
    # misc. shared variables
    axlabel = '[I$^{-}_{aq}$] (nM)'
    # setup PDf
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
    # colours to use?
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
        # if dates present in DataFrame, update axis
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
        # plot whole dataset
        obs_arr = pro_df[var2plot].values
        ax = sns.distplot(obs_arr, axlabel=axlabel,
                          color='k', label='Whole dataset')
        # plot just new data
        ax = sns.distplot(tmp_df[var2plot], axlabel=axlabel, label=Cruise,
                          color='red', ax=ax)
        # force y axis extend to be correct
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
        # plot whole dataset
        obs_arr = pro_df[var2plot].values
        ax = sns.distplot(obs_arr, axlabel=axlabel,
                          color='k', label='Whole dataset')
        # plot just new data
        ax = sns.distplot(tmp_df[var2plot], axlabel=axlabel, label=Cruise,
                          color='red', ax=ax)
        # force y axis extend to be correct
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
        # plot whole dataset
        obs_arr = pro_df[var2plot].values
        ax = sns.distplot(obs_arr, axlabel=axlabel,
                          color='k', label='Whole dataset')
        # plot just new data
        ax = sns.distplot(tmp_df[var2plot], axlabel=axlabel, label=Cruise,
                          color='red', ax=ax)
        # force y axis extend to be correct
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
        # plot whole dataset
        obs_arr = pro_df[var2plot].values
        ax = sns.distplot(obs_arr, axlabel=axlabel,
                          color='k', label='Whole dataset')
        # plot just new data
        ax = sns.distplot(tmp_df[var2plot], axlabel=axlabel, label=Cruise,
                          color='red', ax=ax)
        # force y axis extend to be correct
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
    # misc. shared variables
    axlabel = '[I$^{-}_{aq}$] (nM)'
    # setup PDf
    savetitle = 'Oi_prj_obs_plots'
    if inc_all_extract_vars:
        savetitle += '_all_extract_vars'
        include_hexbin_plots = True
    pdff = AC.plot2pdfmulti(title=savetitle, open=True, dpi=dpi)
    # colours to use?
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

    # setup color dictionary
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
        # if dates present in DataFrame, update axis
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
    # plot 1st model...
    point_name = 'Obs.'
    arr = point_ars_dict[point_name]
    ax = sns.distplot(arr, axlabel=axlabel, label=point_name,
                      color=colour_dict[point_name])
    # Add MacDonald, Chance...
    for point_name in point_data_names:
        arr = point_ars_dict[point_name]
        ax = sns.distplot(arr, axlabel=axlabel, label=point_name,
                          color=colour_dict[point_name])
    # force y axis extend to be correct
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
    # plot 1st model...
    point_name = 'Obs.'
    arr = point_ars_dict[point_name]
    ax = sns.distplot(arr, axlabel=axlabel, label=point_name,
                      color=colour_dict[point_name],
                      hist_kws=dict(cumulative=True), kde_kws=dict(cumulative=True))
    # Add MacDonald, Chance...
    for point_name in point_data_names:
        arr = point_ars_dict[point_name]
        ax = sns.distplot(arr, axlabel=axlabel, label=point_name,
                          color=colour_dict[point_name],
                          hist_kws=dict(cumulative=True), kde_kws=dict(cumulative=True))
    # force y axis extend to be correct
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
                            log=False, title=title, add_ODR_trendline2plot=True)
    #        plt.show()
            # Save to PDF and close plot
            AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
            plt.close()

    # -- Save entire pdf
    AC.plot2pdfmulti(pdff, savetitle, close=True, dpi=dpi)


def plot_PDF_iodide_obs_mod(bins=10):
    """
    plot up PDF of predicted values vs. observations
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    # Location of data to plot
    folder = utils.get_file_locations('data_root')
    f = 'Iodine_obs_WOA.csv'
    df = pd.read_csv(folder+f, encoding='utf-8')

    # Just select non-coastal data
    print(df.shape)
    df = df[~(df['Coastal'] == True)]
#    df = df[ ~(df['Coastal']==True) ]
    # Salinity greater than 30
#    df = df[ (df['Salinity'] > 30 ) ]
    print(df.shape)

    # Plot up data
    # Macdonaly et al 2014 values
    ax = sns.distplot(df['MacDonald2014_iodide'],
                      label='MacDonald2014_iodide', bins=bins)

    # Chance et al 2014 values
    ax = sns.distplot(df['Chance2014_STTxx2_I'],
                      label='Chance2014_STTxx2_I', bins=bins)
    # Iodide obs.
    ax = sns.distplot(df['Iodide'], label='Iodide, nM', bins=bins)

    # Update aesthetics and show plot?
    plt.xlim(-50, 400)
    plt.legend(loc='upper right')
    plt.show()


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
    # sub select variables of interest.
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
    # filename to save values
    filename = 'Oi_prj_global_predicted_vals_vs_lat_only_obs_locs'
    # include iodide observations too?
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
    # plot groups
    for var_ in params2plot:
        # Get quartiles
        Q1 = groups_des[var_]['25%'].values
        Q3 = groups_des[var_]['75%'].values
        # Add median
        ax.plot(X, groups_des[var_]['50%'].values,
                color=color_d[var_], label=rename_titles[var_])
        # add shading for Q1/Q3
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


def plot_up_data_locations_OLD_and_new(save_plot=True, show_plot=False):
    """
    Plot up old and new data on map
    """
    import seaborn as sns
    sns.reset_orig()
    # - Setup plot
    figsize = (11, 5)
    fig, ax = plt.subplots(figsize=figsize, dpi=320)
    p_size = 25
    alpha = 0.5
    window = True
    axis_titles = False
    # - Get all observational data
    df, md_df = obs.get_iodide_obs()
    # Seperate into new and old
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
    m = AC.plot_lons_lats_spatial_on_map(lons=lons1, lats=lats1,
                                         fig=fig, ax=ax, color='blue', label=label,
                                         alpha=alpha,
                                         window=window, axis_titles=axis_titles,
                                         return_axis=True, p_size=p_size)

    # - Add in new data following Chance2014?
    # this is ~ 5 samples from the Atlantic (and some from Indian ocean?)
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
    m.scatter(lons2, lats2, edgecolors=color, c=color, marker='o',
              s=p_size, alpha=alpha, label=label)
    # - Save out / show
    leg = plt.legend(fancybox=True, loc='upper right')
    leg.get_frame().set_alpha(0.95)
    if save_plot:
        plt.savefig('Oi_prj_Obs_locations.png', bbox_inches='tight')
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
#    df = df[ df['Coastal'] == 1.0  ] # select coastal locations
#    df = df[ df['Coastal'] == 0.0  ]  # select non coastal locations
    # only consider locations with salinity > 30
    df = df[df['Salinity'] > 30.0]  # select coastal locations
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
    # plot up white background
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
                                 vmin, vmax], ax=ax1, no_cb=True, resolution='c',
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
                # add one to counter
                counter += 1

            plt.legend()
            # save chunk...
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

    # get details of parameterisations
#    filename='Chance_2014_Table2_PROCESSED_17_04_19.csv'
    filename = 'Chance_2014_Table2_PROCESSED.csv'
    folder = utils.get_file_locations('data_root')
    param_df = pd.read_csv(folder+filename)

    # only consider non-coastal locations?
    print(df.shape)
#    df = df[ df['Coastal'] == 1.0  ] # select coastal locations
#    df = df[ df['Coastal'] == 0.0  ]  # select non coastal locations
    # only consider locations with salinity > 30
    df = df[df['Salinity'] > 30.0]  # select coastal locations
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

        # show/save?
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
    # units dict?
    units_dict = {
        'SeaWIFs_ChlrA': "mg m$^{-3}$",  # Chance et al uses micro g/L
        'WOA_Salinity': 'PSU',  # https://en.wikipedia.org/wiki/Salinity
        'WOA_Nitrate':  "$\mu$M",
        'WOA_TEMP': '$^{o}$C',
    }
    # Colors to use
    CB_color_cycle = AC.get_CB_color_cycle()
    # set the order the dict keys are accessed
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
    # plot up
    ax.scatter(X, Y, edgecolors=color, facecolors='none', s=5)
    # label Y axis
    if plot_n in np.arange(1, 6)[::2]:
        ax.set_ylabel('Extracted')
    # title the plots
    title = 'Salinity ($\geq$ 30, PSU)'.format(units_dict[var2plot])
    ax.text(0.5, 1.05, title, horizontalalignment='center',
            verticalalignment='center', transform=ax.transAxes)
    # Add N value
    stats_str = 'N={} \nRMSE={:.3g}'.format(N_, RMSE_)
    ax.text(0.05, 0.9, stats_str, horizontalalignment='left',
            verticalalignment='center', transform=ax.transAxes)
    # add a 1:1 line
    ax_max = df_tmp.max().max()
    ax_max = AC.myround(ax_max, 1, round_up=True) * 1.05
    ax_min = 29
    x_121 = np.arange(ax_min, ax_max*1.5)
    ax.plot(x_121, x_121, alpha=0.5, color='k', ls='--')
    # add ODR line
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
        # plot up
        ax.scatter(X, Y, edgecolors=color, facecolors='none', s=5)
        # label Y axis
        if plot_n in np.arange(1, 6)[::2]:
            ax.set_ylabel('Extracted')
        # title the plots
        title = '{} ({})'.format(obs_var_dict[var2plot], units_dict[var2plot])
        ax.text(0.5, 1.05, title, horizontalalignment='center',
                verticalalignment='center', transform=ax.transAxes)
        # Add N value
        stats_str = 'N={} \nRMSE={:.3g}'.format(N_, RMSE_)
        ax.text(0.05, 0.9, stats_str, horizontalalignment='left',
                verticalalignment='center', transform=ax.transAxes)
        # add a 1:1 line
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
    # plot up
    ax.scatter(X, Y, edgecolors=color, facecolors='none', s=5)
    # label Y axis
    if plot_n in np.arange(1, 6)[::2]:
        ax.set_ylabel('Extracted')
    ax.set_xlabel('Observed')
    # title the plots
    title = 'ChlrA (all, {})'.format(units_dict[var2plot])
    ax.text(0.5, 1.05, title, horizontalalignment='center',
            verticalalignment='center', transform=ax.transAxes)
    # Add N value
    stats_str = 'N={} \nRMSE={:.3g}'.format(N_, RMSE_)
    ax.text(0.05, 0.9, stats_str, horizontalalignment='left',
            verticalalignment='center', transform=ax.transAxes)
    # add a 1:1 line
    ax_max = df_tmp.max().max()
    ax_max = AC.myround(ax_max, 5, round_up=True) * 1.05
    ax_min = df_tmp.min().min()
    ax_min = ax_min - (ax_max*0.05)
    x_121 = np.arange(ax_min, ax_max*1.5)
    ax.plot(x_121, x_121, alpha=0.5, color='k', ls='--')
    # add ODR line
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
    # plot up
    ax.scatter(X, Y, edgecolors=color, facecolors='none', s=5)
    # label Y axis
    if plot_n in np.arange(1, 6)[::2]:
        ax.set_ylabel('Extracted')
    ax.set_xlabel('Observed')
    # title the plots
    units = units_dict[var2plot]
    title = 'ChlrA ($\leq$5 {})'.format(units)
    ax.text(0.5, 1.05, title, horizontalalignment='center',
            verticalalignment='center', transform=ax.transAxes)
    # Add N value
    stats_str = 'N={} \nRMSE={:.3g}'.format(N_, RMSE_)
    ax.text(0.05, 0.9, stats_str, horizontalalignment='left',
            verticalalignment='center', transform=ax.transAxes)
    # add a 1:1 line
    ax_max = df_tmp.max().max()
    ax_max = AC.myround(ax_max, 1, round_up=True) * 1.05
    ax_min = df_tmp.min().min()
    ax_min = ax_min - (ax_max*0.05)
    x_121 = np.arange(ax_min, ax_max*1.5)
    ax.plot(x_121, x_121, alpha=0.5, color='k', ls='--')
    # add ODR line
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
    # sort dataframe by latitude
#    df = df.sort_values('Latitude', axis=0, ascending=True)

    # set the order the dict keys are accessed
    vars_sorted = list(sorted(obs_var_dict.keys()))[::-1]

    # Setup pdf
    if save2pdf:
        savetitle = 'Oi_prj_Chance2014_Obs_params_vs_NEW_extracted_params'
        pdff = AC.plot2pdfmulti(title=savetitle, open=True, dpi=dpi)

        # - Get variables and confirm which datasets are being used for plot
        dfs = {}
    for key_ in vars_sorted:
        print(obs_var_dict[key_], key_)
        # drop nans...
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
        # drop nans...
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
    # get raw obs
    raw_df = get_core_Chance2014_obs()
    # don't consider iodide values above 30
    raw_df = raw_df[raw_df['Iodide'] > 30.]

    # - get processed obs.
    pro_df = obs.get_processed_df_obs_mod()
    restrict_data_max, restrict_min_salinity = True, True
    if restrict_data_max:
        #        pro_df = pro_df[ pro_df['Iodide'] < 450. ] # used for July Oi! mtg.
        # restrict below 400 (per. com. RJC)
        pro_df = pro_df[pro_df['Iodide'] < 400.]
    if restrict_min_salinity:
        pro_df = pro_df[pro_df['WOA_Salinity'] > 30.]


    # - Plots with raw obs.
    # Plot up "linear" fit of iodide and temperature. (Chance et al 2014)
    # plot up Chance
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
    # general stats on the iodide numbers
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
    for ds in list(set(md_df['Data_Key'])):
        N0 = df.loc[df['Data_Key'] == ds, :].shape[0]
        N1 = df2.loc[df2['Data_Key'] == ds, :].shape[0]
        IsChance = list(set(df.loc[df['Data_Key'] == ds, ChanceStr]))[0]
        prt_str = "DS: '{}' (Chance2014={}) has changed by {} to {} ({} vs. {})"
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
    # save the resultant file out
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
    # split data into all, Coastal and Non-Coastal
    dfs = {}
    dfs['All'] = df.copy()
    dfs['Coastal'] = df.loc[df['Coastal'] == 1, :]
    dfs['Non-coastal'] = df.loc[df['Coastal'] != 1, :]
    # if hist=True, use a count instead of density
    hist = False
    # Loop and plot
    axlabel = '[I$^{-}_{aq}$] (nM)'
    fig, ax = plt.subplots()
    vars2plot = dfs.keys()
    for key in vars2plot:
        sns.distplot(dfs[key]['Iodide'].values, ax=ax,
                     axlabel=axlabel, label=key, hist=hist)
        # force y axis extend to be correct
        ax.autoscale()
    # Add a legend
    plt.legend()
    # Add a label for the Y axis
    plt.ylabel('Density')
    # save plot
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
            'GMAO_ABSH', 'GMAO_PSFC', 'GMAO_SURF', 'GMAO_TEMP', 'GMAO_UWND', 'GMAO_VWND'
        ]
        slist = slist + met_vars
        for key_ in dfs.keys():
            print(key_, dfs[key_].shape)
            df = dfs[key_].dropna()
            print(df.shape)
            # add TYPE flag
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
            # make sure rows are in date order
            df.sort_values(['datetime', 'PRESS'], ascending=True, inplace=True)
            # now output files
            AC.prt_PlaneFlight_files(df=df, slist=slist)
    # Make planeflight files for DataFrames of cruises data
    # (outputting surface values)
    else:
        met_vars = [
            'GMAO_ABSH', 'GMAO_PSFC', 'GMAO_SURF', 'GMAO_TEMP', 'GMAO_UWND', 'GMAO_VWND'
        ]
        assert isinstance(num_tracers, int), 'num_tracers must be an integer'
        slist = ['TRA_{:0>3}'.format(i) for i in np.arange(1, num_tracers+1)]
        species = ['OH', 'HO2', 'GLYX']
        slist = slist + species + met_vars
        for key_ in dfs.keys():
            print(key_)
            df = dfs[key_].dropna()
            # add TYPE flag
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
    # file to save?
    savetitle = 'GC_pf_input_iodide_cruises'
    dpi = 320
    pdff = AC.plot2pdfmulti(title=savetitle, open=True, dpi=dpi)
    vars2test = ['LON', 'LAT']
    for key_ in dfs.keys():
        df = dfs[key_]
        for var_ in vars2test:
            # -- Plot X vs Y plot
            df_tmp = df[['datetime', var_]]
            # calc NaNs
            VAR_dropped_N = int(df_tmp.shape[0])
            df_tmp = df_tmp.dropna()
            VAR_N_data = int(df_tmp.shape[0])
            VAR_dropped_N = VAR_dropped_N-VAR_N_data
            # plot
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
        # possibly one of several cruises in this program
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
        # convert dates to datetime
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
                                             name='{} ({})'.format(key_, extra_str))
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
            get_test_plots_surface_pf_output(df=df,
                                             name='{} ({})'.format(
                                                 key_, extra_str),
                                             specs=specs, units='molec cm$^{-2}$',
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
    # get lon lat array of time in troposphere
    TPS = AC.get_GC_output(wd=wd+'/', vars=['TIME_TPS__TIMETROP'],
                           trop_limit=True)
    # convert this to boolean (<1 == not strat)
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
        # special condition for where there is more than column set
        # for a timestamp
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
        # select everyother value?
        # lon select locations
        LAT_ = tmp_df['LAT'].values
        LON_ = tmp_df['LON'].values
        # check there is only one lat and lon
        assert len(set(LAT_)) == 1
        assert len(set(LON_)) == 1
        # - Select 3D vars from ctm.nc file
        # get LON, LAT index of box
        LON_ind = AC.get_gc_lon(LON_[0], res=res)
        LAT_ind = AC.get_gc_lat(LAT_[0], res=res)
#        time_ind = AC.find_nearest( CTM_EPOCH, (ts-EPOCH).total_seconds() )
        time_ind = AC.find_nearest(CTM_months, ts.month)
        # tropspause height? ('TIME_TPS__TIMETROP)
        TPS_ = TPS[LON_ind, LAT_ind, :, time_ind]
        # Select surface area of grid box
        S_AREA_ = S_AREA[LON_ind, LAT_ind, 0]
        # comput column by spec
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
    df_col.index = dates  # timestamps[::1000]
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
    Test model output at surface for Indian sgip cruises
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
                                     lons=df['LON'].values, lats=df['LAT'].values,
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
    Make data files for the indian ocean surface iodide paper
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
            # select the minimum area for the areas
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
def Do_analysis_and_mk_plots_for_EGU19_poster():
    """
    Driver function for analysis and plotting for EGU poster
    """
    # - Get data
    # data locations and names as a dictionary
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
    # plot up emissions
    plot_up_surface_emissions(dsDH=dsDH)

    # - Do diferences plots
    # - look at the HOI/I2 surface values and IO.
    # species to look at?
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
    # surface ozone figure - made in powerpoint for now...

    # Plot up emissions for EGU presentation
    BASE = 'ML_Iodide'
    DIFF1 = 'Chance2014'
    DIFF2 = 'Macdonald2014'
    plot_up_EGU_fig05_emiss_change(ds_dict=dsD, BASE=BASE, DIFF1=DIFF1, DIFF2=DIFF2,
                                   update_PyGChem_format2COARDS=True)


def plot_up_EGU_fig05_emiss_change(ds_dict=None, levs=[1], specs=[],
                                   BASE='', DIFF1='',  DIFF2='', prefix='IJ_AVG_S__',
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
    # plot up map with mask present
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
                                       REF2=None, REF_wd=None, res='4x5', trop_limit=True,
                                       save2csv=True, prefix='GC_', run_names=None,
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
    # v/v scaling?
    ppbv_unit = 'ppbv'
    ppbv_scale = 1E9
    pptv_unit = 'pptv'
    pptv_scale = 1E12
    # Get shared variables from a single model run
    if isinstance(REF_wd, type(None)):
        REF_wd = wds[0]
    # get time in the troposphere diagnostic
    t_p = AC.get_GC_output(wd=REF_wd, vars=[u'TIME_TPS__TIMETROP'],
                           trop_limit=True)
    # Temperature
    K = AC.get_GC_output(wd=REF_wd, vars=[u'DAO_3D_S__TMPU'], trop_limit=True)
    # airmass
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
    # convert to N equivalent
    ars = [i/AC.species_mass('NO2')*AC.species_mass('N') for i in ars]
    df[NO2_varname] = ars

    # Get NO burden
    NO_varname = 'NO burden ({})'.format(mass_unit)
    ars = [AC.get_trop_burden(spec='NO', t_p=t_p, wd=i, all_data=False).sum()
           for i in wds]
    # convert to N equivalent
    ars = [i/AC.species_mass('NO')*AC.species_mass('N') for i in ars]
    df[NO_varname] = ars

    # Combine NO and NO2 to get NOx burden
    NOx_varname = 'NOx burden ({})'.format(mass_unit)
    df[NOx_varname] = df[NO2_varname] + df[NO_varname]

    # Get HOI burden
    varname = 'HOI burden ({})'.format(mass_unit)
    ars = [AC.get_trop_burden(spec='HOI', t_p=t_p, wd=i, all_data=False).sum()
           for i in wds]
    # convert to I equivalent
    ars = [i/AC.species_mass('HOI')*AC.species_mass('I') for i in ars]
    df[varname] = ars

    # Get I2 burden
    varname = 'I2 burden ({})'.format(mass_unit)
    ars = [AC.get_trop_burden(spec='I2', t_p=t_p, wd=i, all_data=False).sum()
           for i in wds]
    # convert to I equivalent
    ars = [i/AC.species_mass('I2')*AC.species_mass('I') for i in ars]
    df[varname] = ars

    # Get I2 burden
    varname = 'IO burden ({})'.format(mass_unit)
    ars = [AC.get_trop_burden(spec='IO', t_p=t_p, wd=i, all_data=False).sum()
           for i in wds]
    # convert to I equivalent
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


def Check_sensitivity_of_HOI_I2_param2WS():
    """
    Check the sensitivity of the Carpenter et al 2013 parameterisation to wind speed
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
