

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

    # Setup PDF
    savetitle = 'Oi_prj_data_root_exploration_{}'.format(ext_str)
    dpi = 320
    pdff = AC.plot2pdfmulti(title=savetitle, open=True, dpi=dpi)
    # Colours to use?
#    current_palette = sns.color_palette()
    current_palette = sns.color_palette("colorblind")

    # --- --- --- --- --- --- --- ---
    # ---- Add in extra varibles
    # Iodide / iodate
    I_div_IO3_var = 'I$^{-}$/IO$_{3}^{-}$ (ratio)'
    df[I_div_IO3_var] = df['Iodide'] / df['Iodate']
    # Total iodide
    I_plus_IO3 = 'I$^{-}$+IO$_{3}^{-}$'
    df[I_plus_IO3] = df['Iodide'] + df['Iodate']

    # --- Add ocean basin to dataframe
    area_var = 'Region'
    df[area_var] = None
    # Setup a dummy column

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
    # Plot up
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

    # Plot up with no limits
    df.plot(kind='scatter', y=I_div_IO3_var, x='Latitude')
    # Beautify
    plt.title(I_div_IO3_var + ' ({}, y axis unlimited)'.format(ext_str))
    # Save to PDF and close plot
    AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
    if show_plot:
        plt.show()
    plt.close()

    # Plot up with limits at 3
    ylimits = 1.5, 0.75, 0.5,
    for ylimit in ylimits:
        df.plot(kind='scatter', y=I_div_IO3_var, x='Latitude')
        # Beautify
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
    # For southern ocean use the files below
    # For  rest https://www.nodc.noaa.gov/woce/woce_v3/wocedata_1/woce-uot/summary/bound.htm
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

    # If latitude below 60S, overwrite to be Southern ocean
    varname = 'Southern Ocean'
    df.loc[df['Latitude'] < -60, area_var] = varname

    # --- --- --- --- --- --- --- ---
    # ---  locations of data
    sns.reset_orig()
    # Loop regions
    for var_ in list(set(df[area_var].tolist())):
        # Select data for area
        df_tmp = df[df[area_var] == var_]
        # Locations ?
        lons = df_tmp[u'Longitude'].tolist()
        lats = df_tmp[u'Latitude'].tolist()
        # Now plot
        AC.plot_lons_lats_spatial_on_map(lons=lons, lats=lats)
        plt.title('{} ({})'.format(var_, ext_str))
        if show_plot:
            plt.show()

        # Save to PDF and close plot
        AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
        plt.close()

    # --- --- --- --- --- --- --- ---
    # --- Iodide to iodide ratio
    import seaborn as sns
    sns.set(color_codes=True)
    current_palette = sns.color_palette("colorblind")

    # Loop regions
    for var_ in list(set(df[area_var].tolist())):
        # Select data for area
        df_tmp = df[df[area_var] == var_]
        # Plot up with no limits
        df_tmp.plot(kind='scatter', y=I_div_IO3_var, x='Latitude')
        # Beautify
        plt.title(I_div_IO3_var + ' ({}, y axis unlimited)'.format(var_))
        # Save to PDF and close plot
        AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
        if show_plot:
            plt.show()
        plt.close()
        # Plot up with limits at 3
        ylimits = 1.5, 0.75, 0.5
        for ylimit in ylimits:
            df_tmp.plot(kind='scatter', y=I_div_IO3_var, x='Latitude')
            # Beautify
            title = ' ({}, y axis limit: {})'.format(var_, ylimit)
            plt.title(I_div_IO3_var + title)
            plt.ylim(-0.05, ylimit)
            # Save to PDF and close plot
            AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
            if show_plot:
                plt.show()
            plt.close()

    # --- --- --- --- --- --- --- ---
    # --- Iodide + iodide
    import seaborn as sns
    sns.set(color_codes=True)
    current_palette = sns.color_palette("colorblind")

    # Loop regions
    for var_ in list(set(df[area_var].tolist())):
        # Select data for area
        df_tmp = df[df[area_var] == var_]
        # Plot up with no limits
        df_tmp.plot(kind='scatter', y=I_plus_IO3, x='Latitude')
        # Beautify
        plt.title(I_plus_IO3 + ' ({}, y axis unlimited)'.format(var_))
        # Save to PDF and close plot
        AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
        if show_plot:
            plt.show()
        plt.close()

        # Plot up with limits at 3
#         ylimits = 1.5, 0.75, 0.5
#         for ylimit in ylimits:
#             df.plot(kind='scatter', y=I_plus_IO3, x='Latitude' )
#             # Beautify
#             title= ' ({}, y axis limited to {})'.format(var_, ylimit)
#             plt.title( I_plus_IO3 + title )
#             plt.ylim(-0.05, ylimit )
#             # Save to PDF and close plot
#             AC.plot2pdfmulti( pdff, savetitle, dpi=dpi )
#             if show_plot: plt.show()
#             plt.close()

        # Plot up with limits on y
        ylimits = [100, 600]
#        for ylimit in ylimits:
        df_tmp.plot(kind='scatter', y=I_plus_IO3, x='Latitude')
        # Beautify
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
    # Make sure the dataset has units
    ds = add_units2ds(ds)
    # Use appropriate plotting settings for resolution
    if res == '0.125x0.125':
        centre = True
    else:
        centre = False
    # Get data
    arr = ds[var].mean(dim='time').values
    # Colour in values above and below threshold (works)
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
                           fillcontinents=fillcontinents, centre=centre,
                           units=units,
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
    # - Get data
    df = get_core_Chance2014_obs()
    # Select data of interest
    # ( later add a color selection based on coastal values here? )
    vars = ['Iodide', 'Latitude']
    print(df)
    # And select coastal/open ocean
    df_coastal = df[df['Coastal'] == True][vars]
    df_open_ocean = df[~(df['Coastal'] == True)][vars]
    # - Now plot Obs.
    # Plot coastal
    ax = df_coastal.plot(kind='scatter', x='Latitude', y='Iodide', marker='D',
                         color='blue', alpha=0.1,
                         )
    # Plot open ocean
    ax = df_open_ocean.plot(kind='scatter', x='Latitude', y='Iodide',
                            marker='D', color='blue', alpha=0.5, ax=ax,
                            )
    # Update aesthetics of plot
    plt.ylabel('[Iodide], nM')
    plt.xlabel('Latitude, $^{o}$N')
    plt.ylim(-5, 500)
    plt.xlim(-80, 80)
    # Save or show?
    if show_plot:
        plt.show()
    plt.close()


def plot_up_ln_iodide_vs_Nitrate(show_plot=True):
    """
    Reproduce Fig. 11 in Chance et al (2014)

    Original caption:

    Ln[iodide] concentration plotted against observed ( ) and
    climatological ( ) nitrate concentration obtained from the World
    Ocean Atlas as described in the text for all data (A) and nitrate
    concentrations below 2 mM (B) and above 2 mM (C). Dashed lines in B
    and C show the relationships between iodide and nitrate adapted from
    Campos et al.41 by Ganzeveld et al.27
    """
    # - Location of data to plot
    df = obs.get_processed_df_obs_mod()
    # Take log of iodide
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
    Reproduce Fig. 8 in Chance et al (2014)

    Original caption:

    Ln[iodide] concentration plotted against observed sea surface
    temperature ( ) and climatological sea surface temperature ( ) values
    obtained from the World Ocean Atlas as described in the text.
    """
    # - Location of data to plot
    folder = utils.get_file_locations('data_root')
    f = 'Iodine_obs_WOA.csv'
    df = pd.read_csv(folder+f, encoding='utf-8')
    # Take log of iodide
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
    # - Location of data to plot
    folder = utils.get_file_locations('data_root')
    f = 'Iodine_obs_WOA.csv'
    df = pd.read_csv(folder+f, encoding='utf-8')
    # Just select non-coastal data
#    df = df[ ~(df['Coastal']==True) ]
    # Take log of iodide
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
    # Make a kde plot
    def make_kde(*args, **kwargs):
        sns.kdeplot(*args, cmap=next(make_kde.cmap_cycle), **kwargs)
    # Define colormap to cycle
    make_kde.cmap_cycle = cycle(('Blues_r', 'Greens_r', 'Reds_r', 'Purples_r'))
    # Plot a pair plot
    pg = sns.PairGrid(data, vars=vars_list)


def plot_up_data_locations_OLD_and_new(save_plot=True, show_plot=False,
                                       extension='eps', dpi=720):
    """
    Plot up old and new observational data locations on map
    """
    import seaborn as sns
    sns.reset_orig()
    # - Setup plot
    figsize = (11, 5)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    p_size = 25
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
    m = AC.plot_lons_lats_spatial_on_map(lons=lons1, lats=lats1,
                                         fig=fig, ax=ax, color='blue',
                                         label=label,
                                         alpha=alpha,
                                         window=window,
                                         axis_titles=axis_titles,
                                         return_axis=True, p_size=p_size)

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
    m.scatter(lons2, lats2, edgecolors=color, c=color, marker='o',
              s=p_size, alpha=alpha, label=label)
    # - Save out / show
    leg = plt.legend(fancybox=True, loc='upper right')
    leg.get_frame().set_alpha(0.95)
    if save_plot:
        savename = 'Oi_prj_Obs_locations.{}'.format(extension)
        plt.savefig(savename, bbox_inches='tight', dpi=dpi)
    if show_plot:
        plt.show()


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
    # Macdonald et al 2014 values
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

