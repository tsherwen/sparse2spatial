#!/usr/bin/python
# -*- coding: utf-8 -*-
"""

Process input data (sea-surface iodide) to a DataFrame for modelling.

This file is intended to run as a standard alone script. It generates the file of observational data used both in the paper that describes the sea-surface iodide observations [Chance et al 2019] and the work that uses this to build a monthly sea-surface field [Sherwen et al 2019].

To add new observations to the compiled BODC dataset, please use the process_new_observations.py script and follow the instructions there.

e.g.

python observations.py

Citations
-------

 - Data descriptor paper on sea-surface observations
Chance, R.J., Tinel, L., Sherwen, T., Baker, A.R., Bell, T., Brindle, J., Campos, M.L.A., Croot, P., Ducklow, H., Peng, H. and Hopkins, F., 2019. Global sea-surface iodide observations, 1967–2018. Scientific data, 6(1), pp.1-8.

 - Observational data archived at British Oceanographic Data Centre (BODC)
Chance R.; Tinel L.; Sherwen T.; Baker A.; Bell T.; Brindle J.; Campos M.L.A.M.; Croot P.; Ducklow H.; He P.; Hoogakker B.; Hopkins F.E.; Hughes C.; Jickells T.; Loades D.; Reyes Macaya D.A.; Mahajan A.S.; Malin G.; Phillips D.P.; Sinha A.K.; Sarkar A.; Roberts I.J.; Roy R.; Song X.; Winklebauer H.A.; Wuttig K.; Yang M.; Zhou P.; Carpenter L.J.(2019). Global sea-surface iodide observations, 1967-2018. British Oceanographic Data Centre - Natural Environment Research Council, UK. doi:10/czhx.

Notes
-------
 - Abbreviations used here:
obs. = observations

"""
import numpy as np
import pandas as pd
# import AC_tools (https://github.com/tsherwen/AC_tools.git)
import AC_tools as AC
import sparse2spatial as s2s
import sparse2spatial.utils as utils
from sparse2spatial.utils import set_backup_month_if_unknown
#from sea_surface_iodide import mk_iodide_test_train_sets
from sparse2spatial.ancillaries2grid_oversample import extract_ancillaries_from_compiled_file


def get_processed_df_obs_mod(reprocess_params=False,
                             filename='Iodine_obs_WOA.csv',
                             rm_Skagerrak_data=False,
                             file_and_path='./sparse2spatial.rc',
                             verbose=True, debug=False):
    """
    Get the processed observation and model output

    Parameters
    -------
    file_and_path (str): folder and filename with location settings as single str
    rm_Skagerrak_data (boolean): remove the single data from the Skagerrak region
    reprocess_params (bool):
    filename (str): name of the input file of processed observational data
    verbose (bool): print verbose statements
    debug (bool): print debug statements

    Returns
    -------
    (pd.DataFrame)
    """
    # Read in processed csv file
    folder = utils.get_file_locations('data_root', file_and_path=file_and_path)
    folder += '/Iodide/'
    df = pd.read_csv(folder+filename, encoding='utf-8')
    # Add ln of iodide too
    df['ln(Iodide)'] = df['Iodide'].map(np.ma.log)
    # Add SST in Kelvin too
    if 'WOA_TEMP_K' not in df.columns:
        df['WOA_TEMP_K'] = df['WOA_TEMP_K'].values + 273.15
    # Make sure month is numeric (if not given)
    month_var = 'Month'
    NaN_months_bool = ~np.isfinite(df[month_var].values)
    NaN_months_df = df.loc[NaN_months_bool, :]
    N_NaN_months = NaN_months_df.shape[0]
    if N_NaN_months > 1:
        print_str = 'DataFrame contains NaNs for {} months - '
        print_str += 'Replacing these with month # 3 months '
        print_str += 'before (hemispheric) summer solstice'
        if verbose:
            print(print_str.format(N_NaN_months))
        NaN_months_df[month_var] = NaN_months_df.apply(lambda x:
                                                       set_backup_month_if_unknown(
                                                           lat=x['Latitude'],
                                                           # main_var=var2use,
                                                           # var2use=var2use,
                                                           #
                                                           # Data_key_ID_=Data_key_ID_,
                                                           debug=False),
                                                           axis=1)
        # Add back into DataFrame
        df.loc[NaN_months_bool, month_var] = NaN_months_df[month_var].values
    # Re-process the parameterisations (Chance et al etc + ensemble)?
    if reprocess_params:
                # Add predictions from literature
        df = get_literature_predicted_iodide(df=df)
        # Add ensemble prediction
        df = get_ensemble_predicted_iodide(
            rm_Skagerrak_data=rm_Skagerrak_data
        )
    return df


def fill_years4cruises():
    """
    Fill the years for the cruises with missing years with values from the metadata sheet
    """
    # Only update cells where no year is already present
    bool1 = ~np.isfinite(df['Year'])
    # For the Wong_B_1977 dataset set the year to 1973 (values: 'Dec 73?')
    bool_tmp = df['Data_Key'] == 'Wong_B_1977'
    df.loc[bool1 & bool_tmp,'Year'] = 1973
    # For the Tsunogai_H_1971 dataset (Notes on metadata sheet: "1968-1969")
    bool_tmp = df['Data_Key'] == 'Tsunogai_H_1971'
    df.loc[bool1 & bool_tmp,'Year'] = 1968
    # For the Wong_C_1998 dataeset (Notes on metadata sheet: "~1998?")
    bool_tmp = df['Data_Key'] == 'Wong_C_1998'
    df.loc[bool1 & bool_tmp,'Year'] = 1998
    # For the Luther_1988 dataeset (Notes on metadata sheet: "1987")
    bool_tmp = df['Data_Key'] == 'Luther_1988'
    df.loc[bool1 & bool_tmp,'Year'] = 1987
    return df


def add_datetime2Iodide_df(df=None, DateVar='Datetime'):
    """
    Add datetime to iodide obs. dataframe
    """
    # Use a local function to add datetime values to dataframe
    def add_dt2_df(Year=None, Month=None, Day=None):
        print( Year, Month, Day)
        # Set the Date to the middle of the month (#=15) if not known value
        if isinstance(Day, type(None)) or ~np.isfinite(Day):
            Day = 15
        return datetime.datetime(int(Year), int(Month), int(Day))
    # Now apply the function on the whole dataframe
    df[DateVar] = df.apply(lambda x: add_dt2_df(Year=x['Year'],
                                                Month=x['Month'],
                                                Day=x['Day']), axis=1)
    return df


def process_iodide_obs_ancillaries_2_csv(rm_Skagerrak_data=False,
                                         add_ensemble=False,
                                         file_and_path='./sparse2spatial.rc',
                                         target='Iodide', verbose=True):
    """
    Create a csv files of iodide observation and ancilllary observations

    Parameters
    -------
    file_and_path (str): folder and filename with location settings as single str
    add_ensemble (bool): add the ensemble prediction to input data dataframe
    rm_Skagerrak_data (boolean): remove the single data from the Skagerrak region
    target (str): Name of the target variable (e.g. iodide)
    verbose (bool): print verbose statements

    Returns
    -------
    (pd.DataFrame)

    Notes
    -----
     -  Workflow assumes that this step will be run to compile the data
    """
    # Get iodide observations (as a dictionary/DataFrame)
    obs_data_df, obs_metadata_df = get_iodide_obs()
    # Add ancillary obs.
    obs_data_df = extract_ancillaries_from_compiled_file(df=obs_data_df)
    # Save the intermediate file
    folder = utils.get_file_locations('data_root', file_and_path=file_and_path)
    folder += '/{}/'.format(target)
    filename = 'Iodine_obs_WOA_v8_5_1_TEMP_TEST.csv'
    obs_data_df.to_csv(folder+filename, encoding='utf-8')
    # - Add predicted iodide from MacDonald and Chance parameterisations
    obs_data_df = get_literature_predicted_iodide(df=obs_data_df)
    # - Add ensemble prediction by averaging predictions at obs. locations.?
    if add_ensemble:
        print('NOTE - models must have already been provided via RFR_dict')
        RFR_dict = build_or_get_models(
            rm_Skagerrak_data=rm_Skagerrak_data,
        )
        # Now extract for
        obs_data_df = get_ensemble_predicted_iodide(df=obs_data_df,
                                                    use_vals_from_NetCDF=False,
                                                    RFR_dict=RFR_dict,
                                            rm_Skagerrak_data=rm_Skagerrak_data
                                                    )
    # - Join dataframes and save as csv.
#    filename = 'Iodine_obs_WOA.csv'
#    filename = 'Iodine_obs_WOA_v8_1_PLUS_ENSEMBLE.csv'
#    filename = 'Iodine_obs_WOA_v8_5_1_PLUS_ENSEMBLE_8_3_ENSEMBLE.csv'
    filename = 'Iodine_obs_WOA_v8_5_1_ENSEMBLE_csv__avg_nSkag_nOutliers.csv'
#    filename = 'Iodine_obs_WOA_v8_2_PLUS_PARAMS.csv'
    if verbose:
        print(obs_data_df.shape, obs_data_df.columns)
    obs_data_df.to_csv(folder+filename, encoding='utf-8')
    if verbose:
        print('File saved to: ', folder+filename)


def get_core_Chance2014_obs(debug=False, file_and_path='./sparse2spatial.rc'):
    """
    Get core observation data from Chance2014

    Parameters
    -------
    file_and_path (str): folder and filename with location settings as single str
    debug (bool): print debugging to screen

    Returns
    -------
    (pd.DataFrame)

    Notes
    -----
     - This assumes that core data is "surface data" above 20m
     - only considers rows of csv where there is iodine data.
    """
    # - Get file observational file
    # Directory to use?
    folder = utils.get_file_locations('data_root', file_and_path=file_and_path)
    folder = '{}/Iodide/inputs/'.format(folder)
    # Filename for <20m iodide data?
    filename = 'Iodide_data_above_20m.csv'
    # Open data as DataFrame
    df = pd.read_csv(folder+filename)
    # - Process the input observational data
    # list of core variables
    core_vars = [
        'Ammonium', 'Chl-a', 'Cruise', 'Data_Key', 'Data_Key_ID', 'Date',
        'Day',
        'Depth', 'Iodate', 'Iodide', 'Latitude', 'Longitude', 'MLD',  'Month',
        'MLD(vd)', 'Nitrate', 'Nitrite', 'O2', 'Organic-I', 'Salinity',
        'Station',
        'Temperature', 'Time', 'Total-I', 'Unique id', 'Year',
        u'Method', u'ErrorFlag',
    ]
    # 2nd iteration excludes 'MLD(vd)', so remove this.
    core_vars.pop(core_vars.index('MLD(vd)'))
    # 2nd iterations includes new flag columns. Add these.
    core_vars += ['Coastal', 'LocatorFlag', 'Province', ]
    # Just select core variables
    df = df[core_vars]
    # Remove datapoints that are not floats
    def make_sure_values_are_floats(x):
        """
        Some values in the dataframes are "nd" or "###?". remove these.
        """
        try:
            x = float(x)
        except:
            x = np.NaN
        return x
    # TODO: Make this more pythonic
    make_data_floats = [
        'Ammonium', 'Chl-a', 'Iodate', 'Iodide', 'Latitude', 'Longitude',
        'MLD',
        'MLD(vd)', 'Month', 'Nitrate', 'Nitrite', 'O2', 'Organic-I',
        'Salinity', 'Temperature', 'Total-I'
    ]
    # 2nd iteration excludes 'MLD(vd)', so remove this.
    make_data_floats.pop(make_data_floats.index('MLD(vd)'))
    # 2nd iterations includes new flag columns. Add these.
    make_data_floats += ['Coastal', 'LocatorFlag', 'Province', ]
    # v8.4 had further updates.
    make_data_floats += ['ErrorFlag', ]
    for col in make_data_floats:
        df[col] = df[col].map(make_sure_values_are_floats)[:]
    # Only consider rows where there is iodide data (of values from <20m N=930)
    if debug:
        print('I- df shape (inc. NaNs): {}'.format(str(df.shape)))
    df = df[np.isfinite(df['Iodide'])]
    if debug:
        print("I- df post rm'ing NaNs: {}".format(str(df.shape)))
    return df


def get_iodide_obs_metadata(file_and_path='./sparse2spatial.rc'):
    """
    Extract and return metadata from metadata csv
    """
    # What is the location of the iodide data?
    folder = utils.get_file_locations('data_root', file_and_path=file_and_path)
    folder += '/Iodide/inputs/'
    # Filename?
    filename = 'Iodine_climatology_Submitted_data_list_formatted_TMS.xlsx'
    # Extract
    df = pd.read_excel(folder+filename, sheetname='Full')
    # return as DataFrame
    return df


def get_iodide_obs(just_use_submitted_data=False,
                   use_Chance2014_core_data=True,
                   analyse_iodide_values2drop=False,
                   process_new_iodide_obs_file=False,
                   file_and_path='./sparse2spatial.rc',
                   limit_depth_to=20, verbose=True, debug=False):
    """
    Extract iodide observations from the (re-formated) file from Chance2014

    Parameters
    -------
    just_use_submitted_data (bool), just use the data submitted for Chance et al 2014
    use_Chance2014_core_data (bool), just use the code data in Chance2014's analysis
    analyse_iodide_values2drop (bool), check which values should be removed
    process_new_iodide_obs_file (bool), make a new iodide obs. file?
    file_and_path (str): folder and filename with location settings as single str
    limit_depth_to (float), depth (m) to limit inclusion of data to
    verbose (bool): print verbose statements to screen
    debug (bool): print debugging statements to screen

    Returns
    -------
    (pd.DataFrame)

    Notes
    -----
    """
    # What is the location of the iodide data?
    folder = utils.get_file_locations('data_root', file_and_path=file_and_path)
    folder += '/Iodide/inputs/'
    # Name to save file as
    filename = 'Iodide_data_above_20m.csv'
    # - Get Metadata (and keep as a seperate DataFrame )
    metadata_df = get_iodide_obs_metadata()
    # Process new iodide obs. (data) file?
    if process_new_iodide_obs_file:
        # - Extract data?
        # To test processing... just use submitted data?
        if just_use_submitted_data:
            Data_Keys = metadata_df['Data_Key'][metadata_df['source'] == 's']
            print(Data_Keys)
            # Add bodc data
            bool_ = metadata_df['source'] == 'bodc'
            bodc_Data_Keys = metadata_df['Data_Key'].loc[bool_]
            Data_Keys = list(Data_Keys)
            bodc_Data_Keys = list(bodc_Data_Keys)
            print(bodc_Data_Keys)
            Data_Keys = Data_Keys + bodc_Data_Keys
            print(Data_Keys)
        else:  # use all data
            Data_Keys = metadata_df['Data_Key']
        # - Loop by the datasets ("Data_Keys")
        # Setup list to store dataframes
        dfs = []
        # Loop data keys for sites
        for n_Data_Key, Data_Key in enumerate(Data_Keys):
            pcent = float(n_Data_Key)/len(Data_Keys)*100
            if verbose:
                print(n_Data_Key, Data_Key, pcent)
            # Extract data
            df = extract_templated_excel_file(Data_Key=Data_Key,
                                          metadata_df=metadata_df,
                                          limit_depth_to=limit_depth_to)
            # Save to list
            dfs += [df]
        # Combine dataframes.
        main_df = pd.concat(dfs)
        # Analyse the datapoints that are being removed.
        if analyse_iodide_values2drop:
            # Loop indexes and save out values that are "odd"
            ind2save = []
            tmp_var = 'temp #'
            main_df[tmp_var] = np.arange(main_df.shape[0])
            for ind in main_df[tmp_var].values:
                df_tmp = main_df.loc[main_df[tmp_var] == ind, :]
                try:
                    pd.to_numeric(df_tmp['Iodide'])
                except:
                    ind2save += [ind]
        # Make sure core values are numeric
        core_numeric_vars = [
            u'Ammonium', u'Chl-a', u'Depth', u'Iodate', u'Iodide', u'Latitude',
            u'Longitude',
            u'Nitrate', u'Nitrite', u'O2', u'Organic-I', u'Salinity',
            u'Total-I',
            u'Temperature', u'\u03b4Ammonium', u'\u03b4Chl-a', u'\u03b4Iodate',
            u'\u03b4Iodide', u'\u03b4Nitrate', u'\u03b4Nitrite',
            u'\u03b4Org-I',
            u'\u03b4Total-I'
        ]
        for var in core_numeric_vars:
            main_df[var] = pd.to_numeric(main_df[var].values, errors='coerce')
        # Save to disk
        main_df.to_csv(folder+filename, encoding='utf-8')
    # - Just use existing file
    else:
        try:
            # Just open existing file
            if use_Chance2014_core_data:
                main_df = get_core_Chance2014_obs()
            else:
                main_df = pd.read_csv(folder+filename, encoding='utf-8')
        except:
            print('Error opening processed iodide data file')
    # Return DataFrames
    return main_df, metadata_df


def extract_templated_excel_file(limit_depth_to=20, Data_Key=None,
                                 metadata_df=None, use_inclusive_limit=False,
                                 file_and_path='./sparse2spatial.rc',
                                 verbose=True, debug=False):
    """
    Extract an excel file in the iodide template format & return as DataFrame

    Parameters
    -------
    file_and_path (str): folder and filename with location settings as single str
    filename (str): name of the csv file or archived data from BODC
    limit_depth_to (float), depth (m) to limit inclusion of data to
    use_inclusive_limit (bool), limit depth (limit_depth_to) in a inclusive way
    debug (bool), print debugging statements to screen

    Returns
    -------
    (pd.DataFrame)
    """
    # limit_depth_to=20; Data_Key=None; metadata_df=None; debug=False
    # -  Get file details
    # Load metadata file as a DataFrame
    Data_Key_meta = metadata_df[metadata_df.Data_Key == Data_Key]
    # Use TMS updated variable for filename
#    filename = Data_Key_meta['File name'].values[0]
    filename = Data_Key_meta['File_name_UPDATED'].values[0]
    source = Data_Key_meta['source'].values[0]
    InChance2014 = Data_Key_meta['In Chance2014?'].values[0] == 'Y'
    # - Get directory which contains files
    # Data submitted directly for preparation
    # (as publish by Chance et al (2014) )
    # New data, acquired since 2017
    folder = utils.get_file_locations('data_root', file_and_path=file_and_path)
    folder = '/{}/Iodide/'.format(folder)
    if (not InChance2014):
        folder +=  '/inputs/new_data/'
    elif ((source == 's') or (source == 'bodc')) and (InChance2014):
        folder += '/inputs/submitted_data/'
    # Data digitalised for Chance et al (2014)
    elif (source == 'd') and (InChance2014):
        folder += '/inputs/digitised_data/'
    else:
        print("Source received ('') unknown?!".format(source))
        sys.exit()
    # File specific reading settings?
    read_csv_settings = read_csv_settings_4_data_key_file(Data_Key=Data_Key)
    skiprows, file_extension = read_csv_settings
    # - Read file and process
    if verbose:
        print('reading: {}'.format(filename), Data_Key)
    df = pd.read_excel(folder+filename+file_extension, sheet_name='Data',
                       skiprows=skiprows)
    # Force use of 'Index' column as index to preserve ordering.
    df.index = df['Index'].values
    # Only consider values with a depth value lower than x (e.g. 100m)
    # From Chance et al (2014): On ship-based campaigns, ‘surface’ water is
    # usually collected from an underway pumped seawater inlet
    # (typically at a depth of around 6 m on a 100 m length research
    # ship), and/or sampling bottles mounted on a CTD rosette and
    # closed within a few metres of the sea surface, but during some
    # eld campaigns (e.g. winter samples in the Antarctic73), only
    # data from 15 m depth was available. In most cases, the water
    # column is thought to be sufficiently homogenous between 0 and
    # 20 m that this choice of depth can be assumed to be representative
    # of concentrations in the top few metres of the water
    # column (see Section 3.4 for a description of the changes in
    # iodine speciation with depth).
    if verbose:
        print(df.columns)
    if use_inclusive_limit:
        df = df.loc[df['Depth'] <= limit_depth_to, :] # consider values inclusively
    else:
        df = df.loc[df['Depth'] < limit_depth_to, :] # only consider values less than X
    # Add a column to be a unique identifier and column index
    def get_unique_Data_Key_label(x, Data_Key=Data_Key):
        # Use the index as the number (which now starts from 1)
        x = int(x)
        return '{}_{:0>4}'.format(Data_Key, x)
    # Map to index, then assign to be index
    df['Data_Key_ID'] = df['Index'].map(get_unique_Data_Key_label)
#    df.index = df['Data_Key_ID']
    # Also add column for data key
    df['Data_Key'] = Data_Key
    return df


def read_csv_settings_4_data_key_file(Data_Key=None):
    """
    get skiprows/extension of observational excel files

    Returns
    -------
    (tuple)
    """
    # Dictionary values: Data Key : ( skiprows, file extension)
    d = {
        'Wong_Z_2003': (1, '.xls'),
        'Tsunogai_1971': (1, '.xls'),
        'Truesdale_B_2002': (1, '.xlsx'),
        'Chance_UnP_I': (1, '.xls'),
        'Wong_Z_1992': (1, '.xls'),
        'Ullman_1990': (1, '.xls'),
        'Tian_1995_1996': (1, '.xls'),
        'Baker_UnP': (1, '.xlsx'),
        'Tsunogai_H_1971': (1, '.xls'),
        'McTaggart_1994': (1, '.xls'),
        'Bluhm_2011': (1, '.xlsx'),
        'Truesdale_2000': (1, '.xls'),
        'Waite_2006': (1, '.xls'),
        #    'Wong_1985' : (1, '.xlsx'),
        'Wong_1985': (1, '.xls'),
        'Truesdale_2001': (1, '.xlsx'),
        'Liss_H_G_1973': (1, '.xls'),
        'Truesdale_2003_II': (1, '.xlsx'),
        'Rue_1997': (1, '.xls'),
        'Truesdale_2003_I': (1, '.xls'),
        'Wong_B_1974': (1, '.xlsx'),
        'Chance_2010': (1, '.xlsx'),
        'Schwehr_S_2003': (1, '.xlsx'),
        'Hou_2007': (1, '.xlsx'),
        'Elderfield_T_1980': (1, '.xlsx'),
        'Luther_C_1991': (1, '.xls'),
        'Luther_UnP': (1, '.xls'),
        'Jickells_1988': (1, '.xls'),
        'Hou_2013': (1, '.xls'),
        'Wong_B_1977': (1, '.xls'),
        'Wong_1977': (1, '.xls'),
        'Truesdale_1978': (1, '.xlsx'),
        'Farrenkopf_2002': (1, '.xlsx'),
        'Truesdale_U_2003': (1, '.xls'),
        'Woittiez_1991': (1, '.xlsx'),
        'Luther_1988': (1, '.xls'),
        'Chance_2007': (1, '.xls'),
        #    'Chance_2007' : (2, '.xls'),
        'Ducklow_2018': (1, '.xlsx'),
        'Campos_1999': (1, '.xlsx'),
        'Campos_1996': (1, '.xlsx'),
        'Chance_UnP_II': (1, '.xlsx'),
        'Wong_C_2008': (1, '.xlsx'),
        'Truesdale_J_2000': (1, '.xls'),
        'Nakayama_1989_1985': (1, '.xls'),
        'Wong_C_1998': (1, '.xlsx'),
        'Huang_2005': (1, '.xlsx'),
        'He_2013': (1, '.xlsx'),
        'He_2014': (1, '.xlsx'),
        'Chance_2018_I': (1, '.xlsx'),
        'Chance_2018_II': (1, '.xlsx'),
        'Chance_2018_III': (1, '.xlsx'),
        'Chance_UnP_III': (1, '.xlsx'),
        'Croot_UnP': (1, '.xlsx'),
        'Cutter_2018': (1, '.xlsx'),
        'Zhou_2017': (1, '.xlsx'),
        'Campos_UnP': (1, '.xlsx'),
        'Chance_UnP_IV': (1, '.xlsx'),
        'Chance_UnP_V': (1, '.xlsx'),
    }
    return d[Data_Key]


def build_comparisons_MASTER_obs_vs_extracted_data(dpi=320,
                                                   show_plot=False,):
    """
    Check the extract data against the values used previously

    Parameters
    -------
    show_plot (bool): show the plot on screen
    dpi (int): dots per inch of saved plot
    """
    import seaborn as sns
    sns.set(color_codes=True)
    current_palette = sns.color_palette("colorblind")
    sns.set_style("darkgrid")
    # Get data from Chance2014
    df_obs = get_MASTER_Chance2014_iodide_obs_file(sheetname='<20 m all data')
    # Get meta data on observations
    metadata_df = get_iodide_obs_metadata()
    # Get the processed obs+extracted data ...
    df = get_processed_df_obs_mod()  # NOTE this df contains values >400nM
    # Use the 'unique ID' variable (not unique!) to map Data_Key_ID to obs. data
    var2map = 'Unique id'
    TMS_ID_KEY = u'Data_Key_ID'
    mapping_dict = dict(zip(df[var2map].values, df[TMS_ID_KEY].values))
    df_obs[TMS_ID_KEY] = df_obs[var2map].map(mapping_dict)
    # Map Chance2014 variable names against TMS' extracted ones
    vars_dict = {
        u'vd MLD': 'WOA_MLDvd',
        u'pt MLD': 'WOA_MLDpt',
        u'pd MLD': 'WOA_MLDpd',
        'WOA05 SST': 'WOA_TEMP',
        'WOA01 salinity': 'WOA_Salinity',
        u'WOA05 Nitrate': 'WOA_Nitrate',
        'Latitude': 'Latitude',
        u'Longitude': 'Longitude',
        #    u'Coastal':'Coastal',
    }
    # Rename the variables that share names
    EXCEL_names = list(sorted(vars_dict.keys()))
    NEW_names = [vars_dict[i] for i in EXCEL_names]
    UPDATED_EXCEL_names = [i+' (Chance2014)' for i in NEW_names]
    vars_dict = dict(zip(UPDATED_EXCEL_names, NEW_names))
    UPDATED_EXCEL_names_d = dict(zip(EXCEL_names, UPDATED_EXCEL_names))
    df_obs.rename(columns=UPDATED_EXCEL_names_d, inplace=True)
    vars2plot = vars_dict.keys()
    # Reindex df to use df_obs
    df_obs.index = df_obs[TMS_ID_KEY].values
    # Select data keys inDataFrame that are not duplicated
    Data_Key_IDs = df_obs[TMS_ID_KEY].drop_duplicates(keep=False)
    df_obs = df_obs.loc[df_obs[TMS_ID_KEY].isin(Data_Key_IDs), :]
    df_obs.index = df_obs[TMS_ID_KEY].values
    df_obs.sort_index(inplace=True)
    # Only consider extract values where  'unique ID' vairable is present
    df.index = df[TMS_ID_KEY].values
    df = df.loc[df[TMS_ID_KEY].isin(Data_Key_IDs), :]
    df.sort_index(inplace=True)
    # Now combine
    df = pd.concat([df, df_obs[UPDATED_EXCEL_names]], axis=1)
    # Setup PDF to save plots to
    savetitle = 'Oi_prj_Compare_extracted_Chance2014_vs_ML_work'
    pdff = AC.plot2pdfmulti(title=savetitle, open=True, dpi=dpi)
    # Plot up vairables
    for var_ in vars2plot:
        print(var_, vars_dict[var_])
        # Just look at variables of interest
        df_tmp = df[[var_, vars_dict[var_]]]
        if 'MLD' in var_:
            for col in df_tmp.columns:
                df_tmp.loc[df_tmp[col] == -99.9, col] = np.NaN
        # Drop NaNs (otherwise regression plot will fail
        df_tmp = df_tmp.dropna()
        ax = sns.regplot(x=var_, y=vars_dict[var_], data=df_tmp)
        title = 'Comparison of extracted values used in Chance2014 vs. ML work'
        plt.title(title)
        # Save to PDF and close plot
        AC.plot2pdfmulti(pdff, savetitle, dpi=dpi)
        if show_plot:
            plt.show()
        plt.close()
    # Save entire pdf
    AC.plot2pdfmulti(pdff, savetitle, close=True, dpi=dpi)


def get_MASTER_Chance2014_iodide_obs_file(sheetname='S>30 data set',
                                          skiprows=1,
                               file_and_path='./sparse2spatial.rc',):
    """
    To check on the correlations between the newly extract climatological
    values, this funtion extracts the details from Chance2014's master
    spreadsheet to perform comparisons.

    Parameters
    -------
    sheetname (str): name of the excel sheet to use
    skiprows (int): number of rows to skip when reading sheet
    file_and_path (str): folder and filename with location settings as single str

    Returns
    -------
    (pd.DataFrame)
    """
    # Location and filename?
    filename = 'Iodide_correlations_310114_MASTER_TMS_EDIT.xlsx'
    folder = utils.get_file_locations('data_root', file_and_path=file_and_path)
    folder += 'Iodide/inputs/RJC_spreadsheets/'
    # Extract MASTER excel spreadsheet from Chance2014
    df = pd.read_excel(folder+filename, sheetname=sheetname, skiprows=skiprows)
    return df


def add_extra_vars_rm_some_data(df=None, target='Iodide',
                                restrict_data_max=False,
                                restrict_min_salinity=False,
                                use_median4chlr_a_NaNs=False,
                                median_4MLD_when_NaN_or_less_than_0=False,
                                median_4depth_when_greater_than_0=False,
                                rm_LOD_filled_data=False,
                                add_modulus_of_lat=False,
                                rm_Skagerrak_data=False, rm_outliers=False,
                                verbose=True, debug=False):
    """
    Add, process, or remove (requested) derivative variables for use with ML code

    Parameters
    -------
    restrict_data_max (bool): restrict the obs. data to a maximum value?
    restrict_min_salinity (bool): restrict the obs. data to a minimum value of salinity?
    use_median4chlr_a_NaNs (bool): use median values for Chl-a if it is a NaN
    median_4depth_when_greater_than_0 (bool): use median values for depth if it is <0
    median_4MLD_when_NaN_or_less_than_0 (bool): use median values for MLD if it is <0
    rm_LOD_filled_data (bool): remove the observational values below LOD
    add_modulus_of_lat (bool): add the modulus of lat to dataframe
    rm_Skagerrak_data (bool): remove the data from the Skagerrak region
    rm_outliers (bool): remove the observational outliers from the dataframe
    verbose (bool): print verbose statements to screen
    debug (bool): print debugging statements to screen

    Returns
    -------
    (pd.DataFrame)
    """
    # - Apply choices & Make user aware of choices applied to data
    Shape0 = str(df.shape)
    N0 = df.shape[0]
    if rm_outliers:
        bool = df[target] < utils.get_outlier_value(df=df, var2use=target)
        df_tmp = df.loc[bool]
        prt_str = 'Removing outlier {} values. (df {}=>{},{})'
        N = int(df_tmp.shape[0])
        if verbose:
            print(prt_str.format(target, Shape0, str(df_tmp.shape), N0-N))
        df = df_tmp
    # - Remove the outliers (N=19 for v8.1)
    if restrict_data_max:
        df_tmp = df[df[target] < 400.]  # updated on 180611 (& commented out)
        prt_str = 'Restricting max {} values. (df {}=>{},{})'
        N = int(df_tmp.shape[0])
        if verbose:
            print(prt_str.format(target, Shape0, str(df_tmp.shape), N0-N))
        df = df_tmp
    # - Remove the Skagerrak data
    if rm_Skagerrak_data:
        bool1 = df['Data_Key'].values == 'Truesdale_2003_I'
        index2drop = df.loc[bool1, :].index
        df_tmp = df.drop(index2drop)
        N = int(df_tmp.shape[0])
        prt_str = 'Removing Skagerrak data (Truesdale_2003_I). (df {}=>{}, {})'
        if verbose:
            print(prt_str.format(Shape0, str(df_tmp.shape), N0-N))
        df = df_tmp
    # - Remove the Skagerrak data
    if rm_LOD_filled_data:
        # ErrorFlag = 7 - Value <LoD; replaced with substitute value
        bool1 = df['ErrorFlag'].values == 7
        index2drop = df.loc[bool1, :].index
        df_tmp = df.drop(index2drop)
        N = int(df_tmp.shape[0])
        prt_str = 'Removing LOD filled data. (df {}=>{}, {})'
        if verbose:
            print(prt_str.format(Shape0, str(df_tmp.shape), N0-N))
        df = df_tmp
    # - Restrict the minimum salinity ( coastal vs. open-ocean)
    if restrict_min_salinity:
        df_tmp = df[df['WOA_Salinity'] > 30.]
        prt_str = 'Restricting min Salinity values (>30). (df {}=>{})'
        if verbose:
            print(prt_str.format(str(df.shape), str(df.shape)))
        df = df_tmp
    # - Chlorophyll arrays are prone to NaNs...
    if use_median4chlr_a_NaNs:
        # Average value
        avg = df['SeaWIFs_ChlrA'].median()
        # Function to map

        def swp_NaN4_median(input, avg=avg):
            """ swap NaNs for median """
            if np.isfinite(input):
                return input
            else:
                return avg
        # Map function...
        numerics = pd.to_numeric(df['SeaWIFs_ChlrA'].values, errors='coerce')
        N_finites = numerics[np.isfinite(numerics)].shape[0]
        N_NaNs = df.shape[0] - N_finites
        df['SeaWIFs_ChlrA'] = df['SeaWIFs_ChlrA'].copy().map(swp_NaN4_median)
        ver_str = 'SeaWIFs_ChlrA: Swapped NaNs for median values (N={})'
        if verbose:
            print(ver_str.format(N_NaNs))
    # - MLD values have -100 as fill arrays are prone to NaNs...
    if median_4MLD_when_NaN_or_less_than_0:
        # Get MLD variables
        MLD_vars = [i for i in df.columns if ('WOA_MLD' in i)]
        MLD_vars = [i for i in MLD_vars if ('flagged' not in i)]
        for var_ in MLD_vars:
            bool_ = df[var_] < 0
            df.loc[bool_, var_] = np.NaN
            bool_ = ~np.isfinite(df[var_].values)
            df.loc[bool_, var_] = np.NaN
            # Save number of NaNs
            N_NaNs = df.loc[bool_, var_].shape[0]
            # Get the average
            avg = df[var_].median()
            # Function to map

            def swp_NaN4_median(input, avg=avg):
                """ swap NaNs for median """
                if np.isfinite(input):
                    return input
                else:
                    return avg
            # Swap NaNs for median values
            df[var_] = df[var_].copy().map(swp_NaN4_median)
            # Print to screen any swaps
            ver_str = '{}: Swapped NaNs for median values (N={})'
            if verbose:
                print(ver_str.format(var_, N_NaNs))
    # - Depth values have -100 as fill arrays are prone to NaNs...
    if median_4depth_when_greater_than_0:
        # Get MLD variables
        var_ = 'Depth_GEBCO'
        bool_ = df[var_] > 0
        df.loc[bool_, var_] = np.NaN
        bool_ = ~np.isfinite(df[var_].values)
        df.loc[bool_, var_] = np.NaN
        # Save number of NaNs
        N_NaNs = df.loc[bool_, var_].shape[0]
        # Get the average
        avg = df[var_].median()
        # Function to map

        def swp_NaN4_median(input, avg=avg):
            """ swap NaNs for median """
            if np.isfinite(input):
                return input
            else:
                return avg
        # Swap NaNs for median values
        df[var_] = df[var_].copy().map(swp_NaN4_median)
        #
        ver_str = '{}: Swapped NaNs for median values (N={})'
        if verbose:
            print(ver_str.format(var_, N_NaNs))
    # - Add temperature in Kelvin
    new_var = 'WOA_TEMP_K'
    if (new_var not in df.columns):
        df[new_var] = df['WOA_TEMP'] + 273.15
        if verbose:
            print('Added Temp in K to df')
    # - Add modulus of latitude
    new_var = 'Latitude (MOD)'
    if (new_var not in df.columns) and add_modulus_of_lat:
        df[new_var] = np.sqrt(df['Latitude']**2)
        if verbose:
            print('Added  modulus of latitude to df')
    # If the dataframe has been updated, it must be re-indexed!
    if Shape0 != str(df.shape):
        pstr = 'WARNING:'*20, 'df shape {}=>{})'
        if debug:
            print(pstr.format(starting_shape, df.shape))
        if debug:
            print('now must be reindexed for ML!!!')
        df.index = np.arange(df.shape[0])
    return df


def convert_old_Data_Key_names2new(df, var2use='Data_Key'):
    """
    Convert Data_Keys in old files to be sama as data desp. paper

    Returns
    -------
    (pd.DataFrame)
    """
    # Variables for version v8.3
    NIU, md_df = get_iodide_obs()
    Data_Keys_8_5_1 = md_df['Data_Key'].values
    # Variables for version v8.2
    filename = 'Iodine_climatology_Submitted_data_list_formatted_TMS_v8_2.xlsx'
    md_df2 = pd.read_excel(folder + filename, sheetname='Full')
    Data_Keys_8_2 = md_df2['Data_Key'].values
    # Map together as dictionary
    d = dict(zip(Data_Keys_8_2, Data_Keys_8_5_1))
    # Add the misspelling
    d['Elderfeild_T_1980'] = 'Elderfield_T_1980'
    # Setup as a mappable function to update columns
    def nename_col(input):
        if input in d.keys():
            return d[input]
        else:
            prt_str = "Input Data_Key '{}' not in dictionary"
            print(prt_str.format(input))
            sys.exit()
    # Map the updates then return dataframe
    df[var2use] = df[var2use].map(nename_col)
    return df


def add_all_Chance2014_correlations(df=None, debug=False, verbose=False):
    """
    Add Chance et al 2014 parameterisations to df (from processed .csv)
    """
    # get details of parameterisations
#    filename='Chance_2014_Table2_PROCESSED_17_04_19.csv'
    filename = 'Chance_2014_Table2_PROCESSED.csv'
    folder = utils.get_file_locations('data_root')
    folder += '/Iodide/'
    param_df = pd.read_csv(folder+filename)
    # map input variables
    input_dict = {
        'C': 'WOA_TEMP',
        'ChlorA': 'SeaWIFs_ChlrA',
        'K': 'WOA_TEMP_K',
        'Lat': 'Latitude',
        'MLDpd': 'WOA_MLDpd',
        'MLDpt': 'WOA_MLDpt',
        'MLDvd': 'WOA_MLDvd',
        'MLDpd_max': 'WOA_MLDpd_max',
        'MLDpt_max': 'WOA_MLDpt_max',
        'MLDvd_max': 'WOA_MLDvd_max',
        'MLDpd_sum': 'WOA_MLDpd_sum',
        'MLDpt_sum': 'WOA_MLDpt_sum',
        'MLDvd_sum': 'WOA_MLDvd_sum',
        'NO3': 'WOA_Nitrate',
        'Salinity': 'WOA_Salinity',
    }
    # - Loop parameterisations and add to dataframe
    for param in param_df['TMS ID'].values:
        sub_df = param_df[param_df['TMS ID'] == param]
        if debug:
            print(sub_df)
        # extract variables
        data = df[input_dict[sub_df.param.values[0]]].values
        #  Function to use?
        func2use = str(sub_df.function.values[0])
        if debug:
            print(func2use)
        # Do any functions on the data
        if func2use == 'None':
            pass
        elif func2use == 'abs':
            data = abs(data)
        elif func2use == 'inverse':
            data = 1./data
        elif func2use == 'square':
            data = data**2
#        elif func2use == 'max':
#            print 'Need to add max option!'
#        elif func2use == 'sum':
#            print 'Need to add sum option!'
        else:
            print('function not in list')
            sys.exit()
#        if not isinstance(func2use, type(None) ):
#            data = func2use(data)
        # apply linear scaling
        m, c = [sub_df[i].values[0] for i in ['m', 'c']]
#        print [ (type(i), i) for i in m, c,data ]
        data = (m*data) + c
        # now add to dictionary
        df[param] = data
    return df


def get_literature_predicted_iodide(df=None, verbose=True, debug=False):
    """
    Get predicted iodide from literature parametersations
    """
    # Set local variables
    TEMPvar = 'WOA_TEMP'  # temperature
    # Add temperature in Kelvin to array
    TEMPvar_K = TEMPvar+'_K'
    try:
        df[TEMPvar_K]
    except KeyError:
        print('Adding temperature in Kelvin')
        df[TEMPvar+'_K'] = df[TEMPvar].values+273.15
    # Add Modulus to Dataframe, if not present
    MOD_LAT_var = "Latitude (Modulus)"
    try:
        df[MOD_LAT_var]
    except KeyError:
        print('Adding modulus of Latitude')
        df[MOD_LAT_var] = np.sqrt(df["Latitude"].copy()**2)
    # Other variables used in module
    NO3_var = u'WOA_Nitrate'
    sumMLDpt_var = 'WOA_MLDpt_sum'
    salinity_var = u'WOA_Salinity'
    # - Function to calculate Chance et al. (2014) correlation
    # functions to calculate (main) Chance et al correlation
    # In order of table 2 from Chance et al. (2014)
    # Add two main parameterisations to dataframe
    # Chance et al. (2014)
    var2use = 'Chance2014_STTxx2_I'
    try:
        df[var2use]
    except KeyError:
        df[var2use] = df[TEMPvar].map(utils.calc_I_Chance2014_STTxx2_I)
    # MacDonald et al. (2014)
    var2use = 'MacDonald2014_iodide'
    try:
        df[var2use]
    except KeyError:
        df[var2use] = df[TEMPvar].map(utils.calc_I_MacDonald2014)
    # Add all parameterisations from Chance et al (2014) to dataframe
    df = add_all_Chance2014_correlations(df=df, debug=debug)
#    print df.shape
    # Add multivariate parameterisation too (Chance et al. (2014))
    # Chance et al. (2014). multivariate
    var2use = 'Chance2014_Multivariate'
    try:
        df[var2use]
    except KeyError:
        df[var2use] = df.apply(lambda x:
                               utils.calc_I_Chance2014_multivar(NO3=x[NO3_var],
                                                                sumMLDpt=x[sumMLDpt_var],
                                                                MOD_LAT=x[MOD_LAT_var],
                                                                TEMP=x[TEMPvar],
                                                                salinity=x[salinity_var]),
                                                                axis=1)
    return df

