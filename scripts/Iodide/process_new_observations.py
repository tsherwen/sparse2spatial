"""

Process input data (sea-surface iodide) to a DataFrame for modelling.

This file is intended to run as a standard alone script. It generates a file of
observational data including any new observational files in the provided template file
and combines them with the existing dataset archived with BODC.

To run this script, just execute at command line.

e.g.

python process_new_observations.py

Citations
-------

 - Data descriptor paper on sea-surface observations
“Global sea-surface iodide observations, 1967-2018”, R. J. Chance, L. Tinel, T. Sherwen , et al., in review, 2019

 - observational data archived at British Oceanographic Data Centre (BODC)
Chance R.; Tinel L.; Sherwen T.; Baker A.; Bell T.; Brindle J.; Campos M.L.A.M.; Croot P.; Ducklow H.; He P.; Hoogakker B.; Hopkins F.E.; Hughes C.; Jickells T.; Loades D.; Reyes Macaya D.A.; Mahajan A.S.; Malin G.; Phillips D.P.; Sinha A.K.; Sarkar A.; Roberts I.J.; Roy R.; Song X.; Winklebauer H.A.; Wuttig K.; Yang M.; Zhou P.; Carpenter L.J.(2019). Global sea-surface iodide observations, 1967-2018. British Oceanographic Data Centre - Natural Environment Research Council, UK. doi:10/czhx.

Notes
-------
 - Abbreviations used here:
obs. = observations

"""
import numpy as np
import pandas as pd
import glob
import sparse2spatial as s2s
import sparse2spatial.utils as utils
from sparse2spatial.utils import set_backup_month_if_unknown


def main():
    """
    Driver to process sea-surface iodide observations into a single .csv file
    """
    # - The below lines update the BODC iodide dataset to include new obs.
    # Get the latest data from BODC
    df = get_iodide_data_from_BODC()

    # Extract any new files of iodide (excel files)
    use_test_data = True # Unhash this line to run with example new data
#    use_test_data = False # Unhash this line to run with new .xlsx files
    df_new = extract_new_observational_excel_files(use_test_data=use_test_data)

    # Process these to contain all of the information of the BODC file
    df_new = process_files_to_BODC_format(df_new, use_test_data=use_test_data)

    # Only include columns in BODC file
    df_new = df_new[[i for i in df_new.columns if i in df.columns]]

    # Combine with existing BODC sea-surface iodide file
    df = pd.concat([df, df_new], ignore_index=True, sort=False)

    # Save this new combined file as a .csv
    filename = 'seasurface_iodide_feilds_UPDATED.csv'
    df.to_csv( filename )


def extract_new_observational_excel_files(file_and_path='./sparse2spatial.rc',
                                          limit_depth_to=20,
                                          use_test_data=True):
    """
    Extract new observations in template excel file

    Parameters
    -------
    file_and_path (str): folder and filename with location settings as single str
    filename (str): name of the csv file or archived data from BODC
    limit_depth_to (float), depth (m) to limit inclusion of data to
    debug (bool), print debug statements

    Returns
    -------
    (pd.DataFrame)
    """
    # Look for files in script folder for a given species
#    folder = utils.get_file_locations('s2s_root', file_and_path=file_and_path)
    folder = './'
    # Run with example new data file?
    if use_test_data:
        # Make a dictionary of data Keys and filenames (without file extensions)
        Data_Key_dict = {
            'EXAMPLE_2019':'Iodine_climatologyXX_NAME_OF_CRUISE'
        }

    else:
        # Check for .xslx files in directory
        filenames = glob.glob(folder+'Iodine_climatology*.xlsx')
        print('FOUND: {} .xlsx files in directory'.format(len(filesnames)))
        # NOTE: below lines will need to be updated when new files are added.
        # Make a dictionary of data Keys and filenames (without file extensions)
        Data_Key_dict = {
            'EXAMPLE_2019':'Iodine_climatologyXX_NAME_OF_CRUISE'
        }
    # Loop keys (Data Keys) for the files for datasets
    dfs = []
    for Data_Key in Data_Key_dict.keys():
        filename = Data_Key_dict[Data_Key]
        # extract and save to list of dataframes
        df = extract_observational_excel_file(folder=folder, filename=filename,
                                              limit_depth_to=limit_depth_to)
        dfs += [df]
        del df
    # Merge into a single dataframe and return
    df = pd.concat(dfs, axis=1)
    return df


def extract_observational_excel_file(folder=None,filename=None, file_extension='.xlsx',
                                     limit_depth_to=20, use_inclusive_limit=False,
                                     debug=False):
    """
    Extract observational data from a template excel file as a dataframe

    Parameters
    -------
    file_and_path (str): folder and filename with location settings as single str
    filename (str): name of the csv file or archived data from BODC
    limit_depth_to (float), depth (m) to limit inclusion of data to
    use_inclusive_limit (bool), limit depth (limit_depth_to) in a inclusive way
    debug (bool), print debug statements

    Returns
    -------
    (pd.DataFrame)
    """
    # Read in data sheet of excel file following template
    skiprows = 1
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
    if debug:
        print(df.columns)
    if use_inclusive_limit:
        df = df.loc[df['Depth'] <= limit_depth_to, :]  # only consider values
    else:
        df = df.loc[df['Depth'] < limit_depth_to, :]  # only consider values
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


def process_files_to_BODC_format(df, use_test_data=False ):
    """
    Process the files to the same format (e.g. columns) as the BODC dataset
    """
    # Get a dictionary of extra variables for each dataset (by its "Data_Key")
    if use_test_data:
        # Reference for dataset
        Reference_dict = {
        'EXAMPLE_2019':'Example et al., 2019'
        }
        # Methods for each new dataset
        Method_dict = {
        'EXAMPLE_2019': 3
        }
        # Whether the data is coastal?
        Coastal_dict = {
        'EXAMPLE_2019': 1
        }
    else:
        # Update dictionary below to contain the references for new datasets
        Reference_dict = {
        'EXAMPLE_2019':'Example et al., 2019'
        }
        # Add methods for each new dataset too
        Method_dict = {
        'EXAMPLE_2019': 3
        }
        # And whether the data is coastal
        Coastal_dict = {
        'EXAMPLE_2019': 1
        }
    # Add the extra column for each of the new datasets
    for key in list(set(df['Data_Key'])):
        # Reference
        df.loc[df['Data_Key'] == key, 'Reference'] = Reference_dict[key]
        # Method
        df.loc[df['Data_Key'] == key, 'Method'] = Method_dict[key]
        # Coastal
        df.loc[df['Data_Key'] == key, 'Coastal'] = Method_dict[key]
        # etc...
    # Return updated dataframe
    return df


def get_iodide_data_from_BODC(file_and_path='./sparse2spatial.rc',
                              filename = 'Global_Iodide_obs_surface.csv'):
    """
    Get the latest iodide data from .csv file archived with BODC

    Parameters
    -------
    file_and_path (str): folder and filename with location settings as single str
    filename (str): name of the csv file or archived data from BODC
    debug (bool), print debug statements

    Returns
    -------
    (pd.DataFrame)
    """
    # print instructions to mannually download data.
    prt_str = 'WARNING: automated download from BODC not yet setup \n'
    prt_str += 'Please mannually download the lastest data from BODC \n'
    prt_str += '*.csv file availble from https://doi.org/10/czhx \n'
    print(prt_str)
    # Location of data
    folder = utils.get_file_locations('s2s_root', file_and_path=file_and_path)
    folder += '/Iodide/inputs/'
    # open .csv file and return
    df = pd.read_csv(folder+filename)
    return df





