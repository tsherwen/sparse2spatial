"""

Process input data (sea-surface iodide) to a DataFrame for modelling.

This file is intended to run as a standard alone script. It generates the file of
observational data including any new observational files in the provided template file
and combines them with the existing dataset archived with BODC.

To run this script, just execute at command line.

e.g.

python observations.py

Citations
-------

“Global sea-surface iodide observations, 1967-2018”, R. J. Chance, L. Tinel, T. Sherwen , et al., in review, 2019

Sherwen, T., Chance, R. J., Tinel, L., Ellis, D., Evans, M. J., and Carpenter, L. J.: A machine learning based global sea-surface iodide distribution, Earth Syst. Sci. Data Discuss., https://doi.org/10.5194/essd-2019-40, in review, 2019.

Notes
-------
 - Abbreviations used here:
obs. = observations

"""
import numpy as np
import pandas as pd
import sparse2spatial as s2s
from sparse2spatial.utils import get_file_locations
from sparse2spatial.utils import set_backup_month_if_unkonwn


def main(add_ancillaries=True):
    """
    Driver to process sea-surface iodide observations into a single .csv file

    Parameters
    -------
    add_ancillaries (bool): Inc. ancillaries in .csv file for obs. locations?

    Returns
    -------
    (None)
    """
    # - Below lines are to update the BODC iodide dataset to include new obs.
    # Get the latest data from BODC
    df = get_iodide_data_from_BODC()

    # Extract any new files of iodide (excel files)
    df_new = extract_new_observational_excel_files()

    # process these to contain all of the information of the BODC file
    df_new = process_files_to_BODC_format(df_new)

    # Combine with existing BODC sea-surface iodide file
    df = pd.concat([df, df_new, axis=1)

    # Save this new combined file as a .csv
    filename = 'seasurface_iodide_feilds_UPDATED.csv'
    df.to_csv( filename )


def extract_new_observational_excel_files():
    """
    Extract new observations in template excel file
    """
    pass


def process_files_to_BODC_format()
    """
    Process the files to the same format as the BODC dataset
    """
    pass


def get_iodide_data_from_BODC(file_and_path='./sparse2spatial.rc'):
    """
    Get the latest iodide data from .csv file archived with BODC

    Parameters
    -------
    file_and_path (str), folder and filename with location settings as single str
    debug (bool): print debug statements

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
    folder = get_file_locations('data_root', file_and_path=file_and_path)
    filename = 'BODC_seasurface_iodide_data.csv'
    # open .csv file and return
    df = pd.read_csv(folder+filename)
    return df





