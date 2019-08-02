#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Get the data for ancillary variables (e.g. WOA temperature)
"""
from __future__ import print_function
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sparse2spatial.utils as utils
import gc

def main():
    """
    Driver to download all of the ancillary data files
    """
    # Get data from World Ocean (WOA) 2013 version 2
#    get_WOA13_data()
    # Get data from NODC (Levitus) World Ocean Atlas 1994
#    get_WOA94_data()
    # Get data from NODC World Ocean Atlas 2001
#    get_WOA01_data()
    # GEBCO’s gridded bathymetric data set
#    get_GEBCO_data()
    # Get data for Dissolved Organic Carbon (DOC)
#    get_DOC_data()
    # Get data for Shortwave radiation (Large and Yeager, 2009)
#    get_SWrad_data()
    # Get data for chlorophyll-a from SeaWIFS
#    get_SeaWIFS_data()
    # Get data for Productivity (Behrenfeld and Falkowski, 1997)
#    get_productivity_data()
    # Get the data from World Ocean Atlas 2018
    get_WOA18_data()

def get_WOA18_data(automatically_download=False, target='Iodide'):
    """
    Get data from World Ocean (WOA) 2018 version 2

    Notes
    -------
    https://www.nodc.noaa.gov/OC5/woa18/woa18-preliminary-notes.html
    """
    # Use the data settings for Iodide
    file_and_path = './{}/sparse2spatial.rc'.format(target)
    data_root = utils.get_file_locations('data_root', file_and_path=file_and_path)
    folder = '{}/data/{}/'.format(data_root, 'WOA18')
    # Now loop through the list of variables to donwload
    vars_dict2download = store_of_values2download4WOA18()
    for n in vars_dict2download.keys():
        print(n, vars_dict2download[n].items())
        # Extract variables
        d = vars_dict2download[n]
        var = d['var']
        res = d['res']
        period =d['period']
        # Which specific subfolder to save data to?
        sfolder = '{}/{}/'.format(folder, var)
        # If seasonal data (decadal averaged) then download the monthd
        if (period == 'decav') or (period == 'all'):
            # get monthly and seasonal files
            seasons = ['{:0>2}'.format(i+1) for i in np.arange(16)]
            # download files for season
            for season in seasons:
                WOA18_data4var_period(folder=sfolder, season=season, period=period,
                                      res=res, var=var)
        # If decadal split data, down load by season
        else:
            # just get seasonal files
            seasons = ['{:0>2}'.format(i) for i in [13,14,15,16]]
            # download files for season
            for season in seasons:
                WOA18_data4var_period(folder=sfolder, season=season, period=period,
                                      res=res, var=var)


def store_of_values2download4WOA18():
    """
    Store a dictionary of which files to download at which resolutions
    """
    d = {
    # Climatological data fields - temperature
    1 : {'var':'temperature', 'res':'0.25', 'period':'decav'},
    # Decadal temperature fields - temperature
    2 : {'var':'temperature', 'res':'0.25', 'period':'5564'},
    3 : {'var':'temperature', 'res':'0.25', 'period':'6574'},
    4 : {'var':'temperature', 'res':'0.25', 'period':'7584'},
    5 : {'var':'temperature', 'res':'0.25', 'period':'8594'},
    6 : {'var':'temperature', 'res':'0.25', 'period':'95A4'},
    7 : {'var':'temperature', 'res':'0.25', 'period':'A5B7'},
    # Climatological data fields - salinity
    8 : {'var':'salinity', 'res':'0.25', 'period':'decav'},
    # Decadal salinity fields - salinity
    9 : {'var':'salinity', 'res':'0.25', 'period':'5564'},
    10 : {'var':'salinity', 'res':'0.25', 'period':'6574'},
    11 : {'var':'salinity', 'res':'0.25', 'period':'7584'},
    12 : {'var':'salinity', 'res':'0.25', 'period':'8594'},
    13 : {'var':'salinity', 'res':'0.25', 'period':'95A4'},
    14 : {'var':'salinity', 'res':'0.25', 'period':'A5B7'},
    # Climatological data fields - nitrate
    15 : {'var':'nitrate', 'res':'1.00', 'period':'all'},
    # Climatological data fields - phosphate
    16 : {'var':'phosphate', 'res':'1.00', 'period':'all'},
    # Climatological data fields - oxygen
    17 : {'var':'oxygen', 'res':'1.00', 'period':'all'},
    # Climatological data fields - silicate
    18 : {'var':'silicate', 'res':'1.00', 'period':'all'},
    }
    return d


def WOA18_data4var_period(var='temperature', res='0.25', period='decav', season='16',
                          res_str=None, folder=None, verbose=False):
    """
    Download a specific file via opendap from the NOAA/NCEI host for WOA
    """
    # Print a status message on which files are being downloaded
    if verbose:
        ptr_str = "ATTEMPTING: access+download of '{}' for '{}' (res={}, season={})"
        print(locals())
    # Root URL str
    URL_root = 'https://data.nodc.noaa.gov/'
    # Folder name
    folder_str = '/thredds/dodsC/ncei/woa/{var}/{period}/{res}/'
    folder_str = folder_str.format(var=var, period=period, res=res)
    # Get the resolution string used in filename str for given folder res.
    if isinstance(res_str, type(None)):
        res_str = get_res_str4WOA18(res)
    # Get the prefix for a given variable
    prefix = get_prefix4WOA18(var)
    # Setup the filename to downlaod (e.g. woa18_decav_t16_04.nc)
    filestr = 'woa18_{}_{}{}_{}.nc'.format(period, prefix, season, res_str )
    # Using xarray (issues found with NASA OpenDAP data model - via PyDAP)
    url_str = URL_root+folder_str+filestr
    print( url_str )
    ds = xr.open_dataset(url_str, decode_times=False)
    # Print a status message on which files are being downloaded
    if verbose:
        print('RETRIEVED: {}'.format(url_str))
    # Save the dataset locally.
    ds.to_netcdf(folder+filestr)
    # Print a status message on which files are being downloaded
    if verbose:
        print('SAVED: {}'.format(folder+filestr))
    # Remove the dataset from memory and call the garbage collector
    del ds
    gc.collect()


def get_res_str4WOA18(input):
    """
    Convert WOA folder resolution str into filename's resolution str.
    """
    d = {
    '0.25': '04',
    '1.00': '01',
    '5deg': '5d',
    }
    return d[input]


def get_prefix4WOA18(input):
    """
    Convert WOA18 variable name string into the prefix used on files
    """
    d = {
    'temperature': 't',
    'salinity': 's',
    'phosphate': 'p',
    'nitrate':'n',
    'oxygen':'o',
    'silicate':'i',
    }
    return d[input]


def get_WOA13_data(automatically_download=False):
    """
    Get data from World Ocean (WOA) 2013 version 2
    """
    # Local variables describing host
    dataset = 'WOA13'
    host = 'https://www.nodc.noaa.gov/OC5/woa13/'
    reference = "Locarnini, R. A., A. V. Mishonov, J. I. Antonov, T. P. Boyer, H. E. Garcia, O. K. Baranova, M. M. Zweng, C. R. Paver, J. R. Reagan, D. R. Johnson, M. Hamilton, and D. Seidov, 2013: World Ocean Atlas 2013, Volume 1: Temperature. S. Levitus, Ed., A. Mishonovte"
    # Download or print that the data needs to be got directly from the data originator
    if automatically_download:
        print("WARNING: automatic download not setup for '{}'".format(dataset))
    else:
        print_data_not_archived_offline(dataset=dataset, host=host,
                                        reference=reference)


def get_WOA94_data(automatically_download=False):
    """
    Get data from NODC (Levitus) World Ocean Atlas 1994
    """
    # Local variables describing host
    dataset = 'WOA94'
    host = 'https://www.esrl.noaa.gov/psd/data/gridded/data.nodc.woa94.html'
    reference = 'Monterey, G. I., and S. Levitus, 1997:  Climatological cycle of mixed layer depth in the world ocean.  U.S. Gov. Printing Office, NOAA NESDIS, 5pp.'
    # Download or print that the data needs to be got directly from the data originator
    if automatically_download:
        print("WARNING: automatic download not setup for '{}'".format(dataset))
    else:
        print_data_not_archived_offline(dataset=dataset, host=host,
                                        reference=reference)


def get_WOA01_data(automatically_download=False):
    """
    Get data from NODC World Ocean Atlas 2001
    """
    # Local variables describing host
    dataset = 'WOA01'
    host = 'https://www.nodc.noaa.gov/OC5/WOA01/pr_woa01.html'
    reference = 'Conkright, M.E., R. A. Locarnini, H.E. Garcia, T.D. O’Brien, T.P. Boyer, C. Stephens, J.I. Antonov,2002: World Ocean Atlas 2001: Objective Analyses, Data Statistics, and Figures, CD-ROM Documentation. National Oceanographic Data Center, Silver Spring, MD, 17 pp.'
    # Download or print that the data needs to be got directly from the data originator
    if automatically_download:
        print("WARNING: automatic download not setup for '{}'".format(dataset))
    else:
        print_data_not_archived_offline(dataset=dataset, host=host,
                                        reference=reference)


def get_GEBCO_data(automatically_download=False):
    """
    Get GEBCO’s gridded bathymetric data set
    """
    # Local variables describing host
    dataset = 'WOA01'
    host = 'https://www.gebco.net/data_and_products/gridded_bathymetry_data/'
    reference = None
    # Download or print that the data needs to be got directly from the data originator
    if automatically_download:
        print("WARNING: automatic download not setup for '{}'".format(dataset))
    else:
        print_data_not_archived_offline(dataset=dataset, host=host,
                                        reference=reference)


def get_DOC_data(automatically_download=False):
    """
    Get data for Dissolved Organic Carbon (DOC)
    """
    # Local variables describing host
    dataset = 'DOC'
    host = None
    reference = 'Roshan, S. and DeVries, T.: Efficient dissolved organic carbon production and export in the oligotrophic ocean, Nature Communications, 8, 2036, https://doi.org/10.1038/s41467-017-02227-3, 2017.'
    # Download or print that the data needs to be got directly from the data originator
    if automatically_download:
        print("WARNING: automatic download not setup for '{}'".format(dataset))
    else:
        print_data_not_archived_offline(dataset=dataset, host=host,
                                        reference=reference)


def get_productivity_data(automatically_download=False):
    """
    Get data for Productivity (Behrenfeld and Falkowski, 1997)
    """
    # Local variables describing host
    dataset = 'Productivity'
    host = None
    reference = "Behrenfeld, M.J. and Falkowski, P.G., 1997. Photosynthetic rates derived from satellite‐based chlorophyll concentration. Limnology and oceanography, 42(1), pp.1-20."
    # Download or print that the data needs to be got directly from the data originator
    if automatically_download:
        print("WARNING: automatic download not setup for '{}'".format(dataset))
    else:
        print_data_not_archived_offline(dataset=dataset, host=host,
                                        reference=reference)


def get_SeaWIFS_data(automatically_download=False):
    """
    Get data for chlorophyll-a from SeaWIFS
    """
    # Local variables describing host
    dataset = 'SeaWIFS'
    host = None
    reference = "OBPG: NASA Goddard Space Flight Center, Ocean Biology Processing Group: Sea-viewing Wide Field-of-view Sensor (SeaWiFS) Ocean Color Data, NASA OB.DAAC, Greenbelt, MD, USA. Maintained by NASA Ocean Biology Distributed Active Archive Center (OB.DAAC), Goddard Spa, https://doi.org/http://doi.org/10.5067/ORBVIEW-2/SEAWIFS_OC.2014.0, 2014."
    # Download or print that the data needs to be got directly from the data originator
    if automatically_download:
        print("WARNING: automatic download not setup for '{}'".format(dataset))
    else:
        print_data_not_archived_offline(dataset=dataset, host=host,
                                        reference=reference)


def get_SWrad_data(automatically_download=False):
    """
    Get data for Shortwave radiation (Large and Yeager, 2009)
    """
    # Local variables describing host
    dataset = 'SWrad'
    host = None
    reference = "Large, W. G. and Yeager, S. G.: The global climatology of an interannually varying air–sea flux data set, Climate Dynamics, 33, 341–364, https://doi.org/10.1007/s00382-008-0441-3, 2009."
    # Download or print that the data needs to be got directly from the data originator
    if automatically_download:
        print("WARNING: automatic download not setup for '{}'".format(dataset))
    else:
        print_data_not_archived_offline(dataset=dataset, host=host,
                                        reference=reference)


def print_data_not_archived_offline(dataset=None, host=None, reference=None):
    """
    Print that the data must be obtained from the data provider directly
    """
    # Print the where to go to find the data
    print('WARNING: Data not archived as it is the property of the data owner')
    print("         Please download the data for '{}' directly".format(dataset))
    if not isinstance(host, type(None)):
        print("         Please find the data hosted here: {}".format(host))
    if not isinstance(reference, type(None)):
        print("         The reference for this data is: {}".format(reference))
    print( '\n')



if __name__ == "__main__":
    main()

