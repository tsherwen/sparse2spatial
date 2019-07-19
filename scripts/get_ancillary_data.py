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


def main():
    """
    Driver to download all of the ancillary data files
    """
    # Get data from World Ocean (WOA) 2013 version 2
    get_WOA13_data()
    # Get data from NODC (Levitus) World Ocean Atlas 1994
    get_WOA94_data()
    # Get data from NODC World Ocean Atlas 2001
    get_WOA01_data()
    # GEBCO’s gridded bathymetric data set
    get_GEBCO_data()
    # Get data for Dissolved Organic Carbon (DOC)
    get_DOC_data()
    # Get data for Shortwave radiation (Large and Yeager, 2009)
    get_SWrad_data()
    # Get data for chlorophyll-a from SeaWIFS
    get_SeaWIFS_data()
    # Get data for Productivity (Behrenfeld and Falkowski, 1997)
    get_productivity_data()


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

