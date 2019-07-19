#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Function to process sklearn RandomForestRegressors .pkl files to csv files for TreeSurgeon
which can then be read in by forrester's nope.js plotter functions.


NOTE(s):
 - The TreeSurgeon plotting code is archived separately
https://github.com/wolfiex/TreeSurgeon
"""
from __future__ import print_function
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# s2s modules imports
import sparse2spatial.RFRbuild as build
import sparse2spatial.RFRanalysis as analysis

def main( target=None ):
    """
    Driver to make summary csv files from sklearn RandomForestRegressor models

    Parameters
    -------
    target (str), Name of the target variable (e.g. iodide)
    """
    # Get dictionaries of feature variables, model names etc...
    RFR_dict = get_RFR_dictionary_local()
    # Otherwise could use Iodide in the scripts folder
#    RFR_dict = build_or_get_models_iodide(rebuild=False)
    # Extract the pickled sklearn RandomForestRegressor models to .dot files
    analysis.extract_trees_to_dot_files( target='Iodide', folder=None )

    # Analyse the nodes in the models
    analysis.analyse_nodes_in_models( RFR_dict=RFR_dict )


def get_RFR_dictionary_local():
    """
    Read in RandomForestRegressor variables

    Returns
    -------
    (dict)

    Notes
    -------
     - This is just pseudo code listing the vaiables that are required to be in the
     dictionary
     - This pseudo code can be used or code from the scripts directory can be used
    """
    # Setup a dictionary object
    RFR_dict = {}
    # Add model names and models
    # RFR_dict['models_dict'] = {'name of model': model, ...}
    # Add testing features for models
    # RFR_dict['testing_features_dict'] = {'name of model': testing features of model, ...}
    # Add list of the topmodels (models to analyse)
    # RFR_dict['topmodels'] = [...]

    return RFR_dict


if __name__ == "__main__":
    main()

