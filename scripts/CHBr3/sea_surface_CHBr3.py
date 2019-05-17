#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Module to hold processing/analysis functions for CHBr3 work

Notes
----
ML = Machine Learning
target = the value aiming to be estimated or provided in training
feature = a induivual conpoinet of a predictor vector assigned to a target
( could be called an attribute )
predictor = vector assigned to a target value
"""

import numpy as np
import sparse2spatial as s2s
import sparse2spatial.utils as utils
#import sparse2spatial.ancillaries2grid_oversample as ancillaries2grid
a#import sparse2spatial.archiving as archiving
import sparse2spatial.RTRbuild as build
import sparse2spatial.RFRanalysis as analysis
from sparse2spatial.RTRbuild import build_or_get_current_models

# Get iodide specific functions
from observations import get_dataset_processed4ML


def main():
    """
    Driver for module's man if run directly from command line. unhash
    functionalitliy to call.
    """
    # - Set core local variables
    target = 'CH3Br'








if __name__ == "__main__":
    main()