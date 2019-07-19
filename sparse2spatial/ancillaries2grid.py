"""

Function for interpolating all variables onto high resolution base grid (~1km-12km)

"""
import numpy as np
import pandas as pd
import xarray as xr
import gc
import xesmf as xe
import sparse2spatial.utils as utils
