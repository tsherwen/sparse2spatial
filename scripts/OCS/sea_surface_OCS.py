

import numpy as np
import pandas as pd
# import AC_tools (https://github.com/tsherwen/AC_tools.git)
import AC_tools as AC
import matplotlib
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
# s2s imports
import sparse2spatial.RFRanalysis as RFRanalysis
import sparse2spatial.analysis as analysis
import sparse2spatial.RFRbuild as build
import sparse2spatial.utils as utils
from sparse2spatial.RFRbuild import build_or_get_models
from sparse2spatial.RFRbuild import get_top_models
#from sparse2spatial.RFRanalysis import get_stats_on_models
#from sparse2spatial.RFRanalysis import get_stats_on_multiple_global_predictions
# Local modules specific to OCS work
import observations as obs



def explore_values_per_hour(df, target='OCS', dpi=320, debug=False):
    """
    Explore the concentrations of OCS on a hourly basis
    """
    import seaborn as sns
    sns.set(style="whitegrid")
    # - get the data
    df = obs.get_OCS_obs()
    N0 = float(df.shape[0])
    # drop the NaNs
    df.dropna()

    # - plot up the values by hour
    hrs = df['Hour'].value_counts(dropna=True)
    hrs = hrs.sort_index()
    N = float(hrs.sum())
    ax = hrs.plot.bar(x='hour of day', y='#', rot=0)
    title_str = '{} data that includes measured hour \n (N={}, {:.2f}% of all data)'
    plt.title( title_str.format(target, int(N), N/N0*100 ) )
    # Update the asthetics
    time_labels = hrs.index.values
    # make sure the values with leading zeros drop these
    index = [float(i) for i in time_labels]
    # Make the labels into strings of integers
    time_labels = [str(int(i)) for i in time_labels]
    if len(index) < 6:
        ax.set_xticks(index)
        ax.set_xticklabels(time_labels)
    else:
        ax.set_xticks(index[2::3])
        ax.set_xticklabels(time_labels[2::3])
    xticks = ax.get_xticks()
    if debug:
        print((xticks, ax.get_xticklabels()))
    # Save the plot
    plt.savefig( 's2s_obs_data_by_hour_{}'.format(target), dpi=dpi)
    plt.close('all')

    # - Plot the data X vs. Y for the obs vs. hour
    x_var, y_var = 'Hour', 'OCS'
    df_tmp = df[ [x_var, y_var] ]
    X = df_tmp[x_var].values
    Y = df_tmp[y_var].values
    # Drop NaNs

    # Plot all data
    fig = plt.figure(dpi=dpi, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(111)
    alpha = 0.5
    plt.scatter(X, Y, color='red', s=3, facecolor='none', alpha=alpha)
    # plot linear fit line


    # Now plot
#    AC.plt_df_X_vs_Y( df=df_tmp, x_var=x_var, y_var=y_var, save_plot=True )
    png_filename = 'X_vs_Y_{}_vs_{}'.format(x_var, y_var)
    png_filename = AC.rm_spaces_and_chars_from_str(png_filename)
    plt.savefig(png_filename, dpi=dpi)


    plt.close('all')
