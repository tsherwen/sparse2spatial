"""

Analysis output from RandomForestRegressor algorithms

"""

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
# import AC_tools (https://github.com/tsherwen/AC_tools.git)
import AC_tools as AC
# Internal loads within s2s
#from sparse2spatial.utils import *
import sparse2spatial.utils as utils
from sparse2spatial.utils import get_df_stats_MSE_RMSE


def get_stats4mulitple_model_builds(model_name=None, RFR_dict=None,
                                    features_used=None, df=None, target='Iodide',
                                    verbose=False):
    """
    Get stats on performance of mutliple model builds on obs. testset

    Parameters
    -------
    target (str), Name of the target variable (e.g. iodide)
    features_used (list), list of the features within the model_name model
    RFR_dict (dict), dictionary of core variables and data
    model_name (str), name of model to build
    df (pd.dataframe), dataframe containing of target and features
    verbose (boolean), print out verbose output?

    Returns
    -------
    (pd.DataFrame)
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.externals import joblib
    from sklearn.metrics import mean_squared_error
    # - Local variables
    # Get unprocessed input data at observation points
    if isinstance(df, type(None)):
        if isinstance(RFR_dict, type(None)):
            RFR_dict = build_or_get_models()
        df = RFR_dict['df']
    # - Get the data
    # get processed data
    # Which "features" (variables) to use
    if isinstance(features_used, type(None)):
        #        model_name = 'ALL'
        #        model_name = 'RFR(TEMP+DEPTH+SAL)'
        features_used = get_model_features_used_dict(model_name)
    # Fix the extra_str variable for now
    extr_str = ''

    # - local variables
    # dictionary of test set variables
    # NOTE: default increase of the base number of n_estimators from 10 to 500
    # Random states to use (to make the plot reproducibility
    random_states = np.arange(25, 45, 1)
    # Location of data
    data_root = utils.get_file_locations('data_root')
    folder = '{}/{}/models/LIVE/'.format(data_root, target)
    # - Predict multiple models and save these
    dfs = {}
    # get random state to use
    for random_state in random_states:
        prt_str = 'Using: random_state = {} to get stats for model = {}'
        if verbose:
            print(prt_str.format(random_state, model_name))
        # Set the training and test sets
        # Stratified split by default, unless random var in name
        returned_vars = mk_test_train_sets(df=df,
                                           rand_20_80=False,
                                           features_used=features_used,
                                           random_state=random_state,
                                           rand_strat=True,
                                           nsplits=4,
                                           )
        train_set, test_set, test_set_targets = returned_vars
        # Set the training and test sets
        train_features = df[features_used].loc[train_set.index]
        train_labels = df[[target]].loc[train_set.index]
        test_features = df[features_used].loc[test_set.index]
        test_labels = df[[target]].loc[test_set.index]
        # Get testset
        # build the model - NOTE: THIS MUST BE RE-DONE
        # ( otherwise the model is being re-trained )
        # Open the already built model model
        model_savename = "my_model_{}_{}.pkl".format(model_name, random_state)
        b_modelname = model_savename.split('my_model_')[-1][:-3]
        loc2use = '{}/{}{}/'.format(folder, '/ENSEMBLE_REPEAT_BUILD', extr_str)
        model = joblib.load(loc2use + model_savename)
        # Predict with model for the test conditions
        predictions = model.predict(test_features)
        # Calculate stats (inc. RMSE) for testset and save
        MSE = mean_squared_error(test_labels, predictions)
        df_tmp = pd.DataFrame(predictions).describe()
        df_tmp = df_tmp.T
        df_tmp['RMSE'] = np.sqrt(MSE)
        df_tmp = df_tmp.T
        df_tmp.columns = [b_modelname]
        dfs[b_modelname] = df_tmp
        # Remove the model from memory
        del model
    # Return a single data frame
    return pd.concat([dfs[i].T for i in dfs.keys()], axis=0)


def get_stats_on_multiple_global_predictions(model_name=None, target='Iodide',
                                             RFR_dict=None, res='0.125x0.125',
                                             rm_Skagerrak_data=False):
    """
    Get stats on the mutiple global predictions per model

    Parameters
    -------
    target (str), name of the target variable being predicted by the feature variables
    model_name (str), name of the RFR model being used to predict the target variable
    RFR_dict (dict), dictionary of models, data and shared variables
    res (str), horizontal resolution of dataset (e.g. 4x5)
    rm_Skagerrak_data (boolean), Remove specific data
    (above argument is a iodide specific option - remove this)

    Returns
    -------
    (pd.DataFrame)
    """
    # Get key data as a dictionary
    if isinstance(RFR_dict, type(None)):
        RFR_dict = build_or_get_models()
        # set the extr_str if rm_Skagerrak_data set to True
    if rm_Skagerrak_data:
        extr_str = '_No_Skagerrak'
    else:
        extr_str = ''
    # Location of data
#    data_root = './' # KLUDGE: use currentfolderwhilst testing
    data_root = utils.get_file_locations('data_root')
    folder_str = '{}/{}/models/LIVE/ENSEMBLE_REPEAT_BUILD{}/'
    folder = folder_str.format(data_root, target, extr_str)
    # Get the folder and filename to use
    file_str = folder + '*{}*ENSEMBLE_BUILDS*{}*.nc'
    file2use = glob.glob(file_str.format(res, model_name))
    assert_str = "There aren't any file for the model! ({})"
    assert len(file2use) != 0, assert_str.format(model_name)
    assert len(file2use) == 1, 'There should only be one file per model!'
    file2use = file2use[0]
    filename = file2use.split('/')[-1]
    folder = '/'.join(file2use.split('/')[:-1]) + '/'
    # Use different drivers depending on resolution
    if res == '0.125x0.125':
        df = get_stats_on_spatial_predictions_0125x0125(filename=filename,
                                                        folder=folder,
                                                        just_return_df=True,
                                                        ex_str=model_name)
    else:
        df = get_stats_on_spatial_predictions_4x5_2x25(filename=filename,
                                                       folder=folder,
                                                       just_return_df=True,
                                                       ex_str=model_name)
    # Remove the values that aren't for a specific model
    df = df[[i for i in df.columns if model_name in i]]
    # return the DataFrame
    return df


def build_the_same_model_mulitple_times(model_name, n_estimators=500,
                                        features_used=None, target='Iodide', df=None,
                                        RFR_dict=None,
                                        testset='Test set (strat. 20%)',
                                        rm_Skagerrak_data=False):
    """
    Build a set of 20 random models based on a single model

    Parameters
    -------
    target (str), Name of the target variable (e.g. iodide)
    testset (str), Testset to use, e.g. stratified sampling over quartiles for 20%:80%
    RFR_dict (dict), dictionary of core variables and data
    model_name (str), name of model to build
    features_used (list), list of the features within the model_name model
    n_estimators (int), number of estimators (decision trees) to use
    df (pd.dataframe), dataframe containing of target and features
    rm_Skagerrak_data (boolean), Remove specific data
    (above argument is a iodide specific option - remove this)

    Returns
    -------
    (None)
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.externals import joblib
    # ----- Local variables
    # Get unprocessed input data at observation points
    if isinstance(RFR_dict, type(None)):
        RFR_dict = build_or_get_models(
            rm_Skagerrak_data=rm_Skagerrak_data
        )
    if isinstance(df, type(None)):
        df = RFR_dict['df']
        # extr_str
    if rm_Skagerrak_data:
        extr_str = '_No_Skagerrak'
    else:
        extr_str = ''
    # ---- get the data
    # get processed data
    # Which "features" (variables) to use
    if isinstance(features_used, type(None)):
        #        model_name = 'RFR(TEMP+DEPTH+SAL)'
        #        features_used = get_model_features_used_dict(model_name)
        print('please provided features_used to build_the_same_model_mulitple_times')
        sys.exit()
    # dictionary of test set variables
    # NOTE: default increase of the base number of n_estimators from 10 to 500
    # Random states to use (to make the plot reproducibility
    random_states = np.arange(25, 45, 1)
    #  location of data
    data_root = utils.get_file_locations('data_root')
    folder_str = '{}/{}/models/LIVE/ENSEMBLE_REPEAT_BUILD{}/'
    folder = folder_str.format(data_root, target, extr_str)

    # - build multiple models and save these
    # get random state to use
    for random_state in random_states:
        prt_str = 'Using: random_state = {} to build model = {}'
        print(prt_str.format(random_state, model_name))
        # set the training and test sets
        # Stratified split by default, unless random var in name
        returned_vars = mk_test_train_sets(df=df,
                                           rand_20_80=False,
                                           features_used=features_used,
                                           random_state=random_state,
                                           rand_strat=True,
                                           nsplits=4,
                                           )
        train_set, test_set, test_set_targets = returned_vars
        # set the training and test sets
        train_features = df[features_used].loc[train_set.index]
        train_labels = df[[target]].loc[train_set.index]
        test_features = df[features_used].loc[test_set.index]
        test_labels = df[[target]].loc[test_set.index]
        # Get testset
        # build the model - NOTE THIS MUST BE RE-DONE!
        # ( otherwise the model is being re-trained )
        model = RandomForestRegressor(random_state=random_state,
                                      n_estimators=n_estimators, criterion='mse')
#            , oob_score=oob_score)
        # fit the model
        model.fit(train_features, train_labels)
        # Save the newly built model model
        model_savename = "my_model_{}_{}.pkl".format(model_name, random_state)
        joblib.dump(model, folder+model_savename)
        # remove the model from memory
        del model


def run_tests_on_testing_dataset_split_quantiles(model_name=None,
                                                 features_used=None, target='Iodide',
                                                 df=None,
                                                 n_estimators=500):
    """
    Run tests on the sensitivity of model to test/training choices

    Parameters
    -------
    target (str), Name of the target variable (e.g. iodide)
    features_used (list), list of the features within the model_name model


    Returns
    -------

    Notes
    -----
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.externals import joblib
    # ----- Local variables
    # Get unprocessed input data at observation points
    if isinstance(df, type(None)):
        df = get_processed_df_obs_mod()  # NOTE this df contains values >400nM
    # ---- get the data
    # get processed data
    # Which "features" (variables) to use
    if isinstance(features_used, type(None)):
        #        model_name = 'ALL'
        model_name = 'RFR(TEMP+DEPTH+SAL)'
        features_used = get_model_features_used_dict(model_name)

    # --- local variables
    # dictionary of test set variables
    random_split_var = 'rn. 20%'
    strat_split_var = 'strat. 20%'
    # NOTE: increase default the base number of n_estimators from 10 to 100
    # Random states to use (to make the plot reproducibility
    random_states = np.arange(25, 45, 1)
    # Formatted variable name for target
    if target == 'Iodide':
        Iaq = '[I$^{-}_{aq}$]'
    else:
        Iaq = target

    # --- set testset to evaulte
    TSETS = {}
    TSETS_N = {}
    TSETS_nsplits = {}
    # - no vals above 400
    Tname = '{}<400'.format(Iaq)
    tmp_ts = df.loc[df[target] < 400][features_used+[target]].copy()
    TSETS_N[Tname] = tmp_ts.shape[0]
    TSETS[Tname] = tmp_ts
    TSETS_nsplits[Tname] = 5

    # - add teast for quartiles choosen
    Tname2copy = '{}<400'.format(Iaq)
    nsplits = np.arange(1, 11, 1)
    for nsplit in nsplits:
        Tname = '{} \n (Q={})'.format(Tname2copy, nsplit)
        TSETS[Tname] = TSETS[Tname2copy].copy()
        TSETS_N[Tname] = TSETS[Tname2copy].shape[0]
        TSETS_nsplits[Tname] = nsplit
    # remove the now double up of nsplit=5 for Tname2copy
    del TSETS[Tname2copy]

    # ---  build models using testsets
    # setup Dataframe to store values
    RMSE_df = pd.DataFrame()
    # Now loop TSETS
    for Tname in TSETS.keys():
        # Initialise lists to store data in
        RMSE_l = []
        # get random state to use
        for random_state in random_states:
            print('Using: random_state={}'.format(random_state))
            # Get testset
            df_tmp = TSETS[Tname].copy()
            # force index to be a range
            df_tmp.index = range(df_tmp.shape[0])
            print(Tname, df_tmp.shape)
            # Stratified split by default, unless random var in name
            rand_strat = True
            rand_20_80 = False
            # get the training and test set
            returned_vars = mk_test_train_sets(df=df_tmp,
                                               rand_20_80=rand_20_80,
                                               random_state=random_state,
                                               nsplits=TSETS_nsplits[Tname],
                                               rand_strat=rand_strat,
                                               features_used=features_used,
                                               )
            train_set, test_set, test_set_targets = returned_vars
            # set the training and test sets
            train_features = df_tmp[features_used].loc[train_set.index]
            train_labels = df_tmp[[target]].loc[train_set.index]
            test_features = df_tmp[features_used].loc[test_set.index]
            test_labels = df_tmp[[target]].loc[test_set.index]
            # build the model - NOTE THIS MUST BE RE-DONE!
            # ( otherwise the model is being re-trained )
            model = RandomForestRegressor(random_state=random_state,
                                          n_estimators=n_estimators, criterion='mse')
    #            , oob_score=oob_score)
            # fit the model
            model.fit(train_features, train_labels)
            # predict the values
            df_tmp[Tname] = model.predict(df_tmp[features_used].values)
            # get the stats against the test group
            df_tmp = df_tmp[[Tname, target]].loc[test_set.index]
            # get MSE and RMSE
            MSE = (df_tmp[target]-df_tmp[Tname])**2
            MSE = np.mean(MSE)
            std = np.std(df_tmp[Tname].values)

            # return stats on bias and variance
            # (just use RMSE and std dev. for now)
            RMSE_l += [np.sqrt(MSE)]
    #        s = pd.Series( {'MSE': MSE, 'RMSE': np.sqrt(MSE), 'std':std } )
    #        RMSE_df[Tname] = s
            #
            del df_tmp, train_features, train_labels, test_features, test_labels
            del model
#        gc.collect()
        #
        RMSE_df[Tname] = RMSE_l

    # --- Get stats on the ensemble values
    # Get general stats on ensemble
    RMSE_stats = pd.DataFrame(RMSE_df.describe().copy()).T
    RMSE_stats.sort_values(by='mean', inplace=True)
    # sort the main Dataframe by the magnitude of the mean
    RMSE_df = RMSE_df[list(RMSE_stats.index)]
    # work out the deviation from mean of the ensemble
    pcent_var = '% from mean'
    means = RMSE_stats['mean']
    pcents = ((means - means.mean()) / means.mean() * 100).values
    RMSE_stats[pcent_var] = pcents
    # print to screen
    print(RMSE_stats)
    pstr = '{:<13} - mean: {:.2f} (% from ensemble mean: {:.2f})'
    for col in RMSE_stats.T.columns:
        vals2print = RMSE_stats.T[col][['mean', pcent_var]].values
        print(pstr.format(col.replace("\n", ""), *vals2print))
    # Also add the deviation
    RMSE_stats['Q'] = [i.split('Q=')[-1][:-1] for i in RMSE_stats.index]
    # Save to csv
    RMSE_stats.to_csv('Oi_prj_test_training_selection_quantiles.csv')

    # --- Setup the datafframes for plotting ( long form needed )
    RMSE_df = RMSE_df.melt()
    # rename columns
    ylabel_str = 'RMSE (nM)'
    RMSE_df.rename(columns={'value': ylabel_str}, inplace=True)

    # --- Plot up the test runs
    CB_color_cycle = AC.get_CB_color_cycle()
    import seaborn as sns
    sns.set(color_codes=True)
    sns.set_context("paper")
    dpi = 320

    # --- plot up the results as violin plots
    fig, ax = plt.subplots(figsize=(10, 4), dpi=dpi)
    # plot up these values
    ax = sns.violinplot(x='variable', y=ylabel_str, data=RMSE_df,
                        palette=CB_color_cycle, width=1.05, ax=ax,
                        order=RMSE_stats.index)
    # remove the variable label from the x axis
    ax.xaxis.label.set_visible(False)
    # force yaxis extent
    ymax = AC.myround(RMSE_df[ylabel_str].max(), base=10, round_up=True)
#    ax.set_ylim(15, ymax+5 )
    # add N value to plot
#    f_size =10
    xlabels = [i.get_text() for i in ax.get_xticklabels()]
    # set locations for N lael
    if len(xlabels) == 7:
        x_l = np.linspace(0.041, 0.9025, len(xlabels))
    else:
        x_l = np.linspace(0.035, 0.9075, len(xlabels))
    # loop and add N value
#     for xlabel_n, xlabel  in enumerate( xlabels ):
#         N = TSETS_N[xlabel]
#         # Set location for label
#         alt_text_x = x_l[xlabel_n]
# #        alt_text_x = 0.5
# #        alt_text_y = 0.035
#         alt_text_y = 0.5
#         # Setup label and plot
#         alt_text = 'N={}'.format( N )
#         ax.annotate( alt_text , xy=(alt_text_x, alt_text_y), \
#             textcoords='axes fraction', )
    # Adjust positions of subplot
    bottom = 0.095
    top = 0.975
    left = 0.075
    right = 0.975
    fig.subplots_adjust(bottom=bottom, top=top, left=left, right=right,)
    # save the plot
    png_name = 'Oi_prj_test_training_selection_sensitivity_violin_quantiles.png'
    plt.savefig(png_name, dpi=dpi)
    plt.close()


def run_tests_on_model_build_options(df=None,
#                                     use_choosen_model=True,
                                     target='Iodide',
                                     testset='Test set (strat. 20%)',
                                     features_used=None,
                                     model_name='TEST_MODEL'):
    """
    Test feature and hyperparameter options for model

    Parameters
    -------
    target (str), Name of the target variable (e.g. iodide)
    testset (str), Testset to use, e.g. stratified sampling over quartiles for 20%:80%
    features_used (list), list of the features within the model_name model

    Returns
    -------

    Notes
    -----
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.externals import joblib
    # - Get the data/variables
    # get processed data
    if isinstance(df, type(None)):
        df = get_dataset_processed4ML()
    # Use the model selected from the feature testing
#     if use_choosen_model:
#         mdict = get_choosen_model_from_features_selection()
#         features_used = mdict['features_used']
#         model = mdict['model']
#         model_name = mdict['name']
    # Which "features" (variables) to use
    if isinstance(features_used, type(None)):
        model_name = 'ALL'
        features_used = get_model_features_used_dict(model_name)
    # Select just the testing features, target, and  testset split
    df = df[features_used+[target, testset]]
    # - Select training dataset
    test_set = df.loc[df[testset] == True, :]
    train_set = df.loc[df[testset] == False, :]
    # also sub select all vectors for input data
    # ( Making sure to remove the target!!! )
    train_set_full = df[features_used].loc[train_set.index]
    train_set_targets = df[target].loc[train_set.index]
    test_set_full = df[features_used].loc[test_set.index]
    test_set_targets = df[target].loc[test_set.index]

    # - Preparing input data for ML algorythm
    # Make sure that the values are within a reasonable range
    # (almost all ML algorythims won't work without standardisation )
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    # Setup as pipeline (only one operation... so this is overkill ATM.)
    num_pipeline = Pipeline([('std_scaler', StandardScaler()), ])
    # not biniarisation or labels... so full_pipeline just equals pipeline
    full_pipeline = num_pipeline
    # transform data
    if do_not_transform_feature_data:
        print('WARNING! '*5, 'Not transforming feature data')
        print('No transform assumed, as not needed for Decision tree regressor')
        train_set_tr = train_set_full
    else:
        train_set_tr = num_pipeline.fit_transform(train_set_full)

    # - ...
    # Plot up variable (e.g. # trees) vs. RMSE (or oob error?),
    # use this to argue for # trees etc...
    from sklearn.model_selection import GridSearchCV
    # Get the param_grid (with params to test)
    param_grid = define_hyperparameter_options2test(
        features_used=features_used)
    # initialise RFR (using a fixed random state)
    forest_reg = RandomForestRegressor(random_state=42, criterion='mse')
    # train across 5 folds, that's a total of (12+6)*5=90 rounds of training
    grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                               scoring='neg_mean_squared_error')
    # set verbosity to 99
    grid_search.verbose = 99
    # Now fit models
    grid_search.fit(train_set_tr, train_set_targets)
    # print main results
    grid_search.best_params_
    grid_search.best_estimator_
    # print all results
    cvres = grid_search.cv_results_
    #
    df = pd.DataFrame(cvres)
    sqrt_neg_mean = 'sqrt neg mean'
    df[sqrt_neg_mean] = np.sqrt(-df['mean_test_score'])
    #
    df.sort_values(['rank_test_score', sqrt_neg_mean], inplace=True)
    # evaluate best parameters
    attributes = df.columns.values
    feature_importances = grid_search.best_estimator_.feature_importances_
    sorted(zip(feature_importances, attributes), reverse=True)
    # show which is best model
    # <not pasted code here yet... >
    final_model = grid_search.best_estimator_

    # - Test the performance of the models
    for model_name in models.keys():
        model = models[model_name]
        df[model_name] = get_model_predictions4obs_point(model=model)


def get_feature_importance(RFR_dict=None):
    """
    Get the feature variable inmportance for current models
    """
    # set models to compare...
    models2compare = []
    topmodels = get_top_models(RFR_dict=RFR_dict, vars2exclude=['DOC', 'Prod'])
    models2compare = topmodels
    # Get data
    if isinstance(RFR_dict, type(None)):
        RFR_dict = build_or_get_models()
    # select dataframe with observations and predictions in it
    df = RFR_dict['df']
    features_used_dict = RFR_dict['features_used_dict']
    models_dict = RFR_dict['models_dict']
    # - Print feature importances to screen
    df_feats = pd.DataFrame()
    for modelname in models2compare:
        # get model
        model = models_dict[modelname]
        # Get feature importance
        feature_importances = model.feature_importances_
        # Get testing features
        features_used = features_used_dict[modelname].split('+')
        #
        s = pd.Series(dict(zip(features_used, feature_importances)))
        df_feats[modelname] = s
    # Save as .csv
    df_feats.T.to_csv('Oi_prj_feature_importances.csv')


def get_core_stats_on_current_models(df=None, testset='Test set (strat. 20%)',
                                     target='Iodide', inc_ensemble=False,
                                     param_names=[],
                                     analysis4coastal=False,
                                     plot_up_model_performance=True, RFR_dict=None,
                                     add_sklean_metrics=False, save2csv=True,
                                     verbose=True, debug=False):
    """
    Get Core statistics on the models built

    Parameters
    -------
    target (str), Name of the target variable (e.g. iodide)
    testset (str), Testset to use, e.g. stratified sampling over quartiles for 20%:80%
    inc_ensemble (bool), include the ensemble (var2use) in the analysis
    analysis4coastal (bool), include analysis for coastal vs. non-coastal regions
    plot_up_model_performance (bool), plot up the model performance
    add_sklean_metrics (bool), include core sklearn metrics
    RFR_dict (dict), dictionary of core variables and data
    save2csv (bool), save calculated statistics as a .csv file
    analysis4coastal (bool), include analysis for coastal vs. non-coastal regions
    param_names (list), list of parameters to calculate performance of
    debug (boolean), print out debugging output?

    Returns
    -------
    (pd.DataFrame)
    """
    # --- Get data
    if isinstance(RFR_dict, type(None)):
        RFR_dict = build_or_get_models()
    # select dataframe with observations and predictions in it
    if isinstance(df, type(None)):
        df = RFR_dict['df']
    # model names
    model_names = RFR_dict['model_names']
    features_used_dict = RFR_dict['features_used_dict']
    N_features_used = RFR_dict['N_features_used']
    oob_scores = RFR_dict['oob_scores']
    # Calculate performance
    stats = calc_performance_of_params(df=df, target=target,
                                       params=param_names+model_names)
    # Just test on test set
    df_tmp = df.loc[df[testset] == True, :]
    stats_sub1 = get_df_stats_MSE_RMSE(params=param_names+model_names,
                                       df=df_tmp[[target]+model_names +
                                                 param_names], dataset_str=testset,
                                       target=target,
                                       add_sklean_metrics=add_sklean_metrics).T
    stats2concat = [stats, stats_sub1]
    # Combine all stats (RMSE and general stats)
    stats = pd.concat(stats2concat)
    # Add number of features too
    stats = stats.T
    feats = pd.DataFrame(index=model_names)
    N_feat_Var = '# features'
    feats[N_feat_Var] = [N_features_used[i] for i in model_names]
    # and the feature names
    feat_Var = 'features_used'
    feats[feat_Var] = [features_used_dict[i] for i in model_names]
    # and the oob score
    feats['OOB score'] = [oob_scores[i] for i in model_names]
    # combine with the rest of the stats
    stats = pd.concat([stats, feats], axis=1)
    # which vars to sort by
#    var2sortby = ['RMSE (all)', N_feat]
#    var2sortby = 'RMSE (all)'
    var2sortby = 'RMSE ({})'.format(testset)
    # print useful vars to screen
    vars2inc = [
        'RMSE (all)', 'RMSE ({})'.format(testset),
        #    'MSE ({})'.format(testset),'MSE (all)',
    ]
    vars2inc += feats.columns.tolist()
    # Sort df by RMSE
    stats.sort_values(by=var2sortby, axis=0, inplace=True)
    # sort columns
    first_columns = [
        'mean', 'std', '25%', '50%', '75%',
        'RMSE ({})'.format(testset),  'RMSE (all)',
    ]
    rest_of_columns = [i for i in stats.columns if i not in first_columns]
    stats = stats[first_columns + rest_of_columns]
    # Rename columns (50% to median and ... )
    df.rename(columns={'50%': 'median', 'std': 'std. dev.'})
    # Set filename and save detail on models
    csv_name = 'Oi_prj_stats_on_{}_models_built_at_obs_points'.format(target)
    if save2csv:
        stats.round(2).to_csv(csv_name+'.csv')
    # Also print to screen
    if verbose:
        print(stats[vars2inc+[N_feat_Var]])
    if verbose:
        print(stats[vars2inc])
    # Without testing features
    vars2inc.pop(vars2inc.index('features_used'))
    if verbose:
        print(stats[vars2inc])
    if verbose:
        print(stats[['RMSE ({})'.format(testset), 'OOB score', ]])
    # Save statistics to csv
    csv_name += '_selected'
    if save2csv:
        stats[vars2inc].round(2).to_csv(csv_name+'.csv')
    return stats


def get_stats_on_models(df=None, testset='Test set (strat. 20%)',
                        target='Iodide', inc_ensemble=False,
#                        save_CHOOSEN_MODEL=False,
                        analysis4coastal=False, var2use='RFR(Ensemble)',
                        plot_up_model_performance=True, RFR_dict=None,
                        add_sklean_metrics=False, verbose=True, debug=False):
    """
    Analyse the stats on of params and obs.

    Parameters
    -------
    analysis4coastal (bool), include analysis of data split by coastal/non-coastal
    target (str), Name of the target variable (e.g. iodide)
    testset (str), Testset to use, e.g. stratified sampling over quartiles for 20%:80%
    inc_ensemble (bool), include the ensemble (var2use) in the analysis
    debug (boolean), print out debugging output?

    Returns
    -------

    Notes
    -----
    """
    # --- Get data
    if isinstance(RFR_dict, type(None)):
        RFR_dict = build_or_get_models()
    # select dataframe with observations and predictions in it
    if isinstance(df, type(None)):
        df = RFR_dict['df']
    # model names
    model_names = RFR_dict['model_names']
    features_used_dict = RFR_dict['features_used_dict']
    N_features_used = RFR_dict['N_features_used']
    oob_scores = RFR_dict['oob_scores']
    # - Evaluate performance of models (e.g. Root Mean Square Error (RMSE) )
    # Also evaluate parameterisations
    param_names = []
    if target == 'Iodide':
        param_names += [u'Chance2014_STTxx2_I', u'MacDonald2014_iodide',
                        # u'Chance2014_Multivariate',
                        ]
    # Aslo include the ensemble parameters
    if inc_ensemble:
        param_names += [var2use]
    # Calculate performance
    stats = calc_performance_of_params(df=df,
                                       params=param_names+model_names)
    # Just test on test set
    df_tmp = df.loc[df[testset] == True, :]
    stats_sub1 = get_df_stats_MSE_RMSE(params=param_names+model_names,
                                       df=df_tmp[[target]+model_names + param_names],
                                       dataset_str=testset,
                                       target=target,
                                       add_sklean_metrics=add_sklean_metrics).T
    stats2concat = [stats, stats_sub1]
    if analysis4coastal:
        # Add testing on coastal
        dataset_split = 'Coastal'
        df_tmp = df.loc[(df['Coastal'] == 1), :]
        stats_sub2 = get_df_stats_MSE_RMSE(params=param_names+model_names,
                                           df=df_tmp[[target]+model_names+param_names],
                                           target=target,
                                           dataset_str=dataset_split,
                                           add_sklean_metrics=add_sklean_metrics).T
        # Add testing on non-coastal
        dataset_split = 'Non coastal'
        df_tmp = df.loc[(df['Coastal'] == 0), :]
        stats_sub3 = get_df_stats_MSE_RMSE(params=param_names+model_names,
                                           df=df_tmp[[target]+model_names+param_names],
                                           target=target,
                                           dataset_str=dataset_split,
                                           add_sklean_metrics=add_sklean_metrics).T
        # Add testing on coastal
        dataset_split = 'Coastal ({})'.format(testset)
        df_tmp = df.loc[(df['Coastal'] == 1) & (df[testset] == True), :]
        stats_sub4 = get_df_stats_MSE_RMSE(params=param_names+model_names,
                                           df=df_tmp[[target]+model_names+param_names],
                                           target=target,
                                           dataset_str=dataset_split,
                                           add_sklean_metrics=add_sklean_metrics).T
        # Add testing on non-coastal
        dataset_split = 'Non coastal ({})'.format(testset)
        df_tmp = df.loc[(df['Coastal'] == 0) & (df[testset] == True), :]
        stats_sub5 = get_df_stats_MSE_RMSE(params=param_names+model_names,
                                           df=df_tmp[[target]+model_names+param_names],
                                           target=target,
                                           dataset_str=dataset_split,
                                           add_sklean_metrics=add_sklean_metrics).T
        # Statistics to concat
        stats2concat += [stats_sub2, stats_sub3, stats_sub4, stats_sub5, ]
    # Combine all stats (RMSE and general stats)
    stats = pd.concat(stats2concat)
    # Add number of features too
    stats = stats.T
    feats = pd.DataFrame(index=model_names)
    N_feat_Var = '# features'
    feats[N_feat_Var] = [N_features_used[i] for i in model_names]
    # and the feature names
    feat_Var = 'features_used'
    feats[feat_Var] = [features_used_dict[i] for i in model_names]
    # and the oob score
    feats['OOB score'] = [oob_scores[i] for i in model_names]
    # combine with the rest of the stats
    stats = pd.concat([stats, feats], axis=1)
    # which vars to sort by
#    var2sortby = ['RMSE (all)', N_feat]
#    var2sortby = 'RMSE (all)'
    var2sortby = 'RMSE ({})'.format(testset)
    # print useful vars to screen
    vars2inc = [
        'RMSE (all)', 'RMSE ({})'.format(testset),
        #    'MSE ({})'.format(testset),'MSE (all)',
    ]
    vars2inc += feats.columns.tolist()
    # Sort df by RMSE
    stats.sort_values(by=var2sortby, axis=0, inplace=True)
    # sort columns
    first_columns = [
        'mean', 'std', '25%', '50%', '75%',
        'RMSE ({})'.format(testset),  'RMSE (all)',
    ]
    rest_of_columns = [i for i in stats.columns if i not in first_columns]
    stats = stats[first_columns + rest_of_columns]
    # Rename columns (50% to median and ... )
    df.rename(columns={'50%': 'median', 'std': 'std. dev.'})
    # Set filename and save detail on models
    csv_name = 'Oi_prj_stats_on_{}_models_built_at_obs_points'.format(target)
    stats.round(2).to_csv(csv_name+'.csv')
    # Also print to screen
    if verbose:
        print(stats[vars2inc+[N_feat_Var]])
    if verbose:
        print(stats[vars2inc])
    # Without testing features
    vars2inc.pop(vars2inc.index('features_used'))
    if verbose:
        print(stats[vars2inc])
    if verbose:
        print(stats[['RMSE ({})'.format(testset), 'OOB score', ]])
    # Save statistics to csv
    csv_name += '_selected'
    stats[vars2inc].round(2).to_csv(csv_name+'.csv')

    # - also save a version that doesn't include the derived dataset
    params2inc = stats.T.columns
    params2inc = [i for i in params2inc if 'DOC' not in i]
    params2inc = [i for i in params2inc if 'Prod' not in i]
    # select these variables from the list
    tmp_stats = stats.T[params2inc].T
    # save a reduced csv
    vars2inc_REDUCED = [
        'mean', 'std', '25%', '50%', '75%',
        'RMSE ({})'.format(testset),  'RMSE (all)',
    ]
    # add the coastal testsets to the data?
    if analysis4coastal:
        vars2inc_REDUCED += [
            u'RMSE (Coastal)', u'RMSE (Non coastal)',
            'RMSE (Coastal (Test set (strat. 20%)))',
            u'RMSE (Non coastal (Test set (strat. 20%)))',
        ]
    # Save a csv with reduced infomation
    csv_name = 'Oi_prj_models_built_stats_on_models_at_obs_points'
    csv_name += '_REDUCED_NO_DERIVED.csv'
    tmp_stats[vars2inc_REDUCED].round(2).to_csv(csv_name)

    # - Select the best model based of criteria
#     if save_CHOOSEN_MODEL:
#         # Set a criteria?
#         # Select model at the top of the the sorted table for now...
#         CHOOSEN_MODEL_NAME = stats.head(1).index[0]
#         CHOOSEN_MODEL = models_dict[CHOOSEN_MODEL_NAME]
#         # Save best estimator model
#         model_savename = "my_model_{}.pkl".format(CHOOSEN_MODEL_NAME)
#         joblib.dump(CHOOSEN_MODEL, wrk_dir+'/CHOOSEN_MODEL/' + model_savename)
#         # Print detail on choosen model
#         features_used = model_feature_dict[CHOOSEN_MODEL_NAME]
#         zip(features_used, CHOOSEN_MODEL.feature_importances_)

    # --- plot up model performance against the testset
    if plot_up_model_performance:
        #
        rename_titles = {u'Chance2014_STTxx2_I': 'Chance et al. (2014)',
                         u'MacDonald2014_iodide': 'MacDonald et al. (2014)',
                         'Ensemble_Monthly_mean': 'RFR(Ensemble)',
                         'Iodide': 'Obs.',
                         #                         u'Chance2014_Multivariate': 'Chance et al. (2014) (Multi)',
                         }
        stats.rename(index=rename_titles,  inplace=True)
        # also compare existing parameters
        params = [
            #         u'Chance2014_STTxx2_I','Chance2014_Multivariate',
            #        u'MacDonald2014_iodide',
            #            'Chance et al. (2014) (Multi)',
            'Chance et al. (2014)',
            'MacDonald et al. (2014)',
        ]
        CB_color_cycle = AC.get_CB_color_cycle()
        import seaborn as sns
        sns.set(color_codes=True)
        sns.set_context("paper")
        dpi = 320
        fig, ax = plt.subplots(dpi=dpi)
        # select the top ten and twenty models
        topmodels = stats.head(20).index
        params2plot = list(topmodels) + params
        # select the data
        stat_var = 'RMSE ({})'.format(testset)
        df_tmp = stats.T[params2plot].T[stat_var][::-1]
        # plot up these values
        X = range(len(params2plot))
        ax.scatter(X, df_tmp.values, color=CB_color_cycle[0])
    #    plt.title( 'Top models and existing parameters')
        ax.set_xticks(X)
        ax.set_xticklabels(params2plot[::-1], rotation=90)
        plt.ylim(25, 85)
        plt.ylabel('RMSE (nM)',  color=CB_color_cycle[0])

        # - Plot up top models amongst all datasets
        # select testset
        df_tmp = df.rename(columns=rename_titles)
        df_tmp = df_tmp.loc[df_tmp[testset] == True, :][params2plot+['Obs.']]
        var = df_tmp.var()[params2plot[::-1]]
        std = df_tmp.describe().T['std'].T[params2plot[::-1]]
#        for i in df_tmp:
#            df_tmp[i] = abs(df_tmp[i].values - df_tmp['Iodide'])
#        std_of_E = df_tmp.describe().T['std'].T[params2plot[::-1]]
        ax2 = ax.twinx()
        ax2.grid(False)
    #    ax2.scatter(X, std_of_E, color=CB_color_cycle[1] )
        ax2.scatter(X, std, color=CB_color_cycle[1])
    #    ax2.scatter(X, var, color=CB_color_cycle[1] )
    #    plt.ylim( 25, 85 )
        plt.ylabel('std. dev. (nM)',  color=CB_color_cycle[1])
    #    plt.ylabel( 'std. dev. of abs. error (nM)',  color=CB_color_cycle[1] )
    #    plt.ylabel( 'variance',  color=CB_color_cycle[1] )
        plt.tight_layout()
        plt.savefig('Oi_prj_model_performance.png', dpi=dpi)
        plt.close()

        # - Also plot the top values for models without DOC and prod
        # list params not including DOC and Prod
        params2plot = stats.T.columns
        params2plot = [i for i in params2plot if 'Ensemble' not in i]
        params2plot = [i for i in params2plot if 'DOC' not in i]
        params2plot = [i for i in params2plot if 'Prod' not in i]
        # select the top 20
        params2plot = params2plot[:20] + params
        topten = params2plot[:10]

        # select the data for params2plot
        stat_var = 'RMSE ({})'.format(testset)
        df_tmp = stats.T[params2plot].T[stat_var][::-1]
        # plot up these values
        X = range(len(params2plot))
        fig, ax = plt.subplots(dpi=dpi)
        ax.scatter(X, df_tmp.values, color=CB_color_cycle[0])
    #    plt.title( 'Top models and existing parameters')
        ax.set_xticks(X)
        ax.set_xticklabels(params2plot[::-1], rotation=90)
        plt.ylim(25, 85)
        plt.ylabel('RMSE (nM)',  color=CB_color_cycle[0])
        # get standard deviation within testset
        df_tmp = df.rename(columns=rename_titles)
        df_tmp = df_tmp.loc[df_tmp[testset] == True, :][params2plot+['Obs.']]
        var = df_tmp.var()[params2plot[::-1]]
        std = df_tmp.describe().T['std'].T[params2plot[::-1]]
#        for i in df_tmp:
#            df_tmp[i] = abs(df_tmp[i].values - df_tmp['Iodide'])
#        std_of_E = df_tmp.describe().T['std'].T[params2plot[::-1]]
        ax2 = ax.twinx()
        ax2.grid(False)
    #    ax2.scatter(X, std_of_E, color=CB_color_cycle[1] )
        ax2.scatter(X, std, color=CB_color_cycle[1])
    #    ax2.scatter(X, var, color=CB_color_cycle[1] )
    #    plt.ylim( 25, 85 )
        plt.ylabel('std. dev. (nM)',  color=CB_color_cycle[1])
    #    plt.ylabel( 'std. dev. of abs. error (nM)',  color=CB_color_cycle[1] )
    #    plt.ylabel( 'variance',  color=CB_color_cycle[1] )
        # Set the top ten and old params as bold
        set2bold = [
            n for n, i in enumerate(params2plot[::-1]) if (i in params+topten)
        ]
        for ntick, tick in enumerate(ax2.xaxis.get_major_ticks()):
            if ntick in set2bold:
                tick.label.set_fontweight('bold')
                prt_str = 'Set tick to bold - {}{}'
#                print( prt_str.format( ntick, params2plot[::-1][ntick]) )
        for ntick, tick in enumerate(ax.xaxis.get_major_ticks()):
            if ntick in set2bold:
                tick.label.set_fontweight('bold')
                prt_str = 'Set tick to bold - {}{}'
#                print( prt_str.format( ntick, params2plot[::-1][ntick]) )
        # Now update the layout and save
        plt.tight_layout()
        plt.savefig('Oi_prj_model_performance_NO_DERIV.png', dpi=dpi)
        plt.close()
    # return dataframe of stats regardless
    return stats


def test_performance_of_params(target='Iodide', features_used=None):
    """
    Test the performance of the parameters

    Parameters
    -------
    target (str), Name of the target variable (e.g. iodide)
    features_used (list), list of the features within the model_name model

    Returns
    -------
    (None)
    """
    # - Get the data
    # get processed data
    # settings for incoming feature data
    restrict_data_max = False
    restrict_min_salinity = False
    use_median4chlr_a_NaNs = True
    add_modulus_of_lat = False
    # apply transforms to  data?
    do_not_transform_feature_data = True
    # just use the forest out comes
    use_forest_without_optimising = True
    # Which "features" (variables) to use
    if isinstance(features_used, type(None)):
        features_used = [
            'WOA_TEMP_K',
            'WOA_Salinity',
            'Depth_GEBCO',
        ]
    # Local variables for use in this function
    param_rename_dict = {
        u'Chance2014_STTxx2_I': 'Chance2014',
        u'MacDonald2014_iodide': 'MacDonald2014',
        u'Iodide': 'Obs.',
    }
    param_names = param_rename_dict.keys()
    param_names.pop(param_names.index('Iodide'))
    # Set-up a dictionary of test set variables
    random_split_var = 'rn. 20%'
    strat_split_var = 'strat. 20%'
    model_names_dict = {
        'TEMP+DEPTH+SAL (rs)': random_split_var,
        'TEMP+DEPTH+SAL': strat_split_var,
    }
    model_names = model_names_dict.keys()
    # Get data as a DataFrame
    df = get_processed_df_obs_mod()  # NOTE this df contains values >400nM
    # Add extra variables and remove some data.
    df = add_extra_vars_rm_some_data(df=df,
                                     restrict_data_max=restrict_data_max,
                                     restrict_min_salinity=restrict_min_salinity,
                                     add_modulus_of_lat=add_modulus_of_lat,
                                     use_median4chlr_a_NaNs=use_median4chlr_a_NaNs,
                                     )    # add
    # add boolean for test and training dataset
    splits_dict = {
        random_split_var: (True, False), strat_split_var: (False, True),
    }
    # Loop test sets
    for test_split in splits_dict.keys():
        rand_20_80, rand_strat = splits_dict[test_split]
        # Select just the features used and the target variable
        df_tmp = df[features_used+['Iodide']].copy()
        # split into the training and test sets
        returned_vars = mk_test_train_sets(df=df_tmp,
                                           rand_20_80=rand_20_80,
                                           rand_strat=rand_strat,
                                           features_used=features_used,
                                           )
        train_set, test_set, test_set_targets = returned_vars
        # Add this to the dataframe using the passed shape as a template
        dummy = np.zeros(df.shape[0])
        dummy[test_set.index] = True
        df['test ({})'.format(test_split)] = dummy

    # Add model predictions
    for model_name in model_names:
        df[model_name] = get_model_predictions4obs_point(df=df,
                                                         model_name=model_name)

    # - Get stats on whole dataset?
    stats = calc_performance_of_params(df=df,
                                       params=param_names+model_names)

    # - Get stats for model on just its test set dataset
    model_stats = []
    for modelname in model_names:
        test_set = model_names_dict[modelname]
        dataset_str = 'test ({})'.format(test_set)
        print(modelname, test_set, dataset_str)
        df_tmp = df.loc[df[dataset_str] == True]
        print(df_tmp.shape, df_tmp[target].mean())
        model_stats.append(get_df_stats_MSE_RMSE(
            df=df_tmp[[target, modelname]+param_names],
            params=param_names+[modelname],
            dataset_str=test_set, target=target).T)
    # Add these to core dataset
    stats = pd.concat([stats] + model_stats)

    # - get stats for coastal values
    # Just ***NON*** coastal values
    df_tmp = df.loc[df['coastal_flagged'] == False]
    test_set = '>30 Salinty'
    print(df_tmp.shape)
    # Calculate...
    stats_open_ocean = get_df_stats_MSE_RMSE(
        df=df_tmp[[target]+model_names+param_names],
        params=param_names+model_names,
        dataset_str=test_set, target=target).T
    # Just  ***coastal*** values
    df_tmp = df.loc[df['coastal_flagged'] == True]
    test_set = '<30 Salinty'
    print(df_tmp.shape)
    # Calculate...
    stats_coastal = get_df_stats_MSE_RMSE(
        df=df_tmp[[target]+model_names+param_names],
        params=param_names+model_names,
        dataset_str=test_set, target=target).T
    # Add these to core dataset
    stats = pd.concat([stats] + [stats_coastal, stats_open_ocean])

    # - Minor processing and save
    # rename the columns for re-abliiity
    stats.rename(columns=param_rename_dict, inplace=True)
    # round the columns to one dp.
    stats = stats.round(1)
    print(stats)
    # Save as a csv
    stats.to_csv('Oi_prj_param_performance.csv')


def calc_performance_of_params(df=None, target='Iodide', params=[]):
    """
    Calculate stats on performance of parameters in DataFrame

    Parameters
    -------
    target (str), Name of the target variable (e.g. iodide)
    df (pd.dataframe), dataframe containing of target and features
    params (list), list of parameters to calculate performance of
    """
    # Initialise with generic stats
    stats = [df[i].describe() for i in params + [target]]
    stats = pd.DataFrame(stats).T
    # - Now add own stats
    new_stats = get_df_stats_MSE_RMSE(df=df, target=target, params=params,
                                      dataset_str='all')
    # Add new stats to standard stats
    stats = pd.concat([stats, new_stats.T])
    # - add other stats? (mean, standard deviation )
    return stats


def extract_trees4models(N_trees2output=10, RFR_dict=None, max_depth=7, target='Iodide',
                         ouput_random_tree_numbers=False, verbose=True, ):
    """
    Extract individual trees from models

    Parameters
    -------
    target (str), Name of the target variable (e.g. iodide)
    N_trees2output (int), number of trees to extract to .csv files
    RFR_dict (dict), dictionary of core variables and data
    max_depth (int), maximum depth of tree branch to extract
    ouput_random_tree_numbers (bool), randomly select trees to output
    verbose (boolean), print out verbose output?

    Returns
    -------
    (None)

    Notes
    -----
     - This is a file processor for the TreeSurgeon java/node.js plotter
    https://github.com/wolfiex/TreeSurgeon
    """
    # Get the dictionary
    if isinstance(RFR_dict, type(None)):
        RFR_dict = build_or_get_models()
    # Get the top model names
    topmodels = get_top_models(RFR_dict=RFR_dict, vars2exclude=['DOC', 'Prod'], n=10)
    # Set the folder
    data_root = utils.get_file_locations('data_root')
    folder = '{}/{}/models/LIVE/TEMP_MODELS/'.format(data_root, target)
    # Get the file names for these
    modelnames = glob.glob(folder+'*.pkl')
    modelname_d = dict(zip(RFR_dict['model_names'], modelnames))
    # Get testing features dictionary
    features_used_dict = RFR_dict['features_used_dict']
    # Loop by model and
    for modelname in topmodels:
        if verbose:
            print(modelname)
        # Get name of model's file (ex. directory)  and testing features
        model_filename = modelname_d[modelname].split('/')[-1]
        features_used = features_used_dict[modelname].split('+')
        # Extract the trees to dot files
        extract_trees_to_dot_files(folder=folder,
                                   model_filename=model_filename,
                                   N_trees2output=N_trees2output,
                                   ouput_random_tree_numbers=ouput_random_tree_numbers,
                                   max_depth=max_depth,
                                   extr_str=modelname, features_used=features_used)


def extract_trees_to_dot_files(folder=None, model_filename=None, target='Iodide',
                               features_used=None, N_trees2output=10, max_depth=7,
                               ouput_random_tree_numbers=False, extr_str=''):
    """
    Extract individual model trees to .dot files to be plotted in d3

    Parameters
    -------
    target (str), Name of the target variable (e.g. iodide)
    features_used (list), list of the features within the model_name model
    N_trees2output (int), number of trees to extract to .csv files
    max_depth (int), maximum depth of tree branch to extract
    ouput_random_tree_numbers (bool), randomly select trees to output
    verbose (boolean), print out verbose output?
    model_filename (str), filename of the model to extract
    folder (str), location of file (model_filename) to extract
    extr_str (str), string to add to outputted dot file

    Returns
    -------
    (None)

    Notes
    -----
     - This is a file processor for the TreeSurgeon java/node.js plotter
     https://github.com/wolfiex/TreeSurgeon
    """
    from sklearn.externals import joblib
    from sklearn import tree
    import os
    # Get the location of the saved model.
    if isinstance(folder, type(None)):
        data_root = utils.get_file_locations('data_root')
        folder = '{}/{}/models/'.format(data_root, target)
    # Create a file name for model if not provided
    if isinstance(model_filename, type(None)):
        model_filename = "my_model_{}.pkl".format(extr_str)
    # Provide feature names?
    if isinstance(features_used, type(None)):
        features_used = [
            #        u'Longitude',
            #       'Latitude',
            'WOA_TEMP_K',
            'WOA_Salinity',
            #       'WOA_Nitrate',
            'Depth_GEBCO',
            #       'SeaWIFs_ChlrA',
            #     u'month',
        ]
    # Open as sklearn rf object
    rf = joblib.load(folder+model_filename)
    #
    if ouput_random_tree_numbers:
        np.random.seed(42)
        my_list = list(np.arange(0, 500))
        np.random.shuffle(my_list)
        nums2plot = my_list[:N_trees2output]
    else:
        nums2plot = np.arange(len(rf))
    # Save all trees to disk
    for n, rf_unit in enumerate(rf):
        # Save file if N within list
        if (n in nums2plot):
            # Save out trees
            out_file = 'tree_{}_{:0>4}.dot'.format(extr_str, n)
            print("Saving {} for '{}' in '{}'".format(n, extr_str, out_file))
            tree.export_graphviz(rf_unit, out_file=out_file,
                                 max_depth=max_depth,
                                 feature_names=features_used)


def analyse_nodes_in_models(RFR_dict=None, depth2investigate=5):
    """
    Analyse the nodes in a RFR model

    Parameters
    -------
    RFR_dict (dict), dictionary of core variables and data
    depth2investigate (int), the depth of branches to investigate to

    Returns
    -------
    (None)

    Notes
    -----
     - This is a file processor for the TreeSurgeon java/node.js plotter
     https://github.com/wolfiex/TreeSurgeon
    """
    import glob
    # ---
    # get dictionary of data if not provided as arguement
    if isinstance(RFR_dict, type(None)):
        RFR_dict = build_or_get_models()
    # models to analyse?
    models2compare = []
    topmodels = get_top_models(RFR_dict=RFR_dict, vars2exclude=['DOC', 'Prod'], n=10)
    models2compare = topmodels
    # get strings to update variable names to
    name_dict = convert_fullname_to_shortname(rtn_dict=True)
    # Loop and analyse models2compare
    for model_name in models2compare:
        print(model_name)
        get_decision_point_and_values_for_tree(model_name=model_name,
                                               RFR_dict=RFR_dict,
                                               depth2investigate=depth2investigate)
    # Loop and update the variable names
    for model_name in models2compare:
        print(model_name)
        # Now rename variables in columns
        filestr = 'Oi_prj_features_of*{}*{}*.csv'
        filestr = filestr.format(model_name, depth2investigate)
        csv_files = glob.glob(filestr)
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            # Update the names for the variables
            feature_columns = [i for i in df.columns if 'feature' in i]
            for col in feature_columns:
                for key, value in name_dict.items():
                    df[col] = df[col].str.replace(key, value)
            # save the .csv
            df.to_csv(csv_file)


def get_decision_point_and_values_for_tree(depth2investigate=3,
                                           model_name='RFR(TEMP+DEPTH+SAL)',
                                           RFR_dict=None, verbose=True,
                                           debug=False):
    """
    Get the variables driving decisions at each point

    Parameters
    -------
    depth2investigate (int), the depth of branches to investigate to
    RFR_dict (dict), dictionary of core variables and data
    model_name (str), name of model to get decision points for
    verbose (boolean), print out verbose output?
    debug (boolean), print out debugging output?

    Returns
    -------
    (None)

    Notes
    -----
     - This is a file processor for the TreeSurgeon java/node.js plotter
     https://github.com/wolfiex/TreeSurgeon
     - Details on unfold approach
    link: http://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html
    # The decision estimator has an attribute called tree_  which stores the entire
    # tree structure and allows access to low level attributes. The binary tree
    # tree_ is represented as a number of parallel arrays. The i-th element of each
    # array holds information about the node `i`. Node 0 is the tree's root. NOTE:
    # Some of the arrays only apply to either leaves or split nodes, resp. In this
    # case the values of nodes of the other type are arbitrary!
    #
    # Among those arrays, we have:
    #   - left_child, id of the left child of the node
    #   - right_child, id of the right child of the node
    #   - feature, feature used for splitting the node
    #   - threshold, threshold value at the node
    #
    """
    from sklearn.externals import joblib
    from sklearn import tree
    import os
    # get dictionary of data if not provided as arguement
    if isinstance(RFR_dict, type(None)):
        RFR_dict = build_or_get_models()
    # extra variables needed from RFR_dict
    models_dict = RFR_dict['models_dict']
    features_used_dict = RFR_dict['features_used_dict']
    # Extract model from dictionary
    model = models_dict[model_name]
    # Get training_features
    training_features = features_used_dict[model_name].split('+')
    # Core string for saving data to.
    filename_str = 'Oi_prj_features_of_{}_for_depth_{}{}.{}'
    # Intialise a DataFrame to store values in
    df = pd.DataFrame()
    # Loop by estimator in model
    for n_estimator, estimator in enumerate(model):
        # Extract core variables of interest
        n_nodes = estimator.tree_.node_count
        children_left = estimator.tree_.children_left
        children_right = estimator.tree_.children_right
        feature = estimator.tree_.feature
        threshold = estimator.tree_.threshold
        n_node_samples = estimator.tree_.n_node_samples
        # The tree structure can be traversed to compute various properties such
        # as the depth of each node and whether or not it is a leaf.
        node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
        is_leaves = np.zeros(shape=n_nodes, dtype=bool)
        stack = [(0, -1)]  # seed is the root node id and its parent depth
        # Now extract data
        while len(stack) > 0:
            node_id, parent_depth = stack.pop()
            node_depth[node_id] = parent_depth + 1
            # If we have a test node
            if (children_left[node_id] != children_right[node_id]):
                stack.append((children_left[node_id], parent_depth + 1))
                stack.append((children_right[node_id], parent_depth + 1))
            else:
                is_leaves[node_id] = True
        # - Work out which nodes are required.
        # NOTE: numbering is 1=># of nodes (zero is the first node)
        # add the initial node to a dictionary
        nodes2save = {}
        depth = 0
        n_node = 0
        nodes2save[depth] = {n_node: [children_left[0], children_right[0]]}
        num2node = {0: 0}
        # For depth in depths
        for depth in range(depth2investigate)[:-1]:
            nodes4depth = {}
            new_n_node = max(nodes2save[depth].keys())+1
            for n_node in nodes2save[depth].keys():
                # Get nodes from the children of each node (LH + RH)
                for ChildNum in nodes2save[depth][n_node]:
                    # Get the children of this node
                    LHnew = children_left[ChildNum]
                    RHnew = children_right[ChildNum]
                    # save to temp. dict
                    nodes4depth[new_n_node] = [LHnew, RHnew]
                    # increment the counter and
                    new_n_node += 1
            # Save the new nodes for depth with assigned number
            nodes2save[depth+1] = nodes4depth
        # Get node numbers to save as a dict
        for d in range(depth2investigate)[1:]:
            if debug:
                print(d, nodes2save[d])
            for n in nodes2save[d-1].keys():
                if debug:
                    print(n, nodes2save[d-1][n])
                for nn in nodes2save[d-1][n]:
                    newnum = max(num2node.keys()) + 1
                    num2node[newnum] = nn
        # Make a series of values for estimators
        s = pd.Series()
        for node_num in sorted(num2node.keys()):
            # get index of node of interest
            idx = num2node[node_num]
            # save threadhold value
            var_ = 'N{:0>4}: threshold '.format(node_num)
            s[var_] = threshold[idx]
            # save feature (and convert index to variable name)
            var_ = 'N{:0>4}: feature '.format(node_num)
            s[var_] = training_features[feature[idx]]
            # save feature (and convert index to variable name)
            var_ = 'N{:0>4}: n_node_samples '.format(node_num)
            s[var_] = n_node_samples[idx]
            # save right hand children
            var_ = 'N{:0>4}: RH child '.format(node_num)
            s[var_] = children_right[idx]
            # save the left hand children
            var_ = 'N{:0>4}: LH child '.format(node_num)
            s[var_] = children_left[idx]
        # Also add general details for estimator
        s['n_nodes'] = n_nodes
        # now save to main DataFrame
        df[n_estimator] = s.copy()
    # Set index to be the estimator number
    df = df.T
    # Save the core data on the estimators
    filename = filename_str.format(model_name, depth2investigate, '_ALL', '')
    df.to_csv(filename+'csv')
    # - Print a summary to a file screen
    dfs = {}
    for node_num in sorted(num2node.keys()):
        # get index of node of interest
        idx = num2node[node_num]
        vars_ = [i for i in df.columns if 'N{:0>4}'.format(node_num) in i]
        # get values of inteest for nodes
        FEATvar = [i for i in vars_ if 'feature' in i][0]
        THRESvar = [i for i in vars_ if 'threshold' in i][0]
        SAMPLEvar = [i for i in vars_ if 'n_node_samples' in i][0]
#        RHChildvar = [i for i in vars_ if 'RH child' in i][0]
#        LHChildvar = [i for i in vars_ if 'LH child' in i][0]
#            print FEATvar, THRESvar
        # Get value counts
        val_cnts = df[FEATvar].value_counts()
        df_tmp = pd.DataFrame(val_cnts)
        # Store the features and rename the # of tress column
        df_tmp['feature'] = df_tmp.index
        df_tmp.rename(columns={FEATvar: '# of trees'}, inplace=True)
        # Calc percent
        df_tmp['%'] = val_cnts.values / float(val_cnts.sum()) * 100.
        # Save the children for node
#        df_tmp['RH child'] = df[RHChildvar][idx]
#        df_tmp['LH child'] = df[LHChildvar][idx]
        # intialise series objects to store stats
        s_mean = pd.Series()
        s_median = pd.Series()
        s_std = pd.Series()
        node_feats = list(df_tmp.index)
        s_samples_mean = pd.Series()
        s_samples_median = pd.Series()
        # Now loop and get values fro features
        for feat_ in node_feats:
            # - Get threshold value for node + stats on this
            thres_val4node = df[THRESvar].loc[df[FEATvar] == feat_]
            # make sure the value is a float
            thres_val4node = thres_val4node.astype(np.float)
            # convert Kelvin to degrees for readability
            if feat_ == 'WOA_TEMP_K':
                thres_val4node = thres_val4node - 273.15
            # exact stats of interest
            stats_ = thres_val4node.describe().T
            s_mean[feat_] = stats_['mean']
            s_median[feat_] = stats_['50%']
            s_std[feat_] = stats_['std']
            # - also get avg. samples
            sample_val4node = df[SAMPLEvar].loc[df[FEATvar] == feat_]
            # make sure the value is a float
            sample_val4node = sample_val4node.astype(np.float)
            stats_ = sample_val4node.describe().T
            s_samples_mean = stats_['mean']
            s_samples_median = stats_['50%']
        # Add stats to tmp DataFrame
        df_tmp['std'] = s_std
        df_tmp['median'] = s_median
        df_tmp['mean'] = s_mean
        # Set the depth value for each node_num
        if node_num == 0:
            depth = node_num
        elif node_num in range(1, 3):
            depth = 1
        elif node_num in range(3, 3+(2**2)):
            depth = 2
        elif node_num in range(7, 7+(3**2)):
            depth = 3
        elif node_num in range(16, 16+(4**2)):
            depth = 4
        elif node_num in range(32, 32+(5**2)):
            depth = 5
        elif node_num in range(57, 57+(6**2)):
            depth = 6
        elif node_num in range(93, 93+(7**2)):
            depth = 7
        elif node_num in range(129, 129+(8**2)):
            depth = 8
        else:
            print('Depth not setup for > n+8')
            sys.exit()
        df_tmp['depth'] = depth
        df_tmp['node #'] = node_num
        df_tmp['# samples (mean)'] = s_samples_mean
        df_tmp['# samples (median)'] = s_samples_median
        # Set the index to just a range
        df_tmp.index = range(len(df_tmp.index))
        # Save to main DataFrame
        dfs[node_num] = df_tmp.copy()
    # Loop and save info to files
    filename = filename_str.format(model_name, depth2investigate, '', 'txt')
    a = open(filename, 'w')
    for depth in range(depth2investigate):
        # print summary
        header = '--- At depth {:0>3}:'.format(depth)
        if verbose:
            print(header)
            print(dfs[depth])
        # save
        print(header, file=a)
        print(dfs[depth], file=a)
    # Close file to save data
    a.close()
    # - Build a DataFrame with details on a node by node basis
    # combine by node
    keys = sorted(dfs.keys())
    dfn = dfs[keys[0]].append([dfs[i] for i in keys[1:]])
    # re index and order by
    dfn.index = range(len(dfn.index))
    dfn.sort_values(by=['node #'], ascending=True, inplace=True)
    filename = filename_str.format(model_name, depth2investigate, '', 'csv')
    dfn.to_csv(filename)
