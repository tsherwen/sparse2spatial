"""

Build ensemble models from ensemble of RandomForestRegressor models

"""
import sys
import numpy as np
import pandas as pd
import xarray as xr
import datetime as datetime
import sklearn as sk
from sklearn.ensemble import RandomForestRegressor
import glob

# s2s imports
import sparse2spatial.utils as utils


def build_or_get_models(df=None, testset='Test set (strat. 20%)',
                        save_model_to_disk=False, read_model_from_disk=True,
                        target='Iodide', model_names=None,
                        delete_existing_model_files=False, rm_outliers=True,
                        #                                rm_Skagerrak_data=False,
                        model_sub_dir='/TEMP_MODELS/', random_state=42,
                        rm_LOD_filled_data=False, model_feature_dict=None,
                        debug=False):
    """
    Build (or read from disc) various models (diff. features) to test comparisons

    Parameters
    -------
    df (pd.DataFrame), DataFrame of target and features values for point locations
    testset (str), Testset to use, e.g. stratified sampling over quartiles for 20%:80%
    save_model_to_disk (boolean), Save the models to disc as pickled binaries?
    read_model_from_disk (boolean), read the models from disc if they are already built?
    target (str), Name of the target variable (e.g. iodide)
    model_names (list), List of model names to build/read
    random_state (int), the seed used by the random number generator
    delete_existing_model_files (boolean), delete the existing model binaries in folder?
    rm_outliers (boolean), remove the outliers from the observational dataset
    rm_LOD_filled_data (boolean), remove the limit of detection (LOD) filled values?
    model_feature_dict (dict), dictionary of features used in each model
    model_sub_dir (str), the sub directory in which the models are to be saved/read
    debug (boolean), run and debug function/output

    Returns
    -------
    (dict)
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.externals import joblib
    import gc
    # - Get processed data
    if isinstance(df, type(None)):
        print('Dictionary of model names and features must be provided!')
        sys.exit()

    # - Get local variables
    # Location to save models
    s2s_root = utils.get_file_locations('s2s_root')
    folder = '{}/{}/models/LIVE/{}/'.format(s2s_root, target, model_sub_dir)
    if debug:
        print('Using models from {}'.format(folder))
    # Get details on model setups to use
    if isinstance(model_feature_dict, type(None)):
        print('Dictionary of model names and features must be provided!')
        sys.exit()
    if isinstance(model_names, type(None)):
        model_names = list(sorted(model_feature_dict.keys()))
    # Set a hyperparameter settings
    hyperparam_dict = utils.get_hyperparameter_dict()
    # Setup dictionaries to save model detail into
    N_features_used = {}
    features_used_dict = {}
    oob_scores = {}
    models_dict = {}

    # - Loop model input variable options and build models
    if not read_model_from_disk:
        for n_model_name, model_name in enumerate(model_names):
            print(n_model_name, model_name)
            # Get testing features and hyperparameters to build model
            features_used = model_feature_dict[model_name]
            n_estimators = hyperparam_dict['n_estimators']
            oob_score = hyperparam_dict['oob_score']
            # select and split variables in the training and test dataset
            train_set_tr = df.loc[df[testset] != True, features_used]
            train_set_tr_labels = df.loc[df[testset] != True, target]
            # Build model (Setup and fit)
            model = RandomForestRegressor(random_state=random_state,
                                          n_estimators=n_estimators,
                                          oob_score=oob_score,
                                          criterion='mse')
            # Provide the model with the features (features_used) and
            # The labels ( target, train_set_tr_labels)
            model.fit(train_set_tr, train_set_tr_labels)
            # Save model in temporary folder?
            if save_model_to_disk:
                # Check if there are any existing files...
                pkls_in_dir = glob.glob(folder+'*.pkl')
                Npkls = len(pkls_in_dir)
                if delete_existing_model_files and (n_model_name == 0):
                    import os
                    [os.remove(i) for i in pkls_in_dir]
                    print('WARNING: deleted existing ({}) pkls'.format(Npkls))
                elif(not delete_existing_model_files) and (n_model_name == 0):
                    assert Npkls == 0, 'WARNING: model files exist!'
                else:
                    pass
                # Save models...
                model_savename = "my_model_{:0>4}.pkl".format(n_model_name)
                try:
                    joblib.dump(model, folder+model_savename)
                except FileNotFoundError:
                    utils.check_or_mk_directory_struture()
            # Also keep models online in dictionary
            models_dict[model_name] = model
            # force local tidy of garbage
            gc.collect()

    # -  Loop model and predict for all values
    # If time to make models too great, then read-in here and 'rm' from above
    for n_model_name, model_name in enumerate(model_names):
        # Get testing features and hyperparameters to build model
        features_used = model_feature_dict[model_name]
        print(n_model_name, model_name, features_used)
        # read from disk
        if (not save_model_to_disk) and (read_model_from_disk):
            model_savename = "my_model_{:0>4}.pkl".format(n_model_name)
            model = joblib.load(folder+model_savename)
            models_dict[model_name] = model
        else:
            model = models_dict[model_name]
        # Predict target for all observation locations
        df[model_name] = model.predict(df[features_used].values)
        # Save number of features used too
        N_features_used[model_name] = len(features_used)
        features_used_dict[model_name] = '+'.join(features_used)
        try:
            oob_scores[model_name] = model.oob_score_
        except:
            oob_scores[model_name] = np.NaN
        models_dict[model_name] = model

    # - Return models and predictions in a dictionary structure
    RFR_dict = {}
    RFR_dict['models_dict'] = models_dict
    RFR_dict['model_names'] = model_names
    RFR_dict['df'] = df
    RFR_dict['features_used_dict'] = features_used_dict
    RFR_dict['N_features_used'] = N_features_used
    RFR_dict['oob_scores'] = oob_scores
    return RFR_dict


def get_features_used_by_model(models_list=None, RFR_dict=None):
    """
    Get the (set of) features used by a list of models

    Parameters
    -------

    Returns
    -------
    (dict)

    Notes
    -----

    """
    # Get dictionary of shared data if not provided
    if isinstance(RFR_dict, type(None)):
        RFR_dict = build_or_get_models()
    # get models to use (assume top models, if not provided)
    if isinstance(models_list, type(None)):
        models_list = get_top_models(RFR_dict=RFR_dict, NO_DERIVED=True)
    # now plot up in input variables
    features_used_dict = RFR_dict['features_used_dict']
    vars2use = []
    for model_name in models_list:
        vars2use += [features_used_dict[model_name].split('+')]
    # remove double ups
    vars2use = [j for i in vars2use for j in i]
    return list(set(vars2use))


def get_top_models(n=10, stats=None, RFR_dict=None, NO_DERIVED=True,
                   exclude_ensemble=True, verbose=True):
    """
    retrieve the names of the top 10 models

    Parameters
    -------

    Returns
    -------
    (dict)

    Notes
    -----
    """
    # Get stats on models in RFR_dict
    if isinstance(RFR_dict, type(None)):
        RFR_dict = build_or_get_models()
    if isinstance(stats, type(None)):
        stats = get_core_stats_on_current_models(
            RFR_dict=RFR_dict, verbose=False)
    # Don't count the Ensemble in the topten
    if exclude_ensemble:
        var_ = 'RFR(Ensemble)'
        try:
            stats = stats.T[[i for i in stats.T.columns if var_ not in i]].T
            if verbose:
                print('removed {} from list'.format(var_))
        except:
            if verbose:
                print('failed to remove {} from list'.format(var_))
    # Return the top model's names (with ot without derivative values)
    if NO_DERIVED:
        params2inc = stats.T.columns
        params2inc = [i for i in params2inc if 'DOC' not in i]
        params2inc = [i for i in params2inc if 'Prod' not in i]
        return list(stats.T[params2inc].T.head(n).index)
    else:
        return list(stats.head(n).index)


# def get_choosen_model_from_features_selection(rtn_features=True):
#     """
#     Load choosen model and retrieve its testing features
#
#     Parameters
#     -------
#
#     Returns
#     -------
#
#     Notes
#     -----
#     """
#     from sklearn.externals import joblib
#     import glob
#     # load best estimator model
#     s2s_root = utils.get_file_locations('s2s_root')
#     folder = '{}/{}/models/LIVE/CHOOSEN_MODEL/'.format(s2s_root, target)
#     prefix = 'my_model_'
#     model_savename = glob.glob(folder+prefix+"*.pkl")
#     N = len(model_savename)
#     assert N == 1, 'There should be only one choosen model! Not {}'.format(N)
#     model_savename = model_savename[0]
#     name = model_savename.split(prefix)[1][:-4]
#     model = joblib.load(model_savename)
#     # Return model and variables as dict
#     mdict = {'model': model, 'name': name}
#     if rtn_features:
#         mdict['features_used'] = get_model_features_used_dict(name)
#     return mdict


def Hyperparameter_Tune4choosen_models(RFR_dict=None, target='Iodide',
                                       testset='Test set (strat. 20%)'):
    """
    Driver to tune mutiple RFR models

    Parameters
    -------

    Returns
    -------

    Notes
    -----
    """
    from sklearn.externals import joblib
    # Get the data for the models
    if isinstance(RFR_dict, type(None)):
        RFR_dict = build_or_get_models()
    # Set models to optimise
    models2compare = get_top_models(RFR_dict=RFR_dict, NO_DERIVED=True)
    # Get variables needed from core dictionary
    features_used_dict = RFR_dict['features_used_dict']
    models_dict = RFR_dict['models_dict']
    # Set folder to use for optimised models
    s2s_root = utils.get_file_locations('s2s_root')
    folder = '{}/{}/models/LIVE/OPTIMISED_MODELS/'.format(s2s_root, target)
    # Use X fold cross validation (e.g. 5 or 7)
    cv = 7
    # Loop and save optimised model
    # NOTE: this could be speed up by using more cores
    for model_name in models2compare:
        print('Optimising model: {}'.format(model_name))
        # Get model
        model = models_dict[model_name]
        # get testing features
        features_used = features_used_dict[model_name].split('+')
        # Tune parameters
        BE = Hyperparameter_Tune_model(model=model, use_choosen_model=False,
                                       save_best_estimator=True, model_name=model_name,
                                       RFR_dict=RFR_dict,
                                       features_used=features_used, cv=cv)

    # - Test the tuned models against the test set
    test_the_tuned_models = False
    if test_the_tuned_models:
        # Get the core data
        df = RFR_dict['df']
        # get the data
        test_set = df.loc[df[testset] == True, :]
        train_set = df.loc[df[testset] == False, :]
        # Test the improvements in the optimised models?
        for model_name in models2compare:
            # - Get existing model
            model = models_dict[model_name]
            # Get testing features
            features_used = features_used_dict[model_name].split('+')
            # -  Get the data
            # ( Making sure to remove the target!!! )
    #        train_features = df[features_used].loc[ train_set.index  ]
    #        train_labels = df[[target]].loc[ train_set.index  ]
            test_features = df[features_used].loc[test_set.index]
            test_labels = df[[target]].loc[test_set.index]
            # - test the existing model
            print(' ---------------- '*3)
            print(' ---------------- {}: '.format(model_name))
            print(' - Base values: ')
            quick_model_evaluation(model, test_features, test_labels)
            # - Get optimised model
            try:
                model_savename = "my_model_{}.pkl".format(model_name)
                OPmodel = joblib.load(folder + model_savename)
                #
                print(' - Optimised values: ')
                quick_model_evaluation(OPmodel, test_features, test_labels)
            except:
                pass
    # - Test the tuned models against the training set
        # Get the core data
        df = RFR_dict['df']
        # get the data
        test_set = df.loc[df[testset] == True, :]
        train_set = df.loc[df[testset] == False, :]
        # Test the improvements in the optimised models?
        for model_name in models2compare:
            # - Get existing model
            model = models_dict[model_name]
            # get testing features
            features_used = features_used_dict[model_name].split('+')
            # -  Get the data
            # ( Making sure to remove the target!!! )
            train_features = df[features_used].loc[train_set.index]
            train_labels = df[[target]].loc[train_set.index]
#            test_features = df[features_used].loc[ test_set.index ]
#            test_labels = df[[target]].loc[ test_set.index ]
            # - test the existing model
            print(' ---------------- '*3)
            print(' ---------------- {}: '.format(model_name))
            print(' - Base values: ')
            quick_model_evaluation(model, train_features, train_labels)
            # - Get optimised model
            try:
                model_savename = "my_model_{}.pkl".format(model_name)
                OPmodel = joblib.load(folder + model_savename)
                #
                print(' - Optimised values: ')
                quick_model_evaluation(OPmodel, train_features, train_labels)
            except:
                pass


def Hyperparameter_Tune_model(use_choosen_model=True, model=None,
                              RFR_dict=None, df=None, cv=3,
                              testset='Test set (strat. 20%)', target='Iodide',
                              features_used=None, model_name=None,
                              save_best_estimator=True):
    """
    Driver to tune hyperparmeters of model

    Parameters
    -------

    Returns
    -------

    Notes
    -----
    """
    from sklearn.externals import joblib
    from sklearn.ensemble import RandomForestRegressor
    # Get data to test
    if isinstance(df, type(None)):
        #        df = get_dataset_processed4ML()
        df = RFR_dict['df']

    # Use the model selected from the feature testing
    if use_choosen_model:
        assert_str = "model name not needed as use_choosen_model selected!"
        assert isinstance(model, type(None)), assert_str
        # select a single chosen model
        mdict = get_choosen_model_from_features_selection()
        features_used = mdict['features_used']
        model = mdict['model']
        model_name = mdict['name']

    # - extract training dataset
    test_set = df.loc[df[testset] == True, :]
    train_set = df.loc[df[testset] == False, :]
    # also sub select all vectors for input data
    # ( Making sure to remove the target!!! )
    train_features = df[features_used].loc[train_set.index]
    train_labels = df[[target]].loc[train_set.index]
    test_features = df[features_used].loc[test_set.index]
    test_labels = df[[target]].loc[test_set.index]

    # - Make the base model for comparisons
    base_model = RandomForestRegressor(n_estimators=10, random_state=42,
                                       criterion='mse')
    base_model.fit(train_features, train_labels)
    quick_model_evaluation(base_model, test_features, test_labels)

    # - First make an intial explore of the parameter space
    rf_random = Use_RS_CV_to_explore_hyperparams(cv=cv,
                                                 train_features=train_features,
                                                 train_labels=train_labels,
                                                 features_used=features_used
                                                 #                                                 test_features=test_features,
                                                 #                                                 test_labels=test_labels
                                                 )
    # Check the performance by Random searching (RandomizedSearchCV)
    best_random = rf_random.best_estimator_
    best_params_ = rf_random.best_params_
    print(rf_random.best_params_)
    quick_model_evaluation(best_random, test_features, test_labels)

    # - Now do a more focused optimisation
    # get the parameters based on the RandomizedSearchCV output
    param_grid = define_hyperparameter_options2test(
        features_used=features_used, best_params_=best_params_,
        param_grid_based_on_RandomizedSearchCV=True)
    # Use GridSearchCV
    grid_search = use_GS_CV_to_tune_Hyperparams(cv=cv,
                                                train_features=train_features,
                                                param_grid=param_grid,
                                                train_labels=train_labels,
                                                features_used=features_used,
                                                #                                               test_features=test_features,
                                                #                                                test_labels=test_labels
                                                )
    print(grid_search.best_params_)
    # Check the performance of grid seraching searching
    BEST_ESTIMATOR = grid_search.best_estimator_
    quick_model_evaluation(BEST_ESTIMATOR, test_features, test_labels)

    # Save the best estimator now for future use
    if save_best_estimator:
        s2s_root = utils.get_file_locations('s2s_root')
        folder = '{}/{}/models/LIVE/OPTIMISED_MODELS/'.format(s2s_root, target)
        model_savename = "my_model_{}.pkl".format(model_name)
        joblib.dump(BEST_ESTIMATOR, folder + model_savename)
    else:
        return BEST_ESTIMATOR


def Use_RS_CV_to_explore_hyperparams(train_features=None,
                                     train_labels=None,
                                     features_used=None,
                                     test_features=None,
                                     test_labels=None,
                                     scoring='neg_mean_squared_error',
                                     cv=3):
    """
    Intial test of parameter space using RandomizedSearchCV


    Parameters
    -------

    Returns
    -------

    Notes
    -----
    """
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.ensemble import RandomForestRegressor
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=10, stop=1000, num=10)]
    # Number of features to consider at every split
#    max_features = ['auto', 'sqrt']
    max_features = range(1, 30)
    if not isinstance(features_used, type(None)):
        max_features = [i for i in max_features if
                        i <= len(features_used)]
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
#    bootstrap = [True, False]
    bootstrap = [True]  # Force use of bootstrapping
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    rf = RandomForestRegressor(random_state=42, criterion='mse')
    # Random search of parameters, using 3 fold cross validation,
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator=rf,
                                   param_distributions=random_grid, n_iter=100, cv=cv,
                                   verbose=2,
                                   random_state=42, n_jobs=-1, scoring=scoring)
    # Fit the random search model
    rf_random.fit(train_features, train_labels)
    return rf_random


def use_GS_CV_to_tune_Hyperparams(param_grid=None,
                                  train_features=None, train_labels=None,
                                  features_used=None, \
                                  #                                  test_features=None, test_labels=None
                                  scoring='neg_mean_squared_error', cv=3,
                                  ):
    """
    Refine hyperparameters using (GridSearchCV)

    Parameters
    -------

    Returns
    -------

    Notes
    -----
    """
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import RandomForestRegressor
    # Create a based model
    rf = RandomForestRegressor(random_state=42, criterion='mse')
    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                               cv=cv, n_jobs=-1, verbose=2, scoring=scoring)
    # Fit the grid search to the data
    grid_search.fit(train_features, train_labels)
    return grid_search


def quick_model_evaluation(model, test_features, test_labels):
    """
    Perform a quick model evaluation
    """
    from sklearn.metrics import mean_squared_error
    predictions = model.predict(test_features)
#    mse = np.mean( (predictions - test_labels.values) **2 )
    MSE = mean_squared_error(test_labels, predictions)
    RMSE = np.sqrt(MSE)
    ME = np.mean(abs(predictions - test_labels.values))
    print('Model Performance')
    print('Mean squared error (MAE): {:0.4f} nM'.format(MSE))
    print('Mean absolute error (MAE): {:0.4f} nM'.format(ME))
    print('RMSE = {:0.2f}'.format(RMSE))
    return RMSE


def define_hyperparameter_options2test(features_used=None,
                                       param_grid_based_on_RandomizedSearchCV=True,
                                       best_params_=None,
                                       param_grid_based_on_intial_guesses=True,
                                       ):
    """
    Define a selction of test groups

    Parameters
    -------

    Returns
    -------

    Notes
    -----
    """
    # - Shared variables in grid
    vals2test = {
        'n_estimators': [10, 50, 75, 100, 125, 200, 300, 500],
        'max_features': [1, 2, 3, 4, 5],
        'max_depth': [3, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 3, 10],
        'oob_score':  [True],
        'bootstrap': [True],
    }
    if param_grid_based_on_RandomizedSearchCV:
        if not isinstance(best_params_, type(None)):
            vals2test_ASSUMED = vals2test.copy()
            vals2test = {}
            for key in best_params_:
                value = best_params_[key]
                # 'n_estimators' / trees
                if (key == 'n_estimators'):
                    values = [value+(i*10) for i in range(0, 4)]
                    values += [value+(i*10) for i in range(-4, 0)]
                    # only allow values greater than zero
                    values = [i for i in values if i > 0]
                    # add sorted values
                    vals2test[key] = sorted(values)
                # max depth
                elif (key == 'max_depth'):
                    # value is either a number of "None".
                    if is_number(value):
                        values = [value+(i*5) for i in range(0, 2)]
                        values += [value+(i*5) for i in range(-2, 0)]
                        # only allow values greater than zero
                        values = [i for i in values if i > 0]
                        # add sorted values
                        vals2test[key] = sorted(values)
                    else:  # If None, just use None.
                        vals2test[key] = [value]
                # 'min_samples_split'
                elif (key == 'min_samples_leaf'):
                    if value == 1:
                        values = range(value, value+3)
                    else:
                        values = [value, value+1, value+2]
                    # add sorted values
                    vals2test[key] = list(sorted(values))
                # 'min_samples_split'
                elif (key == 'min_samples_split'):
                    values = [value, value+1, value+2]
                    # add sorted values
                    vals2test[key] = list(sorted(values))
                # Add bootstrap and 'max_features' as recived
                elif (key == 'bootstrap') or (key == 'max_features'):
                    vals2test[key] = [value]
                # Check the key has settings intialised for
                else:
                    print('No settings setup for {}'.format(key))
                    sys.exit()

            # check all the values in best_params_ are in dict
            new_keys = best_params_.keys()
            old_keys = vals2test_ASSUMED.keys()
            extra_keys = [i for i in old_keys if i not in new_keys]
            print('WARNING: adding standard keys for: ', extra_keys)
            for key in extra_keys:
                vals2test[key] = vals2test_ASSUMED[key]
            # check all values in
            all_in_dict = any([i not in vals2test.keys() for i in new_keys])
            assert (not all_in_dict), 'Missing keys from provided best_params_'

        else:
            vals2test = {
                'n_estimators': [80+(i*10) for i in range(8)],
                'max_features': [1, 2, 3, 4, 5],
                'max_depth': [90+(i*5) for i in range(5)],
                'min_samples_split': [4, 5, 6],
                'min_samples_leaf': [1, 2, 3],
                'oob_score':  [True],
                'bootstrap': [True],
            }
    # Check the number of variations being tested

    def prod(iterable):
        import operator
        return reduce(operator.mul, iterable, 1)
    len_of_values = [len(vals2test[i]) for i in vals2test.keys()]
    print('WARNING: # of variations undertest = {}'.format(prod(len_of_values)))
    # Make sure the max features isn't set to more features_used that known
    if not isinstance(features_used, type(None)):
        max_features = vals2test['max_features']
        max_features = [i for i in max_features if
                        i <= len(features_used)]
        vals2test['max_features'] = max_features
    # --- Setup a parameter grid for testings
    param_grid = [
        # - # of trees (“n_estimators”, test=10, 25, 50, 100, 250, 500)
        #     {
        #     'bootstrap': [True],
        #     'n_estimators': vals2test['n_estimators'],
        #     'oob_score': [True],
        #     },
        #     # - # of features/”variables” (“max_features”, test= 2,3,4, None)
        #     {
        #     'bootstrap': [True],
        #     'max_features': vals2test['max_features2test'],
        #     'oob_score': [True],
        #     },
        #     # - both of the above
        #     {
        #     'bootstrap': [True],
        #     'n_estimators': vals2test['n_estimators'],
        #     'max_features': vals2test['max_features'],
        #     'oob_score': [True],
        #     },
        #     # - Minimum samples per leaf
        #     {
        #     'bootstrap': [True],
        #     "min_samples_leaf": vals2test['min_samples_leaf'],
        #     'oob_score': [True],
        #     },
        #     # - Depth
        #     {
        #     'bootstrap': [True],
        #     "max_depth": max_depth2test,
        #     'oob_score': [True],
        #     },
        #     # - Split?
        #     {
        #     'bootstrap': [True],
        #     "min_samples_split": vals2test['min_samples_split'],
        #     'oob_score': [True],
        #     },
        # - all of the above
        {
            'bootstrap': vals2test['bootstrap'],
            'n_estimators': vals2test['n_estimators'],
            'max_features': vals2test['max_features'],
            "min_samples_split": vals2test['min_samples_split'],
            "min_samples_leaf": vals2test['min_samples_leaf'],
            "max_depth": vals2test['max_depth'],
            'oob_score': vals2test['oob_score'],
        },
    ]
    if param_grid_based_on_intial_guesses:
        return param_grid
    elif return_random_informed_grid:
        return param_grid_based_on_RandomizedSearchCV


def mk_predictions_NetCDF_4_many_builds(model2use, res='4x5',
                                        models_dict=None, features_used_dict=None,
                                        RFR_dict=None, target='Iodide',
                                        stats=None, plot2check=False,
                                        rm_Skagerrak_data=False,
                                        debug=False):
    """
    Make a NetCDF file of predicted variables for a given resolution

    Parameters
    -------

    Returns
    -------

    Notes
    -----
    """
    from sklearn.externals import joblib
    import gc
    import glob
    # - local variables
    # extract the models...
    if isinstance(RFR_dict, type(None)):
        RFR_dict = build_or_get_models(
            rm_Skagerrak_data=rm_Skagerrak_data
        )
    # Get the variables required here
    if isinstance(features_used_dict, type(None)):
        features_used_dict = RFR_dict['features_used_dict']
    # Set the extr_str if rm_Skagerrak_data set to True
    if rm_Skagerrak_data:
        extr_str = '_No_Skagerrak'
    else:
        extr_str = ''
    # Get location to save file and set filename
    data_root = utils.get_file_locations('data_root')
    filename = 'Oi_prj_feature_variables_{}.nc'.format(res)
    dsA = xr.open_dataset(data_root + filename)
    # Get location to save ensemble builds of models
    s2s_root = utils.get_file_locations('s2s_root')
    folder_str = '{}/{}/models/LIVE/ENSEMBLE_REPEAT_BUILD{}/'
    folder = folder_str.format(s2s_root, target, extr_str)
    # - Make a da for each model
    ds_l = []
    # Get list of twenty models built
    models_str = folder + '*{}*.pkl'.format(model2use)
    builds4model = glob.glob(models_str)
    print(builds4model, models_str)
    # Print a string to debug the output
    db_str = "Found {} saved models for '{} - glob str:{}'"
    print(db_str.format(len(builds4model), model2use, models_str))
    # Get the numbers for the models in directory
    b_modelnames = [i.split('my_model_')[-1][:-3] for i in builds4model]
    # Check the number of models selected
    ast_str = "There aren't models for {} in {}"
    assert len(b_modelnames) > 1, ast_str.format(model2use, folder)
    # Now loop by model built for ensemble member and predict values
    for n_modelname, b_modelname in enumerate(b_modelnames):
        # Load the model
        model = joblib.load(builds4model[n_modelname])
        # Get testinng features
        features_used = features_used_dict[model2use].split('+')
        # Make a DataSet of predicted values
        ds_l += [mk_da_of_predicted_values(model=model, res=res, dsA=dsA,
                                           modelname=b_modelname,
                                           features_used=features_used)]
        # Force local tidy of garbage
        gc.collect()
    # Combine datasets
    ds = xr.merge(ds_l)
    # - Also get values for existing parameterisations
    if target == 'Iodide':
        # Chance et al (2013)
        param = u'Chance2014_STTxx2_I'
        arr = calc_iodide_chance2014_STTxx2_I(dsA['WOA_TEMP'].values)
        ds[param] = ds[b_modelname]  # use existing array as dummy to fill
        ds[param].values = arr
        # MacDonald et al (2013)
        param = 'MacDonald2014_iodide'
        arr = calc_iodide_MacDonald2014(dsA['WOA_TEMP'].values)
        ds[param] = ds[b_modelname]  # use existing array as dummy to fill
        ds[param].values = arr
    # Do a test diagnostic plot?
    if plot2check:
        for var_ in ds.data_vars:
            # Do a quick plot to check
            arr = ds[var_].mean(dim='time')
            AC.map_plot(arr, res=res)
            plt.title(var_)
            plt.show()
    # Save to NetCDF
    save_name = 'Oi_prj_predicted_{}_{}_ENSEMBLE_BUILDS_{}_{}.nc'
    ds.to_netcdf(save_name.format(target, res, model2use, extr_str))


def get_model_predictions4obs_point(df=None, model_name='TEMP+DEPTH+SAL',
                                    model=None, features_used=None):
    """
    Get model predictions for all observed points

    Parameters
    -------

    Returns
    -------

    Notes
    -----
    """
    # Model name?
    if isinstance(model, type(None)):
        print('Model now must be provided get_model_predictions4obs_point')
        sys.exit()
    # Testing features to use
    if isinstance(features_used, type(None)):
        func_name = 'get_model_predictions4obs_point'
        print("The model's features must be provided to {}".format(func_name))
    # Now predict for the given testing features
    target_predictions = model.predict(df[features_used])
    return target_predictions


def mk_testing_training_sets(df=None, target='Iodide',
                             rand_strat=True, features_used=None,
                             random_state=42, rand_20_80=False,
                             nsplits=4, verbose=True, debug=False):
    """
    Make a test and training dataset for ML algorithms

    Parameters
    -------

    Returns
    -------

    Notes
    -----
    """
    # - make Test and training set
    # to make this approach's output identical at every run
    np.random.seed(42)
    # --- Standard random selection:
    if rand_20_80:
        from sklearn.model_selection import train_test_split
        # Use a standard 20% test set.
        train_set, test_set = train_test_split(df, test_size=0.2,
                                               random_state=random_state)
        # also sub select all vectors for input data
        # ( Making sure to remove the target!!! )
        train_set = df[features_used].loc[train_set.index]
        test_set = df[features_used].loc[test_set.index]
        test_set_targets = df[[target]].loc[test_set.index]

    # - Use a random split
    if rand_strat:
        from sklearn.model_selection import StratifiedShuffleSplit
        # Add in "SPLIT_GROUP" metric
        SPLITvar = 'SPLIT_GROUP'
        use_ceil_of_log = False  # This approach was only used
        if use_ceil_of_log:
            # Original approach taken for AGU work etc
            ceil_ln_limited = np.ceil(np.log(df[target]))
            # push bottom end values into lower bin
            ceil_ln_limited[ceil_ln_limited <= 2] = 2
            # push top end values in higher bin
            ceil_ln_limited[ceil_ln_limited >= 5] = 5
            df[SPLITvar] = ceil_ln_limited
        else:
            # Use decals and put the bins with high values to together
            # NOTE: use quartile cut! (pd.qcut, not pd.cut)
            #            df[SPLITvar] = pd.cut(df[target].values,10).codes.astype(int)
            # Combine the lesser populated higher 5 bins into the 5th bin
            #            df.loc[ df[SPLITvar] >= 4, SPLITvar ] = 4
            # qcut will split the data into N ("nsplits") bins (e.g. quintiles)
            #            pd.qcut(df[target].values,5).value_counts()
            df[SPLITvar] = pd.qcut(df[target].values, nsplits).codes
            if verbose:
                print(df[SPLITvar].value_counts())
        # setup the split
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2,
                                       random_state=random_state)
        # Now split
        for train_index, test_index in split.split(df, df[SPLITvar]):
            train_set = df.loc[train_index]
            test_set = df.loc[test_index]
            test_set_targets = df[[target]].loc[test_index]
        # Gotcha for changes in array index
        Na = df[~df.index.isin(train_index.tolist() + test_index.tolist())]
#        assert (Na.shape[0] < 0), 'WARING: thre now NaNs in the arrays! - {}'
        if (Na.shape[0] < 0):
            print('WARNING'*20)
            print(Na)
        # Print out the split of the bins...
        if verbose:
            dfs = {
                'ALL data': df, 'test data': test_set, 'train data': train_set
            }
            for key_ in dfs.keys():
                print('data split in: {}'.format(key_))
                print(dfs[key_][SPLITvar].value_counts() / dfs[key_].shape[0])
        # Now remove the SPLIT group
        for set_ in train_set, test_set:
            #            if (SPLITvar in set_.columns):
            set_.drop(SPLITvar, axis=1, inplace=True)
        # re-name (TODO mak naming consistent!)
#        train_set_full = train_set
#        test_set_full = test_set
#        test_set_targets =test_set_targets

#        print('NOT setup!'*100)
#        sys.exit()
#    else:
#        print('select a method for making the test dataset')
#        sys.exit()
    return train_set, test_set, test_set_targets


# def get_current_model(dir=None, extr_str=''):
#     """
#     Load the saved model being used by this work - REDUNDENT
#     """
#     from sklearn.externals import joblib
#     # Get the location of the saved model and load it
#     if isinstance(dir, type(None)):
#         s2s_root = utils.get_file_locations('s2s_root')
#         folder = '{}/{}/models/'.format(s2s_root, target)
#     model_savename = "my_model_{}.pkl".format(extr_str)
#     return joblib.load(folder+model_savename)


# def get_predict_lat_lon_array(res='4x5', month=9):
#     """
#     Load extracted predictor array - REDUNDENT
#
#     Parameters
#     -------
#     res (str), horizontal resolution of dataset (e.g. 4x5)
#     month (int), integer month between 1 (Jan) and 12 (Dec)
#
#     Returns
#     -------
#     (pd.DataFrame)
#     """
#     # get file of predictor values for res and month
#     filename = 'Oi_prj_predictor_values_{}_month_num_{}'.format(res, month)
#     if res == '4x5':
#         filename += '_TEST'
#     filename += '.csv'
#     folder = utils.get_file_locations('data_root')
#     folder += 'Oi_prj_predictor_files_by_month_{}/'.format(res)
#     df = pd.read_csv(folder+filename)
#     # make sure the variables have the same names
#     rename_dict = {'lat': 'Latitude', 'lon': 'Longitude'}
#     df.rename(columns=rename_dict, inplace=True)
#     # Make sure the input variables are numeric
#     num_params = ['WOA_Nitrate', 'WOA_Salinity', 'WOA_TEMP', 'SeaWIFs_ChlrA']
#     for param in num_params:
#         df[param] = pd.to_numeric(df[param].values, errors='coerce')
#     # Add modulus if not in array
#     mod_var = 'Latitude (MOD)'
#     if not (mod_var in df.columns):
#         df[mod_var] = (df[u'Latitude']**2)**0.5
#     # Add TEMP_K if not in array
#     TEMP_var = 'WOA_TEMP'
#     if not (TEMP_var+'_K' in df.columns):
#         df[TEMP_var+'_K'] = df[TEMP_var].values + 273.15
#     # Force values to be numerics
#     for col in df.columns:
#         df[col] = pd.to_numeric(df[col], errors='coerce')
#     # Drop NaNs (e.g. bits over land...
#     df_dropped = df.dropna()
#     if (df.shape != df_dropped.shape):
#         ptr_str = 'WARNING droppped values {} => {}'
#         print(ptr_str.format(df.shape, df_dropped.shape))
#         df = df_dropped
#     # Return DataFrane
#     return df


def mk_predictions_for_3D_features(dsA=None, RFR_dict=None, res='4x5',
                                   models_dict=None, features_used_dict=None,
                                   stats=None, folder=None, target='Iodide',
                                   use_updated_predictor_NetCDF=False,
                                   save2NetCDF=False, plot2check=False,
                                   models2compare=[], topmodels=None,
                                   xsave_str='',
                                   add_ensemble2ds=False,
                                   verbose=True, debug=False):
    """
    Make a NetCDF file of predicted target from feature variables for a given resolution

    Parameters
    ----------
    dsA (xr.Dataset), dataset object with variables to interpolate
    RFR_dict (dict), dictionary of core variables and data
    res (res), horizontal resolution (e.g. 4x5) of Dataset
    save2NetCDF (boolean), save interpolated Dataset to as a NetCDF?
    features_used_dict (dict), dictionary of feature variables in models
    models_dict (dict), dictionary of RFR models and there names
    stats (pd.DataFrame), dataframe of statistics on models in models_dict
    folder (str), location of NetCDF file of feature variables
    target (str), name of the species being predicted
    models2compare (list), list of models to make spatial predictions for (rm: double up?)
    topmodels (list), list of models to make spatial predictions for
    xsave_str (str), string to include as suffix in filename used for saved NetCDF
    add_ensemble2ds (boolean), calculate std. dev. and mean for list of topmodels
    verbose (boolean), print out verbose output?
    debug (boolean), print out debugging output?

    Returns
    -------
    (xr.Dataset)
    """
    # Make sure the core dictionary is provided
    assert (type(RFR_dict) ==
            dict), 'Core variables must be provided as dict (RFR_dict)'
    # Make sure a full list of models was provided
    assert (len(models2compare) > 0), 'List of models to must be provided!'
    # Inc. all the topmodels in the list of models to compare if they have been provided.
    if isinstance(topmodels, type(list)):
        models2compare += topmodels
    # Remove any double ups in list of of models to predict
    models2compare = list(set(models2compare))
    # Get the variables required here
    if isinstance(models_dict, type(None)):
        models_dict = RFR_dict['models_dict']
    if isinstance(features_used_dict, type(None)):
        features_used_dict = RFR_dict['features_used_dict']
    # Get location to save file and set filename
    if isinstance(folder, type(None)):
        folder = utils.get_file_locations('data_root')
    if isinstance(dsA, type(None)):
        filename = 'Oi_prj_feature_variables_{}.nc'.format(res)
        dsA = xr.open_dataset(folder + filename)
    # - Make a dataset of predictions for each model
    ds_l = []
    for modelname in models2compare:
        # get model
        model = models_dict[modelname]
        # get testinng features
        features_used = utils.get_model_features_used_dict(modelname)
        # Make a DataSet of predicted values
        ds_tmp = utils.mk_da_of_predicted_values(dsA=dsA, model=model, res=res,
                                                 modelname=modelname,
                                                 features_used=features_used)
        #  Add attributes to the prediction
        ds_tmp = utils.add_attrs2target_ds(ds_tmp, add_global_attrs=False,
                                           varname=modelname)
        # Save to list
        ds_l += [ds_tmp]
    # Combine datasets
    ds = xr.merge(ds_l)
    # - Also get values for parameterisations
    if target == 'Iodide':
        # Chance et al (2013)
        param = u'Chance2014_STTxx2_I'
        arr = utils.calc_iodide_chance2014_STTxx2_I(dsA['WOA_TEMP'].values)
        ds[param] = ds[modelname]  # use existing array as dummy to fill
        ds[param].values = arr
        # MacDonald et al (2013)
        param = 'MacDonald2014_iodide'
        arr = utils.calc_iodide_MacDonald2014(dsA['WOA_TEMP'].values)
        ds[param] = ds[modelname]  # use existing array as dummy to fill
        ds[param].values = arr
    # Add ensemble to ds too
    if add_ensemble2ds:
        print('WARNING: Using topmodels for ensemble as calculated here')
        var2template = list(ds.data_vars)[0]
        ds = analysis.add_ensemble_avg_std_to_dataset(ds=ds, res=res, target=target,
                                                      RFR_dict=RFR_dict,
                                                      topmodels=topmodels,
                                                      var2template=var2template,
                                                      save2NetCDF=False)
    # add global attributes
    ds = utils.add_attrs2target_ds(ds, add_varname_attrs=False)
    # Save to NetCDF
    if save2NetCDF:
        filename = 'Oi_prj_predicted_{}_{}{}.nc'.format(target, res, xsave_str)
        ds.to_netcdf(filename)
    else:
        return ds
