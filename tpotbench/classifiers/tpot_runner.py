"""
This file runs the TPOTClassifier job and is completely dependant on
the config file it is passed. It also defines the default parameters
that the model is run off through the function `tpot_params`. Users can
overwrite these defaults or any other parameters by providing 'model_params' in
the config.
"""
import os
import sys
import json
import pickle

import numpy as np
from tpot import TPOTClassifier
from tpot.config import classifier_config_dict

from tpotbench.runner_util import get_task_splits

def tpot_params(time, seed, algorithm_family, checkpoint_folder, cpus, logfile,
                model_params):
    families = {
        'KNN': ['sklearn.neighbors.KNeighborsClassifier'],
        'LR': ['sklearn.linear_model.LogisticRegression'],
        'MLP': ['sklearn.neural_network.MLPClassifier'],
        'SGD': ['sklearn.linear_model.SGDClassifier'],
        'XGB': ['xgboost.XGBClassifier'],
        'SVM': ['sklearn.svm.LinearSVC'],
        'NB': ['sklearn.naive_bayes.GaussianNB',
               'sklearn.naive_bayes.BernoulliNB',
               'sklearn.naive_bayes.MultinomialNB'],
        'TR': ['sklearn.tree.DecisionTreeClassifier',
               'sklearn.ensemble.ExtraTreesClassifier',
               'sklearn.ensemble.RandomForestClassifier',
               'sklearn.ensemble.GradientBoostingClassifier'],
    }
    algorithms = families[algorithm_family]
    core_params = {
        'generations': None,
        'population_size': 100,
        'offspring_size': None,
        'mutation_rate': 0.9,
        'crossover_rate': 0.1,
        'scoring': 'accuracy',
        'cv': 5,
        'subsample': 1.0,
        'n_jobs': cpus,
        'max_time_mins': time,
        'max_eval_time_mins': 15,
        'random_state': seed,
        'config_dict': {
            algorithm: classifier_config_dict[algorithm]
            for algorithm in algorithms
        },
        'template': None,
        'warm_start': True,
        'memory': 'auto',
        'use_dask': False,
        'periodic_checkpoint_folder': checkpoint_folder,
        'early_stop': None,
        'verbosity': 3,
        'disable_update_check': False,
        'log_file': logfile
    }
    return {**core_params, **model_params}

def run(config_path):

    config = {}
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)

    started_at = config['started_at']
    os.system(f'date > {started_at}')

    files = config['files']
    folders = config['folders']
    seed = config['seed']
    time = config['time']
    cpus = config['cpus']
    splits = config['splits']
    task_id = config['task_id']
    algorithm_family = config['algorithm_family']
    model_params = config.get('model_params', {})

    data_splits = get_task_splits(task_id, seed, splits)
    X_train, y_train = data_splits['algo_train']
    selector_X_train, selector_y_train = data_splits['selector_train']
    X_test, y_test = data_splits['test']

    params = tpot_params(time=time,
                         seed=seed,
                         algorithm_family=algorithm_family,
                         checkpoint_folder=folders['checkpoints'],
                         cpus=cpus,
                         logfile=files['tpot_log'],
                         model_params=model_params)
    tpot = TPOTClassifier(**params)

    tpot.fit(X_train, y_train)

    # Save best pipeline
    pickle.dump(tpot.fitted_pipeline_, files['model'])

    # Save classifications
    train_classifications = tpot.predict(X_train)
    test_classifications = tpot.predict(X_test)
    selector_training_classifications = tpot.predict(selector_X_train)

    np.save(files['train_classifications'], train_classifications)
    np.save(files['test_classifications'], test_classifications)
    np.save(files['selector_training_classifications'],
            selector_training_classifications)

    try:
        train_probabilities = tpot.predict_proba(X_train)
        test_probabilities = tpot.predict_proba(X_test)
        selector_training_probabilities = tpot.predict_proba(selector_X_train)

        np.save(files['train_probabilities'], train_probabilities)
        np.save(files['test_probabilities'], test_probabilities)
        np.save(files['selector_training_probabilities'],
                selector_training_probabilities)

    except RuntimeError as err:
        print('\n\n Probably could not use predict_proba \n\n')
        print(err)
        print('\n\n')

    tpot.export(files['export'])


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise RuntimeError('Please provide a config\n'
                           + f'{sys.argv[0]} /path/to/config')
    run(sys.argv[1])
