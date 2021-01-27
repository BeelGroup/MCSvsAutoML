"""
This file runs the AutoSklearnBaselineJob and is completely dependant on
the config file it is passed. It also defines the default parameters
that the model is run off through the function `autosklearn_params`. Users can
overwrite these defaults or any other parameters by providing 'model_params' in
the config.
"""
import sys
import json
import pickle

import numpy as np
from autosklearn.classification import AutoSklearnClassifier

from tpotbench.runners.util import (  # type: ignore[no-name-in-module]
    get_task_split
)

def autosklearn_params(time, seed, cpus, memory, model_params):
    params = {
        'time_left_for_this_task': time * 60,
        'seed': seed,
        'memory_limit': memory,
        'n_jobs': cpus,
        **model_params
    }
    return params


def run(config_path):
    config = {}
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)

    files = config['files']

    # Get the training and test data splits
    data_split = get_task_split(task=config['task'],
                                seed=config['seed'],
                                split=config['split'])

    X_train, y_train = data_split['baseline_train']
    X_test, y_test = data_split['baseline_test']

    # Create a new automodel that will be trained
    params = autosklearn_params(time=config['time'],
                                seed=config['seed'],
                                cpus=config['cpus'],
                                memory=config['memory'],
                                model_params=config['model_params'])
    automodel = AutoSklearnClassifier(**params)

    # Providing the X_test and y_test to allow for overtime
    # predictions
    print(f'Fitting model with params {params=}')
    automodel.fit(X_train, y_train)

    # Save the classification and probability output of the models
    training_classifications = automodel.predict(X_train)
    training_probabilities = automodel.predict_proba(X_train)

    test_classifications = automodel.predict(X_test)
    test_probabilities = automodel.predict_proba(X_test)

    # Save data
    file_paths_to_save_to = {
        files['training_classifications']: training_classifications,
        files['training_probabilities']: training_probabilities,
        files['test_classifications']: test_classifications,
        files['test_probabilities']: test_probabilities,
    }
    for path, data in file_paths_to_save_to.items():
        np.save(path, data)

    # Save the model
    with open(files['model'], 'wb') as f:
        pickle.dump(automodel, f)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise RuntimeError('Please provide a config\n'
                           + f'{sys.argv[0]} /path/to/config')
    run(sys.argv[1])
