"""
This file runs the AutoSklearnSelector job and is completely dependant on
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

from tpotbench.runner_util import (  # type: ignore[no-name-in-module]
    get_task_splits, classifier_predictions_to_selector_labels
)


def autosklearn_params(time, seed, cpus, memory, model_params):
    core_params = {
        'time_left_for_this_task': time,
        'seed': seed,
        'memory_limit': memory,
        # This is forced to 0 by default with autosklearn2classifier
        # Must set manually with normal autosklearnclassifier
        'initial_configurations_via_metalearning': 0,
        'n_jobs': cpus,
    }
    return {**core_params, **model_params}


def run(config_path):
    config = {}
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)

    files = config['files']

    # Get the training and test data splits
    data_splits = get_task_splits(task_id=config['task_id'],
                                  seed=config['seed'],
                                  splits=config['splits'])

    selector_X_train, selector_y_train = data_splits['selector_train']
    X_test, y_test = data_splits['test']

    # Loading the training classifications over the models to save memory
    clfs_selector_training_classifications = [
        np.load(clf_files['selector_training_classifications'])
        for clf_name, clf_files
        in files['classifiers']
    ]
    selector_training_labels = classifier_predictions_to_selector_labels(
        clfs_selector_training_classifications, selector_y_train
    )

    # Create a new automodel that will be trained
    params = autosklearn_params(time=config['time'],
                                seed=config['seed'],
                                cpus=config['cpus'],
                                memory=config['memory'],
                                model_params=config['model_params'])
    automodel = AutoSklearnClassifier(**params)

    # Providing the X_test and y_test to allow for overtime
    # predictions
    automodel.fit(selector_X_train, selector_training_labels)

    # Save the classification and probability output of the models
    selector_training_classifications = automodel.predict(selector_X_train)
    np.save(files['selector_training_classifications'],
            selector_training_classifications)

    selector_training_probabilities = automodel.predict_proba(selector_X_train)
    np.save(files['selector_training_probabilities'],
            selector_training_probabilities)

    test_classifications = automodel.predict(X_test)
    np.save(files['test_classifications'], test_classifications)

    test_probabilities = automodel.predict_proba(X_test)
    np.save(files['test_probabilities'], test_probabilities)

    # Save the model
    with open(files['model'], 'wb') as f:
        pickle.dump(automodel, f)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise RuntimeError('Please provide a config\n'
                           + f'{sys.argv[0]} /path/to/config')
    run(sys.argv[1])
