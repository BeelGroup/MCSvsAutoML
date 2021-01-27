import sys
import json
import pickle

import numpy as np
from deslib.des.meta_des import METADES  # type: ignore

from tpotbench.runners.util import (  # type: ignore[no-name-in-module]
    get_task_split, deslib_competences, deslib_selections
)


def metades_params(seed, model_params):
    return {
        'random_state': seed,
        **model_params
    }


def run(config_path):
    config = {}
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)

    files = config['files']

    # Get the training and test data splits
    data_split = get_task_split(task=config['task'],
                                seed=config['seed'],
                                split=config['split'])

    selector_X_train, selector_y_train = data_split['selector_train']
    X_test, y_test = data_split['test']

    # Load in the pool of classifier models
    classifier_pool = []
    for clf_files in files['classifiers'].values():
        with open(clf_files['model'], 'rb') as model_file:
            model = pickle.load(model_file)
            classifier_pool.append(model)

    # Make and fit model
    params = metades_params(seed=config['seed'],
                            model_params=config['model_params'])
    metades_model = METADES(classifier_pool, **params)

    metades_model.fit(selector_X_train, selector_y_train)

    # Get the competences of the model
    # TODO the api for deslib is rather odd for getting competences
    #  I've wrapped it in runner_util but I should help the author with the
    #  public API
    selector_training_classifier_competences = deslib_competences(metades_model,
                                                                  selector_X_train)
    selector_training_classifier_selections = deslib_selections(metades_model,
                                                                selector_X_train)

    test_classifier_selections = deslib_competences(metades_model, X_test)
    test_classifier_competences = deslib_selections(metades_model, X_test)

    # Save data
    file_paths_to_save_to = {
        files['selector_training_classifier_selections']: selector_training_classifier_selections,
        files['selector_training_classifier_competences']: selector_training_classifier_competences,
        files['test_classifier_selections']: test_classifier_selections,
        files['test_classifier_competences']: test_classifier_competences,
    }
    for path, data in file_paths_to_save_to.items():
        np.save(path, data)

    # Save the model
    with open(files['model'], 'wb') as f:
        pickle.dump(metades_model, f)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise RuntimeError('Please provide a config\n'
                           + f'{sys.argv[0]} /path/to/config')
    run(sys.argv[1])
