"""
This file runs the AutoKerasSelector job and is completely dependant on
the config file it is passed. It also defines the default parameters
that the model is run off through the function `autokeras_params`. Users can
overwrite these defaults or any other parameters by providing 'model_params' in
the config.
"""
import sys
import json
import pickle

import numpy as np  # type: ignore[no-name-in-module]
from autokeras import StructuredDataClassifier

from tpotbench.runners.util import (  # type: ignore[no-name-in-module]
    get_task_split, classifier_predictions_to_selector_labels
)


# TODO: Assuming that nothing needs to be specified for multiclass as it
#   can be detected from using a classifier on a n-d array of 1/0.
#   The API lists:
#       * `num_classes` -- inferred (n_classifiers)
#       * `multi_label` -- inferred (True?)
#       * `loss` -- inferred (`categorical_crossentropy`)
#       * `objective` -- inferred (`val_accuracy`)
#   https://autokeras.com/structured_data_classifier/
def autokeras_model_params(seed, memory, directory, model_params):
    params = {
        'seed': seed,
        'max_model_size': int(memory * 0.75),
        'max_trials': 100,
        'overwrite': True,
        **model_params
    }
    return params


# Autokeras: https://autokeras.com/structured_data_classifier/#fit
# Basemodel: https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit
def autokeras_fit_params(cpus):
    params = {
        'validation_split': 0.2,
        'use_multiprocessing': False,
        'workers': cpus
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

    # Loading the training classifications over the models to save memory
    print('Loading classifiers')
    clfs_selector_training_classifications = [
        np.load(clf_files['selector_training_classifications'])
        for clf_name, clf_files
        in files['classifiers'].items()
    ]
    selector_training_labels = classifier_predictions_to_selector_labels(
        clfs_selector_training_classifications, selector_y_train
    )

    # Create a new automodel that will be trained
    params = autokeras_model_params(seed=config['seed'],
                                    cpus=config['cpus'],
                                    memory=config['memory'],
                                    model_params=config['model_params'])
    time = config['time']
    automodel = StructuredDataClassifier(**params)

    # Providing the X_test and y_test to allow for overtime
    # predictions
    print(f'Fitting model with params {params=}')
    print(
        f'Training on \n\tX={selector_X_train.shape}\n\ty={selector_training_labels.shape}')
    automodel.fit(selector_X_train, selector_training_labels)

    # Save the classification and probability output of the models
    selector_training_classifier_selections = automodel.predict(
        selector_X_train)
    selector_training_classifier_competences = automodel.predict_proba(
        selector_X_train)

    test_classifier_selections = automodel.predict(X_test)
    test_classifier_competences = automodel.predict_proba(X_test)

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
        pickle.dump(automodel, f)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise RuntimeError('Please provide a config\n'
                           + f'{sys.argv[0]} /path/to/config')
    run(sys.argv[1])
