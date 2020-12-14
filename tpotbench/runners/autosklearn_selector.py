import os
import sys
import json

import numpy as np
from autosklearn.experimental.askl2 import AutuoSklearn2Classifier
from sklearn.metrics import accuracy_score

from util import get_task_splits, instance_wise_algorithm_correct_vectors

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise RuntimeError('Please provide a config\n'
                           + f'{sys.argv[0]} /path/to/config')

    seed = config['seed']
    times = config['times']
    splits = config['splits']
    folders = config['folders']
    files = config['files']
    task_id = config['openml_task_id']
    autosklearn_params = config['tpot_params']
    algorithms = config['algorithms']

    # Get the training and test data splits
    data_splits = get_task_splits(task_id, seed, splits)
    selector_X_train, selector_y_train = data_splits['selector_train']
    X_test, y_test = data_splits['test']

    training_classifications_by_time = algorithms['selector_training_classifications']
    testing_classifications_by_time = algorithms['test_classifications']

    automodel = AutuoSklearn2Classifier(**autosklearn_params)

    for time in times:

        # Get the instance wise algorithm correct vectors
        # These are a row of (1) or (0) depending on if the algorithm
        # correctly classified the instance.
        train_correct_vectors = instance_wise_algorithm_correct_vectors(
            training_classifications_by_time, time, selector_y_train
        )
        test_correct_vectors = instance_wise_algorithm_correct_vectors(
            testing_classifications_by_time, time, y_test
        )

        # Providing the X_test and y_test to allow for overtime
        # predictions
        automodel.fit(selector_X_train, train_correct_vectors,
                      X_test=X_test, y_test=test_correct_vectors)

        train_classifications = automodel.predict(selector_X_train)
        classifications = automodel.predict(X_test)
        classification_path = os.path.join(
            folders['classifications'], f'classifications_{time}.npy'
        )
        np.save(classification_path, classifications)

        predictions = automodel.predict_proba(X_test)
        prediction_path = os.path.join(
            folders['predictions'], f'predictions_{time}.npy'
        )
        np.save(prediction_path, predictions)

        train_score = accuracy_score(classifications, train_correct_vectors)
        train_score = tpot.score(train_classifications, test_correct_vectors)
        test_score = tpot.score(classifications, test_correct_vectors)
        print(f'\n\nTrain Score: {train_score}\n\n')
        print(f'\n\nTest Score: {test_score}\n\n')
