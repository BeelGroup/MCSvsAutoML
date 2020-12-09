import os
import sys
import json

import numpy as np
from tpot import TPOTClassifier

from util import get_task_splits

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise RuntimeError('Please provide a config\n'
                           + f'{sys.argv[0]} /path/to/config')

    config = {}
    config_path = sys.argv[1]
    with open(sys.argv[1], 'r') as config_file:
        config = json.load(config_file)

    seed = config['seed']
    times = config['times']
    splits = config['splits']
    folders = config['folders']
    files = config['files']
    task_id = config['openml_task_id']
    tpot_params = config['tpot_params']

    data_splits = get_task_splits(task_id, seed, splits)
    X_train, y_train = data_splits['algo_train']
    selector_X_train, selector_y_train = data_splits['selector_train']
    X_test, y_test = data_splits['test']

    tpot = TPOTClassifier(**tpot_params)

    for time in times:

        tpot.fit(X_train, y_train)

        # Save classficiations for test data
        test_classifications = tpot.predict(X_test)
        test_classification_path = os.path.join(
            folders['classifications'], f'classifications_{time}.npy'
        )
        np.save(test_classification_path, test_classifications)

        # Save classificaitons for selector training
        selector_training_classifications = tpot.predict(selector_X_train)
        selector_training_classification_path = os.path.join(
            folders['selector_training_classifications'],
            f'selector_training_classifications_{time}.npy'
        )
        np.save(selector_training_classification_path,
                selector_training_classifications)

        try:
            # Save predictions for test data
            predictions = tpot.predict_proba(X_test)
            prediction_path = os.path.join(
                folders['predictions'], f'predictions_{time}.npy'
            )
            np.save(prediction_path, predictions)

            # Save predictions for selector training
            selector_training_predictions = tpot.predict_proba(selector_X_train)
            selector_training_prediction_path = os.path.join(
                folders['selector_training_predictions'],
                f'selector_training_predictions_{time}.npy'
            )
            np.save(selector_training_prediction_path,
                    selector_training_predictions)

        except RuntimeError as err:
            print('\n\n Probably could not use predict_proba \n\n')
            print(err)
            print('\n\n')

        print(f'\n\nTrain Score: {tpot.score(X_train, y_train)=}\n\n')
        print(f'\n\nTest Score: {tpot.score(X_test, y_test)=}\n\n')

        export_path = os.path.join(
            folders['exports'], f'export_{time}.py'
        )
        tpot.export(export_path)
