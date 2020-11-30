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

    # We train the baseline on the same amount of data that single algorithms
    # and selector would be trained on
    algo_split = splits[0]
    selector_split = splits[1]
    test_split = splits[2]
    new_split = (algo_split + selector_split, test_split)

    data_splits = get_task_splits(task_id, seed, new_split)
    X_train, y_train = data_splits['baseline_train']
    X_test, y_test = data_splits['baseline_test']

    tpot = TPOTClassifier(**tpot_params)

    print('got here')
    for time in times:
        print('loop time')

        tpot.fit(X_train, y_train)

        classifications = tpot.predict(X_test)
        classification_path = os.path.join(
            folders['classifications'], f'classifications_{time}.npy'
        )
        np.save(classification_path, classifications)

        try:
            predictions = tpot.predict_proba(X_test)
            prediction_path = os.path.join(
                folders['predictions'], f'predictions_{time}.npy'
            )
            np.save(prediction_path, predictions)
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
