import os
import sys
import json

import numpy as np
from tpot import TPOTClassifier

from util import get_task_splits, instance_wise_algorithm_correct_vectors

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
    algorithms = config['algorithms']

    # Get the training and test data splits
    data_splits = get_task_splits(task_id, seed, splits)
    selector_X_train, selector_y_train = data_splits['selector_train']
    X_test, y_test = data_splits['test']

    training_classifications_by_time = algorithms['selector_training_classifications']
    testing_classifications_by_time = algorithms['test_classifications']

    tpot = TPOTClassifier(**tpot_params)

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

        # TPOT can be 'hackily' `refit` such that it keeps the same genetic
        # population but resets the scores for each found model.
        # This is needed as the algorithms that give the predictions could
        # change at each timestep so keeping the previous selector model might
        # not be ideal. Ideally the models in the pool are suited for selection
        # and the rescoring should help identify good candidates.
        #
        # Modified hack for refit:
        # https://github.com/EpistasisLab/tpot/issues/881#issuecomment-504421537
        #
        # Modified not to set anything if it doesn't exists
        # ========
        if hasattr(tpot, '._pop'):
            for ind in tpot._pop:
                del ind.fitness.values

        if hasattr(tpot, '._last_optimized_pareto_front'):
            tpot._last_optimized_pareto_front = None

        if hasattr(tpot, '._last_optimized_pareto_front_n_gens'):
            tpot._last_optimized_pareto_front_n_gens = None

        if hasattr(tpot, '._pareto_front'):
            tpot._pareto_front = None
        # ========

        tpot.fit(selector_X_train, train_correct_vectors)

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

        train_score = tpot.score(selector_X_train, train_correct_vectors)
        test_score = tpot.score(X_test, test_correct_vectors)
        print(f'\n\nTrain Score: {train_score}\n\n')
        print(f'\n\nTest Score: {test_score}\n\n')

        export_path = os.path.join(
            folders['exports'], f'export_{time}.py'
        )
        tpot.export(export_path)
