import pandas
import openml
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import ShuffleSplit


def split_data(X, y, split_percentage, seed):
    splitter = ShuffleSplit(1, test_size=split_percentage, random_state=seed)
    split_1_idxs, split_2_idxs = next(splitter.split(X))
    return {
        'split_1': (X[split_1_idxs], y[split_1_idxs]),
        'split_2': (X[split_2_idxs], y[split_2_idxs])
    }


def get_task_split(task, seed, split):
    if not (len(split) == 2 or len(split) == 3):
        raise ValueError(f'Splits must be either 2 or 3 floats\n{split=}')

    task = openml.tasks.get_task(task)
    X, y, categorical_mask, _ = task.get_dataset().get_data(task.target_name)
# Process labels
    if y is not None:
        if y.dtype == 'category' or y.dtype == object:
            y = LabelEncoder().fit_transform(y.values)
        elif y.dtype == bool:
            y = y.astype('int')

    if isinstance(y, pandas.Series):
        y = y.to_numpy()

    # Process Features
    for col in X.columns:
        mode = X[col].mode()[0]
        X[col].fillna(mode, inplace=True)

    encoding_frames = []
    for col in list(X.columns[categorical_mask]):
        encodings = pandas.get_dummies(X[col], prefix=col, prefix_sep='_')
        X.drop(col, axis=1, inplace=True)
        encoding_frames.append(encodings)

    X = pandas.concat([X, *encoding_frames], axis=1)
    X = X.to_numpy()

    # Create split
    if len(split) == 2:
        train_split = split[0]
        test_split = split[1]
        splits = split_data(X, y, test_split, seed)
        return {
            'baseline_train': splits['split_1'],
            'baseline_test': splits['split_2']
        }
    else:
        algo_split = split[0]
        selector_split = split[1]
        test_split = split[2]

        # Split data between testing and training
        train_test_split = split_data(X, y, test_split, seed)

        # Further divide the train split of train_test_split to be between
        # the algorithm and the selector
        selector_relative_split = selector_split / (algo_split + selector_split)

        X_train, y_train = train_test_split['split_1']
        train_splits = split_data(
            X_train, y_train, selector_relative_split, seed)
        return {
            'algo_train': train_splits['split_1'],
            'selector_train': train_splits['split_2'],
            'test': train_test_split['split_2']
        }


def predictions_to_selector_labels(
    all_model_classifications,
    y
):
    """
    classifiers_classifications: (k, n)
    y: (n)

    Returns
    =======
        (n, k)
    """
    correct_vectors = np.asarray([
        np.equal(model_classifications, y).astype(int)
        for model_classifications in all_model_classifications
    ])
    return np.transpose(correct_vectors)


def deslib_competences(model, X):
    distances, neighbors = model._get_region_competence(X)

    if hasattr(model, 'estimate_competence_from_proba') and model.needs_proba:
        classifier_probabilities = model._predict_proba_base(X)
        competences = model.estimate_competence(
            query=X, neighbors=neighbors, probabilities=classifier_probabilities,
            distances=distances)
    else:
        classifier_predictions = model._predict_proba_base(X)
        competences = model.estimate_competence_from_proba(
            query=X, neighbors=neighbors, predicitons=classifier_predictions,
            distances=distances)

    return competences


def deslib_selections(model, X):
    competences = deslib_competences(model, X)
    selections = model.select(competences)
    return selections
