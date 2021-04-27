import sys
import json

from tpotbench.models import selector_classes
from tpotbench.util import get_task_split, predictions_to_selector_labels
from tpotbench.models.selectors import DESSelectorModel


def run(config_path):
    config = {}
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)

    print(f'Running\n\nConfig\n------\n{config}')

    # Create the selector model
    algo_type = config['algo_type']
    selector_class = selector_classes[algo_type]

    selector = selector_class(name=config['name'],
                              model_params=config['model_params'],
                              classifier_paths=config['classifiers'])

    data_split = get_task_split(task=config['task'],
                                seed=config['seed'],
                                split=config['split'])

    X, y = data_split['selector_train']
    classifiers = selector.classifiers
    print(classifiers)

    # The DESlib models will perform this step themselves
    if not isinstance(selector, DESSelectorModel):
        classifier_predictions = selector.classifier_predictions(X)
        labels = predictions_to_selector_labels(classifier_predictions, y)
    else:
        labels = y

    # Fit and then save model
    print(f'{labels.shape=}')
    print(f'{X.shape=}')
    selector.fit(X, labels)
    selector.save(path=config['model_path'])


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise RuntimeError('Please provide a config\n'
                           + f'{sys.argv[0]} /path/to/config')
    run(sys.argv[1])
