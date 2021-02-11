import sys
import json

from tpotbench.models import selector_classes
from tpotbench.util import get_task_split, predictions_to_selector_labels


def run(config_path):
    config = {}
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)

    print(f'Running\n\nConfig\n------\n{config}')

    # Create the selector model
    algo_type = config['algo_type']
    selector_class = selector_classes[algo_type]

    selector = selector_class(name=config['name'],
                              classifiers=config['classifiers'],
                              model_params=config['model_params'])

    data_split = get_task_split(task=config['task'],
                                seed=config['seed'],
                                split=config['split'])

    X, y = data_split['selector_train']
    classifier_predictions = selector.classifier_predictions(X)
    labels = predictions_to_selector_labels(classifier_predictions, y)

    # Fit and then save model
    selector.fit(X, labels)
    selector.save(path=config['model_path'])


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise RuntimeError('Please provide a config\n'
                           + f'{sys.argv[0]} /path/to/config')
    run(sys.argv[1])
