import sys
import json

from tpotbench.models import classifier_classes
from tpotbench.util import get_task_split


def run(config_path):
    config = {}
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)

    print(f'Running\n\nConfig\n------\n{config}')

    # Create the selector model
    algo_type = config['algo_type']
    classifier_class = classifier_classes[algo_type]

    classifier = classifier_class(name=config['name'],
                                  model_params=config['model_params'])

    data_split = get_task_split(task=config['task'],
                                seed=config['seed'],
                                split=config['split'])

    X, y = data_split['algo_train']

    # Fit and save model
    classifier.fit(X, y)
    classifier.save(config['model_path'])


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise RuntimeError('Please provide a config\n'
                           + f'{sys.argv[0]} /path/to/config')
    run(sys.argv[1])
