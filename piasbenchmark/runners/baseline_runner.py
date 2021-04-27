import sys
import json

from tpotbench.models import baseline_classes
from tpotbench.util import get_task_split


def run(config_path):
    config = {}
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)

    print(f'Running\n\nConfig\n------\n{config}')

    # Create the selector model
    algo_type = config['algo_type']
    baseline_class = baseline_classes[algo_type]

    baseline = baseline_class(name=config['name'],
                              model_params=config['model_params'])

    # Split needs to be modified for baseline
    # The split for classifiers and selector are added together
    split = config['split']
    split = (split[0] + split[1], split[2])
    data_split = get_task_split(task=config['task'],
                                seed=config['seed'],
                                split=split)

    X, y = data_split['baseline_train']

    # Fit and save model
    baseline.fit(X, y)
    baseline.save(config['model_path'])


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise RuntimeError('Please provide a config\n'
                           + f'{sys.argv[0]} /path/to/config')
    run(sys.argv[1])
