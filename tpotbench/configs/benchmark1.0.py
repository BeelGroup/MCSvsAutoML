import os
import json
from itertools import product

benchmark_name = 'benchmark1.0'

current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, f'{benchmark_name}.json')

tasks = [3, 6, 11, 12, 14, 15, 16, 18, 22, 23, 28, 29, 31, 32, 37, 43, 45, 49,
         53, 219, 2074, 2079, 3021, 3022, 3481, 3549, 3560, 3573, 3902, 3903,
         3904, 3913, 3917, 3918, 7592, 9910, 9946, 9952, 9957, 9960, 9964, 9971,
         9976, 9977, 9978, 9981, 9985, 10093, 10101, 14952, 14954, 14965, 14969,
         14970, 125920, 125922, 146195, 146800, 146817, 146819, 146820, 146821,
         146822, 146824, 146825, 167119, 167120, 167121, 167124, 167125, 167140, 
         167141]
times_in_mins = [120]
seeds = [5]
splits = [[0.5, 0.3, 0.2]]
cpus = 4
memory_classifiers = 12000
memory_selectors = 20000

tpot_classifiers = ['NB', 'TR', 'KNN', 'MLP', 'LR', 'XGB', 'SVM', 'SGD']
selectors = ['autosklearn']

config = {
    'id': f'{benchmark_name}',
    'path': f'./{benchmark_name}',
    'env': {
        'type': 'slurm',
        'username': 'eb130475'
    },
    'classifiers': [
        {
            'type': 'TPOT',
            'name': f'T-{clf}_{task}_{time}_{seed}',
            'classifier': clf,
            'seed': seed,
            'split': split,
            'time': time,
            'task': task,
            'cpus': cpus,
            'memory': memory_classifiers,
            'model_params': {}
        }
        for seed, split, time, task, clf
        in product(seeds, splits, times_in_mins, tasks[0:20], tpot_classifiers)
    ],
    'selectors': [
        {
            'type': 'autosklearn',
            'name': f'ASK-{task}_{time}_{seed}',
            'seed': seed,
            'split': split,
            'time': time,
            'task': task,
            'cpus': cpus,
            'memory': memory_selectors,
            'model_params': {},
            'classifiers': [
                f'T-{clf}_{task}_{time}_{seed}'
                for clf in tpot_classifiers
            ]
        }
        for seed, split, time, task
        in product(seeds, splits, times_in_mins, tasks[0:20])
    ]
}

with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)

