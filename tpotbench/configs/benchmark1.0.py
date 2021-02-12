import os
import json
from itertools import product

benchmark_name = 'benchmark1.0'

current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, f'{benchmark_name}.json')

all_tasks = [3, 6, 11, 12, 14, 15, 16, 18, 22, 23, 28, 29, 31, 32, 37, 43, 45, 49,
         53, 219, 2074, 2079, 3021, 3022, 3481, 3549, 3560, 3573, 3902, 3903,
         3904, 3913, 3917, 3918, 7592, 9910, 9946, 9952, 9957, 9960, 9964, 9971,
         9976, 9977, 9978, 9981, 9985, 10093, 10101, 14952, 14954, 14965, 14969,
         14970, 125920, 125922, 146195, 146800, 146817, 146819, 146820, 146821,
         146822, 146824, 146825, 167119, 167120, 167121, 167124, 167125, 167140, 
         167141]
tasks = all_tasks[0:20]
times_in_mins = [120]
seed = 5
split = [0.5, 0.3, 0.2]
cpus = 4
memory_classifiers = 12000
memory_selectors = 24000
tpot_classifiers = ['NB', 'TR', 'KNN', 'MLP', 'LR', 'XGB', 'SVM', 'SGD']

# https://github.com/EpistasisLab/tpot/issues/1170
tpot_classifiers_modified = ['NB', 'TR', 'KNN', 'MLP', 'LR', 'XGB']

selectors = ['autosklearn', 'metades']

config = {
    'id': f'{benchmark_name}',
    'path': f'./{benchmark_name}',
    'env': {
        'type': 'slurm',
        'username': 'eb130475'
    },
    'split': split,
    'seed': seed,
    'tasks': tasks,
    'classifier': [
        {
            'algo_type': 'tpot',
            'name': f'T-{clf}_{task}_{time}_{seed}',
            'time': time,
            'task': task,
            'cpus': cpus,
            'memory': memory_classifiers,
            'model_config': {
                'algorithm_family': clf,
            }
        }
        for time, task, clf
        in product(times_in_mins, all_tasks[0:40], tpot_classifiers_modified)
    ],
    'selector': [
        {
            'algo_type': 'autosklearn',
            'name': f'ASK-{task}_{time}_{seed}',
            'time': time,
            'task': task,
            'cpus': cpus,
            'memory': memory_selectors,
            'model_config': {},
            'classifiers': [
                f'T-{clf}_{task}_{time}_{seed}'
                for clf in tpot_classifiers_modified
            ]
        }
        for time, task
        in product(times_in_mins, tasks[0:20])
    ] + [
        {
            'algo_type': 'metades',
            'name': f'MDES-{task}_{time}_{seed}',
            'time': time,
            'task': task,
            'cpus': cpus,
            'memory': memory_selectors,
            'model_config': {},
            'classifiers': [
                f'T-{clf}_{task}_{time}_{seed}'
                for clf in tpot_classifiers_modified
            ]
        }
        for time, task
        in product(times_in_mins, tasks[0:20])
    ],
    'baseline': [
        {
            'algo_type': 'autosklearn',
            'name': f'bASK-{task}_{time}_{seed}',
            'time': time,  # time should be for 8 single classifiers and selector
            'task': task,
            'cpus': cpus,
            'memory': memory_selectors,
            'model_config': {},
        }
        for time, task
        in product(times_in_mins, tasks[0:20])
    ]
}

with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)

