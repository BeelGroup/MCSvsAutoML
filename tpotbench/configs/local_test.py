import os
import json
from itertools import product

benchmark_name = 'local_test'
current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, f'{benchmark_name}.json')

tasks = [3, 6, 11]
times_in_mins = [5]
seeds = [5]
splits = [[0.5, 0.3, 0.2]]
cpus = 4
memory_classifiers = 12000
memory_selectors = 20000
tpot_classifiers = ['NB', 'TR', 'LR']
selectors = ['autosklearn', 'metades']

config = {
    'id': f'{benchmark_name}',
    'path': f'./{benchmark_name}',
    'env': { 'type': 'local', },
    'classifiers': [
        {
            'type': 'TPOT',
            'name': f'T-{clf}_{task}_{time}_{seed}',
            'seed': seed,
            'split': split,
            'time': time,
            'task': task,
            'cpus': cpus,
            'memory': memory_classifiers,
            'model_params': {
                'algorithm_family': clf
            }
        }
        for seed, split, time, task, clf
        in product(seeds, splits, times_in_mins, tasks[0:1], tpot_classifiers)
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
        in product(seeds, splits, times_in_mins, tasks[0:1])
    ] + [
        {
            'type': 'metades',
            'name': f'MDES-{task}_{time}_{seed}',
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
        in product(seeds, splits, times_in_mins, tasks[0:1])
    ],
    'baselines' : [
        {
            'type': 'autosklearn',
            'name': f'bASK-{task}_{time}_{seed}',
            'seed': seed,
            'split': [split[0] + split[1], split[2]],
            'time': time, # time should be for 8 single classifiers and selector
            'task': task,
            'cpus': cpus,
            'memory': memory_selectors,
            'model_params': {},
        }
        for seed, split, time, task
        in product(seeds, splits, times_in_mins, tasks[0:1])
    ]
}

with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)

