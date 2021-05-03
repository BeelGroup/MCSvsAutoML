import os
import json
from itertools import product

benchmark_name = 'local_test'
current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, f'{benchmark_name}.json')

tasks = [3, 6, 11]
times_in_mins = [1]
seed = 5
split = [0.5, 0.3, 0.2]
cpus = 4
memory_classifiers = 12000
memory_selectors = 20000
tpot_classifiers = ['NB', 'TR', 'LR']
selectors = ['autosklearn', 'metades']
baselines = ['autosklearn', 'tpot']

config = {
    'id': f'{benchmark_name}',
    'path': f'./{benchmark_name}',
    'env': {'type': 'local', },
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
                'algorithm_family': clf
            }
        }
        for time, clf, task
        in product(times_in_mins, tpot_classifiers, tasks)
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
                for clf in tpot_classifiers
            ]
        }
        for time, task
        in product(times_in_mins, tasks)
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
                for clf in tpot_classifiers
            ]
        }
        for time, task
        in product(times_in_mins, tasks)
    ] + [
        {
            'algo_type': 'desrrc',
            'name': f'DESRRC-{task}_{time}_{seed}',
            'time': time,
            'task': task,
            'cpus': cpus,
            'memory': memory_selectors,
            'model_config': {},
            'classifiers': [
                f'T-{clf}_{task}_{time}_{seed}'
                for clf in tpot_classifiers
            ]
        }
        for time, task
        in product(times_in_mins, tasks)
    ] + [
        {
            'algo_type': 'desknorau',
            'name': f'KNORAU-{task}_{time}_{seed}',
            'time': time,
            'task': task,
            'cpus': cpus,
            'memory': memory_selectors,
            'model_config': {},
            'classifiers': [
                f'T-{clf}_{task}_{time}_{seed}'
                for clf in tpot_classifiers
            ]
        }
        for time, task
        in product(times_in_mins, tasks)
    ] + [
        {
            'algo_type': 'lca',
            'name': f'LCA-{task}_{time}_{seed}',
            'time': time,
            'task': task,
            'cpus': cpus,
            'memory': memory_selectors,
            'model_config': {},
            'classifiers': [
                f'T-{clf}_{task}_{time}_{seed}'
                for clf in tpot_classifiers
            ]
        }
        for time, task
        in product(times_in_mins, tasks)
    ] + [
        {
            'algo_type': 'mla',
            'name': f'MLA-{task}_{time}_{seed}',
            'time': time,
            'task': task,
            'cpus': cpus,
            'memory': memory_selectors,
            'model_config': {},
            'classifiers': [
                f'T-{clf}_{task}_{time}_{seed}'
                for clf in tpot_classifiers
            ]
        }
        for time, task
        in product(times_in_mins, tasks)
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
        in product(times_in_mins, tasks)
    ] + [
        {
            'algo_type': 'tpot',
            'name': f'bTPOT-{task}_{time}_{seed}',
            'time': time,  # time should be for 8 single classifiers and selector
            'task': task,
            'cpus': cpus,
            'memory': memory_selectors,
            'model_config': {},
        }
        for time, task
        in product(times_in_mins, tasks)
    ]
}

"""
+ [
    {
        'algo_type': 'autokeras',
        'name': f'bAUK-{task}_{time}_{seed}',
        'time': time,  # time should be for 8 single classifiers and selector
        'task': task,
        'cpus': cpus,
        'memory': memory_selectors,
        'model_config': {},
    }
    for time, task
    in product(times_in_mins, tasks)
]
"""
with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)
