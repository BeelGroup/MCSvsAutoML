import os
import json

from tpotbench import Benchmark

current_dir = os.path.abspath(os.path.dirname(__file__))
benchmark_config = os.path.join(current_dir, 'configs', 'local_test.json')

bench = Benchmark(benchmark_config)

results = {
    f'task_{task}' : {
        'classifiers': {
            name: classifier.score(X_test, y_test)
            for name, classifier in models['classifiers'].items()
        },
        'selectors': {
            name: selector.score(X_test, y_test)
            for name, selector in models['selectors'].items()
        },
        'baselines': {
            name: baseline.score(X_test, y_test)
            for name, baseline in models['baselines'].items()
        }
    }
    for task, models, X_test, y_test in bench.jobs_and_data_by_task()
}

with open(bench.results_path, 'w') as f:
    json.dump(results, f, indent=2)
