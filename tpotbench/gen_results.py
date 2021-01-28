import os
import sys
import json

from tpotbench import Benchmark

def run(config_path):
    bench = Benchmark(config_path)

    results = {
        f'task_{task}' : {
            'classifiers': {
                name: job.model().score(X_test, y_test)
                if job.complete() else 'INCOMPLETE'
                for name, job in jobs['classifiers'].items()

            },
            'selectors': {
                name: job.model().score(X_test, y_test)
                if job.complete() else 'INCOMPLETE'
                for name, job in jobs['selectors'].items()
            },
            'baselines': {
                name: job.model().score(X_test, y_test)
                if job.complete() else 'INCOMPLETE'
                for name, job in jobs['baselines'].items()
            }
        }
        for task, jobs, X_test, y_test in bench.jobs_and_data_by_task()
    }

    with open(bench.results_path, 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise RuntimeError('Please provide a config\n'
                           + f'{sys.argv[0]} /path/to/config')
    run(sys.argv[1])
