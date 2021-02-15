import sys
import json
from itertools import chain

import numpy as np
from deslib.static.single_best import SingleBest
from deslib.static.oracle import Oracle

from tpotbench import Benchmark


def run(config_path):
    bench = Benchmark(config_path)

    task_results = {}
    for task in bench.tasks:
        # Get information for task
        jobs, data = bench.jobs_and_data_by_task(task)
        X_test, y_test = data['test']

        # Create the static single best and oracle baselines
        # Starting with getting the classifiers that have completed
        classifier_models = [
            clf.model() for clf in jobs['classifiers'] if clf.complete()
        ]

        # TODO: Unsure if random_state is required or an artifact of base class.
        #       Still including it anyway
        # TODO: Technically this should be do on algo_train + selector_train
        #       to be an accurate baseline. However, these algorithms were
        #       only train on algo_train and se we will evaluate the single
        #       best on this smaller subset of data. I don't expect this making
        #       much difference unless one of the algorithms overfits more on a
        #       smaller dataset and would lose accuracy with more data.
        X_train, y_train = data['algo_train']
        single_best = SingleBest(classifier_models, random_state=bench.seed)
        single_best.fit(X_train, y_train)

        # TODO: Oracles requires fitting despite not being a that
        #       actually require fitting, this is used for verification and
        #       creation of meta information
        oracle = Oracle(classifier_models, random_state=bench.seed)
        oracle.fit(X_train, y_train)

        # Generate results, marking with None if the model was not complete
        oracle_acc = oracle.score(X_test, y_test)
        single_best_acc = single_best.score(X_test, y_test)

        # Function for calculating normalized score between single best and ora
        # pylint: disable=cell-var-from-loop
        def normalize(acc):
            return (acc - single_best_acc) / (oracle_acc - single_best_acc)

        # Generate the accuracy and normalized score for model on each task
        task_results[task] = {
            job_type: {
                job.algo_type(): {
                    'acc': (acc := job.model().score(X_test, y_test)
                            if job.complete() else None),
                    'norm_score': normalize(acc) if acc else None
                }
                for job in jobs[job_type]
            }
            for job_type in ['classifiers', 'selectors', 'baselines']
        }

        # Include both the oracle and single best as two baselines
        task_results[task]['baselines']['oracle'] = {
            'acc': oracle_acc, 'norm_score': 1
        }
        task_results[task]['baselines']['single_best'] = {
            'acc': single_best_acc, 'norm_score': 0
        }

    # Finished computing all the results for individual tasks
    # Now to generate a summary of all of them

    # We can't include any tasks when one of the values is None
    # Returns all results of models for a given task as a list
    def task_models(task):
        return chain.from_iterable(list(task_results[task][job_type].values())
                                   for job_type
                                   in ['classifiers', 'selectors', 'baselines'])

    failed_tasks = set(task for task in bench.tasks
                       if any(m['acc'] is None for m in task_models(task)))
    valid_tasks = set(bench.tasks) - failed_tasks

    # Get the name of the algorithms used
    first_task = next(iter(task_results.values()))
    algo_types = {
        'classifiers': list(first_task['classifiers'].keys()),
        'selectors': list(first_task['selectors'].keys()),
        'baselines': list(first_task['baselines'].keys())
    }

    # TODO : Probably easier to read for publication as a for loop
    #       ...but dict comprehension and walrus operator (:=) is so concise...
    #       The dict merging of two dicts ( a = { **{...}, **{...} } ) is made
    #       nicer in python 3.9 as a = {...} | {...} but here we are in 3.8.6
    summary_results = {
        job_type: {
            algo_type: {

                'accuracies': (accuracies := [
                    task_results[task][job_type][algo_type]['acc']
                    for task in valid_tasks]),
                'avg_accuracy': np.average(accuracies),
                'std_accuracy': np.std(accuracies),

                'norm_scores': (norm_scores := [
                    task_results[task][job_type][algo_type]['norm_score']
                    for task in valid_tasks
                ]),
                'avg_norm_score': np.average(norm_scores),
                'std_norm_score': np.std(norm_scores)

            }
            for algo_type in algo_types[job_type]
        }
        for job_type in ['classifiers', 'selectors', 'baselines']
    }

    results = {
        'summary': summary_results,
        'tasks': task_results
    }

    with open(bench.results_path, 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise RuntimeError('Please provide a config\n'
                           + f'{sys.argv[0]} /path/to/config')
    run(sys.argv[1])
