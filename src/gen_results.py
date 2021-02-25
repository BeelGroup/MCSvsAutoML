import sys
import json
from itertools import chain
from collections import Counter

import numpy as np
from deslib.static.single_best import SingleBest
from deslib.static.oracle import Oracle
from deslib.static.stacked import StackedClassifier

from tpotbench import Benchmark
from tpotbench.custom_json_encoder import CustomEncoder


def run(config_path):
    bench = Benchmark(config_path)

    task_results = {}
    for task in bench.tasks:
        # Get information for task
        jobs, data = bench.jobs_and_data_by_task(task)
        X_train, y_train = data['selector_train']
        X_test, y_test = data['test']

        classifier_models = [
            clf.model() for clf in jobs['classifiers'] if clf.complete()
        ]

        # Create the static SingleBest, Oracle and Stacking baselines

        # TODO: Unsure if random_state is required or an artifact of base class.
        #       Still including it anyway
        single_best = SingleBest(classifier_models, random_state=bench.seed)
        single_best.fit(X_train, y_train)

        # TODO: The stacking classifier baseline should really be moved to its
        #       own baseline class as it has to perform LogisticRegression
        #       but this is relatively fast
        stacking = StackedClassifier(classifier_models, random_state=bench.seed)
        stacking.fit(X_train, y_train)

        # TODO: Oracles requires fitting despite not being a that
        #       actually require fitting, this is used for verification and
        #       creation of meta information
        oracle = Oracle(classifier_models, random_state=bench.seed)
        oracle.fit(X_train, y_train)

        # Generate results, marking with None if the model was not complete
        oracle_acc = oracle.score(X_test, y_test)
        single_best_acc = single_best.score(X_test, y_test)
        stacking_acc = stacking.score(X_test, y_test)

        # Function for calculating normalized score between single best and ora
        # pylint: disable=cell-var-from-loop
        def normalize(acc):
            # TODO: In this case, which happens when a single algorithm could
            #       correctly classify all instances which were correctly
            #       classified by any algorithm, the oracle's accuracy is the
            #       the same as the single best. In this case, we simply give
            #       an algorithm a score of 1.
            if oracle_acc == single_best_acc:
                return 1
            else:
                return (acc - single_best_acc) / (oracle_acc - single_best_acc)

        # Generate the accuracy and normalized score for model on each task
        # TODO: Doing it verbosely for clarity, could be compressed
        task_results[task] = {
            'classifiers': {
                classifier.model_config['algorithm_family']: {
                    'acc': (acc := classifier.model().score(X_test, y_test)
                            if classifier.complete() else None),
                    'norm_score': normalize(acc) if acc else None
                }
                for classifier in jobs['classifiers']
            },
            'selectors': {
                selector.algo_type(): {
                    'acc': (acc := selector.model().score(X_test, y_test)
                            if selector.complete() else None),
                    'norm_score': normalize(acc) if acc else None
                }
                for selector in jobs['selectors']
            },
            'baselines': {
                baseline.algo_type(): {
                    'acc': (acc := baseline.model().score(X_test, y_test)
                            if baseline.complete() else None),
                    'norm_score': normalize(acc) if acc else None
                }
                for baseline in jobs['baselines']
            },
            'static_baselines': {
                'oracle': {'acc': oracle_acc, 'norm_score': 1},
                'single_best': {'acc': single_best_acc, 'norm_score': 0}
            }
        }

        # Include stacking as a baseline
        task_results[task]['baselines']['stacking'] = {
            'acc': stacking_acc, 'norm_score': normalize(stacking_acc)
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
        'classifiers': np.asarray(list(first_task['classifiers'].keys())),
        'selectors': np.asarray(list(first_task['selectors'].keys())),
        'baselines': np.asarray(list(first_task['baselines'].keys())),
        'static_baselines': np.asarray(list(first_task['static_baselines'].keys()))
    }

    # Collect accuracies of models over all valid tasks and calculate metrics

    # TODO : Probably easier to read for publication as a for loop
    summary_results = {
        job_type: {
            algo_type: {
                'accuracies': (accuracies := np.asarray([
                    task_results[task][job_type][algo_type]['acc']
                    for task in valid_tasks])),
                'avg_accuracy': np.average(accuracies),
                'std_accuracy': np.std(accuracies),

                'norm_scores': (norm_scores := np.asarray([
                    task_results[task][job_type][algo_type]['norm_score']
                    for task in valid_tasks
                ])),
                'avg_norm_score': np.average(norm_scores),
                'std_norm_score': np.std(norm_scores)
            }
            for algo_type in algo_types[job_type]
        }
        for job_type in ['classifiers', 'selectors',
                         'baselines', 'static_baselines']
    }

    # Get the best within the categories {classifiers} {baselines} {selectors},
    # {baselines + selectors}
    # TODO: This is currently sensitive to order in which things are done,
    #       not stable... make sure the concatenation of results is in the
    #       same order as the names and that the accuracies are in the same
    #       order as the tasks. This is the default way things line up.
    #       In the future, pandas is likely a better solution
    bests = {
        'classifiers': {'by_task': {}, 'by_algo': {}},
        'baselines': {'by_task': {}, 'by_algo': {}},
        'selectors': {'by_task': {}, 'by_algo': {}},
        'selectors_and_baselines': {'by_task': {}, 'by_algo': {}}
    }

    for job_type in ['classifiers', 'baselines', 'selectors']:
        # Collect all accuracies of one category into one big 2d array
        accuracies = np.asarray([
            summary_results[job_type][algo_name]['accuracies']
            for algo_name in algo_types[job_type]
        ])

        # Get the indices of the best
        idxs_of_best = np.argmax(accuracies, axis=0)
        names_of_best = algo_types[job_type][idxs_of_best]

        # Populate name of best algo for task
        bests[job_type]['by_task'] = dict(zip(valid_tasks, names_of_best))

        # Populate with percentage of how often each algo was best
        bests[job_type]['by_algo'] = {
            name: {
                'count': count,
                'perc': float(count) / len(names_of_best),
            }
            for name, count in Counter(names_of_best).items()
        }

    # We need to treat {baselines + selectors} differently
    accuracies = np.asarray(
        [
            summary_results['selectors'][algo_name]['accuracies']
            for algo_name in algo_types['selectors']
        ] + [
            summary_results['baselines'][algo_name]['accuracies']
            for algo_name in algo_types['baselines']
        ]
    )
    # In the same order as they
    selector_names = [f'selector_{name}' for name in algo_types['selectors']]
    baseline_names = [f'baseline_{name}' for name in algo_types['baselines']]
    combined_names = np.asarray(selector_names + baseline_names)

    idxs_of_best = np.argmax(accuracies, axis=0)
    names_of_best = combined_names[idxs_of_best]

    # Populate which algorithm was best for each task
    bests['selectors_and_baselines']['by_task'] = dict(zip(valid_tasks, names_of_best))

    # Populate the count and percent of how often each algo was best
    bests['selectors_and_baselines']['by_algo'] = {
        name: {
            'count': count,
            'perc': float(count) / len(names_of_best),
        }
        for name, count in Counter(names_of_best).items()
    }

    results = {
        'summary': summary_results,
        'tasks': task_results,
        'bests': bests
    }

    with open(bench.results_path, 'w') as f:
        json.dump(results, f, indent=2, cls=CustomEncoder)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise RuntimeError('Please provide a config\n'
                           + f'{sys.argv[0]} /path/to/config')
    run(sys.argv[1])
