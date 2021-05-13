"""
Generates the results of a benchmark.

    Usage: python gen_results.py /path/to/config /path/for/results [--csv]

        Creates the results at the path specified

    Options:
        --csv: to generate a csv view of accuracies and norm scores

    NOTE: This may take a while as it has to load in every trained model
    and evaluate it on the datasets. This could be optimized by simply storing
    the results but requires more management to make sure no stale results get
    used.

"""
from typing import Dict, Any, Tuple, List

import os
import json
import argparse
from itertools import chain
from collections import Counter

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from deslib.static.single_best import SingleBest
from deslib.static.oracle import Oracle
from deslib.static.stacked import StackedClassifier
from sklearn.preprocessing import StandardScaler

from piasbenchmark import Benchmark
from piasbenchmark.custom_json_encoder import CustomEncoder

def normalize(val, high, low):
    if val is None:
        return None
    if high == low: # When oracle has same acc as single_best, prevents x/0
        return 1
    else:
        return (val - low) / (high - low)

def generate_results(bench: Benchmark) -> Dict[str, Any]:
    """ Generates the actual results from a benchmark object

    Args:
        bench: The benchmark for which to generate results from.

    Returns:
        Dict of the results
    """
    task_results = {}
    for task in bench.tasks:
        # Get information for task
        jobs, data = bench.jobs_and_data_by_task(task)
        X_train, y_train = data['selector_train']
        X_test, y_test = data['test']

        # Generate the accuracy and normalized score for model on each task
        # TODO: Doing it verbosely for clarity, could be compressed
        task_results[task] = {
            **{
                f'classifier_{classifier.algo_type()}': {
                    'acc': classifier.model().score(X_test, y_test)
                    if classifier.complete() else None
                }
                for classifier in jobs['classifiers']
            },
            **{
                f'selector_{selector.algo_type()}': {
                    'acc': selector.model().score(X_test, y_test)
                    if selector.complete() else None
                } for selector in jobs['selectors']
            },
            ** {
                f'baseline_{baseline.algo_type()}': {
                    'acc': baseline.model().score(X_test, y_test)
                    if baseline.complete() else None
                } for baseline in jobs['baselines']
            },
            ** {
                'static_baseline_oracle': {},
                'static_baseline_single_best': {}
            }
        }

        classifier_models = [
            clf.model() for clf in jobs['classifiers'] if clf.complete()
        ]
        # Include stacking as a baseline
        # TODO: The stacking classifier baseline should really be moved to its
        #       own baseline class as it has to perform LogisticRegression
        #       but this is relatively fast
        stacking = StackedClassifier(classifier_models, random_state=bench.seed)
        stacking.fit(X_train, y_train)
        stacking_acc = stacking.score(X_test, y_test)

        task_results[task]['baseline_stacking'] = { 'acc': stacking_acc }

        # TODO: Oracles requires fitting despite not being a that
        #       actually require fitting, this is used for verification and
        #       creation of meta information
        oracle = Oracle(classifier_models, random_state=bench.seed)
        oracle.fit(X_train, y_train)
        oracle_acc = oracle.score(X_test, y_test)

        # Manually calculate the single best classifier
        classifier_accuracies = np.asarray([
            task_results[task][algo_name]['acc']
            for algo_name in task_results[task].keys()
            if 'classifier' in algo_name
        ])
        for i in range(len(classifier_accuracies)):
            if classifier_accuracies[i] == None:
                classifier_accuracies[i] = 0

        single_best_acc = np.max(classifier_accuracies)

        task_results[task]['static_baseline_oracle'] = {
                'acc': oracle_acc, 'norm_score': 1 
        }
        task_results[task]['static_baseline_single_best'] = {
                'acc': single_best_acc, 'norm_score': 1 
        }

        # Generate norm scores for the task, minus the two static baselines
        non_static = task_results[task].keys() - ['static_baseline_oracle',
                                                  'static_baseline_single_best']
        for algo in non_static:
            acc = task_results[task][algo]['acc']
            norm_score = normalize(acc, oracle_acc, single_best_acc)
            task_results[task][algo]['norm_score'] = norm_score

    # Get the name of the algorithms used
    algo_names = list(next(iter(task_results.values())).keys())

    # We can't include any tasks when one of the values is None
    failed_tasks = set(
        task for task in bench.tasks
        if any(task_results[task][algo]['acc'] is None for algo in algo_names)
    )
    valid_tasks = set(bench.tasks) - failed_tasks


    # Collect accuracies of models over all valid tasks and calculate metrics
    algorithm_results = {
        algo : {
            'accuracies': np.asarray([
                task_results[task][algo]['acc']
                for task in valid_tasks ]),
            'norm_scores': np.asarray([
                task_results[task][algo]['norm_score']
                for task in valid_tasks ]),
        }
        for algo in algo_names
    }

    return {
        'algo_results': algorithm_results,
        'tasks': task_results,
        'valid_tasks': list(valid_tasks),
        'failed_tasks': list(failed_tasks),
    }

def generate_results_dataframes(results: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """ Creates a pandas dataframe of the accuracys and norm scores

    Uses the results to generate two dataframes of the shape (n,m):
           |  task_1 | ... | task_m |
    --------------------------------|
    algo_1 |                        |
    -------|                        |
      :    |                        |
    -------|                        |
    algo_n |                        |
    ---------------------------------

    Args:
        results: The benchmark results

    Returns:
        (df_accuracy, df_normscore) - a tuple of the two different dataframes
    """
    # Columns
    column_headers = results['valid_tasks']

    algo_results = results['algo_results']
    accuracies = { k: v['accuracies'] for k, v in algo_results.items() }
    normscores = { k: v['norm_scores'] for k, v in algo_results.items() }

    df_accuracies = pd.DataFrame.from_dict(accuracies, orient='index',
                                           columns=column_headers)
    df_normscores = pd.DataFrame.from_dict(normscores, orient='index',
                                           columns=column_headers)

    return df_accuracies, df_normscores

def process_dataframes(df: pd.DataFrame) -> Dict[str, Any]:
    """ Used in create_plots.py and extra_details.py """
    classifier_results = df[df.apply(
        lambda row: 'classifier' in row.name, axis=1
    )]
    selector_results = df[df.apply(
        lambda row: 'selector' in row.name, axis=1
    )]
    baseline_results = df[df.apply(
        lambda row: 'baseline' in row.name and 'static' not in row.name, axis=1
    )]

    classifier_best_names = classifier_results.idxmax()
    classifier_best = classifier_results.max()

    selector_best_names = selector_results.idxmax()
    selector_best = selector_results.max()

    baseline_best_names = baseline_results.idxmax()
    baseline_best = baseline_results.max()

    # shape (n_task, 3), the best result of each category
    best_values = pd.concat(
        [classifier_best, selector_best, baseline_best], axis=1)
    best_names = pd.concat(
        [classifier_best_names, selector_best_names, baseline_best_names],
        axis=1)

    rename_values = {0:'classifiers', 1:'selectors', 2:'baselines'}
    best_values.rename(columns=rename_values, inplace=True)
    best_names.rename(columns=rename_values, inplace=True)

    return {
        'classifier_results': classifier_results,
        'selector_results': selector_results,
        'baseline_results': baseline_results,
        'classifier_best': classifier_best,
        'selector_best': selector_best,
        'baseline_best': baseline_best,
        'classifier_best_names': classifier_best_names,
        'selector_best_names': selector_best_names,
        'baseline_best_names': baseline_best_names,
        'best_values': best_values,
        'best_names': best_names,
    }

def best_by_categories(dfs: Dict[str, pd.DataFrame] , categories: List[str]) -> pd.Series:
    best_values = dfs['best_values'][categories]
    best_names = dfs['best_names'][categories]

    overall_best_values = best_values.idxmax(axis=1)
    overall_best_names = [
        best_names.loc[task, category]
        for task, category
        in zip(overall_best_values.index, overall_best_values.values)
    ]
    return pd.Series(data=overall_best_names, index=overall_best_values.index)

def generate_summary(df_accs: pd.DataFrame, df_nrms: pd.DataFrame) -> Dict[str, Any]:
    summary = {}
    accs_means, accs_std = df_accs.mean(axis=1), df_accs.std(axis=1)
    nrms_means, nrms_std = df_nrms.mean(axis=1), df_nrms.std(axis=1)

    summary['means'] = {
        'acc': { k: v for k, v in accs_means.items() },
        'norm_score': { k: v for k, v in nrms_means.items() }
    }
    summary['std'] = {
        'acc': { k: v for k, v in accs_std.items() },
        'norm_score': { k: v for k, v in nrms_std.items() }
    }

    processed_accs = process_dataframes(df_accs)
    processed_nrms = process_dataframes(df_nrms)

    # Best between all algorithms
    best_overall = best_by_categories(processed_accs, categories=['classifiers',
                                                                  'selectors',
                                                                  'baselines'])
    summary['overall_best'] = { k: v for k,v in Counter(best_overall).items() }

    # Best Classifiers
    best_by_classifiers = best_by_categories(processed_accs,
                                             categories=['classifiers'])
    summary['classifiers_best'] = { k: v for k,v in Counter(best_by_classifiers).items() }

    # Best between baselines and selectors
    best_selectors_baselines = best_by_categories(processed_accs,
                                                  categories=['selectors',
                                                              'baselines'])
    summary['selectors_baselines'] = { k: v for k,v in Counter(best_selectors_baselines).items() }

    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generates results for a benchmark")
    parser.add_argument('-c', '--config', type=str, required=True,
                        help='Path to benchmark config')
    parser.add_argument('-r', '--results', type=str, required=True,
                        help='Path to where results are stored')

    args = parser.parse_args()

    # Outer results var to save from
    results = None
    bench = Benchmark(args.config)

    full_path = os.path.abspath(args.results)
    results_path = os.path.join(full_path, f'{bench.id}_results.json')
    summary_path = os.path.join(full_path, f'{bench.id}_summary.json')
    accuracies_path = os.path.join(full_path, f'{bench.id}_accuracies.csv')
    normscores_path = os.path.join(full_path, f'{bench.id}_normscores.csv')

    if not os.path.exists(full_path):
        raise ValueError(f'Directory {full_path} does not exist')

    if os.path.exists(results_path):
        print('Results already found at:' +
              f'  {results_path}')
        results = json.load(open(results_path, 'r'))
    else:
        results = generate_results(bench)
        json.dump(results, open(results_path, 'w'), indent=2, cls=CustomEncoder)
        print('Results saved at:'+
              f' {results_path}')

    df_accs, df_normscores = generate_results_dataframes(results)
    df_accs.to_csv(accuracies_path)
    df_normscores.to_csv(normscores_path)
    print('csvs generated at:'
          + f'\n  {accuracies_path}'
          + f'\n  {normscores_path}')

    summary = generate_summary(df_accs, df_normscores)
    json.dump(summary, open(summary_path, 'w'))
    print('summary generated at:'
          + f'\n {summary_path}')
