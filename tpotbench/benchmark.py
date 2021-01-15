"""
Manages job objects for the benchmark given a config to work from
"""
# https://stackoverflow.com/questions/33533148/how-do-i-type-hint-a-method-with-the-type-of-the-enclosing-class
from __future__ import annotations

import os
import json
from random import Random
from itertools import chain, product
from typing import (
    Dict, Optional, Iterable, List, Literal, Callable, Any, Mapping, Set
)

from slurmjobmanager import SlurmEnvironment
from slurmjobmanager.job import Job

from .jobs import SingleAlgorithmJob, BaselineJob, SelectorJob, TPOTJob

cdir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
runner_dir = os.path.join(cdir, 'runners')
default_config_template = os.path.join(cdir, 'template_config.json')

class Benchmark:
    """ #TODO """

    filter_types = Literal['ready', 'failed', 'blocked', 'pending', 'running',
                           'complete', 'in_progress']
    job_filters : Dict[filter_types, Callable[[Job], bool]] = {
        'ready': lambda job: job.ready(),
        'failed': lambda job: job.failed(),
        'blocked': lambda job: job.blocked(),
        'pending': lambda job: job.pending(),
        'running': lambda job: job.running(),
        'complete': lambda job: job.complete(),
        'in_progress': lambda job: job.in_progress(),
    }

    runners = {
        'single_algorithm_runner' : os.path.join(runner_dir,
                                                 'single_algorithm_runner.py'),
        'baseline_runner' : os.path.join(runner_dir, 'baseline_runner.py'),
        'selector_runner' : os.path.join(runner_dir, 'autosklearn_selector.py'),
    }

    def __init__(
        self,
        config_path: Optional[str] = None,
        username: Optional[str] = None,
        saveto: Optional[str] = None,
    ) -> None:
        """
        Params
        ======
        config_path : str
            path to the config to load
            Warning, does not do a config check, may result in delayed errors
        """
        assert config_path or (username and saveto), \
            'must specify either a config or at least username and saveto'

        config : Dict[str, Any] = {}

        # User specified config
        if config_path:
            template : Dict[str, Any] = {}
            user_config : Dict[str, Any] = {}

            with open(default_config_template) as default_template:
                template = json.load(default_template)

            with open(config_path, 'r') as config_file:
                user_config = json.load(config_file)

            config = {**template, **user_config}

        # Default config template
        else:
            config_path = default_config_template
            with open(config_path, 'r') as config_file:
                config = json.load(config_file)
                config['username'] = username
                config['dir'] = saveto



        self.env = SlurmEnvironment(config['username'])
        self.config = config

        root_folder = os.path.abspath(os.path.join(config['dir'], config['id']))

        self.folders : Dict[str, str] = {
            'root': root_folder,
            'results': os.path.join(root_folder, 'results'),
            'graphs': os.path.join(root_folder, 'graphs'),
            'plots': os.path.join(root_folder, 'plots')
        }

    def create(self) -> None:
        """ Create the folders required for the benchmark """
        # TODO not sure if the user should have to make the initial dir or not
        if not os.path.exists(self.config['dir']):
            os.mkdir(self.config['dir'])

        if not os.path.exists(self.folders['root']):
            os.mkdir(self.folders['root'])

        if not os.path.exists(self.folders['results']):
            os.mkdir(self.folders['results'])

        if not os.path.exists(self.folders['plots']):
            os.mkdir(self.folders['plots'])

        for job in self.jobs():
            job.create()

    def run_remaining(self) -> None:
        """ Run the remaining jobs """
        raise NotImplementedError

    def jobs(
        self,
        filter_by : Optional[Benchmark.filter_types] = None
    ) -> Iterable[TPOTJob]:
        """
        Params
        =======
        filter_by : Optional[Literal['complete', 'failed', 'blocked', 'ready',
                                     'pending', 'running', 'in_progress']]
            A filter to apply to the list before returning it
        Returns
        =======
        List[TPOTJob]
            returns a list of jobs based on the config used to create the
            benchmark
        """
        job_iter = chain(
            self.algorithm_jobs(), self.selector_jobs(), self.baseline_jobs()
        )

        # Use the filter if correct
        if filter_by:
            job_filter = Benchmark.job_filters[filter_by]
            return list(filter(job_filter, job_iter))

        return list(job_iter)

    def algorithm_jobs(self) -> List[SingleAlgorithmJob]:
        """
        Returns
        =======
        List[SingleAlgorithmJob]
            returns a list of SingleAlgorithmJobs based on the config
            used to create the benchmark
        """
        config = self.config
        root = self.folders['root']

        seeds = config['seeds']
        times = config['times_in_mins']
        tasks = config['openml_tasks']
        splits = config['data_splits']
        algorithms = config['algorithms']
        runner = Benchmark.runners['single_algorithm_runner']

        return [
            SingleAlgorithmJob(self.env, seed, times, task, root, split, algo,
                               runner)
            for seed, task, split, algo
            in product(seeds, tasks, splits, algorithms)
        ]

    def baseline_jobs(self) -> List[BaselineJob]:
        """
        Returns
        =======
        List[BaselineJob]
            returns a list of BaselineJob based on the config
            used to create the benchmark
        """
        config = self.config
        root = self.folders['root']

        seeds = config['seeds']
        times = config['times_in_mins']
        tasks = config['openml_tasks']
        splits = config['data_splits']
        runner = Benchmark.runners['baseline_runner']

        # Scale up times to enable fair compute resource comparisons
        # and reduce to unique times
        scale = config['baselines']['scale']

        baseline_times : Set[int] = set()
        for i in range(1, scale+1):
            # pylint: disable=cell-var-from-loop
            scaled_times = set(map(lambda x: x * i, times))
            baseline_times = baseline_times.union(scaled_times)

        return [
            BaselineJob(self.env, seed, list(baseline_times), task, root, split,
                        runner)
            for seed, task, split
            in product(seeds, tasks, splits)
        ]

    def selector_jobs(self) -> List[SelectorJob]:
        """
        Returns
        =======
        List[SelectorJob]
            returns a list of SingleAlgorithmJobs based on the config
            used to create the benchmark
        """
        functions = {
            'all' : self.selector_jobs_all,
            'top_n': self.selector_jobs_n_top,
            'n_most_coverage': self.selector_jobs_n_most_coverage,
            'n_random_selection': self.selector_jobs_n_random,
            'n_least_overlapping': self.selector_jobs_n_least_overlapping,
        }

        jobs = []
        for selector_config in self.config['selectors']:
            selector_type = selector_config['type']
            get_selector = functions[selector_type]
            jobs += get_selector(selector_config)

        return jobs

    def selector_jobs_all(
        self,
        selector_config: Mapping[str, Any]
    ) -> List[SelectorJob]:
        """
        Returns selector jobs for type "all" which use
        `all` single algorithms in its seleciton training and prediction

        Params
        ======
        selector_config: Mapping[str, Any]
            Config of selector
            {
                "name": "name id for job",
                "type": "all"
            },

        Returns
        =======
        List[SelectorJob]
            The list of selector jobs that will use all algorithms
        """
        config = self.config
        root = self.folders['root']

        seeds = config['seeds']
        times = config['times_in_mins']
        tasks = config['openml_tasks']
        splits = config['data_splits']
        algorithms = config['algorithms']

        single_algorithm_runner = Benchmark.runners['single_algorithm_runner']
        selector_runner = Benchmark.runners['selector_runner']

        selector_name = selector_config['name']

        jobs = []
        for seed, task, split in product(seeds, tasks, splits):
            # Create the algorithms the selector will use the results of
            algorithm_jobs = [
                SingleAlgorithmJob(self.env, seed, times, task, root, split,
                                   algo, single_algorithm_runner)
                for algo in algorithms
            ]

            selector_job =  SelectorJob(self.env, seed, times, task, root,
                                        split, selector_runner, selector_name,
                                        algorithm_jobs)
            jobs.append(selector_job)

        return jobs

    def selector_jobs_n_top(
        self,
        selector_config: Mapping[str, Any],
    ) -> List[SelectorJob]:
        """ #TODO """
        return []

    def selector_jobs_n_least_overlapping(
        self,
        selector_config: Mapping[str, Any],
    ) -> List[SelectorJob]:
        """ # TODO """
        return []

    def selector_jobs_n_most_coverage(
        self,
        selector_config: Mapping[str, Any],
    ) -> List[SelectorJob]:
        # TODO
        return []

    def selector_jobs_n_random(
        self,
        selector_config: Mapping[str, Any],
    ) -> List[SelectorJob]:
        """
        Returns the list of SelectorJob's that will use `n` random single
        algorithms in its training and prediction.

        Will

        Params
        ======
        selector_config: Mapping[str, Any]
            The config to base off of,
                'n': amount of algorithms to choose randomly
                'selection_seeds': the different seeds to use when randomly
                                    selecting algorithms. Creates one job for
                                    every seed.

            Example:
            {
                "name": "myID",
                "type": "n_random_selection",
                "n": 4,
                "selection_seeds": [1,2,3]
            }

        Returns
        =======
        List[SelectorJob]
            The list of selector jobs that will use `n` random algorithms.
        """
        config = self.config
        root = self.folders['root']

        seeds = config['seeds']
        times = config['times_in_mins']
        tasks = config['openml_tasks']
        splits = config['data_splits']
        algorithms = config['algorithms']

        single_algorithm_runner = Benchmark.runners['single_algorithm_runner']
        selector_runner = Benchmark.runners['selector_runner']

        n_algorithms = selector_config['n']
        selection_seeds = selector_config['selection_seeds']
        selector_names = [
            f'{selector_config["name"]}_{selection_seed}'
            for selection_seed in selection_seeds
        ]

        jobs = []
        for selection_seed, name in zip(selection_seeds, selector_names):
            rand = Random(selection_seed * n_algorithms)
            random_algorithms : List[str] = \
                rand.sample(algorithms, n_algorithms)

            for seed, task, split in product(seeds, tasks, splits):
                # Create the algorithms the selector will use the results of
                algorithm_jobs = [
                    SingleAlgorithmJob(self.env, seed, times, task, root, split,
                                       algo, single_algorithm_runner)
                    for algo in random_algorithms
                ]
                selector_job =  SelectorJob(self.env, seed, times, task, root,
                                            split, selector_runner,
                                            name, algorithm_jobs)
                jobs.append(selector_job)

        return jobs
