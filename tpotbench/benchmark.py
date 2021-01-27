"""
Manages job objects for the benchmark given a config to work from
"""
from typing import Dict, Iterable, Optional, List, Tuple, Any

import os
import json

import numpy as np
from slurmjobmanager import LocalEnvironment, SlurmEnvironment

from .slurm import slurm_job_options
from .jobs import job_types, BenchmarkJob
from .runners.util import get_task_split

# TODO
# Finished writing up the autosklearn selector and classifiers, should
# fix it all up and start running it on HORUS.
# After that, I should to begin to get the other selectors up and running
# and then start keeping track of metrics


class Benchmark:

    def __init__(self, config_path):

        # load config
        config_path = os.path.abspath(config_path)
        cfg = {}
        with open(config_path, 'r') as f:
            cfg = json.load(f)

        self.cfg = cfg
        self.seed = cfg['seed']
        self.split = cfg['split']
        self.id = cfg['id']
        self.tasks = cfg['tasks']
        self._jobs = {
            'classifier': {},
            'baseline': {},
            'selector': {}
        }

        self.benchmark_path = os.path.abspath(cfg['path'])
        self.results_path = os.path.join(self.benchmark_path, 'results.json')

        # Setup environment
        self.env = None
        if cfg['env']['type'] == 'slurm':
            username = cfg['env']['username']
            self.env = SlurmEnvironment(username=username)
        else:
            self.env = LocalEnvironment()
            print('No environment specified, running locally.\n'
                  + 'This may take a substantial amount of time')

        # Test if directories can be made
        if not os.path.exists(self.benchmark_path):
            os.mkdir(self.benchmark_path)

        for model_type in ['classifier', 'selector', 'baseline']:

            for model_cfg in cfg[model_type]:
                algo_type = model_cfg['algo_type']
                name = model_cfg['name']

                # Add seed and split to config
                model_cfg['seed'] = self.seed
                model_cfg['split'] = self.split

                basedir = os.path.join(self.benchmark_path, name)

                # If it's a selector, add the classifiers it works from
                if model_type == 'selector':
                    model_cfg['classifiers'] = [
                        self._jobs['classifier'][clf_name]
                        for clf_name
                        in model_cfg['classifiers']
                    ]

                job_class = job_types[model_type][algo_type]
                job = job_class.from_config(model_cfg, basedir)

                self._jobs[model_type][name] = job

    def job_failed(self, job: BenchmarkJob) -> bool:
        if job.complete():
            return False

        if isinstance(self.env, SlurmEnvironment):
            # Check if it's currently running
            in_progress = self.env.pending_jobs() + self.env.running_jobs()
            if job.name() in in_progress:
                return False

            # Check if slurm script has been created, incomplete job
            slurm_opts = slurm_job_options(job)
            script_path = slurm_opts['slurm_script_path']
            if os.path.exists(script_path):
                return True

        # Local env, assumed if it has been recorded as running and hasn't
        # completed then it's failed, very much not foolproof
        else:
            info = self.env.info()
            if job.name() in info['jobs_run']:
                return True

        # If there is no marker of having started then we assume it has not
        # failed
        return False

    def jobs(self) -> List[BenchmarkJob]:
        return list(self._jobs['classifier'].values()) + \
            list(self._jobs['selector'].values()) + \
            list(self._jobs['baseline'].values())

    def jobs_and_data_by_task(
        self
    ) -> Iterable[Tuple[int, Dict[str, Any], np.ndarray, np.ndarray]]:
        for task in self.tasks:
            models = {
                'classifiers': {
                    name: clf.model()
                    for name, clf in self._jobs['classifier'].items()
                    if clf.task == task
                },
                'selectors': {
                    name: sel.model() 
                    for name, sel in self._jobs['selector'].items()
                    if sel.task == task
                },
                'baselines': {
                    name: baseline.model() 
                    for name, baseline in self._jobs['baseline'].items()
                    if baseline.task == task
                }
            }
            data = get_task_split(task, self.seed, self.split)
            X_test, y_test = data['test']
            yield task, models, X_test, y_test

    def status(
        self,
        jobs: Optional[Iterable[BenchmarkJob]] = None
    ) -> Dict[str, List[BenchmarkJob]]:
        if jobs is None:
            jobs = self.jobs()

        # TODO: Could be made quicker by only iterating through jobs once
        results = {
            'complete': [job for job in jobs if job.complete()],
            'failed': [job for job in jobs if self.job_failed(job)],
            'blocked': [job for job in jobs if job.blocked()],
        }
        if isinstance(self.env, SlurmEnvironment):
            info = self.env.info()
            results['pending'] = [
                job for job in jobs if job.name() in info['pending']
            ]
            results['running'] = [
                job for job in jobs if job.name() in info['running']
            ]
            results['ready'] = [
                job for job in jobs
                if job.ready() and not job.name() in info['pending'] + info['running']
            ]
        else:
            results['ready'] = [job for job in jobs if job.ready()]

        return results

    def run(
        self,
        jobs: Optional[Iterable[BenchmarkJob]] = None
    ) -> None:
        # If no jobs specified, collect all the available ones
        if jobs is None:
            jobs = self.jobs()

        # Filter out any complete or failed jobs
        jobs = [
            job for job in jobs if not job.complete() or self.job_failed(job)
        ]

        # Run all jobs that are not blocked
        blocked_jobs = []
        for job in jobs:
            if not job.blocked():
                if isinstance(self.env, SlurmEnvironment):
                    self.env.refresh_info()
                    info = self.env.info()
                    in_progress = info['pending'] + info['running']
                    if not job.name() in in_progress:
                        self.env.run(job, slurm_job_options(job))
                else:
                    print(f'running {job.name()}')
                    self.env.run(job, {})
            else:
                blocked_jobs.append(job)

        # Get count of any jobs that are now ready
        ready_jobs = [job for job in jobs if job.ready()]
        if len(ready_jobs) > 0:
            self.run(blocked_jobs)
        else:
            print('Finished')
