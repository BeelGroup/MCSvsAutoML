"""
Manages job objects for the benchmark given a config to work from
"""
from typing import Dict, Iterable, Optional

import os
import json

from slurmjobmanager import LocalEnvironment, SlurmEnvironment

from .slurm import slurm_job_options
from .selectors import selector_job_map
from .classifiers import classifier_job_map
from .benchmarkjob import BenchmarkJob

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
        self.id = cfg['id']

        # Setup environment
        self.env = None
        if cfg['env']['type'] == 'slurm':
            username = cfg['env']
            self.env = SlurmEnvironment(username=username)
        else:
            self.env = LocalEnvironment()
            print('No environment specified, running locally.\n'
                  + 'This may take a substantial amount of time')

        # Test if directory can be made
        self.benchmark_path = os.path.abspath(cfg['path'])
        if not os.path.exists(self.benchmark_path):
            os.mkdir(self.benchmark_path)

        # Ensure names of jobs are unique
        names = set()
        models = cfg.get('classifiers', []) + cfg.get('selectors', [])
        for model in models:
            if model['name'] in names:
                raise RuntimeError(f'Duplicate name, {model["name"]}')

            names.add(model['name'])

        # Create classifier jobs
        self.classifier_jobs = {}
        for model_config in cfg['classifiers']:

            clf_type = model_config['type']
            name = model_config['name']

            basedir = os.path.join(self.benchmark_path, name)
            job_class, runner = classifier_job_map[clf_type]

            job = job_class.from_config(model_config, basedir)
            self.classifier_jobs[name] = job

        # Create selector jobs
        self.selector_jobs = {}
        for model_config in cfg['selectors']:

            selector_type = model_config['type']
            name = model_config['name']

            basedir = os.path.join(self.benchmark_path, name)
            job_class, runner = selector_job_map[selector_type]

            # Replace the named classifiers with their job
            model_config['classifier_jobs'] = [
                self.classifier_jobs[clf_name]
                for clf_name in model_config['classifiers']
            ]

            job = job_class.from_config(model_config, basedir)
            self.selector_jobs[name] = job

    def job_failed(self, job: BenchmarkJob) -> bool:
        if job.complete():
            return False

        if isinstance(self.env, SlurmEnvironment):
            # Check if it's currently running
            in_progress = self.env.pending_jobs() + self.env.running_jobs()
            if job in in_progress:
                return False

            # Check if slurm script has been created, incomplete job
            slurm_opts = slurm_job_options(job)
            script_path = slurm_opts['script_script_path']
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

    def failed_jobs(self) -> Dict[str, BenchmarkJob]:
        return {
            name: job
            for name, job
            in self.jobs().items()
            if self.job_failed(job)
        }

    def jobs(self) -> Dict[str, BenchmarkJob]:
        return {**self.classifier_jobs, **self.selector_jobs}

    def run(
        self,
        jobs: Optional[Dict[str, BenchmarkJob]] = None
    ) -> None:
        # If no jobs specified, collect all the available ones
        if jobs is None:
            jobs = self.jobs()

        # Filter out any complete or failed jobs
        jobs = {
            name: job for name, job in jobs.items()
            if not job.complete() or self.job_failed(job)
        }

        # Run all jobs that are not blocked
        blocked_jobs = {}
        for name, job in jobs.items():
            if not job.blocked():
                if isinstance(self.env, SlurmEnvironment):
                    self.env.run(job, slurm_job_options(job))
                else:
                    print(f'running {name}')
                    self.env.run(job, {})
            else:
                blocked_jobs[name] = job

        # Get count of any jobs that are now ready
        n_ready = len(list(filter(lambda job: job.ready(), jobs.values())))
        if n_ready > 0:
            self.run(blocked_jobs)
        else:
            print('Finished')
