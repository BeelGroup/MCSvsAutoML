from typing import Dict, Any, Type

import os

from .benchmark_job import BenchmarkJob
from ..models.baselines import (
    AutoSklearnBaselineModel,
    #AutoKerasBaselineModel,
    TPOTBaselineModel
)


class BaselineJob(BenchmarkJob):

    # Same __init__

    @classmethod
    def job_type(cls) -> str:
        return 'baseline'

    def blocked(self) -> bool:
        return False

    def config(self) -> Dict[str, Any]:
        return {
            'name': self.name(),
            'algo_type': self.algo_type(),
            'seed': self.seed,
            'split': self.split,
            'task': self.task,
            'model_path': self.model_path,
            'model_params': self.model_params(),
        }


class AutoSklearnBaselineJob(BaselineJob):

    @classmethod
    def model_cls(cls) -> Type[AutoSklearnBaselineModel]:
        return AutoSklearnBaselineModel

    def model_params(self) -> Dict[str, Any]:
        return {
            'time_left_for_this_task': self.time * 60,
            'seed': self.seed,
            'memory_limit': int(self.memory * 0.75),
            'n_jobs': self.cpus,
            **self.model_config
        }

    @classmethod
    def algo_type(cls):
        return 'autosklearn'

"""
class AutoKerasBaselineJob(BaselineJob):

    @classmethod
    def model_cls(cls) -> Type[AutoKerasBaselineModel]:
        return AutoKerasBaselineModel

    def model_params(self) -> Dict[str, Any]:
        # TODO: Unfortuantly there's no params for time, kind of difficult
        #       to work around, have to guess how long it will take
        params = {
            #'max_trials': 100,
            'directory': self.basedir,
            'project_name': self.name(),
            'multi_label': False,
            #'overwrite': True,
            #'seed': self.seed,
            #'max_model_size': int(0.25 * self.memory),
            # Extra params that are passed specfically for `fit`
            #'fit_params': {
            #    'use_multiprocessing': True,
            #    #'workers': self.cpus
            #}
        }
        return params

    @classmethod
    def algo_type(cls):
        return 'autokeras'
"""

class TPOTBaselineJob(BaselineJob):

    _default_params = {
        'generations': 9999,
        'population_size': 100,
        'offspring_size': None,
        'mutation_rate': 0.9,
        'crossover_rate': 0.1,
        'scoring': 'accuracy',
        'cv': 5,
        'subsample': 1.0,
        'max_eval_time_mins': 15,
        'template': None,
        'warm_start': True,
        'memory': 'auto',
        'use_dask': False,
        'config_dict': None, # Uses default
        'early_stop': 15,
        'verbosity': 3,
        'disable_update_check': False,
    }

    @classmethod
    def model_cls(cls) -> Type[TPOTBaselineModel]:
        return TPOTBaselineModel

    def model_params(self) -> Dict[str, Any]:
        params = {
            'n_jobs': self.cpus,
            'max_time_mins': self.time,
            'random_state': self.seed,
            'periodic_checkpoint_folder': os.path.join(self.basedir, 'checkpoints'),
            'log_file': os.path.join(self.basedir, 'tpot.log')
        }
        return {**self._default_params, **params}

    @classmethod
    def algo_type(cls):
        return 'tpot'
