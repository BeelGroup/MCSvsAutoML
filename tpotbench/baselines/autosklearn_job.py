from typing import Dict, Any

import os

from .selector_job import SelectorJob


class AutoSklearnBaselineJob(SelectorJob):

    _runner_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'autosklearn_runner.py')
    _baseline_type = 'autosklearn'

    _default_params = {
        'memory': 20000,
        'cpus': 1,
        'model_params': {}
    }

    @classmethod
    def runner_path(cls) -> str:
        return cls._runner_path

    @classmethod
    def baseline_type(cls) -> str:
        return cls._baseline_type

    @classmethod
    def default_params(cls) -> Dict[str, Any]:
        return cls._default_params

