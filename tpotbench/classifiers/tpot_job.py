from typing import Dict, Any, Tuple, Optional
import os

from .classifier_job import ClassifierJob


class TPOTClassifierJob(ClassifierJob):

    _runner_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'tpot_runner.py')

    _classifier_type = 'TPOT'
    _default_params = {
        'memory': 12000,
        'cpus': 1,
        'model_params': {}
    }

    def __init__(
        self,
        name: str,
        seed: int,
        task: int,
        time: int,
        basedir: str,
        split: Tuple[float, float, float],
        memory: int,
        cpus: int,
        model_params: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(name, seed, task, time, basedir, split,
                         memory, cpus, model_params)
        self._paths['folders']['checkpoints'] = os.path.join(basedir,
                                                             'checkpoints')
        self._paths['files']['export'] = os.path.join(basedir, 'export.py')

    @classmethod
    def runner_path(cls) -> str:
        return cls._runner_path

    @classmethod
    def classifier_type(cls) -> str:
        return cls._classifier_type

    @classmethod
    def default_params(cls) -> Dict[str, Any]:
        return cls._default_params
