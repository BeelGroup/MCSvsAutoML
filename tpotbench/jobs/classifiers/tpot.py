from typing import Dict, Any, Tuple, Optional
import os
import json

from .classifier_job import ClassifierJob


class TPOTClassifierJob(ClassifierJob):

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
    def algo_type(cls) -> str:
        return 'tpot'

    # overrides
    def setup(self) -> None:
        if not os.path.exists(self._paths['basedir']):
            os.mkdir(self._paths['basedir'])

        if not os.path.exists(self._paths['folders']['checkpoints']):
            os.mkdir(self._paths['folders']['checkpoints'])

        job_config = self.config()
        with open(self._paths['files']['config'], 'w') as f:
            json.dump(job_config, f, indent=2)
