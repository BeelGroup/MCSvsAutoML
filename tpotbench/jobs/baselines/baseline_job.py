from typing import Tuple, Optional, Dict, Any

import os
import json
import pickle
from abc import ABC
from os.path import join
from shutil import rmtree

from ..benchmarkjob import BenchmarkJob
from ...models import Model


class BaselineJob(BenchmarkJob, ABC):

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
        super().__init__(
            name, seed, task, time, basedir, split, memory, cpus
        )
        self.model_params = model_params
        self._paths: Dict[str, Any] = {
            'basedir': basedir,
            'files': {
                'config': join(basedir, 'config.json'),
                'model': join(basedir, 'model.pkl'),
                'training_classifications': join(
                    basedir, 'training_classifications.npy'
                ),
                'training_probabilities': join(
                    basedir, 'training_probabilities.npy'
                ),
                'test_classifications': join(
                    basedir, 'test_classifications.npy'
                ),
                'test_probabilities': join(
                    basedir, 'test_probabilities.npy'
                ),
            },
            'folders': {}
        }

    @classmethod
    def job_type(cls) -> str:
        return 'baseline'

    def paths(self) -> Dict[str, Any]:
        return self._paths

    def complete(self) -> bool:
        files = self._paths['files']
        classifications = [
            files[f'{t}_classifications']
            for t in ['training', 'test']
        ]
        probabilities = [
            files[f'{t}_probabilities']
            for t in ['training', 'test']
        ]
        model = files['model']
        return all(
            os.path.exists(file)
            for file
            in classifications + probabilities + [model]
        )

    def blocked(self) -> bool:
        return False

    def setup(self) -> None:
        if not os.path.exists(self._paths['basedir']):
            os.mkdir(self._paths['basedir'])

        if not os.path.exists(self._paths['files']['config']):
            job_config = self.config()
            with open(self._paths['files']['config'], 'w') as f:
                json.dump(job_config, f, indent=2)

    def reset(self) -> None:
        rmtree(self._paths['basedir'])

    def config(self) -> Dict[str, Any]:
        paths = self._paths
        model_params = self.model_params if self.model_params else {}
        return {
            'seed': self.seed,
            'time': self.time,
            # Use the training and selector split for training the baseline
            'split': (self.split[0] + self.split[1], self.split[2]),
            'task': self.task,
            'cpus': self.cpus,
            'memory': self.memory,
            'model_params': model_params,
            'files': paths['files'],
            'folders': paths['folders']
        }

    def model(self) -> Model:
        baseline = None
        with open(self._paths['files']['model'], 'rb') as f:
            baseline = pickle.load(f)
        return baseline
