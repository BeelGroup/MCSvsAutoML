from typing import Tuple, Optional, Dict, Any, Iterable

import os
import json
from abc import abstractmethod
from os.path import join
from shutil import rmtree

from ..benchmarkjob import BenchmarkJob


class BaselineJob(BenchmarkJob):

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
                'metrics': join(basedir, 'metrics.json'),
            },
            'folders': {}
        }

    @classmethod
    @abstractmethod
    def default_params(cls) -> Dict[str, Any]:
        pass

    @classmethod
    @abstractmethod
    def baseline_type(cls) -> str:
        pass

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
            'split': self.split,
            'task': self.task,
            'cpus': self.cpus,
            'memory': self.memory,
            'model_params': model_params,
            'files': paths['files'],
            'folders': paths['folders']
        }

    def command(self) -> str:
        config_path = self._paths['files']['config']
        return f'python {self.runner_path()} {config_path}'

    @classmethod
    def from_config(
        cls,
        cfg: Dict[str, Any],
        basedir: str,
    ) -> BenchmarkJob:
        if cfg['type'] != cls.baseline_type():
            raise ValueError(f'Config object not a {cls.baseline_type} '
                             + f'baseline,\n{cfg=}')

        # Remove it as it's not a constructor params
        del cfg['type']

        default_params = cls.default_params()
        baseline_params = {**default_params, **cfg, 'basedir': basedir}
        return cls(**baseline_params)
