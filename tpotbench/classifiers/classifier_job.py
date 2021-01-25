from typing import Tuple, Optional, Dict, Any, Iterable

import os
import json
from abc import abstractmethod
from os.path import join
from shutil import rmtree

from ..benchmarkjob import BenchmarkJob

class ClassifierJob(BenchmarkJob):

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
                'log': join(basedir, 'tpot_log.txt'),
                'model': join(basedir, 'model.pkl'),
                'metrics': join(basedir, 'metrics.json'),
                'train_classifications': join(basedir, 'train_classifications.npy'),
                'train_probabilities': join(basedir, 'train_probabilities.npy'),
                'selector_training_classifications': join(
                    basedir, 'selector_training_classifications.npy'
                ),
                'selector_training_probabilities': join(
                    basedir, 'selector_training_probabilities.npy'
                ),
                'test_classifications': join(basedir, 'test_classifications.npy'),
                'test_probabilities': join(basedir, 'test_probabilities.npy'),
            },
            'folders': { }
        }

    def paths(self) -> Dict[str, Any]:
        return self._paths

    def complete(self) -> bool:
        files = self._paths['files']
        classification_files = [
            files[f'{t}_classifications']
            for t in ['train', 'test', 'selector_training']
        ]
        model = files['model']
        return all(os.path.exists(file)
                   for file in classification_files + [model])

    def blocked(self) -> bool:
        return False

    def setup(self) -> None:
        if not os.path.exists(self._paths['basedir']):
            os.mkdir(self._paths['basedir'])

        if not os.path.exists(self._paths['folders']['checkpoints']):
            os.mkdir(self._paths['folders']['checkpoints'])

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
            'files': paths['files'],
            'folders': paths['folders'],
            'model_params': model_params,
        }

    def command(self) -> str:
        config_path = self._paths['files']['config']
        return f'python {self.runner_path()} {config_path}'

    @classmethod
    @abstractmethod
    def default_params(cls) -> str:
        pass

    @classmethod
    @abstractmethod
    def classifier_type(cls) -> str:
        pass

    @classmethod
    def from_config(
        cls,
        cfg: Dict[str, Any],
        basedir: str,
    ) -> BenchmarkJob:
        if cfg['type'] != cls.classifier_type():
            raise ValueError(f'Config object not a {cls.classifier_type()} type,'
                             + f'\n{cfg=}')

        # Remove it as it's not a constructor argument
        del cfg['type']

        default_params = cls.default_params()
        classifier_params = {**default_params, **cfg, 'basedir': basedir}
        return cls(**classifier_params)
