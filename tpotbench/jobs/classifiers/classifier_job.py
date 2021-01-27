from typing import Tuple, Optional, Dict, Any

import os
import json
import pickle
from abc import ABC
from os.path import join
from shutil import rmtree

from ..benchmarkjob import BenchmarkJob
from ...models import Model


class ClassifierJob(BenchmarkJob, ABC):

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
            'folders': {}
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

    def model(self) -> Model:
        classifier = None
        with open(self._paths['files']['model'], 'rb') as f:
            classifier = pickle.load(f)
        return classifier

    def job_type(cls) -> str:
        return 'classifier'
