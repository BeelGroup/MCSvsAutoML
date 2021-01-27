from typing import Tuple, Optional, Dict, Any, Iterable

import os
import json
from abc import ABC
from os.path import join
from shutil import rmtree

from ..benchmarkjob import BenchmarkJob


class SelectorJob(BenchmarkJob, ABC):

    def __init__(
        self,
        name: str,
        seed: int,
        task: int,
        time: int,
        basedir: str,
        split: Tuple[float, float, float],
        classifiers: Iterable[BenchmarkJob],
        memory: int,
        cpus: int,
        model_params: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(
            name, seed, task, time, basedir, split, memory, cpus
        )
        self.classifiers = classifiers
        self.model_params = model_params
        self._paths: Dict[str, Any] = {
            'basedir': basedir,
            'files': {
                'config': join(basedir, 'config.json'),
                'model': join(basedir, 'model.pkl'),
                'classifiers': {
                    clf.name(): clf.paths()['files']
                    for clf in self.classifiers
                },
                'selector_training_classifier_selections': join(
                    basedir, 'selector_training_classifier_selections.npy'
                ),
                'selector_training_classifier_competences': join(
                    basedir, 'selector_training_competences.npy'
                ),
                'test_classifier_selections': join(
                    basedir, 'test_classifier_selections.npy'
                ),
                'test_classifier_competences': join(
                    basedir, 'test_competences.npy'
                ),
            },
            'folders': {}
        }

    @classmethod
    def job_type(cls) -> str:
        return 'selector'

    def paths(self) -> Dict[str, Any]:
        return self._paths

    def complete(self) -> bool:
        files = self._paths['files']
        classifier_selection_files = [
            files[f'{t}_classifier_selections']
            for t in ['test', 'selector_training']
        ]
        classifier_competence_files = [
            files[f'{t}_classifier_competences']
            for t in ['test', 'selector_training']
        ]
        model = files['model']
        return all(
            os.path.exists(file)
            for file
            in classifier_selection_files + classifier_competence_files + [model]
        )

    def blocked(self) -> bool:
        return any(not clf.complete() for clf in self.classifiers)

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
