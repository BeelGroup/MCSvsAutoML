from typing import Tuple, Optional, Dict, Any, Iterable

import os
from os.path import join
from shutil import rmtree

from ..benchmarkjob import BenchmarkJob


class AutoSklearnSelectorJob(BenchmarkJob):

    _runner_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'autosklearn_runner.py'
    )

    defaults = {
        'memory': 20000,
        'cpus': 1
    }

    def __init__(
        self,
        jobname: str,
        seed: int,
        task_id: int,
        time: int,
        basedir: str,
        splits: Tuple[float, float, float],
        classifier_jobs: Iterable[BenchmarkJob],
        memory: int,
        cpus: int,
        model_params: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(
            jobname, seed, task_id, time, basedir, splits, memory, cpus
        )
        self.classifier_jobs = classifier_jobs
        self.model_params = model_params
        self._paths: Dict[str, Any] = {
            'basedir': basedir,
            'files': {
                'config': join(basedir, 'config.json'),
                'model': join(basedir, 'model.pkl'),
                'classifiers': {
                    clf.name(): clf.paths()['files']
                    for clf in self.classifier_jobs
                },
                'selector_training_classifications': join(
                    basedir, 'selector_training_classifications'
                ),
                'selector_training_probabilities': join(
                    basedir, 'selector_training_probabilities'
                ),
                'test_classifications': join(
                    basedir, 'test_classifications'
                ),
                'test_probabilities': join(
                    basedir, 'test_probabilities'
                ),
            },
            'metrics': join(basedir, 'metrics.json'),
            'folders': {}
        }

    def paths(self) -> Dict[str, Any]:
        return self._paths

    def complete(self) -> bool:
        files = self._paths['files']
        classification_files = [
            files[f'{t}_classifications']
            for t in ['test', 'selector_training']
        ]
        probability_files = [
            files[f'{t}_probabilities']
            for t in ['test', 'selector_training']
        ]
        model = files['model']
        return all(
            os.path.exists(file)
            for file in classification_files + probability_files + [model]
        )

    def blocked(self) -> bool:
        return any(not clf.complete() for clf in self.classifier_jobs)

    def setup(self) -> None:
        if not os.path.exists(self._paths['basedir']):
            os.mkdir(self._paths['folders']['basedir'])

    def reset(self) -> None:
        rmtree(self._paths['basedir'])

    def config(self) -> Dict[str, Any]:
        paths = self._paths
        model_params = self.model_params if self.model_params else {}
        return {
            'seed': self.seed,
            'time': self.time,
            'splits': self.splits,
            'task_id': self.task_id,
            'cpus': self.cpus,
            'model_params': model_params,
            'files': paths['files'],
            'folders': paths['folders']
        }

    def command(self) -> str:
        config_path = self._paths['files']['config']
        return f'python {self.runner_path()} {config_path}'

    @classmethod
    def runner_path(cls) -> str:
        return cls._runner_path

    @classmethod
    def from_config(
        cls,
        cfg: Dict[str, Any],
        basedir: str,
    ) -> BenchmarkJob:
        if cfg['type'] != 'TPOT':
            raise ValueError(f'Config object not a TPOT classifier,\n{cfg=}')

        return cls(jobname=cfg['name'],
                   seed=cfg['seed'],
                   task_id=cfg['task'],
                   time=cfg['time'],
                   basedir=basedir,
                   splits=cfg['splits'],
                   classifier_jobs=cfg['classifier_jobs'],
                   memory=cfg.get('memory', cls.defaults['memory']),
                   cpus=cfg.get('cpus', cls.defaults['cpus']),
                   model_params=cfg.get('model_params', None),
                   )
