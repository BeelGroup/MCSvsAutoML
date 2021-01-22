from typing import Tuple, Dict, Any, Optional

import os
from os.path import join
from shutil import rmtree

from ..benchmarkjob import BenchmarkJob


class TPOTClassifierJob(BenchmarkJob):

    _runner_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'autosklearn_runner.py'
    )

    defaults = {
        'memory': 12000,
        'cpus': 1,
    }

    def __init__(
        self,
        jobname: str,
        seed: int,
        task_id: int,
        time: int,
        basedir: str,
        splits: Tuple[float, float, float],
        algorithm_family: str,
        memory: int,
        cpus: int,
        model_params: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(
            jobname, seed, task_id, time, basedir, splits, memory, cpus
        )
        self.algorithm_family = algorithm_family
        self.model_params = model_params
        self._paths: Dict[str, Any] = {
            'basedir': basedir,
            'files': {
                'config': join(basedir, 'config.json'),
                'log': join(basedir, 'tpot_log'),
                'export': join(basedir, 'export.py'),
                'model': join(basedir, 'model.pkl'),
                'metrics': join(basedir, 'metrics.json'),
                'train_classifications': join(basedir, 'train_classifications'),
                'train_probabilities': join(basedir, 'train_probabilities'),
                'selector_training_classifications': join(
                    basedir, 'selector_training_classifications'
                ),
                'selector_training_probabilities': join(
                    basedir, 'selector_training_probabilities'
                ),
                'test_classifications': join(basedir, 'test_classifications'),
                'test_probabilities': join(basedir, 'test_probabilities'),
            },
            'folders': {
                'checkpoints': join(basedir, 'checkpoints')
            }
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
            os.mkdir(self._paths['folders']['basedir'])

        if not os.path.exists(self._paths['folders']['checkpoints']):
            os.mkdir(self._paths['folders']['checkpoints'])

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
            'algorithm_family': self.algorithm_family,
            'model_params': model_params,
            'files': paths['files'],
            'folders': paths['folders']
        }

    def command(self) -> str:
        config_path = self._paths['files']['config']
        return f'python {self.runner_path} {config_path}'

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
                   algorithm_family=cfg['classifier'],
                   memory=cfg.get('memory', cls.defaults['memory']),
                   cpus=cfg.get('cpus', cls.defaults['cpus']),
                   model_params=cfg.get('model_params', None)
                   )
