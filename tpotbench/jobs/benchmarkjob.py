# https://stackoverflow.com/a/33533514/5332072
from __future__ import annotations

from abc import abstractmethod, ABC
from typing import Mapping, Dict, Any, Tuple

from slurmjobmanager import Job
from ..models.model import Model
from ..runners import runners

# TODO Need to more config path into BenchmarkJob
class BenchmarkJob(Job, ABC):

    @abstractmethod
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
            *args,
            **kwargs
    ) -> None:
        super().__init__()
        self._name = name
        self.seed = seed
        self.task = task
        self.split = split
        self.time = time
        self.basedir = basedir
        self.memory = memory
        self.cpus = cpus

    def name(self) -> str:
        return self._name

    def ready(self) -> bool:
        return not self.blocked() and not self.complete()

    def command(self) -> str:
        config_path = self.paths()['files']['config']
        runner = runners[self.job_type()][self.algo_type()]

        return f'python {runner} {config_path}'

    @classmethod
    def from_config(
        cls,
        cfg: Dict[str, Any],
        basedir: str,
    ) -> BenchmarkJob:
        if cfg['algo_type'] != cls.algo_type():
            raise ValueError(f'Config object not a {cls.algo_type()} '
                             + f'\n{cfg=}')

        # Remove it as it's not a constructor params
        del cfg['algo_type']

        # TODO this will raise mypy errors, need to fix this up
        params = {**cfg, 'basedir': basedir}
        return cls(**params)

    @abstractmethod
    def paths(self) -> Dict[str, Any]:
        pass

    @classmethod
    @abstractmethod
    def job_type(cls) -> str:
        pass

    @classmethod
    @abstractmethod
    def algo_type(cls) -> str:
        pass

    @abstractmethod
    def config(self) -> Mapping[str, Any]:
        pass

    @abstractmethod
    def model(self) -> Model:
        pass
