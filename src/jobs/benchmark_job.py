from typing import Mapping, Dict, Any, Tuple, Type, TypeVar
from abc import abstractmethod, ABC

import os
import json
from shutil import rmtree

from slurmjobmanager import Job

from ..runners import runners
from ..models import Model
from ..custom_json_encoder import CustomEncoder

T = TypeVar('T', bound='BenchmarkJob')


class BenchmarkJob(Job, ABC):

    def __init__(self, name: str, algo_type: str, seed: int, task: int,
                 time: int, basedir: str, split: Tuple[float, float, float],
                 memory: int, cpus: int, model_config: Dict[str, Any],
                 *args, **kwargs) -> None:
        super().__init__()
        if algo_type != self.algo_type():
            raise ValueError(f'Can not construct {self.__class__.__name__}'
                            + f' with {algo_type}')

        self._name = name
        self.seed = seed
        self.task = task
        self.time = time
        self.basedir = basedir
        self.split = split
        self.memory = memory
        self.cpus = cpus
        self.model_config = model_config

    @classmethod
    @abstractmethod
    def algo_type(cls) -> str:
        raise NotImplementedError

    @property
    def config_path(self) -> str:
        return os.path.join(self.basedir, 'config.json')

    @property
    def model_path(self) -> str:
        return os.path.join(self.basedir, 'model.pkl')

    def name(self) -> str:
        return self._name

    def ready(self) -> bool:
        return not self.blocked() and not self.complete()

    def command(self) -> str:
        config_path = self.config_path
        runner = runners[self.job_type()]

        return f'python {runner} {config_path}'

    def setup(self) -> None:
        if not os.path.exists(self.basedir):
            os.mkdir(self.basedir)

        if not os.path.exists(self.config_path):
            job_config = self.config()
            with open(self.config_path, 'w') as f:
                # CustomEncoder is required for ranges
                json.dump(job_config, f, cls=CustomEncoder, indent=2)

    def reset(self) -> None:
        rmtree(self.basedir)

    def complete(self) -> bool:
        return os.path.exists(self.model_path)

    def model(self) -> Model:
        # TODO: Not very type safe seeing as it returns selector models
        #       general models
        if not self.complete():
            raise RuntimeError('This model has not been created yet')

        model_cls = self.model_cls()
        return model_cls.load(self.model_path)

    @classmethod
    def from_config(cls: Type[T], cfg: Dict[str, Any], basedir: str) -> T:
        # pylint: disable-msg=comparison-with-callable
        if cfg['algo_type'] != cls.algo_type():
            raise ValueError(f'Config object not a {cls.algo_type}\n{cfg=}')

        # TODO this will raise mypy errors, need to fix this up
        params = {**cfg, 'basedir': basedir}
        return cls(**params)  # type: ignore

    @abstractmethod
    def config(self) -> Mapping[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def model_params(self) -> Mapping[str, Any]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def job_type(cls) -> str:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def model_cls(cls) -> Type[Model]:
        raise NotImplementedError
