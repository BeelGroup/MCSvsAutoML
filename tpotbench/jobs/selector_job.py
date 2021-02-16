# TODO: Autokeras removed
from typing import Tuple, Dict, Any, Iterable, Type, List

from .benchmark_job import BenchmarkJob
from .classifier_job import ClassifierJob
from ..models import Model
from ..models.selectors import (
    AutoSklearnSelectorModel,
    DESSelectorModel, METADESSelectorModel, DESRRCSelectorModel,
    DESKNORAUSelectorModel, MLASelectorModel, LCASelectorModel
)


class SelectorJob(BenchmarkJob):

    def __init__(self, name: str, algo_type: str, seed: int, task: int,
                 time: int, basedir: str, split: Tuple[float, float, float],
                 memory: int, cpus: int, model_config: Dict[str, Any],
                 classifiers: Iterable[ClassifierJob]) -> None:
        super().__init__(name, algo_type, seed, task, time, basedir, split,
                         memory, cpus, model_config)
        self.classifiers = classifiers

    @classmethod
    def job_type(cls) -> str:
        return 'selector'

    def blocked(self) -> bool:
        return any(not clf.complete() for clf in self.classifiers)

    def classifier_models(self) -> List[Model]:
        return [clf.model() for clf in self.classifiers]

    def config(self) -> Dict[str, Any]:
        return {
            'name': self.name(),
            'seed': self.seed,
            'split': self.split,
            'task': self.task,
            'model_path': self.model_path,
            'algo_type': self.algo_type(),
            'model_params': self.model_params(),
            'classifiers': [
                (clf.algo_type(), clf.model_path)
                for clf in self.classifiers
            ],
        }


# TODO: For now, most selectors can be fit into a single class but we can
#       give them their own individual classes for now incase anything needs to
#       be added
class AutoSklearnSelectorJob(SelectorJob):

    @classmethod
    def algo_type(cls):
        return 'autosklearn'

    @classmethod
    def model_cls(cls) -> Type[AutoSklearnSelectorModel]:
        return AutoSklearnSelectorModel

    def model_params(self) -> Dict[str, Any]:
        return {
            'time_left_for_this_task': self.time * 60,
            'seed': self.seed,
            'memory_limit': int(self.memory * 0.75),
            'n_jobs': self.cpus,
            **self.model_config
        }

"""
class AutoKerasSelectorJob(SelectorJob):

    @classmethod
    def algo_type(cls):
        return 'autokeras'

    @classmethod
    def model_cls(cls) -> Type[AutoKerasSelectorModel]:
        return AutoKerasSelectorModel

    def model_params(self) -> Dict[str, Any]:
        raise NotImplementedError
"""


# TODO: Should probably start seperating out into seperate files
class DESSelectorJob(SelectorJob):

    def model_params(self) -> Dict[str, Any]:
        return {}

class METADESSelectorJob(DESSelectorJob):

    @classmethod
    def model_cls(cls) -> Type[METADESSelectorModel]:
        return METADESSelectorModel

    @classmethod
    def algo_type(cls):
        return 'metades'

class DESRRCSelectorJob(DESSelectorJob):

    @classmethod
    def model_cls(cls) -> Type[DESRRCSelectorModel]:
        return DESRRCSelectorModel

    @classmethod
    def algo_type(cls):
        return 'desrrc'

class DESKNORAUSelectorJob(DESSelectorJob):

    @classmethod
    def model_cls(cls) -> Type[DESKNORAUSelectorModel]:
        return DESKNORAUSelectorModel

    @classmethod
    def algo_type(cls):
        return 'desknorau'

class MLASelectorJob(DESSelectorJob):

    @classmethod
    def model_cls(cls) -> Type[MLASelectorModel]:
        return MLASelectorModel

    @classmethod
    def algo_type(cls):
        return 'mla'

class LCASelectorJob(DESSelectorJob):

    @classmethod
    def model_cls(cls) -> Type[LCASelectorModel]:
        return LCASelectorModel

    @classmethod
    def algo_type(cls):
        return 'lca'
