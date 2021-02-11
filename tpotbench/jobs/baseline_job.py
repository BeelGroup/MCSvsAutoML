from typing import Dict, Any, Type

from .benchmark_job import BenchmarkJob
from ..models.baselines import (
    AutoSklearnBaselineModel,
    AutoKerasBaselineModel,
    TPOTBaselineModel
)


class BaselineJob(BenchmarkJob):

    # Same __init__

    @classmethod
    def job_type(cls) -> str:
        return 'baseline'

    def blocked(self) -> bool:
        return False

    def config(self) -> Dict[str, Any]:
        return {
            'name': self.name(),
            'algo_type': self.algo_type(),
            'seed': self.seed,
            'split': self.split,
            'task': self.task,
            'model_path': self.model_path,
            'model_params': self.model_params(),
        }


class AutoSklearnBaselineJob(BaselineJob):

    @classmethod
    def model_cls(cls) -> Type[AutoSklearnBaselineModel]:
        return AutoSklearnBaselineModel

    def model_params(self) -> Dict[str, Any]:
        return {
            'time_left_for_this_task': self.time * 60,
            'seed': self.seed,
            'memory_limit': int(self.memory * 0.75),
            'n_jobs': self.cpus,
            **self.model_config
        }

    @classmethod
    def algo_type(cls):
        return 'autosklearn'

class AutoKerasBaselineJob(BaselineJob):

    @classmethod
    def model_cls(cls) -> Type[AutoKerasBaselineModel]:
        return AutoKerasBaselineModel

    def model_params(self) -> Dict[str, Any]:
        raise NotImplementedError

    @classmethod
    def algo_type(cls):
        return 'autokeras'

class TPOTBaselineJob(BaselineJob):

    @classmethod
    def model_cls(cls) -> Type[TPOTBaselineModel]:
        return TPOTBaselineModel

    def model_params(self) -> Dict[str, Any]:
        raise NotImplementedError

    @classmethod
    def algo_type(cls):
        return 'tpot'
