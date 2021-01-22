import os
from abc import abstractmethod
from typing import Mapping, Dict, Any, Tuple, Type

from slurmjobmanager import Job

class BenchmarkJob(Job):

    @abstractmethod
    def __init__(
            self,
            jobname: str,
            seed: int,
            task_id: int,
            time: int,
            basedir: str,
            splits: Tuple[float, float, float],
            memory: int,
            cpus: int,
            *args,
            **kwargs
    ) -> None:
        super().__init__()
        self.jobname = jobname
        self.seed = seed
        self.task_id = task_id
        self.splits = splits
        self.time = time
        self.basedir = basedir
        self.memory = memory
        self.cpus = cpus

        # Used to indicate a job having started which is also used to indicate
        # if a job has failed, i.e. not in progress, not complete and started_at
        # exists
        self._started_at = os.path.join(basedir, 'started_at')

    @abstractmethod
    def paths(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def config(self) -> Mapping[str, Any]:
        pass

    @abstractmethod
    @classmethod
    def runner_path(cls) -> str:
        pass

    @abstractmethod
    @classmethod
    def from_config(
        cls: Type[BenchmarkJob],
        cfg: Dict[str, Any],
        basedir: str,
    ) -> BenchmarkJob:
        pass

    def ready(self) -> bool:
        return not self.blocked() and not self.complete()
