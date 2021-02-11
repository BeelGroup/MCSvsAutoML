from .baseline_job import BaselineJob

class AutoSklearnBaselineJob(BaselineJob):

    @classmethod
    def algo_type(cls) -> str:
        return 'autokeras'
