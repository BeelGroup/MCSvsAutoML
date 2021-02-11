from .baseline_job import (
    BaselineJob, AutoKerasBaselineJob, AutoSklearnBaselineJob, TPOTBaselineJob
)

from .selector_job import (
    SelectorJob, AutoKerasSelectorJob, AutoSklearnSelectorJob, DESSelectorJob
)

from .classifier_job import ClassifierJob, TPOTClassifierJob
from .benchmark_job import BenchmarkJob

job_types = {
    'classifier' : {
        'tpot': TPOTClassifierJob,
    },
    'selector': {
        'autokeras': AutoKerasSelectorJob,
        'autosklearn': AutoSklearnSelectorJob,
        'metades': DESSelectorJob,
    },
    'baseline': {
        'tpot': TPOTBaselineJob,
        'autosklearn': AutoSklearnBaselineJob,
        'autokeras': AutoKerasBaselineJob,
    },
}
