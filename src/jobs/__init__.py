# TODO: Autokeras removed due to issues on server
from .baseline_job import (
    BaselineJob, AutoSklearnBaselineJob, TPOTBaselineJob
)

from .selector_job import (
    SelectorJob, AutoSklearnSelectorJob, DESSelectorJob,
    METADESSelectorJob, DESRRCSelectorJob, DESKNORAUSelectorJob,
    LCASelectorJob, MLASelectorJob
)

from .classifier_job import ClassifierJob, TPOTClassifierJob
from .benchmark_job import BenchmarkJob

job_types = {
    'classifier' : {
        'tpot': TPOTClassifierJob,
    },
    'selector': {
        #'autokeras': AutoKerasSelectorJob,
        'autosklearn': AutoSklearnSelectorJob,
        'metades': METADESSelectorJob,
        'desrrc': DESRRCSelectorJob,
        'desknorau': DESKNORAUSelectorJob,
        'lca': LCASelectorJob,
        'mla': MLASelectorJob,
    },
    'baseline': {
        'tpot': TPOTBaselineJob,
        'autosklearn': AutoSklearnBaselineJob,
        #'autokeras': AutoKerasBaselineJob,
    },
}
