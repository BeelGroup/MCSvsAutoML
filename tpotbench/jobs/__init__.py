from .baselines.autosklearn import AutoSklearnBaselineJob

from .classifiers.tpot import TPOTClassifierJob

from .selectors.autosklearn import AutoSklearnSelectorJob
from .selectors.metades import METADESSelectorJob
from .benchmarkjob import BenchmarkJob

job_types = {
    'classifier': {
        clf_cls.algo_type(): clf_cls
        for clf_cls
        in [TPOTClassifierJob]
    },
    'selector': {
        selector_cls.algo_type(): selector_cls
        for selector_cls in [AutoSklearnSelectorJob, METADESSelectorJob]
    },
    'baseline': {
        baseline_cls.algo_type(): baseline_cls
        for baseline_cls in [AutoSklearnBaselineJob]
    }
}
