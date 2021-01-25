import os
from .autosklearn_job import AutoSklearnSelectorJob
from .metades_job import METADESSelectorJob

selector_job_map = {
    job_class.selector_type() : job_class
    for job_class
    in [AutoSklearnSelectorJob, METADESSelectorJob]
}
