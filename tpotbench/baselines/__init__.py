import os
from .autosklearn_job import AutoSklearnBaselineJob

baseline_job_map = {
    job_class.baseline_type(): job_class
    for job_class
    in [AutoSklearnBaselineJob]
}
