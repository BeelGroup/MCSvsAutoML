import os
from .tpot_job import TPOTClassifierJob

classifier_job_map = {
    job_class.classifier_type(): job_class
    for job_class
    in [TPOTClassifierJob]
}
