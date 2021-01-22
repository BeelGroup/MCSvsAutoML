import os
from .tpot_job import TPOTClassifierJob

current_dir = os.path.abspath(os.path.dirname(__file__))
classifier_job_map = {
    'TPOT': (TPOTClassifierJob, os.path.join(current_dir, 'tpot_runner.py'))
}
