import os
from .autosklearn_job import AutoSklearnSelectorJob

current_dir = os.path.abspath(os.path.dirname(__file__))
selector_job_map = {
    'autosklearn': (AutoSklearnSelectorJob,
                    os.path.join(current_dir, 'autosklearn_runner.py'))
}

