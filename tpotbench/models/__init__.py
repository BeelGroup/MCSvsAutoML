import os
from .model import Model
current_dir = os.path.abspath(os.path.dirname(__file__))

baseline_dir = os.path.join(current_dir, 'baselines')
selector_dir = os.path.join(current_dir, 'selectors')
classifier_dir = os.path.join(current_dir, 'classifiers')

runners = {
    'baselines': {
        'autosklearn': os.path.join(baseline_dir, 'autosklearn.py')
    },
    'selectors': {
        'autosklearn': os.path.join(selector_dir, 'autosklearn.py'),
        'metades': os.path.join(selector_dir, 'metades.py')
    },
    'classifiers': {
        'TPOT': os.path.join(classifier_dir, 'tpot.py')
    },
}
