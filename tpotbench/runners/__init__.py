import os
current_dir = os.path.abspath(os.path.dirname(__file__))

baseline_dir = os.path.join(current_dir, 'baselines')
selector_dir = os.path.join(current_dir, 'selectors')
classifier_dir = os.path.join(current_dir, 'classifiers')

runners = {
    'baseline': {
        'autosklearn': os.path.join(baseline_dir, 'autosklearn_runner.py')
    },
    'selector': {
        'autosklearn': os.path.join(selector_dir, 'autosklearn_runner.py'),
        'metades': os.path.join(selector_dir, 'metades_runner.py')
    },
    'classifier': {
        'tpot': os.path.join(classifier_dir, 'tpot_runner.py')
    },
}
