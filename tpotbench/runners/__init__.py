import os
current_dir = os.path.abspath(os.path.dirname(__file__))

runners = {
    'baseline': os.path.join(current_dir, 'baseline_runner.py'),
    'selector': os.path.join(current_dir, 'selector_runner.py'),
    'classifier': os.path.join(current_dir, 'classifier_runner.py')
}
