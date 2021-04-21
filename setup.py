import os
import sys
from distutils.core import setup

if sys.version_info < (3, 8):
    raise ValueError('Requires Python 3.8 or higher')

# Load requirements from requirements.txt
current_dir = os.path.dirname(os.path.realpath(__file__))
requirements_path = os.path.join(current_dir, 'requirements.txt')

install_requires = None
if os.path.isfile(requirements_path):
    with open(requirements_path, 'r') as f:
        install_requires = f.read().splitlines()

setup(name='PerInstanceAlgorithmSelectionBenchmark',
      version='0.1.0',
      description='A benchmark for comparing between AutoML tools and MCS tools for per instance algorithm selection',
      author='Jonathan Doe',
      url='https://github.com/jonathandoe-submission/PerInstanceAlgorithmSelectionBenchmark',
      packages=['src'],
      python_requires='>=3.8',
      install_requires=install_requires,
      extras_require={
        'dev': [
            'mypy',
            'ipython',
            'pylint',
            'autopep8'
        ]
      }
)
