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

setup(name='tpotbench',
      version='0.1.0',
      description='per instance algorithm bench with tpot',
      author='Eddie Bergman',
      author_email='eddiebergmanhs@gmail.com',
      url='https://github.com/eddiebergman/tpotbench',
      packages=['tpotbench'],
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

#slurmjobmanager @ git+git://github.com/eddiebergman/slurmjobmanager.git'
