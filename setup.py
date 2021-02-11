import sys
from distutils.core import setup

if sys.version_info < (3, 8):
    raise ValueError('Requires Python 3.8 or higher')

setup(name='tpotbench',
      version='0.1.0',
      description='per instance algorithm bench with tpot',
      author='Eddie Bergman',
      author_email='eddiebergmanhs@gmail.com',
      url='https://github.com/eddiebergman/tpotbench',
      packages=['tpotbench'],
      python_requires='>=3.8',
      install_requires=[
          'openml',
          'tpot',
          'pandas',
          'xgboost',
          'numpy<=1.19.2',
          'slurmjobmanager @ git+git://github.com/eddiebergman/slurmjobmanager.git',
          'auto-sklearn',
          'autokeras',
          'deslib'
      ],
      extras_require={
        'dev': [
            'mypy',
            'ipython',
            'pylint',
            'autopep8'
        ]
      }
)
