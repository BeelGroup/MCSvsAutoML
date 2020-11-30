import sys
from distutils.core import setup

if sys.version_info < (3, 9):
    raise ValueError('Requires Python 3.9 or higher')

setup(name='tpotbench',
      version='0.1.0',
      description='per instance algorithm bench with tpot',
      author='Eddie Bergman',
      author_email='eddiebergmanhs@gmail.com',
      url='https://github.com/eddiebergman/tpotbench',
      packages=['tpotbench'],
      python_requires='>=3.9',
      install_requires=[
          'openml',
          'tpot',
          'pandas',
          'xgboost',
          'numpy',
          'slurmjobmanager @ git+git://github.com/eddiebergman/slurmjobmanager.git'
      ],
      extras_require={
        'dev': [
            'mypy',
            'ipython',
            'pylint'
        ]
      }
)
