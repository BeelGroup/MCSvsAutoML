from typing import Dict, Any, Type

import os

from tpot.config import classifier_config_dict

from .benchmark_job import BenchmarkJob
from ..models.classifiers import TPOTClassifierModel


class ClassifierJob(BenchmarkJob):

    # Same __init__

    def blocked(self) -> bool:
        return False

    def config(self) -> Dict[str, Any]:
        return {
            'name': self.name(),
            'seed': self.seed,
            'split': self.split,
            'task': self.task,
            'model_path': self.model_path,
            'algo_type': self.algo_type(),
            'model_params': self.model_params(),
        }

    @classmethod
    def job_type(cls) -> str:
        return 'classifier'


class TPOTClassifierJob(ClassifierJob):

    _default_params = {
        'generations': 9999,
        'population_size': 100,
        'offspring_size': None,
        'mutation_rate': 0.9,
        'crossover_rate': 0.1,
        'scoring': 'accuracy',
        'cv': 5,
        'subsample': 1.0,
        'max_eval_time_mins': 15,
        'template': None,
        'warm_start': True,
        'memory': 'auto',
        'use_dask': False,
        'early_stop': 15,
        'verbosity': 3,
        'disable_update_check': False,
    }

    _algo_families = {
        'KNN': ['sklearn.neighbors.KNeighborsClassifier'],
        'LR': ['sklearn.linear_model.LogisticRegression'],
        'MLP': ['sklearn.neural_network.MLPClassifier'],
        'SGD': ['sklearn.linear_model.SGDClassifier'],
        'XGB': ['xgboost.XGBClassifier'],
        'SVM': ['sklearn.svm.LinearSVC'],
        'NB': ['sklearn.naive_bayes.GaussianNB',
               'sklearn.naive_bayes.BernoulliNB',
               'sklearn.naive_bayes.MultinomialNB'],
        'TR': ['sklearn.tree.DecisionTreeClassifier',
               'sklearn.ensemble.ExtraTreesClassifier',
               'sklearn.ensemble.RandomForestClassifier',
               'sklearn.ensemble.GradientBoostingClassifier'],
    }

    @classmethod
    def model_cls(cls) -> Type[TPOTClassifierModel]:
        return TPOTClassifierModel

    def model_params(self) -> Dict[str, Any]:
        algorithms = self._algo_families[self.model_config['algorithm_family']]
        params = {
            'n_jobs': self.cpus,
            'max_time_mins': self.time,
            'random_state': self.seed,
            'config_dict': {
                algorithm: classifier_config_dict[algorithm]
                for algorithm in algorithms
            },
            'periodic_checkpoint_folder': os.path.join(self.basedir, 'checkpoints'),
            'log_file': os.path.join(self.basedir, 'tpot.log')
        }
        return {**self._default_params, **params}

    @classmethod
    def algo_type(cls):
        return 'tpot'

