from typing import Dict, Any, List
from abc import abstractmethod

import os

import numpy as np
from numpy import ndarray

from .. import Model
from .. import classifier_classes


class SelectorModel(Model):

    def __init__(
        self,
        name: str,
        model_params: Dict[str, Any],
        classifier_paths: Dict[str, str],
    ) -> None:
        super().__init__(name, model_params)

        self._classifiers: List[Model] = []

        for clf_type, clf_path in classifier_paths.items():
            if not os.path.exists(clf_path):
                raise RuntimeError(f'No {clf_type} classifier at {clf_path}'
                                   + f'for selector {self.name()}')

            clf_class = classifier_classes[clf_type]
            clf_model = clf_class.load(clf_path)
            self.classifiers.append(clf_model)

    @property
    def classifiers(self) -> List[Model]:
        return self._classifiers

    @classmethod
    @abstractmethod
    def ensemble_selector(cls) -> bool:
        raise NotImplementedError

    @abstractmethod
    def selections(self, X: ndarray) -> ndarray:
        raise NotImplementedError

    @abstractmethod
    def competences(self, X: ndarray) -> ndarray:
        raise NotImplementedError

    def classifier_predictions(self, X: ndarray) -> ndarray:
        return np.asarray([clf.predict(X) for clf in self.classifiers])

    def classifier_probabilities(self, X: ndarray) -> ndarray:
        return np.asarray([clf.predict_proba(X) for clf in self.classifiers])
