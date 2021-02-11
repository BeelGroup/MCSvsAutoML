from typing import Any, Dict, cast, Iterable, Tuple

import pickle

import numpy as np
from autosklearn.classification import AutoSklearnClassifier

from .selector_model import SelectorModel


class AutoSklearnSelectorModel(SelectorModel):

    def __init__(
        self,
        name: str,
        model_params: Dict[str, Any],
        classifier_paths: Iterable[Tuple[str, str]],
    ) -> None:
        super().__init__(name, model_params, classifier_paths)
        self.selector = AutoSklearnClassifier(**model_params)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.selector.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        # TODO: Can optimize and make cleaner
        return np.asarray([
            self.classifiers[i].predict(instance.reshape(1, -1))
            for i, instance in zip(self.selections(X), X)
        ])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # TODO: Can optimize and make cleaner
        return np.asarray([
            self.classifiers[i].predict_proba(instance.reshape(1, -1))
            for i, instance in zip(self.selections(X), X)
        ])

    def selections(self, X: np.ndarray) -> np.ndarray:
        competences = self.competences(X)
        return np.argmax(competences, axis=1)

    def competences(self, X: np.ndarray) -> np.ndarray:
        return self.selector.predict_proba(X)

    def save(self, path: str) -> None:
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def ensemble_selector(cls) -> bool:
        return False

    @classmethod
    def load(cls, path: str):
        # Inherits typing from parent
        with open(path, 'rb') as file:
            return cast(AutoSklearnSelectorModel, pickle.load(file))
