from typing import List, Any

import numpy as np

from .selector_model import SelectorModel
from ..model import Model

class AutoSklearnSelectorModel(SelectorModel):

    def __init__(
        self,
        name: str,
        model: Any,
        classifiers: List[Model],
    ):
        super().__init__()
        self._name = name
        self.selector = model
        self._classifiers = classifiers

    def name(self) -> str:
        return self._name

    def ensemble_selector(self) -> bool:
        return False

    def predict(self, X: np.ndarray) -> np.ndarray:
        selections = self.selections(X)
        return [
            self._classifiers[i].predict(instance.reshape(1, -1))
            for i, instance
            in zip(selections, X)
        ]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        selections = self.selections(X)
        return [
            self._classifiers[i].predict_proba(instance.reshape(1, -1))
            for i, instance
            in zip(selections, X)
        ]

    def selections(self, X: np.ndarray) -> np.ndarray:
        competences = self.competences(X)
        return np.argmax(competences, axis=1)

    def competences(self, X: np.ndarray) -> np.ndarray:
        return self.selector.predict_proba(X)

    def classifiers(self) -> List[Model]:
        return self._classifiers
