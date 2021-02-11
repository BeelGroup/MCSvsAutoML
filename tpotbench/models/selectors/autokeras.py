from typing import Any, Dict

import numpy as np

from .selector_model import SelectorModel


class AutoKerasSelectorModel(SelectorModel):

    def __init__(
        self,
        name: str,
        model_params: Dict[str, Any],
        classifiers_paths: Dict[str, str],
    ):
        super().__init__(name, model_params, classifiers_paths)
        self.selector = None
        raise NotImplementedError

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

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path: str):
        raise NotImplementedError

    def save(self, path: str) -> None:
        raise NotImplementedError
