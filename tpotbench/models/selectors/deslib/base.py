from typing import Any, Dict, cast, Iterable, Tuple
from abc import abstractmethod

import pickle

import numpy as np

from ..selector_model import SelectorModel


class DESSelectorModel(SelectorModel):

    @abstractmethod
    def __init__(
        self,
        name: str,
        model_params: Dict[str, Any],
        classifier_paths: Iterable[Tuple[str, str]],
    ) -> None:
        super().__init__(name, model_params, classifier_paths)

    @property
    @abstractmethod
    def selector(self) -> Any:
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.selector.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.selector.predict_proba(X)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.selector.fit(X, y)


    def selections(self, X: np.ndarray) -> np.ndarray:
        competences = self.competences(X)
        selections = self.selector.select(competences)
        return selections

    @classmethod
    def load(cls, path: str):
        with open(path, 'rb') as file:
            model = pickle.load(file)
            return cast(DESSelectorModel, model)

    def save(self, path: str) -> None:
        with open(path, 'wb') as file:
            pickle.dump(self, file)

    def competences(self, X: np.ndarray) -> np.ndarray:
        distances, neighbors = self.selector._get_region_competence(X)

        if self._uses_proba():
            classifier_probabilities = self.selector._predict_proba_base(X)
            competences = self.selector.estimate_competence(
                query=X, neighbors=neighbors, probabilities=classifier_probabilities,
                distances=distances)
        else:
            classifier_predictions = self.selector._predict_proba_base(X)
            competences = self.selector.estimate_competence_from_proba(
                query=X, neighbors=neighbors, predicitons=classifier_predictions,
                distances=distances)

        return competences

    def _uses_proba(self):
        return (
            hasattr(self.selector, 'estimate_competence_from_proba')
            and self.selector.needs_proba
        )
