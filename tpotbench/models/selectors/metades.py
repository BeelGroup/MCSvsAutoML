from typing import List, Any

import numpy as np

from .selector_model import SelectorModel
from ..model import Model


class METADESSelectorModel(SelectorModel):

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
        return True

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.selector.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.selector.predict_proba(X)

    def selections(self, X: np.ndarray) -> np.ndarray:
        competences = self.competences(X)
        selections = self.selector.select(competences)
        return selections

    def competences(self, X: np.ndarray) -> np.ndarray:
        distances, neighbors = self.selector._get_region_competence(X)
        classifier_probabilities = self.selector._predict_proba_base(X)

        competences = self.selector.estimate_competence_from_proba(
            query=X, neighbors=neighbors,
            probabilities=classifier_probabilities,
            distances=distances)

        return competences

    def classifiers(self) -> List[Model]:
        return self._classifiers
