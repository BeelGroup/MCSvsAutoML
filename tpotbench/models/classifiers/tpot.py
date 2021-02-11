from typing import Dict, Any, cast

import pickle

import numpy as np
from tpot import TPOTClassifier

from ..model import Model


class TPOTClassifierModel(Model):

    def __init__(
        self,
        name: str,
        model_params: Dict[str, Any],
    ) -> None:
        super().__init__(name, model_params)
        self._model = TPOTClassifier(**model_params)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model.fit(X, y)

    def save(self, path: str) -> None:
        with open(path, 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, path: str):
        with open(path, 'rb') as file:
            model = pickle.load(file)
            return cast(TPOTClassifierModel, model)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)
