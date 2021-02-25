from typing import Dict, Any, cast

import pickle

import numpy as np
from tpot import TPOTClassifier

from ..model import Model


class TPOTBaselineModel(Model):

    def __init__(
        self,
        name: str,
        model_params: Dict[str, Any],
    ) -> None:
        super().__init__(name, model_params)
        self._model = TPOTClassifier(**model_params)

    def _force_fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model.fit(X, y)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if not isinstance(self._model, TPOTClassifier):
            raise RuntimeError('Due to TPOT being unpickelable, saving this'
                               + ' means only the actual sklearn.Pipeline'
                               + ' was saved. Calling fit will fit this pipeline'
                               + ' rather than the TPOT algorithm. If this is'
                               + ' desired behaviour, please use `_force_fit`'
                               + ' instead')
        self._force_fit(X, y)

    def save(self, path: str) -> None:
        # See comment above class
        if isinstance(self._model, TPOTClassifier):
            self._model = self._model.fitted_pipeline_

        with open(path, 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, path: str):
        with open(path, 'rb') as file:
            model = pickle.load(file)
            return cast(TPOTBaselineModel, model)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # TODO: May cause issues if SVG or SVM model is best
        return self._model.predict_proba(X)
