from typing import Dict, Any

import os
import pickle

import numpy as np
from autokeras import StructuredDataClassifier
from tensorflow import keras

from ..model import Model

# TODO: Breaks on server where experiments were running

class AutoKerasBaselineModel(Model):

    def __init__(
        self,
        name: str,
        model_params: Dict[str, Any],
    ) -> None:
        super().__init__(name, model_params)
        #self._fit_params = model_params.pop('fit_params')
        self._fit_params = {}
        self._model = StructuredDataClassifier(**model_params)

    def _force_fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model.fit(X, y, **self._fit_params)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if not isinstance(self._model, StructuredDataClassifier):
            raise RuntimeError('Due to AutoKeras being unsaveable, saving this'
                               + ' means only the best keras model was'
                               + ' saved. Calling fit will fit this pipeline'
                               + ' rather than the AutoKeras algorithm. If this'
                               + ' is desired behaviour, please use '
                               + '`_force_fit` instead')
        self._force_fit(X, y)

    def save(self, path: str) -> None:
        # Get the best model that can be saved
        best_model = self._model.export_model()
        best_model.save(path)

        # Also store the model wrapper after removing autokeras model
        # as it can't be saved
        self._model = None
        wrapper_path = path + '.wrapper'
        with open(wrapper_path, 'wb') as file:
            pickle.dump(self, file)


    @classmethod
    def load(cls, path: str):
        wrapper_class = None
        # Have to load model wrapper and the underlying keras model seperatly
        # as the autokeras model can't be saved
        wrapper_path = path + '.wrapper'
        with open(wrapper_path, 'rb') as file:
            wrapper_class = pickle.load(file)

        wrapper_class._model = keras.models.load_model(path)
        return wrapper_class


    def predict(self, X: np.ndarray) -> np.ndarray:
        if isinstance(self._model, StructuredDataClassifier):
            return self._model.predict(X)
        else: # Already been exported,
            return np.round(self._model.predict(X))


    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if isinstance(self._model, StructuredDataClassifier):
            exported_model = self._model.export_model()
            return exported_model.predict(X)
        else: # Already been exported
            return self._model.predict(X)
