from typing import Dict, Any, cast

import pickle

import numpy as np
from tpot import TPOTClassifier

from ..model import Model


# TODO: Can't pickle the full TPOT model
#       https://github.com/EpistasisLab/tpot/issues/781
#
#       The work around is to only pickle the fitted pipeline accessible at
#       `tpot.fitted_pipeline_`. However this means the model loaded back
#       in is not exactly the same as the TPOT automodel.
#
#       Before saving the model:
#           isinstance(self._model, TPOTClassifier)
#
#       After loading the model:
#           not isinstance(self._model, TPOTClassifier)
#           isinstance(self._model, sklearn.pipeline.Pipeline,)
#
#       Practically for our uses this was not an issue but could be important
#       to anyone who's managed to find this message
class TPOTClassifierModel(Model):

    def __init__(
        self,
        name: str,
        model_params: Dict[str, Any],
    ) -> None:
        super().__init__(name, model_params)
        self._model = TPOTClassifier(**model_params)

    def _force_fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model.fit(X, y)
        # TODO: This is required by DESlib which is kind of annoying.
        #       They call check_if_fitted(model, 'classes_'), meaning these have
        #       to act more like general sklearn models
        #
        #       If using more classifiers, the creation of a ClassifierModel
        #       base class is probably required to ensure consistency
        self.classes_ = self._model.fitted_pipeline_.classes_

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
            return cast(TPOTClassifierModel, model)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)
