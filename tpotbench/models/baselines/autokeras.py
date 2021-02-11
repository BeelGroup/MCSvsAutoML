from typing import Dict, Any

import numpy as np
#from autokeras import StructuredDataClassifier

from ..model import Model


class AutoKerasBaselineModel(Model):

    def __init__(
        self,
        name: str,
        model_params: Dict[str, Any],
    ) -> None:
        super().__init__(name, model_params)
        raise NotImplementedError

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        raise NotImplementedError

    def save(self, path: str) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path: str):
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError
