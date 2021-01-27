from typing import Protocol

from numpy import ndarray
from sklearn.metrics import accuracy_score


class Model(Protocol):

    def name(self) -> str: ...
    def predict(self, X: ndarray) -> ndarray: ...
    def predict_proba(self, X: ndarray) -> ndarray: ...

    def score(self, X, y, sample_weight=None):
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)
