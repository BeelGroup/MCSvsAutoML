from typing import Mapping, TypeVar, Type, Dict, Any
from abc import ABC, abstractmethod

from numpy import ndarray
from sklearn.metrics import accuracy_score

T = TypeVar('T', bound='Model')


class Model(ABC):

    @abstractmethod
    def __init__(
        self,
        name: str,
        model_params: Dict[str, Any],
        *args,
        **kwargs
    ) -> None:
        self._name = name
        self._model_params = model_params

    @property
    def name(self) -> str:
        return self._name

    @property
    def model_params(self) -> Mapping[str, Any]:
        return self._model_params

    @abstractmethod
    def fit(self, X: ndarray, y: ndarray) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: ndarray) -> ndarray:
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, X: ndarray) -> ndarray:
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str) -> None:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls: Type[T], path: str) -> T:
        raise NotImplementedError

    def score(self, X, y, sample_weight=None):
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)
