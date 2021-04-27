from typing import Any, Dict, Iterable, Tuple

from deslib.des.meta_des import METADES

from .base import DESSelectorModel

class METADESSelectorModel(DESSelectorModel):

    def __init__(
        self,
        name: str,
        model_params: Dict[str, Any],
        classifier_paths: Iterable[Tuple[str, str]],
    ) -> None:
        super().__init__(name, model_params, classifier_paths)
        self._selector = METADES(self.classifiers, **model_params)

    @property
    def selector(self) -> METADES:
        return self._selector

    @classmethod
    def ensemble_selector(cls) -> bool:
        return True
