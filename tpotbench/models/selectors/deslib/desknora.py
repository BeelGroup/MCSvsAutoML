from typing import Any, Dict, Iterable, Tuple

from deslib.des.knora_u import KNORAU

from .base import DESSelectorModel

class DESKNORAUSelectorModel(DESSelectorModel):

    def __init__(
        self,
        name: str,
        model_params: Dict[str, Any],
        classifier_paths: Iterable[Tuple[str, str]],
    ) -> None:
        super().__init__(name, model_params, classifier_paths)
        self._selector = KNORAU(self.classifiers, **model_params)

    @property
    def selector(self) -> KNORAU:
        return self._selector

    @classmethod
    def ensemble_selector(cls) -> bool:
        return True
