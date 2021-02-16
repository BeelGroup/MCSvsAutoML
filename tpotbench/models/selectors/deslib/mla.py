from typing import Any, Dict, Iterable, Tuple

from deslib.dcs.mla import MLA

from .base import DESSelectorModel

class MLASelectorModel(DESSelectorModel):

    def __init__(
        self,
        name: str,
        model_params: Dict[str, Any],
        classifier_paths: Iterable[Tuple[str, str]],
    ) -> None:
        super().__init__(name, model_params, classifier_paths)
        self._selector = MLA(self.classifiers, **model_params)

    @property
    def selector(self) -> MLA:
        return self._selector

    @classmethod
    def ensemble_selector(cls) -> bool:
        return False
