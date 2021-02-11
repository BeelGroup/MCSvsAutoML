from .deslib import deslib_models, DESSelectorModel
from .autosklearn import AutoSklearnSelectorModel
from .autokeras import AutoKerasSelectorModel

selector_classes = {
    'autosklearn': AutoKerasSelectorModel,
    'autokeras': AutoKerasSelectorModel,
    ** deslib_models
}
