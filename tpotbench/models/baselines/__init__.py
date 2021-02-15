# TODO: Autokeras had issues on the server
#from .autokeras import AutoKerasBaselineModel
from .autosklearn import AutoSklearnBaselineModel
from .tpot import TPOTBaselineModel

baseline_classes = {
    'autosklearn': AutoSklearnBaselineModel,
    #'autokeras': AutoKerasBaselineModel,
    'tpot': TPOTBaselineModel
}
