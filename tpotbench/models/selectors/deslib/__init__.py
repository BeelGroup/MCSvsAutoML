from .metades import METADESSelectorModel
from .desrrc import DESRRCSelectorModel
from .desknora import DESKNORAUSelectorModel
from .mla import MLASelectorModel
from .lca import LCASelectorModel
from .base import DESSelectorModel

deslib_models = {
    'metades': METADESSelectorModel,
    'desrrc': DESRRCSelectorModel,
    'desknorau': DESKNORAUSelectorModel,
    'mla': MLASelectorModel,
    'lca': LCASelectorModel
}
