import json
from types import GeneratorType

from numpy import ndarray

class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        # check if we can cast it to a list
        if any(
            isinstance(obj, objtype) for objtype in [GeneratorType, ndarray, range]
        ):
            return list(obj)
        return json.JSONEncoder.default(self, obj)
