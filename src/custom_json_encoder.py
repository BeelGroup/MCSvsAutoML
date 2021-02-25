import json
from types import GeneratorType

import numpy as np

class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        # check if we can cast it to a list
        if any(isinstance(obj, objtype) for objtype in [GeneratorType,
                                                        np.ndarray,
                                                        range]):
            return list(obj)

        if isinstance(obj, np.integer):
            return int(obj)

        if isinstance(obj, np.floating):
            return float(obj)

        return json.JSONEncoder.default(self, obj)
