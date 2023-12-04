import os
from pprint import pprint

from flow360.component.flow360_params.flow360_legacy import Flow360ParamsLegacy

here = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(here, "../../flow360/examples/rotatingSpheres/flow360.json")

model = Flow360ParamsLegacy(path)

updated = model.update_model()

pprint(updated.dict())
