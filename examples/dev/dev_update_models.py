from pprint import pprint

from flow360.component.flow360_params.flow360_legacy import Flow360ParamsLegacy

model = Flow360ParamsLegacy("../../flow360/examples/rotatingSpheres/flow360.json")

updated = model.update_model()

pprint(updated.dict())

