import os

from flow360.component.flow360_params.flow360_params import Flow360ParamsLegacy

here = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(here, "../../flow360/examples/rotatingSpheres/flow360.json")

model = Flow360ParamsLegacy(path)

model.update_model()
