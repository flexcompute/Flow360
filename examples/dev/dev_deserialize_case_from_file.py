import os
from pprint import pprint

from flow360.component.flow360_params.flow360_params import Flow360Params

here = os.path.dirname(os.path.abspath(__file__))

validated = Flow360Params(os.path.join(here, "../../tests/data/flow360_case_om6wing.json"))

pprint(validated.dict())

pprint(Flow360Params.schema())

