from typing import List, Literal, Optional, Tuple, Union

import pydantic as pd

from flow360.component.flow360_params.unit_system import AreaType, LengthType
from flow360.component.simulation.base_model import Flow360BaseModel


class ReferenceGeometry(Flow360BaseModel):
    # Note: Cannot use dimensioned values for now because of V1 pd
    "Contains all geometrical related refrence values"
    moment_center: Optional[tuple[pd.StrictFloat, pd.StrictFloat, pd.StrictFloat]] = pd.Field()
    moment_length: Optional[tuple[pd.StrictFloat, pd.StrictFloat, pd.StrictFloat]] = pd.Field()
    area: Optional[pd.StrictFloat] = pd.Field()
