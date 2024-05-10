from typing import Optional

import pydantic as pd

from flow360.component.simulation.base_model import Flow360BaseModel


class ReferenceGeometry(Flow360BaseModel):
    # Note: Cannot use dimensioned values for now because of V1 pd
    "Contains all geometrical related refrence values"
    moment_center: Optional[tuple[float, float, float]] = pd.Field()
    moment_length: Optional[tuple[float, float, float]] = pd.Field()
    area: Optional[float] = pd.Field()
    mesh_unit: Optional[float] = pd.Field()
