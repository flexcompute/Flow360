"""User defined dynamic model for SimulationParams"""

from typing import Dict, List, Optional

import pydantic as pd

from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.framework.entity_base import EntityList
from flow360.component.simulation.framework.expressions import StringExpression
from flow360.component.simulation.primitives import Cylinder, Surface


class UserDefinedDynamic(Flow360BaseModel):
    """:class:`UserDefinedDynamic` class"""

    name: str = pd.Field()
    input_vars: List[str] = pd.Field()
    constants: Optional[Dict[str, float]] = pd.Field(None)
    output_vars: Optional[Dict[str, StringExpression]] = pd.Field(None)
    state_vars_initial_value: List[StringExpression] = pd.Field()
    update_law: List[StringExpression] = pd.Field()
    input_boundary_patches: Optional[EntityList[Surface]] = pd.Field(None)
    output_target: Optional[Cylinder] = pd.Field(
        None
    )  # Limited to `Cylinder` for now as we have only tested using UDD to control rotation.
