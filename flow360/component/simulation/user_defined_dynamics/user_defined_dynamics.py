from typing import Dict, List, Optional

import pydantic as pd

from flow360.component.simulation.base_model import Flow360BaseModel
from flow360.component.simulation.framework.expressions import StringExpression


class UserDefinedDynamic(Flow360BaseModel):
    """:class:`UserDefinedDynamic` class"""

    name: str = pd.Field()
    input_vars: List[str] = pd.Field()
    constants: Optional[Dict[str, float]] = pd.Field(None)
    output_vars: Optional[Dict[str, StringExpression]] = pd.Field(None)
    state_vars_initial_value: List[StringExpression] = pd.Field()
    update_law: List[StringExpression] = pd.Field()
    input_boundary_patches: EntityList[Surface] = pd.Field(None)
    output_target_name: Optional[str] = pd.Field(None)
