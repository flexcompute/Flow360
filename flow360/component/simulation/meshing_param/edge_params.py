from typing import List, Literal, Optional, Union

import pydantic as pd

from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.framework.entity_base import EntityList
from flow360.component.simulation.primitives import Edge


class Aniso(Flow360BaseModel):
    """Aniso edge"""

    type: str = pd.Field("aniso", frozen=True)
    method: Literal["angle", "height", "aspectRatio"] = pd.Field()
    value: pd.PositiveFloat = pd.Field()
    entities: EntityList[Edge] = pd.Field(alias="edges")


class ProjectAniso(Flow360BaseModel):
    """ProjectAniso edge"""

    type: str = pd.Field("projectAnisoSpacing", frozen=True)
    entities: EntityList[Edge] = pd.Field(alias="edges")


EdgeRefinementTypes = Union[Aniso, ProjectAniso]
