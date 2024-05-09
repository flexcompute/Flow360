from typing import List, Literal, Optional, Union

import pydantic as pd

from flow360.component.simulation.entities_base import EntitiesBase


class Aniso(EntitiesBase):
    """Aniso edge"""

    type: str = pd.Field("aniso", frozen=True)
    method: Literal["angle", "height", "aspectRatio"] = pd.Field()
    value: pd.PositiveFloat = pd.Field()


class ProjectAniso(EntitiesBase):
    """ProjectAniso edge"""

    type: str = pd.Field("projectAnisoSpacing", frozen=True)


EdgeRefinementTypes = Union[Aniso, ProjectAniso]
