"""
Initial condition parameters
"""

from __future__ import annotations

from typing import Dict, Optional, Union

import pydantic as pd
from typing_extensions import Literal

from .params_base import Flow360BaseModel


class InitialCondition(Flow360BaseModel):
    """:class:`InitialCondition` class"""

    type: str


class FreestreamInitialCondition(InitialCondition):
    """:class:`FreestreamInitialCondition` class"""

    type: Literal["freestream"] = pd.Field("freestream", const=True)


class ExpressionInitialConditionBase(InitialCondition):
    """:class:`ExpressionInitialConditionBase` class which can be used to manipulate restart solutions too"""

    constants: Optional[Dict[str, str]] = pd.Field(alias="constants")
    rho: str = pd.Field(displayed="rho [non-dim]")
    u: str = pd.Field(displayed="u [non-dim]")
    v: str = pd.Field(displayed="v [non-dim]")
    w: str = pd.Field(displayed="w [non-dim]")
    p: str = pd.Field(displayed="p [non-dim]")


class ExpressionInitialCondition(ExpressionInitialConditionBase):
    """:class:`ExpressionInitialCondition` class"""

    type: Literal["expression"] = pd.Field("expression", const=True)


class ExpressionRestartInitialCondition(ExpressionInitialConditionBase):
    """:class:`ExpressionRestartInitialCondition` class that can be used to manipulate restart solution"""

    type: Literal["restartManipulation"] = pd.Field("restartManipulation", const=True)

    # pylint: disable=arguments-differ,invalid-name
    def to_solver(self, params, **kwargs) -> ExpressionRestartInitialCondition:
        return super().to_solver(self, exclude=["type"], **kwargs)


InitialConditions = Union[
    FreestreamInitialCondition, ExpressionInitialCondition, ExpressionRestartInitialCondition
]
