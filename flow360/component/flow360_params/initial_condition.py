"""
Initial condition parameters
"""

from __future__ import annotations

from typing import Dict, Optional, Union

import pydantic as pd
from typing_extensions import Literal

from ..utils import process_expressions
from .params_base import Flow360BaseModel


class InitialCondition(Flow360BaseModel):
    """:class:`InitialCondition` class"""

    type: str


class ExpressionInitialCondition(InitialCondition):
    """:class:`ExpressionInitialCondition` class"""

    type: Literal["expression"] = pd.Field("expression", const=True)
    constants: Optional[Dict[str, str]] = pd.Field(alias="constants")
    rho: Optional[str] = pd.Field(displayed="rho [non-dim]")
    u: Optional[str] = pd.Field(displayed="u [non-dim]")
    v: Optional[str] = pd.Field(displayed="v [non-dim]")
    w: Optional[str] = pd.Field(displayed="w [non-dim]")
    p: Optional[str] = pd.Field(displayed="p [non-dim]")

    _processed_rho = pd.validator("rho", allow_reuse=True)(process_expressions)
    _processed_u = pd.validator("u", allow_reuse=True)(process_expressions)
    _processed_v = pd.validator("v", allow_reuse=True)(process_expressions)
    _processed_w = pd.validator("w", allow_reuse=True)(process_expressions)
    _processed_p = pd.validator("p", allow_reuse=True)(process_expressions)


class ModifiedRestartSolution(ExpressionInitialCondition):
    """:class:`ModifiedRestartSolution` class.
    For forked (restart) cases, expressions will be applied  to manipulate the restart solution before it is applied.
    """

    type: Literal["restartManipulation"] = pd.Field("restartManipulation", const=True)


InitialConditions = Union[ModifiedRestartSolution, ExpressionInitialCondition]
