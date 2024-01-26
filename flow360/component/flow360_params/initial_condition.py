"""
Initial condition parameters
"""

from __future__ import annotations

from typing import Dict, Optional

import pydantic as pd

from .params_base import Flow360BaseModel


class InitialCondition(Flow360BaseModel):
    """:class:`InitialCondition` class"""

    constants: Optional[Dict[str, str]]


class ExpressionInitialCondition(InitialCondition):
    """:class:`ExpressionInitialCondition` class"""

    rho: Optional[str] = pd.Field(displayed="rho [non-dim]")
    u: Optional[str] = pd.Field(displayed="u [non-dim]")
    v: Optional[str] = pd.Field(displayed="v [non-dim]")
    w: Optional[str] = pd.Field(displayed="w [non-dim]")
    p: Optional[str] = pd.Field(displayed="p [non-dim]")


InitialConditions = ExpressionInitialCondition
