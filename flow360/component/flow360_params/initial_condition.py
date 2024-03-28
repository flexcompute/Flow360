"""
Initial condition parameters
"""

from __future__ import annotations

from typing import Dict, Optional, Union

import pydantic as pd
from typing_extensions import Literal

from ..utils import process_expression
from .params_base import Flow360BaseModel


class InitialCondition(Flow360BaseModel):
    """:class:`InitialCondition` class"""

    type: str


class FreestreamInitialCondition(InitialCondition):
    """:class:`FreestreamInitialCondition` class"""

    type: Literal["freestream"] = pd.Field("freestream", const=True)


class ExpressionInitialCondition(InitialCondition):
    """:class:`ExpressionInitialCondition` class"""

    type: Literal["expression"] = pd.Field("expression", const=True)
    constants: Optional[Dict[str, str]] = pd.Field(alias="constants")
    rho: Optional[str] = pd.Field(displayed="rho [non-dim]")
    u: Optional[str] = pd.Field(displayed="u [non-dim]")
    v: Optional[str] = pd.Field(displayed="v [non-dim]")
    w: Optional[str] = pd.Field(displayed="w [non-dim]")
    p: Optional[str] = pd.Field(displayed="p [non-dim]")

    # pylint: disable=arguments-differ
    def to_solver(self, params, **kwargs) -> ExpressionInitialCondition:
        """
        process the expressions
        """
        for attr_name in ["rho", "u", "v", "w", "p"]:
            expr = getattr(self, attr_name)
            if expr is None:
                continue
            expr = str(process_expression(expr))
            setattr(self, attr_name, expr)
        return super().to_solver(params, **kwargs)


InitialConditions = Union[FreestreamInitialCondition, ExpressionInitialCondition]
