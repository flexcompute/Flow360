"""
Initial condition parameters
"""

from __future__ import annotations

from typing import Dict, Optional, Union

import pydantic.v1 as pd
from typing_extensions import Literal

from flow360.component.utils import process_expressions
from flow360.component.v1.flow360_legacy import LegacyModel
from flow360.component.v1.params_base import Flow360BaseModel


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


class ExpressionInitialConditionLegacy(LegacyModel):
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

    def update_model(self) -> ExpressionInitialCondition:
        model = {
            "constants": self.constants,
            "rho": self.rho,
            "u": self.u,
            "v": self.v,
            "w": self.w,
            "p": self.p,
        }
        return ExpressionInitialCondition.parse_obj(model)


class ModifiedRestartSolutionLegacy(ExpressionInitialConditionLegacy):
    """:class:`ModifiedRestartSolution` class.
    For forked (restart) cases, expressions will be applied  to manipulate the restart solution before it is applied.
    """

    type: Literal["restartManipulation"] = pd.Field("restartManipulation", const=True)

    def update_model(self) -> ModifiedRestartSolution:
        model = {
            "constants": self.constants,
            "rho": self.rho,
            "u": self.u,
            "v": self.v,
            "w": self.w,
            "p": self.p,
        }
        return ModifiedRestartSolution.parse_obj(model)


InitialConditions = Union[ModifiedRestartSolution, ExpressionInitialCondition]
