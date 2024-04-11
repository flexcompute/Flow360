"""
Contains basic components that composes the `volume` types. Each volume represents a physical phenomena that require a combination of solver features to model.

E.g. 
NavierStokes, turbulence and transition composes FluidDynamics `volume` type

From what I can think of right now most can be reused from flow360_params for example the BETDisk and TransitionModelSolver.
"""

from typing import List, Optional, Union

import pydantic as pd

from flow360.component.flow360_params.flow360_params import (
    ActuatorDisk,
    BETDisk,
    PorousMediumBox,
)
from flow360.component.flow360_params.params_base import Flow360BaseModel


class NavierStokesSolver(Flow360BaseModel):
    pass


class KOmegaSST(Flow360BaseModel):
    pass


class SpalartAllmaras(Flow360BaseModel):
    pass


class TransitionModelSolver(Flow360BaseModel):
    pass
