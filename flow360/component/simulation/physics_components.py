"""
Contains basic components that composes the `volume` types. Each volume represents a physical phenomena that require a combination of solver features to model.

E.g. 
NavierStokes, turbulence and transition composes FluidDynamics `volume` type

From what I can think of right now most can be reused from flow360_params for example the BETDisk and TransitionModelSolver.
"""

from typing import Union

from flow360.component.simulation.base_model import Flow360BaseModel


class NavierStokesSolver(Flow360BaseModel):
    """Same as Flow360Param NavierStokesSolver."""

    pass


class KOmegaSST(Flow360BaseModel):
    """Same as Flow360Param KOmegaSST."""

    pass


class SpalartAllmaras(Flow360BaseModel):
    """Same as Flow360Param SpalartAllmaras."""

    pass


class TransitionModelSolver(Flow360BaseModel):
    """Same as Flow360Param TransitionModelSolver."""

    pass


class HeatEquationSolver(Flow360BaseModel):
    """Same as Flow360Param HeatEquationSolver."""

    pass


TurbulenceModelSolverType = Union[KOmegaSST, SpalartAllmaras]
