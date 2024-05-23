from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.volume_models.components import (
    NavierStokesSolver,
    TurbulenceModelSolverType,
    TransitionModelSolver,
)
from flow360.component.simulation.material.material import Material
import pydantic as pd
from typing import Optional


class MaterialModelBase(Flow360BaseModel):
    """
    Subclass models of this class should be essentially specifying materials for the given entities.

    For now the initial_condition happens to make sense only for `FluidDynamics` and `SolidHeatTransfer` which are MaterialModelBase.
    We may need to move it out when we see fit.
    """

    material: Optional[Material] = pd.Field(None)
    initial_conditions: Optional[dict] = pd.Field(None)


class FluidDynamics(MaterialModelBase):
    """
    General FluidDynamics volume model that contains all the common fields every fluid dynamics zone should have.
    """

    navier_stokes_solver: Optional[NavierStokesSolver] = pd.Field(None)
    turbulence_model_solver: Optional[TurbulenceModelSolverType] = pd.Field(None)
    transition_model_solver: Optional[TransitionModelSolver] = pd.Field(None)

class SolidHeatTransfer(MaterialModelBase):
    """
    General SolidHeatTransfer volume model that contains all the common fields every solid heat transfer zone should have.
    """

    pass