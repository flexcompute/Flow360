"""
Contains basic components(solvers) that composes the `volume` type models.
Each volume model represents a physical phenomena that require a combination of solver features to model.

E.g. 
NavierStokes, turbulence and transition composes FluidDynamics `volume` type

"""

from __future__ import annotations

from abc import ABCMeta
from typing import Annotated, Literal, Optional, Union

import numpy as np
import pydantic as pd
from pydantic import NonNegativeFloat, NonNegativeInt, PositiveFloat, PositiveInt
from typing_extensions import Self

from flow360.component.simulation.framework.base_model import (
    Conflicts,
    Flow360BaseModel,
)
from flow360.component.simulation.framework.entity_base import EntityList
from flow360.component.simulation.primitives import Box

# from .time_stepping import UnsteadyTimeStepping

HEAT_EQUATION_EVAL_MAX_PER_PSEUDOSTEP_UNSTEADY = 40
HEAT_EQUATION_EVALUATION_FREQUENCY_STEADY = 10


class LinearSolver(Flow360BaseModel):
    """:class:`LinearSolver` class for setting up linear solver for heat equation


    Parameters
    ----------

    max_iterations : PositiveInt, optional
        Maximum number of linear solver iterations, by default 50

    absolute_tolerance : PositiveFloat, optional
        The linear solver converges when the final residual of the pseudo steps below this value. Either absolute
        tolerance or relative tolerance can be used to determine convergence, by default 1e-10

    relative_tolerance :
        The linear solver converges when the ratio of the final residual and the initial
        residual of the pseudo step is below this value.

    validation: tolerance settings only available to HeatEquationSolver

    Returns
    -------
    :class:`LinearSolver`
        An instance of the component class LinearSolver.


    Example
    -------
    >>> ls = LinearSolver(
                max_iterations=50,
                absoluteTolerance=1e-10
        )
    """

    max_iterations: PositiveInt = pd.Field(30)
    absolute_tolerance: Optional[PositiveFloat] = pd.Field(None)
    relative_tolerance: Optional[PositiveFloat] = pd.Field(None)

    model_config = pd.ConfigDict(
        conflicting_fields=[Conflicts(field1="absolute_tolerance", field2="relative_tolerance")]
    )


class GenericSolverSettings(Flow360BaseModel, metaclass=ABCMeta):
    """:class:`GenericSolverSettings` class"""

    absolute_tolerance: PositiveFloat = pd.Field(1.0e-10)
    relative_tolerance: NonNegativeFloat = pd.Field(0)
    order_of_accuracy: Literal[1, 2] = pd.Field(2)
    equation_evaluation_frequency: PositiveInt = pd.Field(1)
    linear_solver: LinearSolver = pd.Field(LinearSolver())


class NavierStokesSolver(GenericSolverSettings):
    """:class:`NavierStokesSolver` class for setting up compressible Navier-Stokes solver

    Parameters
    ----------

    absolute_tolerance :
        Tolerance for the NS residual, below which the solver goes to the next physical step

    relative_tolerance :
        Tolerance to the relative residual, below which the solver goes to the next physical step. Relative residual is
        defined as the ratio of the current pseudoStep’s residual to the maximum residual present in the first
        10 pseudoSteps within the current physicalStep. NOTE: relativeTolerance is ignored in steady simulations and
        only absoluteTolerance is used as the convergence criterion

    CFL_multiplier :
        Factor to the CFL definitions defined in “timeStepping” section

    kappa_MUSCL :
        Kappa for the MUSCL scheme, range from [-1, 1], with 1 being unstable. The default value of -1 leads to a 2nd
        order upwind scheme and is the most stable. A value of 0.33 leads to a blended upwind/central scheme and is
        recommended for low subsonic flows leading to reduced dissipation

    update_jacobian_frequency :
        Frequency at which the jacobian is updated.

    equation_evaluation_frequency :
        Frequency at which to update the compressible NS equation in loosely-coupled simulations

    max_force_jac_update_physical_steps :
        When which physical steps, the jacobian matrix is updated every pseudo step

    order_of_accuracy :
        Order of accuracy in space

    limit_velocity :
        Limiter for velocity

    limit_pressure_density :
        Limiter for pressure and density

    numerical_dissipation_factor :
        A factor in the range [0.01, 1.0] which exponentially reduces the dissipation of the numerical flux.
        The recommended starting value for most low-dissipation runs is 0.2

    linear_solver:
        Linear solver settings

    low_mach_preconditioner:
        Uses preconditioning for accelerating low Mach number flows.

    low_mach_preconditioner_threshold:
        For flow regions with Mach numbers smaller than threshold, the input Mach number to the preconditioner is
        assumed to be the threshold value if it is smaller than the threshold.
        The default value for the threshold is the freestream Mach number.

    Returns
    -------
    :class:`NavierStokesSolver`
        An instance of the component class NavierStokesSolver.

    Example
    -------
    >>> ns = NavierStokesSolver(absolute_tolerance=1e-10)
    """

    absolute_tolerance: PositiveFloat = pd.Field(1.0e-10)

    CFL_multiplier: PositiveFloat = pd.Field(1.0)
    kappa_MUSCL: pd.confloat(ge=-1, le=1) = pd.Field(-1)

    numerical_dissipation_factor: pd.confloat(ge=0.01, le=1) = pd.Field(1)
    limit_velocity: bool = pd.Field(False)
    limit_pressure_density: bool = pd.Field(False)

    type_name: Literal["Compressible"] = pd.Field("Compressible", frozen=True)

    low_mach_preconditioner: bool = pd.Field(False)
    low_mach_preconditioner_threshold: Optional[NonNegativeFloat] = pd.Field(None)

    update_jacobian_frequency: PositiveInt = pd.Field(4)
    max_force_jac_update_physical_steps: NonNegativeInt = pd.Field(0)


class SpalartAllmarasModelConstants(Flow360BaseModel):
    """:class:`SpalartAllmarasModelConstants` class"""

    type_name: Literal["SpalartAllmarasConsts"] = pd.Field("SpalartAllmarasConsts", frozen=True)
    C_DES: NonNegativeFloat = pd.Field(0.72)
    C_d: NonNegativeFloat = pd.Field(8.0)


class KOmegaSSTModelConstants(Flow360BaseModel):
    """:class:`KOmegaSSTModelConstants` class"""

    type_name: Literal["kOmegaSSTConsts"] = pd.Field("kOmegaSSTConsts", frozen=True)
    C_DES1: NonNegativeFloat = pd.Field(0.78)
    C_DES2: NonNegativeFloat = pd.Field(0.61)
    C_d1: NonNegativeFloat = pd.Field(20.0)
    C_d2: NonNegativeFloat = pd.Field(3.0)


TurbulenceModelConstants = Annotated[
    Union[SpalartAllmarasModelConstants, KOmegaSSTModelConstants],
    pd.Field(discriminator="type_name"),
]


class TurbulenceModelSolver(GenericSolverSettings, metaclass=ABCMeta):
    """:class:`TurbulenceModelSolver` class for setting up turbulence model solver

    Parameters
    ----------
    absoluteTolerance :
        Tolerance for the NS residual, below which the solver goes to the next physical step

    relativeTolerance :
        Tolerance to the relative residual, below which the solver goes to the next physical step. Relative residual is
        defined as the ratio of the current pseudoStep’s residual to the maximum residual present in the first
        10 pseudoSteps within the current physicalStep. NOTE: relativeTolerance is ignored in steady simulations and
        only absoluteTolerance is used as the convergence criterion

    CFL_multiplier :
        Factor to the CFL definitions defined in “timeStepping” section

    linearIterations :
        Number of linear solver iterations

    updateJacobianFrequency :
        Frequency at which the jacobian is updated.

    equationEvalFrequency :
        Frequency at which to update the NS equation in loosely-coupled simulations

    maxForceJacUpdatePhysicalSteps :
        When which physical steps, the jacobian matrix is updated every pseudo step

    orderOfAccuracy :
        Order of accuracy in space

    reconstruction_gradient_limiter :
        The strength of gradient limiter used in reconstruction of solution variables at the faces (specified in the
        range [0.0, 2.0]). 0.0 corresponds to setting the gradient equal to zero, and 2.0 means no limiting.

    quadratic_constitutive_relation : bool, optional
        Use quadratic constitutive relation for turbulence shear stress tensor instead of Boussinesq Approximation

    DDES : bool, optional
        Enables Delayed Detached Eddy Simulation. Supported for both SpalartAllmaras and kOmegaSST turbulence models,
        with and without AmplificationFactorTransport transition model enabled.

    grid_size_for_LES : Literal['maxEdgeLength', 'meanEdgeLength'], optional
        Specifes the length used for the computation of LES length scale. The allowed inputs are "maxEdgeLength"
        (default) and "meanEdgeLength"

    modeling_constants :
        Here, user can change the default values used for DDES coefficients in the solver:
        SpalartAllmaras: "C_DES" (= 0.72), "C_d" (= 8.0)
        kOmegaSST: "C_DES1" (= 0.78), "C_DES2" (= 0.61), "C_d1" (= 20.0), "C_d2" (= 3.0)
        (values shown in the parentheses are the default values used in Flow360)
        An example with kOmegaSST mode would be: {"C_DES1": 0.85, "C_d1": 8.0}

    Returns
    -------
    :class:`TurbulenceModelSolver`
        An instance of the component class TurbulenceModelSolver.

    Example
    -------
    >>> ts = TurbulenceModelSolver(absolute_tolerance=1e-10)
    """

    CFL_multiplier: PositiveFloat = pd.Field(2.0)
    type_name: str = pd.Field()
    absolute_tolerance: PositiveFloat = pd.Field(1e-8)
    equation_evaluation_frequency: PositiveInt = pd.Field(4)
    DDES: bool = pd.Field(False)
    grid_size_for_LES: Literal["maxEdgeLength", "meanEdgeLength"] = pd.Field("maxEdgeLength")
    reconstruction_gradient_limiter: pd.confloat(ge=0, le=2) = pd.Field(1.0)
    quadratic_constitutive_relation: bool = pd.Field(False)
    modeling_constants: Optional[TurbulenceModelConstants] = pd.Field(discriminator="type_name")
    update_jacobian_frequency: PositiveInt = pd.Field(4)
    max_force_jac_update_physical_steps: NonNegativeInt = pd.Field(0)

    linear_solver: LinearSolver = pd.Field(LinearSolver(max_iterations=20))


class KOmegaSST(TurbulenceModelSolver):
    """:class:`KOmegaSST` class"""

    type_name: Literal["kOmegaSST"] = pd.Field("kOmegaSST", frozen=True)
    modeling_constants: KOmegaSSTModelConstants = pd.Field(KOmegaSSTModelConstants())


class SpalartAllmaras(TurbulenceModelSolver):
    """:class:`SpalartAllmaras` class"""

    type_name: Literal["SpalartAllmaras"] = pd.Field("SpalartAllmaras", frozen=True)
    rotation_correction: bool = pd.Field(False)

    modeling_constants: Optional[SpalartAllmarasModelConstants] = pd.Field(
        SpalartAllmarasModelConstants()
    )
    reconstruction_gradient_limiter: Optional[pd.confloat(ge=0, le=2)] = pd.Field(0.5)


class NoneSolver(Flow360BaseModel):
    """:class:`SolverNone` class"""

    type_name: Literal["None"] = pd.Field("None", frozen=True)


TurbulenceModelSolverType = Annotated[
    Union[NoneSolver, SpalartAllmaras, KOmegaSST], pd.Field(discriminator="type_name")
]


class HeatEquationSolver(GenericSolverSettings):
    """:class:`HeatEquationSolver` class for setting up heat equation solver.


    Parameters
    ----------

    equation_evaluation_frequency : PositiveInt, optional
        Frequency at which to solve the heat equation in conjugate heat transfer simulations


    linear_solver_config : LinearSolver, optional
        Linear solver settings, see LinearSolver documentation.

    Returns
    -------
    :class:`HeatEquationSolver`
        An instance of the component class HeatEquationSolver.


    Example
    -------
    >>> he = HeatEquationSolver(
                equation_evaluation_frequency=10,
                linear_solver_config=LinearSolver(
                    max_iterations=50,
                    absoluteTolerance=1e-10
            )
        )
    """

    type_name: Literal["HeatEquation"] = pd.Field("HeatEquation", frozen=True)
    absolute_tolerance: PositiveFloat = pd.Field(1e-9)
    equation_evaluation_frequency: PositiveInt = pd.Field(10)
    order_of_accuracy: Literal[2] = pd.Field(2)

    linear_solver: LinearSolver = pd.Field(
        LinearSolver(max_iterations=50, absolute_tolerance=1e-10)
    )


class TransitionModelSolver(GenericSolverSettings):
    """:class:`TransitionModelSolver` class for setting up transition model solver

    Parameters
    ----------

    (...)

    Returns
    -------
    :class:`TransitionModelSolver`
        An instance of the component class TransitionModelSolver.

    Example
    -------
    >>> ts = TransitionModelSolver(absolute_tolerance=1e-10)
    """

    type_name: Literal["AmplificationFactorTransport"] = pd.Field(
        "AmplificationFactorTransport", frozen=True
    )
    CFL_multiplier: PositiveFloat = pd.Field(2.0)
    absolute_tolerance: PositiveFloat = pd.Field(1e-7)
    equation_evaluation_frequency: PositiveInt = pd.Field(4)
    turbulence_intensity_percent: Optional[pd.confloat(ge=0.03, le=2.5)] = pd.Field(None)
    # pylint: disable=invalid-name
    N_crit: Optional[pd.confloat(ge=1.0, le=11.0)] = pd.Field(None)
    update_jacobian_frequency: PositiveInt = pd.Field(4)
    max_force_jac_update_physical_steps: NonNegativeInt = pd.Field(0)
    reconstruction_gradient_limiter: Optional[pd.confloat(ge=0.0, le=2.0)] = pd.Field(1.0)

    trip_regions: Optional[EntityList[Box]] = pd.Field(None)

    linear_solver: LinearSolver = pd.Field(LinearSolver(max_iterations=20))

    @pd.model_validator(mode="after")
    def _set_aft_ncrit(self) -> Self:
        """
        Compute the critical amplification factor for AFT transition solver based on
        input turbulence intensity and input Ncrit. Computing Ncrit from turbulence
        intensity takes priority if both are specified.
        """

        if self.turbulence_intensity_percent is not None:
            ncrit_converted = -8.43 - 2.4 * np.log(
                0.025 * np.tanh(self.turbulence_intensity_percent / 2.5)
            )
            self.turbulence_intensity_percent = None
            self.N_crit = ncrit_converted
        elif self.N_crit is None:
            self.N_crit = 8.15

        return self


TransitionModelSolverType = Annotated[
    Union[NoneSolver, TransitionModelSolver], pd.Field(discriminator="type_name")
]
