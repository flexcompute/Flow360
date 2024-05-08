"""
Contains basic components that composes the `volume` types. Each volume represents a physical phenomena that require a combination of solver features to model.

E.g. 
NavierStokes, turbulence and transition composes FluidDynamics `volume` type

From what I can think of right now most can be reused from flow360_params for example the BETDisk and TransitionModelSolver.
"""

from abc import ABCMeta
from typing import Literal, Optional, Union

import pydantic as pd
from typing_extensions import Annotated

from flow360.component.simulation.base_model import Flow360BaseModel
from flow360.component_v2.types import (
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
)


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

    max_iterations: Optional[PositiveInt] = pd.Field(default=50)
    absolute_tolerance: Optional[PositiveFloat] = pd.Field(None)
    relative_tolerance: Optional[PositiveFloat] = pd.Field(None)

    # TODO: Conflict absolute_tolerance, relative_tolerance


class GenericFlowSolverSettings(Flow360BaseModel, metaclass=ABCMeta):
    """:class:`GenericFlowSolverSettings` class"""

    absolute_tolerance: Optional[PositiveFloat] = pd.Field(1.0e-10)
    relative_tolerance: Optional[NonNegativeFloat] = pd.Field(0)
    order_of_accuracy: Optional[Literal[1, 2]] = pd.Field(2)
    CFL_multiplier: Optional[PositiveFloat] = pd.Field(2.0)
    update_jacobian_frequency: Optional[PositiveInt] = pd.Field(4)
    max_force_jac_update_physical_steps: Optional[NonNegativeInt] = pd.Field(0)


class NavierStokesSolver(GenericFlowSolverSettings):
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

    equation_eval_frequency :
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

    CFL_multiplier: Optional[PositiveFloat] = pd.Field(1.0)
    # kappa_MUSCL: Optional[pd.confloat(ge=-1, le=1)] = pd.Field(
    kappa_MUSCL: Optional[float] = pd.Field(-1, ge=-1, le=1)
    equation_eval_frequency: Optional[PositiveInt] = pd.Field(1)

    numerical_dissipation_factor: Optional[float] = pd.Field(1, ge=0.01, le=1)
    limit_velocity: Optional[bool] = pd.Field(False)
    limit_pressure_density: Optional[bool] = pd.Field(False)

    linear_solver: Optional[LinearSolver] = pd.Field(LinearSolver(max_iterations=30))

    model_type: Literal["Compressible"] = pd.Field("Compressible", frozen=True)

    low_mach_preconditioner: Optional[bool] = pd.Field(False)
    low_mach_preconditioner_threshold: Optional[NonNegativeFloat] = pd.Field(None)


class SpalartAllmarasModelConstants(Flow360BaseModel):
    """:class:`SpalartAllmarasModelConstants` class"""

    model_type: Literal["SpalartAllmarasConsts"] = pd.Field("SpalartAllmarasConsts", frozen=True)
    C_DES: Optional[NonNegativeFloat] = pd.Field(0.72)
    C_d: Optional[NonNegativeFloat] = pd.Field(8.0)


class KOmegaSSTModelConstants(Flow360BaseModel):
    """:class:`KOmegaSSTModelConstants` class"""

    model_type: Literal["kOmegaSSTConsts"] = pd.Field("kOmegaSSTConsts", frozen=True)
    C_DES1: Optional[NonNegativeFloat] = pd.Field(0.78)
    C_DES2: Optional[NonNegativeFloat] = pd.Field(0.61)
    C_d1: Optional[NonNegativeFloat] = pd.Field(20.0)
    C_d2: Optional[NonNegativeFloat] = pd.Field(3.0)


TurbulenceModelConstants = Union[SpalartAllmarasModelConstants, KOmegaSSTModelConstants]


class TurbulenceModelSolver(GenericFlowSolverSettings, metaclass=ABCMeta):
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

    model_constants :
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

    model_type: str = pd.Field()
    absolute_tolerance: Optional[PositiveFloat] = pd.Field(1e-8)
    equation_eval_frequency: Optional[PositiveInt] = pd.Field(4)
    DDES: Optional[bool] = pd.Field(False)
    grid_size_for_LES: Optional[Literal["maxEdgeLength", "meanEdgeLength"]] = pd.Field(
        "maxEdgeLength"
    )
    reconstruction_gradient_limiter: Optional[float] = pd.Field(ge=0, le=2)
    quadratic_constitutive_relation: Optional[bool] = pd.Field(False)
    model_constants: Optional[TurbulenceModelConstants] = pd.Field(discriminator="model_type")

    linear_solver: Optional[LinearSolver] = pd.Field(LinearSolver(max_iterations=20))


class KOmegaSST(TurbulenceModelSolver):
    """:class:`KOmegaSST` class"""

    model_type: Literal["kOmegaSST"] = pd.Field("kOmegaSST", frozen=True)
    model_constants: Optional[KOmegaSSTModelConstants] = pd.Field(default=KOmegaSSTModelConstants())
    reconstruction_gradient_limiter: Optional[float] = pd.Field(1.0, ge=0, le=2)


class SpalartAllmaras(TurbulenceModelSolver):
    """:class:`SpalartAllmaras` class"""

    model_type: Literal["SpalartAllmaras"] = pd.Field("SpalartAllmaras", frozen=True)
    rotation_correction: Optional[bool] = pd.Field(False)

    model_constants: Optional[SpalartAllmarasModelConstants] = pd.Field(
        default=SpalartAllmarasModelConstants()
    )
    reconstruction_gradient_limiter: Optional[float] = pd.Field(0.5, ge=0, le=2)


class TransitionModelSolver(Flow360BaseModel):
    pass


class HeatEquationSolver(Flow360BaseModel):
    pass


class ActuatorDisk(Flow360BaseModel):
    pass


class BETDisk(Flow360BaseModel):
    pass


class PorousMediumBox(Flow360BaseModel):
    pass


TurbulenceModelSolverType = Union[KOmegaSST, SpalartAllmaras]
