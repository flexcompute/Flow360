"""
Flow360 solvers parameters
"""
from __future__ import annotations

from abc import ABCMeta
from typing import List, Optional, Union

import pydantic as pd
from typing_extensions import Literal

from ..types import NonNegativeFloat, NonNegativeInt, PositiveFloat, PositiveInt
from .params_base import DeprecatedAlias, Flow360BaseModel


class GenericFlowSolverSettings(Flow360BaseModel, metaclass=ABCMeta):
    """:class:`GenericFlowSolverSettings` class"""

    absolute_tolerance: Optional[PositiveFloat] = pd.Field(alias="absoluteTolerance")
    relative_tolerance: Optional[float] = pd.Field(0, alias="relativeTolerance")

    # pylint: disable=missing-class-docstring,too-few-public-methods
    class Config(Flow360BaseModel.Config):
        deprecated_aliases = [
            DeprecatedAlias(name="linear_solver", deprecated="linearSolverConfig"),
            DeprecatedAlias(name="absolute_tolerance", deprecated="tolerance"),
        ]


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

    max_iterations: Optional[PositiveInt] = pd.Field(alias="maxIterations", default=50)
    absolute_tolerance: Optional[PositiveFloat] = pd.Field(alias="absoluteTolerance", default=1e-10)
    relative_tolerance: Optional[float] = pd.Field(alias="relativeTolerance")

    # pylint: disable=missing-class-docstring,too-few-public-methods
    class Config(Flow360BaseModel.Config):
        deprecated_aliases = [DeprecatedAlias(name="absolute_tolerance", deprecated="tolerance")]


class RandomizerParameter(Flow360BaseModel):
    """:class:`RandomizerParameter` class"""

    standard_deviation: NonNegativeFloat = pd.Field(alias="standardDeviation")
    update_frequency: PositiveInt = pd.Field(alias="updateFrequency")


class LinearIterationsRandomizer(Flow360BaseModel):
    """:class:`LinearIterationsRandomizer` class"""

    linear_iterations: RandomizerParameter = pd.Field(alias="linearIterations")


class PressureCorrectionSolver(Flow360BaseModel):
    """:class:`PressureCorrectionSolver` class"""

    randomizer: LinearIterationsRandomizer = pd.Field()
    linear_solver: LinearSolver = pd.Field(alias="linearSolver", default=LinearSolver())

    # pylint: disable=missing-class-docstring,too-few-public-methods
    class Config(Flow360BaseModel.Config):
        deprecated_aliases = [
            DeprecatedAlias(name="linear_solver", deprecated="linearSolverConfig")
        ]


class NavierStokesSolver(GenericFlowSolverSettings):
    """:class:`NavierStokesSolver` class for setting up Navier-Stokes solver

    Parameters
    ----------

    absoluteTolerance :
        Tolerance for the NS residual, below which the solver goes to the next physical step

    relativeTolerance :
        Tolerance to the relative residual, below which the solver goes to the next physical step. Relative residual is
        defined as the ratio of the current pseudoStep’s residual to the maximum residual present in the first
        10 pseudoSteps within the current physicalStep. NOTE: relativeTolerance is ignored in steady simulations and
        only absoluteTolerance is used as the convergence criterion

    CFLMultiplier :
        Factor to the CFL definitions defined in “timeStepping” section

    kappa_MUSCL :
        Kappa for the MUSCL scheme, range from [-1, 1], with 1 being unstable. The default value of -1 leads to a 2nd
        order upwind scheme and is the most stable. A value of 0.33 leads to a blended upwind/central scheme and is
        recommended for low subsonic flows leading to reduced dissipation

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

    limitVelocity :
        Limiter for velocity

    limitPressureDensity :
        Limiter for pressure and density

    numericalDissipationFactor :
        A factor in the range [0.01, 1.0] which exponentially reduces the dissipation of the numerical flux.
        The recommended starting value for most low-dissipation runs is 0.2

    Returns
    -------
    :class:`NavierStokesSolver`
        An instance of the component class NavierStokesSolver.

    Example
    -------
    >>> ns = NavierStokesSolver(absolute_tolerance=1e-10)
    """

    CFL_multiplier: Optional[PositiveFloat] = pd.Field(
        1.0, alias="CFLMultiplier", displayed="CFL Multiplier"
    )
    linear_iterations: Optional[PositiveInt] = pd.Field(alias="linearIterations")
    kappa_MUSCL: Optional[pd.confloat(ge=-1, le=1)] = pd.Field(
        -1, alias="kappaMUSCL", displayed="Kappa MUSCL"
    )
    update_jacobian_frequency: Optional[PositiveInt] = pd.Field(
        4, alias="updateJacobianFrequency", displayed="Update Jacobian frequency"
    )
    equation_eval_frequency: Optional[PositiveInt] = pd.Field(1, alias="equationEvalFrequency")
    max_force_jac_update_physical_steps: Optional[NonNegativeInt] = pd.Field(
        0, alias="maxForceJacUpdatePhysicalSteps", displayed="Max force JAC update physical steps"
    )
    order_of_accuracy: Optional[Literal[1, 2]] = pd.Field(2, alias="orderOfAccuracy")
    numerical_dissipation_factor: Optional[pd.confloat(ge=0.01, le=1)] = pd.Field(
        1, alias="numericalDissipationFactor"
    )
    linear_solver: Optional[LinearSolver] = pd.Field(
        LinearSolver(max_iterations=30), alias="linearSolver"
    )
    limit_velocity: Optional[bool] = pd.Field(False, alias="limitVelocity")
    limit_pressure_density: Optional[bool] = pd.Field(False, alias="limitPressureDensity")

    @classmethod
    def _get_field_order(cls) -> List[str]:
        return ["*", "linearSolver"]


class TurbulenceModelConstants(Flow360BaseModel):
    """:class:`TurbulenceModelConstants` class"""

    C_DES: Optional[float]
    C_d: Optional[float]
    C_DES1: Optional[float]
    C_DES2: Optional[float]
    C_d1: Optional[float]
    C_d2: Optional[float]


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

    rotation_correction : bool, optional
        Rotation correction for the turbulence model. Only support for SpalartAllmaras

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

    model_type: str = pd.Field(alias="modelType")
    absolute_tolerance: Optional[PositiveFloat] = pd.Field(1e-8, alias="absoluteTolerance")
    linear_solver: Optional[LinearSolver] = pd.Field(
        LinearSolver(max_iterations=20), alias="linearSolver"
    )
    update_jacobian_frequency: Optional[PositiveInt] = pd.Field(4, alias="updateJacobianFrequency")
    equation_eval_frequency: Optional[PositiveInt] = pd.Field(4, alias="equationEvalFrequency")
    max_force_jac_update_physical_steps: Optional[NonNegativeInt] = pd.Field(
        0, alias="maxForceJacUpdatePhysicalSteps"
    )
    order_of_accuracy: Optional[Literal[1, 2]] = pd.Field(2, alias="orderOfAccuracy")
    DDES: Optional[bool] = pd.Field(False, alias="DDES")
    grid_size_for_LES: Optional[Literal["maxEdgeLength", "meanEdgeLength"]] = pd.Field(
        "maxEdgeLength", alias="gridSizeForLES"
    )
    quadratic_constitutive_relation: Optional[bool] = pd.Field(
        False, alias="quadraticConstitutiveRelation"
    )
    reconstruction_gradient_limiter: Optional[pd.confloat(ge=0, le=2)] = pd.Field(
        1.0, alias="reconstructionGradientLimiter"
    )
    model_constants: Optional[TurbulenceModelConstants] = pd.Field(alias="modelConstants")


class TurbulenceModelSolverSST(TurbulenceModelSolver):
    """:class:`TurbulenceModelSolverSST` class"""

    model_type: Literal["kOmegaSST"] = pd.Field("kOmegaSST", alias="modelType", const=True)


class TurbulenceModelSolverSA(TurbulenceModelSolver):
    """:class:`TurbulenceModelSolverSA` class"""

    model_type: Literal["SpalartAllmaras"] = pd.Field(
        "SpalartAllmaras", alias="modelType", const=True
    )
    rotation_correction: Optional[bool] = pd.Field(False, alias="rotationCorrection")


class NoneSolver(Flow360BaseModel):
    """:class:`SolverNone` class"""

    model_type: Literal["None"] = pd.Field("None", alias="modelType", const=True)


TurbulenceModelSolverTypes = Union[TurbulenceModelSolverSA, TurbulenceModelSolverSST, NoneSolver]


class HeatEquationSolver(Flow360BaseModel):
    """:class:`HeatEquationSolver` class for setting up heat equation solver.


    Parameters
    ----------

    equation_eval_frequency : PositiveInt, optional
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
                equation_eval_frequency=10,
                linear_solver_config=LinearSolver(
                    max_iterations=50,
                    absoluteTolerance=1e-10
            )
        )
    """

    absolute_tolerance: Optional[PositiveFloat] = pd.Field(alias="absoluteTolerance")
    equation_eval_frequency: Optional[PositiveInt] = pd.Field(alias="equationEvalFrequency")
    linear_solver: Optional[LinearSolver] = pd.Field(alias="linearSolver", default=LinearSolver())

    # pylint: disable=missing-class-docstring,too-few-public-methods
    class Config(Flow360BaseModel.Config):
        deprecated_aliases = [
            DeprecatedAlias(name="linear_solver", deprecated="linearSolverConfig"),
            DeprecatedAlias(name="absolute_tolerance", deprecated="tolerance"),
        ]


class TransitionModelSolver(GenericFlowSolverSettings):
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

    model_type: Optional[Literal["AmplificationFactorTransport"]] = pd.Field(
        "AmplificationFactorTransport", alias="modelType", const=True
    )
    linear_solver: Optional[LinearSolver] = pd.Field(alias="linearSolver", default=LinearSolver())
    update_jacobian_frequency: Optional[PositiveInt] = pd.Field(alias="updateJacobianFrequency")
    equation_eval_frequency: Optional[PositiveInt] = pd.Field(alias="equationEvalFrequency")
    max_force_jac_update_physical_steps: Optional[NonNegativeInt] = pd.Field(
        alias="maxForceJacUpdatePhysicalSteps"
    )
    order_of_accuracy: Optional[Literal[1, 2]] = pd.Field(alias="orderOfAccuracy")
    turbulence_intensity_percent: Optional[PositiveFloat] = pd.Field(
        alias="turbulenceIntensityPercent"
    )
    N_crit: Optional[PositiveFloat] = pd.Field(alias="Ncrit")
