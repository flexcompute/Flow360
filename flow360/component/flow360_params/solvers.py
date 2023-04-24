"""
Flow360 solvers parameters
"""
from __future__ import annotations

from enum import Enum
from typing import Dict, Optional

import pydantic as pd
from typing_extensions import Literal

from ..types import NonNegativeInt, PositiveFloat, PositiveInt
from .params_base import DeprecatedAlias, Flow360BaseModel


class GenericSolverSettings(Flow360BaseModel):
    """:class:`GenericSolverSettings` class"""

    absolute_tolerance: Optional[PositiveFloat] = pd.Field(alias="absoluteTolerance")
    relative_tolerance: Optional[float] = pd.Field(alias="relativeTolerance")
    linear_iterations: Optional[PositiveInt] = pd.Field(alias="linearIterations")
    update_jacobian_frequency: Optional[PositiveInt] = pd.Field(alias="updateJacobianFrequency")
    equation_eval_frequency: Optional[PositiveInt] = pd.Field(alias="equationEvalFrequency")
    max_force_jac_update_physical_steps: Optional[NonNegativeInt] = pd.Field(
        alias="maxForceJacUpdatePhysicalSteps"
    )
    order_of_accuracy: Optional[Literal[1, 2]] = pd.Field(alias="orderOfAccuracy")
    kappaMUSCL: Optional[pd.confloat(ge=-1, le=1)] = pd.Field()
    randomizer: Optional[Dict] = pd.Field()

    # pylint: disable=missing-class-docstring,too-few-public-methods
    class Config(Flow360BaseModel.Config):
        deprecated_aliases = [DeprecatedAlias(name="absolute_tolerance", deprecated="tolerance")]


class NavierStokesSolver(GenericSolverSettings):
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

    kappaMUSCL :
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


    Returns
    -------
    :class:`NavierStokesSolver`
        An instance of the component class NavierStokesSolver.

    Example
    -------
    >>> ns = NavierStokesSolver(absolute_tolerance=1e-10)
    """

    CFL_multiplier: Optional[PositiveFloat] = pd.Field(alias="CFLMultiplier")
    limit_velocity: Optional[bool] = pd.Field(alias="limitVelocity")
    limit_pressure_density: Optional[bool] = pd.Field(alias="limitPressureDensity")
    extra_dissipation: Optional[float] = pd.Field(alias="extraDissipation")
    viscous_wave_speed_scale: Optional[float] = pd.Field(alias="viscousWaveSpeedScale")


class TurbulenceModelModelType(str, Enum):
    """Turbulence model type"""

    SA = "SpalartAllmaras"
    SST = "kOmegaSST"
    NONE = "None"


class TurbulenceModelConstants(Flow360BaseModel):
    """TurbulenceModelConstants"""

    C_DES: Optional[float]
    C_d: Optional[float]
    C_DES1: Optional[float]
    C_DES2: Optional[float]
    C_d1: Optional[float]
    C_d2: Optional[float]


class TurbulenceModelSolver(GenericSolverSettings):
    """:class:`TurbulenceModelSolver` class for setting up turbulence model solver

    Parameters
    ----------

    model_type :
        Turbulence model type can be: “SpalartAllmaras”, “kOmegaSST” or "None"

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
    >>> ts = TurbulenceModelSolver(model_type='SA', absolute_tolerance=1e-10)
    """

    model_type: TurbulenceModelModelType = pd.Field(
        alias="modelType", default=TurbulenceModelModelType.SA
    )
    reconstruction_gradient_limiter: Optional[pd.confloat(ge=0, le=2)] = pd.Field(
        alias="reconstructionGradientLimiter"
    )
    rotation_correction: Optional[bool] = pd.Field(alias="rotationCorrection")
    quadratic_constitutive_relation: Optional[bool] = pd.Field(
        alias="quadraticConstitutiveRelation"
    )
    DDES: Optional[bool]
    grid_size_for_LES: Optional[Literal["maxEdgeLength", "meanEdgeLength"]] = pd.Field(
        alias="gridSizeForLES"
    )
    model_constants: Optional[TurbulenceModelConstants] = pd.Field(alias="modelConstants")

    @pd.validator("model_type", pre=True, always=True)
    def validate_model_type(cls, v):
        """Turbulence model validator"""
        if v == "SA":
            return TurbulenceModelModelType.SA
        if v == "SST":
            return TurbulenceModelModelType.SST
        if v == "None":
            return TurbulenceModelModelType.NONE
        return v
