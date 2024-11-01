"""
Flow360 solvers parameters
"""

from __future__ import annotations

from abc import ABCMeta
from typing import List, Optional, Union

import numpy as np
import pydantic.v1 as pd
from typing_extensions import Literal

from flow360.component.types import Coordinate
from flow360.component.v1.flow360_legacy import (
    LegacyModel,
    LinearSolverLegacy,
    set_linear_solver_config_if_none,
    try_set,
    try_update,
)
from flow360.component.v1.params_base import (
    Conflicts,
    DeprecatedAlias,
    Flow360BaseModel,
)
from flow360.component.v1.time_stepping import UnsteadyTimeStepping
from flow360.flags import Flags

HEAT_EQUATION_EVAL_MAX_PER_PSEUDOSTEP_UNSTEADY = 40
HEAT_EQUATION_EVAL_FREQUENCY_STEADY = 10


class GenericFlowSolverSettings(Flow360BaseModel, metaclass=ABCMeta):
    """:class:`GenericFlowSolverSettings` class"""

    absolute_tolerance: Optional[pd.PositiveFloat] = pd.Field(1.0e-10, alias="absoluteTolerance")
    relative_tolerance: Optional[pd.NonNegativeFloat] = pd.Field(0, alias="relativeTolerance")
    order_of_accuracy: Optional[Literal[1, 2]] = pd.Field(2, alias="orderOfAccuracy")
    CFL_multiplier: Optional[pd.PositiveFloat] = pd.Field(
        2.0, alias="CFLMultiplier", displayed="CFL Multiplier"
    )
    update_jacobian_frequency: Optional[pd.PositiveInt] = pd.Field(
        4, alias="updateJacobianFrequency"
    )
    max_force_jac_update_physical_steps: Optional[pd.NonNegativeInt] = pd.Field(
        0, alias="maxForceJacUpdatePhysicalSteps", displayed="Max force JAC update physical steps"
    )

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

    absolute_tolerance :pd.PositiveFloat, optional
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

    max_iterations: Optional[pd.PositiveInt] = pd.Field(alias="maxIterations", default=50)
    absolute_tolerance: Optional[pd.PositiveFloat] = pd.Field(alias="absoluteTolerance")
    relative_tolerance: Optional[pd.PositiveFloat] = pd.Field(alias="relativeTolerance")

    # pylint: disable=missing-class-docstring,too-few-public-methods
    class Config(Flow360BaseModel.Config):
        deprecated_aliases = [DeprecatedAlias(name="absolute_tolerance", deprecated="tolerance")]
        conflicting_fields = [Conflicts(field1="absolute_tolerance", field2="relative_tolerance")]


class PressureCorrectionSolver(Flow360BaseModel):
    """:class:`PressureCorrectionSolver` class"""

    linear_solver: LinearSolver = pd.Field(
        LinearSolver(absoluteTolerance=1e-8), alias="linearSolver"
    )

    # pylint: disable=missing-class-docstring,too-few-public-methods
    class Config(Flow360BaseModel.Config):
        deprecated_aliases = [
            DeprecatedAlias(name="linear_solver", deprecated="linearSolverConfig")
        ]


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

    CFL_multiplier: Optional[pd.PositiveFloat] = pd.Field(
        1.0, alias="CFLMultiplier", displayed="CFL Multiplier"
    )
    kappa_MUSCL: Optional[pd.confloat(ge=-1, le=1)] = pd.Field(
        -1, alias="kappaMUSCL", displayed="Kappa MUSCL"
    )
    equation_eval_frequency: Optional[pd.PositiveInt] = pd.Field(1, alias="equationEvalFrequency")

    numerical_dissipation_factor: Optional[pd.confloat(ge=0.01, le=1)] = pd.Field(
        1, alias="numericalDissipationFactor"
    )
    limit_velocity: Optional[bool] = pd.Field(False, alias="limitVelocity")
    limit_pressure_density: Optional[bool] = pd.Field(False, alias="limitPressureDensity")

    linear_solver: Optional[LinearSolver] = pd.Field(
        LinearSolver(max_iterations=30), alias="linearSolver", displayed="Linear solver"
    )

    model_type: Literal["Compressible"] = pd.Field("Compressible", alias="modelType", const=True)

    low_mach_preconditioner: Optional[bool] = pd.Field(False, alias="lowMachPreconditioner")
    low_mach_preconditioner_threshold: Optional[pd.NonNegativeFloat] = pd.Field(
        alias="lowMachPreconditionerThreshold"
    )
    low_dissipation_control_factors: Optional[List[float]] = pd.Field(
        default=[], alias="lowDissipationControlFactors"
    )

    if Flags.beta_features():
        debug_type: Optional[
            Literal[
                "minDensity",
                "minPressure",
                "maxVelocity",
                "maxResCont",
                "maxResMomX",
                "maxResMomY",
                "maxResMomZ",
                "maxResEnergy",
            ]
        ] = pd.Field(alias="debugType")
        debug_point: Optional[Coordinate] = pd.Field(alias="debugPoint")

        # pylint: disable=missing-class-docstring,too-few-public-methods
        class Config(GenericFlowSolverSettings.Config):
            conflicting_fields = [Conflicts(field1="debug_type", field2="debug_point")]

    # pylint: disable=arguments-differ,invalid-name
    def to_solver(self, params, **kwargs) -> NavierStokesSolver:
        """
        Set preconditioner threshold to freestream Mach number
        """

        if self.low_mach_preconditioner:
            if self.low_mach_preconditioner_threshold is None:
                self.low_mach_preconditioner_threshold = params.freestream.to_solver(
                    params, **kwargs
                ).Mach

        return super().to_solver(self, **kwargs)


class IncompressibleNavierStokesSolver(GenericFlowSolverSettings):
    """:class:`IncompressibleNavierStokesSolver` class for setting up incompressible Navier-Stokes solver

    Parameters
    ----------
    pressure_correction_solver :
        Pressure correction solver settings

    linear_solver:
        Linear solver settings

    update_jacobian_frequency :
        Frequency at which the jacobian is updated.

    equation_eval_frequency :
        Frequency at which to update the incompressible NS equation in loosely-coupled simulations

    Returns
    -------
    :class:`IncompressibleNavierStokesSolver`
        An instance of the component class IncompressibleNavierStokesSolver.

    Example
    -------
    >>> ns = IncompressibleNavierStokesSolver(absolute_tolerance=1e-10)
    """

    pressure_correction_solver: Optional[PressureCorrectionSolver] = pd.Field(
        alias="pressureCorrectionSolver", default=PressureCorrectionSolver()
    )
    equation_eval_frequency: Optional[pd.PositiveInt] = pd.Field(1, alias="equationEvalFrequency")
    kappa_MUSCL: Optional[pd.confloat(ge=-1, le=1)] = pd.Field(
        -1, alias="kappaMUSCL", displayed="Kappa MUSCL"
    )
    linear_solver: Optional[LinearSolver] = pd.Field(
        LinearSolver(max_iterations=30), alias="linearSolver", displayed="Linear solver"
    )

    model_type: Literal["Incompressible"] = pd.Field(
        "Incompressible", alias="modelType", const=True
    )


NavierStokesSolverType = Union[NavierStokesSolver, IncompressibleNavierStokesSolver]


class SpalartAllmarasModelConstants(Flow360BaseModel):
    """:class:`SpalartAllmarasModelConstants` class"""

    model_type: Literal["SpalartAllmarasConsts"] = pd.Field(
        "SpalartAllmarasConsts", alias="modelType", const=True
    )
    C_DES: Optional[pd.NonNegativeFloat] = pd.Field(0.72)
    C_d: Optional[pd.NonNegativeFloat] = pd.Field(8.0)
    C_cb1: Optional[pd.NonNegativeFloat] = pd.Field(0.1355)
    C_cb2: Optional[pd.NonNegativeFloat] = pd.Field(0.622)
    C_sigma: Optional[pd.NonNegativeFloat] = pd.Field(2.0 / 3.0)
    C_v1: Optional[pd.NonNegativeFloat] = pd.Field(7.1)
    C_vonKarman: Optional[pd.NonNegativeFloat] = pd.Field(0.41)
    C_w2: Optional[pd.NonNegativeFloat] = pd.Field(0.3)
    C_t3: Optional[pd.NonNegativeFloat] = pd.Field(1.2)
    C_t4: Optional[pd.NonNegativeFloat] = pd.Field(0.5)
    C_min_rd: Optional[pd.NonNegativeFloat] = pd.Field(10.0)


class KOmegaSSTModelConstants(Flow360BaseModel):
    """:class:`KOmegaSSTModelConstants` class"""

    model_type: Literal["kOmegaSSTConsts"] = pd.Field(
        "kOmegaSSTConsts", alias="modelType", const=True
    )
    C_DES1: Optional[pd.NonNegativeFloat] = pd.Field(0.78)
    C_DES2: Optional[pd.NonNegativeFloat] = pd.Field(0.61)
    C_d1: Optional[pd.NonNegativeFloat] = pd.Field(20.0)
    C_d2: Optional[pd.NonNegativeFloat] = pd.Field(3.0)
    C_sigma_k1: Optional[pd.NonNegativeFloat] = pd.Field(0.85)
    C_sigma_k2: Optional[pd.NonNegativeFloat] = pd.Field(1.0)
    C_sigma_omega1: Optional[pd.NonNegativeFloat] = pd.Field(0.5)
    C_sigma_omega2: Optional[pd.NonNegativeFloat] = pd.Field(0.856)
    C_alpha1: Optional[pd.NonNegativeFloat] = pd.Field(0.31)
    C_beta1: Optional[pd.NonNegativeFloat] = pd.Field(0.075)
    C_beta2: Optional[pd.NonNegativeFloat] = pd.Field(0.0828)
    C_beta_star: Optional[pd.NonNegativeFloat] = pd.Field(0.09)


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

    rotation_correction:
        rotation-curvature correction for the turbulence model.

    Returns
    -------
    :class:`TurbulenceModelSolver`
        An instance of the component class TurbulenceModelSolver.

    Example
    -------
    >>> ts = TurbulenceModelSolver(absolute_tolerance=1e-10)
    """

    model_type: str = pd.Field(alias="modelType")
    absolute_tolerance: Optional[pd.PositiveFloat] = pd.Field(1e-8, alias="absoluteTolerance")
    equation_eval_frequency: Optional[pd.PositiveInt] = pd.Field(4, alias="equationEvalFrequency")
    DDES: Optional[bool] = pd.Field(False, alias="DDES", displayed="DDES")
    grid_size_for_LES: Optional[Literal["maxEdgeLength", "meanEdgeLength"]] = pd.Field(
        "maxEdgeLength", alias="gridSizeForLES"
    )
    reconstruction_gradient_limiter: Optional[pd.confloat(ge=0, le=2)] = pd.Field(
        alias="reconstructionGradientLimiter"
    )
    quadratic_constitutive_relation: Optional[bool] = pd.Field(
        False, alias="quadraticConstitutiveRelation"
    )
    model_constants: Optional[TurbulenceModelConstants] = pd.Field(
        alias="modelConstants", discriminator="model_type"
    )

    linear_solver: Optional[LinearSolver] = pd.Field(
        LinearSolver(max_iterations=20), alias="linearSolver", displayed="Linear solver config"
    )
    rotation_correction: Optional[bool] = pd.Field(False, alias="rotationCorrection")


class KOmegaSST(TurbulenceModelSolver):
    """:class:`KOmegaSST` class"""

    model_type: Literal["kOmegaSST"] = pd.Field("kOmegaSST", alias="modelType", const=True)
    model_constants: Optional[KOmegaSSTModelConstants] = pd.Field(
        alias="modelConstants", default=KOmegaSSTModelConstants()
    )
    reconstruction_gradient_limiter: Optional[pd.confloat(ge=0, le=2)] = pd.Field(
        1.0, alias="reconstructionGradientLimiter"
    )


class SpalartAllmaras(TurbulenceModelSolver):
    """:class:`SpalartAllmaras` class"""

    model_type: Literal["SpalartAllmaras"] = pd.Field(
        "SpalartAllmaras", alias="modelType", const=True
    )

    model_constants: Optional[SpalartAllmarasModelConstants] = pd.Field(
        alias="modelConstants", default=SpalartAllmarasModelConstants()
    )
    reconstruction_gradient_limiter: Optional[pd.confloat(ge=0, le=2)] = pd.Field(
        0.5, alias="reconstructionGradientLimiter"
    )


class NoneSolver(Flow360BaseModel):
    """:class:`SolverNone` class"""

    model_type: Literal["None"] = pd.Field("None", alias="modelType", const=True)


TurbulenceModelSolverType = Union[NoneSolver, SpalartAllmaras, KOmegaSST]


class HeatEquationSolver(GenericFlowSolverSettings):
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

    model_type: Literal["HeatEquation"] = pd.Field("HeatEquation", alias="modelType", const=True)
    CFL_multiplier: Optional[pd.PositiveFloat] = pd.Field(
        1.0, alias="CFLMultiplier", displayed="CFL Multiplier"
    )
    update_jacobian_frequency: Optional[pd.PositiveInt] = pd.Field(
        1, alias="updateJacobianFrequency"
    )
    absolute_tolerance: Optional[pd.PositiveFloat] = pd.Field(1e-9, alias="absoluteTolerance")
    relative_tolerance: Optional[pd.NonNegativeFloat] = pd.Field(1e-3, alias="relativeTolerance")
    equation_eval_frequency: Optional[pd.PositiveInt] = pd.Field(alias="equationEvalFrequency")
    order_of_accuracy: Optional[Literal[2]] = pd.Field(2, alias="orderOfAccuracy", const=True)

    linear_solver: Optional[LinearSolver] = pd.Field(LinearSolver(), alias="linearSolver")

    # pylint: disable=missing-class-docstring,too-few-public-methods
    class Config(Flow360BaseModel.Config):
        deprecated_aliases = [
            DeprecatedAlias(name="linear_solver", deprecated="linearSolverConfig"),
            DeprecatedAlias(name="absolute_tolerance", deprecated="tolerance"),
        ]

    # pylint: disable=arguments-differ
    def to_solver(self, params, **kwargs) -> HeatEquationSolver:
        """
        set Default Equation Eval Frequency
        """
        if self.equation_eval_frequency is None:
            if isinstance(params.time_stepping, UnsteadyTimeStepping):
                self.equation_eval_frequency = max(
                    1,
                    params.time_stepping.max_pseudo_steps
                    // HEAT_EQUATION_EVAL_MAX_PER_PSEUDOSTEP_UNSTEADY,
                )
            else:
                self.equation_eval_frequency = HEAT_EQUATION_EVAL_FREQUENCY_STEADY

        return super().to_solver(params, **kwargs)


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

    model_type: Literal["AmplificationFactorTransport"] = pd.Field(
        "AmplificationFactorTransport", alias="modelType", const=True
    )
    absolute_tolerance: Optional[pd.PositiveFloat] = pd.Field(1e-7, alias="absoluteTolerance")
    equation_eval_frequency: Optional[pd.PositiveInt] = pd.Field(4, alias="equationEvalFrequency")
    turbulence_intensity_percent: Optional[pd.confloat(ge=0.03, le=2.5)] = pd.Field(
        alias="turbulenceIntensityPercent"
    )
    N_crit: Optional[pd.confloat(ge=1, le=11)] = pd.Field(alias="Ncrit")
    reconstruction_gradient_limiter: Optional[pd.confloat(ge=0, le=2)] = pd.Field(
        1.0, alias="reconstructionGradientLimiter"
    )

    linear_solver: Optional[LinearSolver] = pd.Field(
        LinearSolver(max_iterations=20), alias="linearSolver", displayed="Linear solver config"
    )

    # pylint: disable=arguments-differ,invalid-name
    def to_solver(self, params, **kwargs) -> TransitionModelSolver:
        """
        Convert turbulenceIntensityPercent to Ncrit
        """

        if self.turbulence_intensity_percent is not None:
            Ncrit_converted = -8.43 - 2.4 * np.log(
                0.025 * np.tanh(self.turbulence_intensity_percent / 2.5)
            )
            self.turbulence_intensity_percent = None
            self.N_crit = Ncrit_converted
        elif self.N_crit is None:
            self.N_crit = 8.15

        return super().to_solver(self, exclude=["turbulence_intensity_percent"], **kwargs)

    # pylint: disable=missing-class-docstring,too-few-public-methods
    class Config(Flow360BaseModel.Config):
        conflicting_fields = [Conflicts(field1="N_crit", field2="turbulence_intensity_percent")]


# Legacy models for Flow360 updater, do not expose
class PressureCorrectionSolverLegacy(PressureCorrectionSolver, LegacyModel):
    """:class:`PressureCorrectionSolverLegacy` class"""

    linear_solver: Optional[LinearSolverLegacy] = pd.Field(
        alias="linearSolver", default=LinearSolverLegacy()
    )

    def update_model(self):
        model = {"linear_solver": try_update(self.linear_solver)}

        return model


class NavierStokesSolverLegacy(NavierStokesSolver, LegacyModel):
    """:class:`NavierStokesModelSolverLegacy` class"""

    viscous_wave_speed_scale: Optional[float] = pd.Field(alias="viscousWaveSpeedScale")
    extra_dissipation: Optional[float] = pd.Field(alias="extraDissipation")
    pressure_correction_solver: Optional[PressureCorrectionSolver] = pd.Field(
        alias="pressureCorrectionSolver"
    )
    linear_solver_config: Optional[LinearSolverLegacy] = pd.Field(alias="linearSolverConfig")
    linear_iterations: Optional[pd.PositiveInt] = pd.Field(alias="linearIterations")

    _processed_linear_solver_config = pd.validator(
        "linear_solver_config", allow_reuse=True, pre=True
    )(set_linear_solver_config_if_none)

    def update_model(self):
        model = {
            "absoluteTolerance": self.absolute_tolerance,
            "relativeTolerance": self.relative_tolerance,
            "CFLMultiplier": self.CFL_multiplier,
            "linearSolverConfig": try_update(self.linear_solver_config),
            "updateJacobianFrequency": self.update_jacobian_frequency,
            "equationEvalFrequency": self.equation_eval_frequency,
            "maxForceJacUpdatePhysicalSteps": self.max_force_jac_update_physical_steps,
            "orderOfAccuracy": self.order_of_accuracy,
            "kappaMUSCL": self.kappa_MUSCL,
            "limitVelocity": self.limit_velocity,
            "limitPressureDensity": self.limit_pressure_density,
            "numericalDissipationFactor": self.numerical_dissipation_factor,
            "lowDissipationControlFactors": self.low_dissipation_control_factors,
            "lowMachPreconditioner": self.low_mach_preconditioner,
            "lowMachPreconditionerThreshold": self.low_mach_preconditioner_threshold,
        }

        if self.linear_iterations is not None and model["linearSolverConfig"] is not None:
            model["linearSolverConfig"]["maxIterations"] = self.linear_iterations

        return model


class TurbulenceModelSolverLegacy(TurbulenceModelSolver, LegacyModel):
    """:class:`TurbulenceModelSolverLegacy` class"""

    kappa_MUSCL: Optional[pd.confloat(ge=-1, le=1)] = pd.Field(alias="kappaMUSCL")
    linear_iterations: Optional[pd.PositiveInt] = pd.Field(alias="linearIterations")
    linear_solver_config: Optional[LinearSolverLegacy] = pd.Field(alias="linearSolverConfig")
    rotation_correction: Optional[bool] = pd.Field(alias="rotationCorrection")

    _processed_linear_solver_config = pd.validator(
        "linear_solver_config", allow_reuse=True, pre=True
    )(set_linear_solver_config_if_none)

    # pylint: disable=no-self-argument
    @pd.root_validator(pre=True)
    def populate_model_constant_type(cls, values):
        """
        Add modelConstants->modelType before updater as it is required by discriminator
        """
        turbulence_model_type = values.get("modelType")
        model_constants = values.get("modelConstants")
        if (
            turbulence_model_type is None
            or model_constants is None
            or "modelType" in model_constants
        ):
            return values
        if turbulence_model_type == SpalartAllmaras.__fields__["model_type"].default:
            values["modelConstants"]["modelType"] = SpalartAllmarasModelConstants.__fields__[
                "model_type"
            ].default
        if turbulence_model_type == KOmegaSST.__fields__["model_type"].default:
            values["modelConstants"]["modelType"] = KOmegaSSTModelConstants.__fields__[
                "model_type"
            ].default
        return values

    def update_model(self):
        if self.model_type == "None":
            return NoneSolver()

        model = {
            "absoluteTolerance": self.absolute_tolerance,
            "relativeTolerance": self.relative_tolerance,
            "modelType": self.model_type,
            "linearSolverConfig": try_update(self.linear_solver_config),
            "updateJacobianFrequency": self.update_jacobian_frequency,
            "equationEvalFrequency": self.equation_eval_frequency,
            "maxForceJacUpdatePhysicalSteps": self.max_force_jac_update_physical_steps,
            "orderOfAccuracy": self.order_of_accuracy,
            "DDES": self.DDES,
            "gridSizeForLES": self.grid_size_for_LES,
            "quadraticConstitutiveRelation": self.quadratic_constitutive_relation,
        }

        if self.model_constants is not None:
            model["modelConstants"] = self.model_constants

        try_set(model, "rotationCorrection", self.rotation_correction)

        if self.reconstruction_gradient_limiter is not None:
            model["reconstructionGradientLimiter"] = self.reconstruction_gradient_limiter

        if self.linear_iterations is not None and model["linearSolverConfig"] is not None:
            model["linearSolverConfig"]["maxIterations"] = self.linear_iterations

        if self.model_type == SpalartAllmaras.__fields__["model_type"].default:
            return SpalartAllmaras(**model)
        if self.model_type == KOmegaSST.__fields__["model_type"].default:
            return KOmegaSST(**model)
        return model


class HeatEquationSolverLegacy(HeatEquationSolver, LegacyModel):
    """:class:`HeatEquationSolverLegacy` class"""

    order_of_accuracy: Optional[Literal[1, 2]] = pd.Field(alias="orderOfAccuracy")
    CFL_multiplier: Optional[pd.PositiveFloat] = pd.Field(alias="CFLMultiplier")
    update_jacobian_frequency: Optional[pd.PositiveInt] = pd.Field(alias="updateJacobianFrequency")
    max_force_jac_update_physical_steps: Optional[pd.NonNegativeInt] = pd.Field(
        alias="maxForceJacUpdatePhysicalSteps"
    )
    linear_solver_config: Optional[LinearSolverLegacy] = pd.Field(alias="linearSolverConfig")

    _processed_linear_solver_config = pd.validator(
        "linear_solver_config", allow_reuse=True, pre=True
    )(set_linear_solver_config_if_none)

    def update_model(self) -> Flow360BaseModel:
        model = {
            "absoluteTolerance": self.absolute_tolerance,
            "linearSolverConfig": try_update(self.linear_solver_config),
            "equationEvalFrequency": self.equation_eval_frequency,
        }

        return HeatEquationSolver.parse_obj(model)


class TransitionModelSolverLegacy(TransitionModelSolver, LegacyModel):
    """:class:`TransitionModelSolverLegacy` class"""

    reconstruction_gradient_limiter: Optional[pd.confloat(ge=0, le=2)] = pd.Field(
        alias="reconstructionGradientLimiter"
    )
    CFL_multiplier: Optional[pd.PositiveFloat] = pd.Field(alias="CFLMultiplier")
    linear_iterations: Optional[pd.PositiveInt] = pd.Field(alias="linearIterations")
    linear_solver_config: Optional[LinearSolverLegacy] = pd.Field(alias="linearSolverConfig")

    _processed_linear_solver_config = pd.validator(
        "linear_solver_config", allow_reuse=True, pre=True
    )(set_linear_solver_config_if_none)

    def update_model(self) -> Flow360BaseModel:
        model = {
            "absoluteTolerance": self.absolute_tolerance,
            "relativeTolerance": self.relative_tolerance,
            "modelType": self.model_type,
            "linearSolver": try_update(self.linear_solver_config),
            "updateJacobianFrequency": self.update_jacobian_frequency,
            "equationEvalFrequency": self.equation_eval_frequency,
            "maxForceJacUpdatePhysicalSteps": self.max_force_jac_update_physical_steps,
            "orderOfAccuracy": self.order_of_accuracy,
            "turbulenceIntensityPercent": self.turbulence_intensity_percent,
            "Ncrit": self.N_crit,
        }

        if self.linear_iterations is not None and model["linearSolver"] is not None:
            model["linearSolver"]["maxIterations"] = self.linear_iterations

        return TransitionModelSolver.parse_obj(model)
