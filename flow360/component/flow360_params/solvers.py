"""
Flow360 solvers parameters
"""

from __future__ import annotations

from abc import ABCMeta
from typing import Optional, Union

import numpy as np
import pydantic as pd
from typing_extensions import Literal

from flow360.flags import Flags

from ..types import NonNegativeFloat, NonNegativeInt, PositiveFloat, PositiveInt
from .flow360_legacy import LegacyModel, try_set, try_update
from .params_base import Conflicts, DeprecatedAlias, Flow360BaseModel


class GenericFlowSolverSettings(Flow360BaseModel, metaclass=ABCMeta):
    """:class:`GenericFlowSolverSettings` class"""

    absolute_tolerance: Optional[PositiveFloat] = pd.Field(1.0e-10, alias="absoluteTolerance")
    relative_tolerance: Optional[NonNegativeFloat] = pd.Field(0, alias="relativeTolerance")
    order_of_accuracy: Optional[Literal[1, 2]] = pd.Field(2, alias="orderOfAccuracy")
    CFL_multiplier: Optional[PositiveFloat] = pd.Field(
        2.0, alias="CFLMultiplier", displayed="CFL Multiplier"
    )
    update_jacobian_frequency: Optional[PositiveInt] = pd.Field(4, alias="updateJacobianFrequency")
    max_force_jac_update_physical_steps: Optional[NonNegativeInt] = pd.Field(
        0, alias="maxForceJacUpdatePhysicalSteps", displayed="Max force JAC update physical steps"
    )
    if Flags.beta_features():
        # pylint: disable=missing-class-docstring,too-few-public-methods
        class Config(Flow360BaseModel.Config):
            deprecated_aliases = [
                DeprecatedAlias(name="linear_solver", deprecated="linearSolverConfig"),
                DeprecatedAlias(name="absolute_tolerance", deprecated="tolerance"),
            ]

    else:

        class Config(
            Flow360BaseModel.Config
        ):  # pylint: disable=missing-class-docstring,too-few-public-methods
            deprecated_aliases = [
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
    ## Reflect that only one of absolute_tolerance and relative_tolerance is allowed in schema. TODO
    absolute_tolerance: Optional[PositiveFloat] = pd.Field(alias="absoluteTolerance")
    relative_tolerance: Optional[PositiveFloat] = pd.Field(alias="relativeTolerance")

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
    kappa_MUSCL: Optional[pd.confloat(ge=-1, le=1)] = pd.Field(
        -1, alias="kappaMUSCL", displayed="Kappa MUSCL"
    )
    equation_eval_frequency: Optional[PositiveInt] = pd.Field(1, alias="equationEvalFrequency")

    numerical_dissipation_factor: Optional[pd.confloat(ge=0.01, le=1)] = pd.Field(
        1, alias="numericalDissipationFactor"
    )
    limit_velocity: Optional[bool] = pd.Field(False, alias="limitVelocity")
    limit_pressure_density: Optional[bool] = pd.Field(False, alias="limitPressureDensity")

    if Flags.beta_features():
        linear_solver: Optional[LinearSolver] = pd.Field(
            LinearSolver(max_iterations=30), alias="linearSolver", displayed="Linear solver"
        )

    else:
        linear_solver_config: Optional[LinearSolver] = pd.Field(
            LinearSolver(max_iterations=30),
            alias="linearSolverConfig",
            displayed="Linear solver config",
        )

    model_type: Literal["Compressible"] = pd.Field("Compressible", alias="modelType", const=True)


if Flags.beta_features():

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
        equation_eval_frequency: Optional[PositiveInt] = pd.Field(1, alias="equationEvalFrequency")
        kappa_MUSCL: Optional[pd.confloat(ge=-1, le=1)] = pd.Field(
            -1, alias="kappaMUSCL", displayed="Kappa MUSCL"
        )
        if Flags.beta_features():
            linear_solver: Optional[LinearSolver] = pd.Field(
                LinearSolver(max_iterations=30), alias="linearSolver", displayed="Linear solver"
            )

        else:
            linear_solver_config: Optional[LinearSolver] = pd.Field(
                LinearSolver(max_iterations=30),
                alias="linearSolverConfig",
                displayed="Linear solver config",
            )

        model_type: Literal["Incompressible"] = pd.Field(
            "Incompressible", alias="modelType", const=True
        )

    NavierStokesSolverType = Union[NavierStokesSolver, IncompressibleNavierStokesSolver]


class TurbulenceModelConstantsSA(Flow360BaseModel):
    """:class:`TurbulenceModelConstantsSA` class"""

    model_type: Literal["SpalartAllmarasConsts"] = pd.Field(
        "SpalartAllmarasConsts", alias="modelType", const=True
    )
    C_DES: Optional[NonNegativeFloat] = pd.Field(0.72)
    C_d: Optional[NonNegativeFloat] = pd.Field(8.0)


class TurbulenceModelConstantsSST(Flow360BaseModel):
    """:class:`TurbulenceModelConstantsSST` class"""

    model_type: Literal[" kOmegaSSTConsts"] = pd.Field(
        "kOmegaSSTConsts", alias="modelType", const=True
    )
    C_DES1: Optional[NonNegativeFloat] = pd.Field(0.78)
    C_DES2: Optional[NonNegativeFloat] = pd.Field(0.61)
    C_d1: Optional[NonNegativeFloat] = pd.Field(20.0)
    C_d2: Optional[NonNegativeFloat] = pd.Field(3.0)


TurbulenceModelConstants = Union[TurbulenceModelConstantsSA, TurbulenceModelConstantsSST]


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

    model_type: str = pd.Field(alias="modelType")
    absolute_tolerance: Optional[PositiveFloat] = pd.Field(1e-8, alias="absoluteTolerance")
    equation_eval_frequency: Optional[PositiveInt] = pd.Field(4, alias="equationEvalFrequency")
    DDES: Optional[bool] = pd.Field(False, alias="DDES", displayed="DDES")
    grid_size_for_LES: Optional[Literal["maxEdgeLength", "meanEdgeLength"]] = pd.Field(
        "maxEdgeLength", alias="gridSizeForLES"
    )
    reconstruction_gradient_limiter: Optional[pd.confloat(ge=0, le=2)] = pd.Field(
        1.0, alias="reconstructionGradientLimiter"
    )
    quadratic_constitutive_relation: Optional[bool] = pd.Field(
        False, alias="quadraticConstitutiveRelation"
    )
    model_constants: Optional[TurbulenceModelConstants] = pd.Field(
        alias="modelConstants", discriminator="model_type"
    )

    if Flags.beta_features():
        linear_solver: Optional[LinearSolver] = pd.Field(
            LinearSolver(max_iterations=20), alias="linearSolver", displayed="Linear solver config"
        )

    else:
        linear_solver_config: Optional[LinearSolver] = pd.Field(
            LinearSolver(max_iterations=20),
            alias="linearSolverConfig",
            displayed="Linear solver config",
        )


class KOmegaSST(TurbulenceModelSolver):
    """:class:`KOmegaSST` class"""

    model_type: Literal["kOmegaSST"] = pd.Field("kOmegaSST", alias="modelType", const=True)
    model_constants: Optional[TurbulenceModelConstantsSST] = pd.Field(alias="modelConstants")


class SpalartAllmaras(TurbulenceModelSolver):
    """:class:`SpalartAllmaras` class"""

    model_type: Literal["SpalartAllmaras"] = pd.Field(
        "SpalartAllmaras", alias="modelType", const=True
    )
    rotation_correction: Optional[bool] = pd.Field(False, alias="rotationCorrection")

    model_constants: Optional[TurbulenceModelConstantsSA] = pd.Field(alias="modelConstants")
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
    CFL_multiplier: Optional[PositiveFloat] = pd.Field(
        1.0, alias="CFLMultiplier", displayed="CFL Multiplier"
    )
    update_jacobian_frequency: Optional[PositiveInt] = pd.Field(1, alias="updateJacobianFrequency")
    absolute_tolerance: Optional[PositiveFloat] = pd.Field(1e-9, alias="absoluteTolerance")
    relative_tolerance: Optional[NonNegativeFloat] = pd.Field(1e-3, alias="relativeTolerance")
    equation_eval_frequency: Optional[PositiveInt] = pd.Field(alias="equationEvalFrequency")
    order_of_accuracy: Optional[Literal[2]] = pd.Field(2, alias="orderOfAccuracy", const=True)

    if Flags.beta_features():
        linear_solver: Optional[LinearSolver] = pd.Field(LinearSolver(), alias="linearSolver")

        # pylint: disable=missing-class-docstring,too-few-public-methods
        class Config(Flow360BaseModel.Config):
            deprecated_aliases = [
                DeprecatedAlias(name="linear_solver", deprecated="linearSolverConfig"),
                DeprecatedAlias(name="absolute_tolerance", deprecated="tolerance"),
            ]

    else:
        linear_solver_config: Optional[LinearSolver] = pd.Field(
            LinearSolver(), alias="linearSolverConfig"
        )

        # pylint: disable=missing-class-docstring,too-few-public-methods
        class Config(Flow360BaseModel.Config):
            deprecated_aliases = [
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

    model_type: Literal["AmplificationFactorTransport"] = pd.Field(
        "AmplificationFactorTransport", alias="modelType", const=True
    )
    absolute_tolerance: Optional[PositiveFloat] = pd.Field(1e-7, alias="absoluteTolerance")
    equation_eval_frequency: Optional[PositiveInt] = pd.Field(4, alias="equationEvalFrequency")
    turbulence_intensity_percent: Optional[pd.confloat(ge=0.03, le=2.5)] = pd.Field(
        alias="turbulenceIntensityPercent"
    )
    N_crit: Optional[pd.confloat(ge=1, le=11)] = pd.Field(alias="Ncrit")
    reconstruction_gradient_limiter: Optional[pd.confloat(ge=0, le=2)] = pd.Field(
        1.0, alias="reconstructionGradientLimiter"
    )

    if Flags.beta_features():
        linear_solver: Optional[LinearSolver] = pd.Field(
            LinearSolver(max_iterations=20), alias="linearSolver", displayed="Linear solver config"
        )

    else:
        linear_solver_config: Optional[LinearSolver] = pd.Field(
            LinearSolver(max_iterations=20),
            alias="linearSolverConfig",
            displayed="Linear solver config",
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


class LinearSolverLegacy(LinearSolver, LegacyModel):
    """:class:`LinearSolverLegacy` class"""

    max_level_limit: Optional[NonNegativeInt] = pd.Field(alias="maxLevelLimit")

    def update_model(self):
        model = {
            "absoluteTolerance": self.absolute_tolerance,
            "relativeTolerance": self.relative_tolerance,
            "maxIterations": self.max_iterations,
        }

        return model


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
    linear_solver_config: Optional[LinearSolverLegacy] = pd.Field(
        alias="linearSolverConfig", default=LinearSolverLegacy()
    )
    linear_iterations: Optional[PositiveInt] = pd.Field(alias="linearIterations")

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
        }

        if self.linear_iterations is not None and model["linearSolverConfig"] is not None:
            model["linearSolverConfig"]["maxIterations"] = self.linear_iterations

        return model


class TurbulenceModelSolverLegacy(TurbulenceModelSolver, LegacyModel):
    """:class:`TurbulenceModelSolverLegacy` class"""

    kappa_MUSCL: Optional[pd.confloat(ge=-1, le=1)] = pd.Field(alias="kappaMUSCL")
    linear_iterations: Optional[PositiveInt] = pd.Field(alias="linearIterations")
    linear_solver_config: Optional[LinearSolverLegacy] = pd.Field(
        alias="linearSolverConfig", default=LinearSolverLegacy()
    )
    rotation_correction: Optional[bool] = pd.Field(alias="rotationCorrection")

    def update_model(self):
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
            "reconstructionGradientLimiter": self.reconstruction_gradient_limiter,
            "modelConstants": self.model_constants,
        }

        try_set(model, "rotationCorrection", self.rotation_correction)

        if self.linear_iterations is not None and model["linearSolverConfig"] is not None:
            model["linearSolverConfig"]["maxIterations"] = self.linear_iterations

        return model


class HeatEquationSolverLegacy(HeatEquationSolver, LegacyModel):
    """:class:`HeatEquationSolverLegacy` class"""

    order_of_accuracy: Optional[Literal[1, 2]] = pd.Field(alias="orderOfAccuracy")
    CFL_multiplier: Optional[PositiveFloat] = pd.Field(alias="CFLMultiplier")
    update_jacobian_frequency: Optional[PositiveInt] = pd.Field(alias="updateJacobianFrequency")
    max_force_jac_update_physical_steps: Optional[NonNegativeInt] = pd.Field(
        alias="maxForceJacUpdatePhysicalSteps"
    )
    linear_solver_config: Optional[LinearSolverLegacy] = pd.Field(
        alias="linearSolverConfig", default=LinearSolverLegacy()
    )

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
    CFL_multiplier: Optional[PositiveFloat] = pd.Field(alias="CFLMultiplier")
    linear_iterations: Optional[PositiveInt] = pd.Field(alias="linearIterations")
    linear_solver_config: Optional[LinearSolverLegacy] = pd.Field(
        alias="linearSolverConfig", default=LinearSolverLegacy()
    )

    def update_model(self) -> Flow360BaseModel:
        model = {
            "absoluteTolerance": self.absolute_tolerance,
            "relativeTolerance": self.relative_tolerance,
            "modelType": self.model_type,
            "linearSolverConfig": try_update(self.linear_solver_config),
            "updateJacobianFrequency": self.update_jacobian_frequency,
            "equationEvalFrequency": self.equation_eval_frequency,
            "maxForceJacUpdatePhysicalSteps": self.max_force_jac_update_physical_steps,
            "orderOfAccuracy": self.order_of_accuracy,
            "turbulenceIntensityPercent": self.turbulence_intensity_percent,
            "Ncrit": self.N_crit,
        }

        if self.linear_iterations is not None and model["linearSolverConfig"] is not None:
            model["linearSolverConfig"]["maxIterations"] = self.linear_iterations

        return TransitionModelSolver.parse_obj(model)
