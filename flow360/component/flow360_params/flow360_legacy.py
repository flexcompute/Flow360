"""
Legacy field definitions (fields that are not
specified in the documentation but can be used internally
during validation, most legacy classes can be updated to
the current standard via the update_model method)
"""
from abc import ABCMeta, abstractmethod

import pydantic as pd
from typing import Optional, Literal, Union, List, Any, Dict

import unyt as u
from pydantic import NonNegativeFloat

from flow360 import (
    SurfaceOutput,
    SliceOutput,
    VolumeOutput,
    BETDisk,
    Flow360Params,
    SlidingInterface,
    Geometry,
    TimeStepping,
    FluidDynamicsVolumeZone,
    VolumeZones
)

from flow360.component.flow360_params.params_base import Flow360BaseModel

from flow360.component.flow360_params.solvers import (
    LinearSolver,
    NavierStokesSolver,
    PressureCorrectionSolver,
    LinearIterationsRandomizer,
    TurbulenceModelSolver,
    HeatEquationSolver,
    TransitionModelSolver,
    NoneSolver,
    TurbulenceModelSolvers
)

from flow360.component.flow360_params.unit_system import DimensionedType

from flow360.component.types import (
    NonNegativeInt,
    PositiveInt,
    PositiveFloat,
    Coordinate
)


class LegacyModel(Flow360BaseModel, metaclass=ABCMeta):
    comments: Optional[Dict] = pd.Field()

    @abstractmethod
    def update_model(self) -> Flow360BaseModel:
        pass



def _try_add_unit(model, key, unit: DimensionedType):
    if model[key] is not None:
        model[key] *= unit


def _try_set(model, key, value):
    if value is not None:
        model[key] = value


def _try_update(field: Optional[LegacyModel]):
    if field is not None:
        return field.update_model()
    return None


class OutputLegacy(pd.BaseModel, metaclass=ABCMeta):
    """:class: Base class for common output parameters"""

    Cp: Optional[bool] = pd.Field()
    grad_w: Optional[bool] = pd.Field(alias="gradW")
    k_omega: Optional[bool] = pd.Field(alias="kOmega")
    Mach: Optional[bool] = pd.Field(alias="Mach")
    mut: Optional[bool] = pd.Field()
    mut_ratio: Optional[bool] = pd.Field(alias="mutRatio")
    nu_hat: Optional[bool] = pd.Field(alias="nuHat")
    primitive_vars: Optional[bool] = pd.Field(alias="primitiveVars")
    q_criterion: Optional[bool] = pd.Field(alias="qcriterion")
    residual_navier_stokes: Optional[bool] = pd.Field(alias="residualNavierStokes")
    residual_transition: Optional[bool] = pd.Field(alias="residualTransition")
    residual_turbulence: Optional[bool] = pd.Field(alias="residualTurbulence")
    s: Optional[bool] = pd.Field()
    solution_navier_stokes: Optional[bool] = pd.Field(alias="solutionNavierStokes")
    solution_turbulence: Optional[bool] = pd.Field(alias="solutionTurbulence")
    solution_transition: Optional[bool] = pd.Field(alias="solutionTransition")
    T: Optional[bool] = pd.Field(alias="T")
    vorticity: Optional[bool] = pd.Field()
    wall_distance: Optional[bool] = pd.Field(alias="wallDistance")
    low_numerical_dissipation_sensor: Optional[bool] = pd.Field(
        alias="lowNumericalDissipationSensor"
    )
    residual_heat_solver: Optional[bool] = pd.Field(alias="residualHeatSolver")


class SurfaceOutputLegacy(SurfaceOutput, OutputLegacy, LegacyModel):
    """:class:`SurfaceOutputLegacy` class"""

    wall_function_metric: Optional[bool] = pd.Field(alias="wallFunctionMetric")
    node_moments_per_unit_area: Optional[bool] = pd.Field(alias="nodeMomentsPerUnitArea")
    residual_sa: Optional[bool] = pd.Field(alias="residualSA")
    coarsen_iterations: Optional[int] = pd.Field(alias="coarsenIterations")

    Cf: Optional[bool] = pd.Field(alias="Cf")
    Cf_vec: Optional[bool] = pd.Field(alias="CfVec")
    Cf_normal: Optional[bool] = pd.Field(alias="CfNormal")
    Cf_tangent: Optional[bool] = pd.Field(alias="CfTangent")
    y_plus: Optional[bool] = pd.Field(alias="yPlus")
    wall_distance: Optional[bool] = pd.Field(alias="wallDistance")
    heat_flux: Optional[bool] = pd.Field(alias="heatFlux")
    node_forces_per_unit_area: Optional[bool] = pd.Field(alias="nodeForcesPerUnitArea")
    node_normals: Optional[bool] = pd.Field(alias="nodeNormals")
    velocity_relative: Optional[bool] = pd.Field(alias="VelocityRelative")

    def update_model(self) -> Flow360BaseModel:
        pass


class SliceOutputLegacy(SliceOutput, OutputLegacy, LegacyModel):
    """:class:`SliceOutputLegacy` class"""
    coarsen_iterations: Optional[int] = pd.Field(alias="coarsenIterations")
    bet_metrics: Optional[bool] = pd.Field(alias="betMetrics")
    bet_metrics_per_disk: Optional[bool] = pd.Field(alias="betMetricsPerDisk")

    def update_model(self) -> Flow360BaseModel:
        pass


class VolumeOutputLegacy(VolumeOutput, OutputLegacy, LegacyModel):
    """:class:`VolumeOutputLegacy` class"""

    write_single_file: Optional[bool] = pd.Field(alias="writeSingleFile")
    write_distributed_file: Optional[bool] = pd.Field(alias="writeDistributedFile")
    residual_components_sa: Optional[bool] = pd.Field(alias="residualComponentsSA")
    wall_distance_dir: Optional[bool] = pd.Field(alias="wallDistanceDir")
    velocity_relative: Optional[bool] = pd.Field(alias="VelocityRelative")
    debug_transition: Optional[bool] = pd.Field(alias="debugTransition")
    debug_turbulence: Optional[bool] = pd.Field(alias="debugTurbulence")
    debug_navier_stokes: Optional[bool] = pd.Field(alias="debugNavierStokes")
    bet_metrics: Optional[bool] = pd.Field(alias="betMetrics")
    bet_metrics_per_disk: Optional[bool] = pd.Field(alias="betMetricsPerDisk")

    def update_model(self) -> Flow360BaseModel:
        pass


class LinearSolverLegacy(LinearSolver, LegacyModel):
    """:class:`LinearSolverLegacy` class"""

    max_level_limit: Optional[NonNegativeInt] = pd.Field(alias="maxLevelLimit")

    def update_model(self) -> Flow360BaseModel:

        model = {
            "absoluteTolerance": self.absolute_tolerance,
            "relativeTolerance": self.relative_tolerance,
            "maxIterations": self.max_iterations
        }

        return LinearSolver.parse_obj(model)


class PressureCorrectionSolverLegacy(PressureCorrectionSolver, LegacyModel):
    """:class:`PressureCorrectionSolverLegacy` class"""

    linear_solver: Optional[LinearSolverLegacy] = pd.Field(
        alias="linearSolver",
        default=LinearSolverLegacy()
    )

    def update_model(self) -> Flow360BaseModel:
        model = {
            "randomizer": self.randomizer,
            "linear_solver": _try_update(self.linear_solver)
        }

        return PressureCorrectionSolver.parse_obj(model)


class NavierStokesSolverLegacy(NavierStokesSolver, LegacyModel):
    """:class:`NavierStokesModelSolverLegacy` class"""

    viscous_wave_speed_scale: Optional[float] = pd.Field(alias="viscousWaveSpeedScale")
    extra_dissipation: Optional[float] = pd.Field(alias="extraDissipation")
    pressure_correction_solver: Optional[PressureCorrectionSolver] = pd.Field(
        alias="pressureCorrectionSolver"
    )
    numerical_dissipation_factor: Optional[pd.confloat(ge=0.01, le=1)] = pd.Field(
        alias="numericalDissipationFactor"
    )
    randomizer: Optional[LinearIterationsRandomizer] = pd.Field()
    linear_solver: Optional[LinearSolverLegacy] = pd.Field(
        alias="linearSolver",
        default=LinearSolverLegacy()
    )

    def update_model(self) -> Flow360BaseModel:
        model = {
            "absoluteTolerance": self.absolute_tolerance,
            "relativeTolerance": self.relative_tolerance,
            "CFLMultiplier": self.CFL_multiplier,
            "linearIterations": self.linear_iterations,
            "linearSolver": _try_update(self.linear_solver),
            "updateJacobianFrequency": self.update_jacobian_frequency,
            "equationEvalFrequency": self.equation_eval_frequency,
            "maxForceJacUpdatePhysicalSteps": self.max_force_jac_update_physical_steps,
            "orderOfAccuracy": self.order_of_accuracy,
            "kappaMUSCL": self.kappa_MUSCL,
            "limitVelocity": self.limit_velocity,
            "limitPressureDensity": self.limit_pressure_density
        }

        return NavierStokesSolver.parse_obj(model)


class TurbulenceModelSolverLegacy(TurbulenceModelSolver, LegacyModel):
    """:class:`TurbulenceModelSolverLegacy` class"""

    randomizer: Optional[LinearIterationsRandomizer] = pd.Field()
    kappa_MUSCL: Optional[pd.confloat(ge=-1, le=1)] = pd.Field(alias="kappaMUSCL")
    linear_iterations: Optional[PositiveInt] = pd.Field(alias="linearIterations")
    linear_solver: Optional[LinearSolverLegacy] = pd.Field(
        alias="linearSolver",
        default=LinearSolverLegacy()
    )

    def update_model(self) -> Flow360BaseModel:
        class TurbulenceModelSolverVariants(pd.BaseModel):
            """Helper class used to create
            the correct solver from dict data"""
            solver: TurbulenceModelSolvers = pd.Field(discriminator="model_type")

        model = {
            "solver": {
                "absoluteTolerance": self.absolute_tolerance,
                "relativeTolerance": self.relative_tolerance,
                "modelType": self.model_type,
                "CFLMultiplier": self.CFL_multiplier,
                "linearSolver": _try_update(self.linear_solver),
                "updateJacobianFrequency": self.update_jacobian_frequency,
                "equationEvalFrequency": self.equation_eval_frequency,
                "maxForceJacUpdatePhysicalSteps": self.max_force_jac_update_physical_steps,
                "orderOfAccuracy": self.order_of_accuracy,
                "DDES": self.DDES,
                "gridSizeForLES": self.grid_size_for_LES,
                "quadraticConstitutiveRelation": self.quadratic_constitutive_relation,
                "reconstructionGradientLimiter": self.reconstruction_gradient_limiter,
                "modelConstants": self.model_constants,
                "rotationCorrection": self.rotation_correction
            }
        }

        return TurbulenceModelSolverVariants.parse_obj(model).solver


class TurbulenceModelSolverSSTLegacy(TurbulenceModelSolverLegacy):
    """:class:`TurbulenceModelSolverSSTLegacy` class"""

    model_type: Literal["kOmegaSST"] = pd.Field("kOmegaSST", alias="modelType", const=True)


class TurbulenceModelSolverSALegacy(TurbulenceModelSolverLegacy):
    """:class:`TurbulenceModelSolverSALegacy` class"""

    model_type: Literal["SpalartAllmaras"] = pd.Field("SpalartAllmaras", alias="modelType", const=True)
    rotation_correction: Optional[bool] = pd.Field(alias="rotationCorrection")


TurbulenceModelSolversLegacy = Union[
    TurbulenceModelSolverSALegacy,
    TurbulenceModelSolverSSTLegacy,
    NoneSolver
]


class HeatEquationSolverLegacy(HeatEquationSolver, LegacyModel):
    """:class:`HeatEquationSolverLegacy` class"""

    order_of_accuracy: Optional[Literal[1, 2]] = pd.Field(alias="orderOfAccuracy")
    model_type: Literal["HeatEquation"] = pd.Field("HeatEquation", alias="modelType")
    CFL_multiplier: Optional[PositiveFloat] = pd.Field(alias="CFLMultiplier")
    update_jacobian_frequency: Optional[PositiveInt] = pd.Field(alias="updateJacobianFrequency")
    max_force_jac_update_physical_steps: Optional[NonNegativeInt] = pd.Field(
        alias="maxForceJacUpdatePhysicalSteps"
    )
    linear_solver: Optional[LinearSolverLegacy] = pd.Field(
        alias="linearSolver",
        default=LinearSolverLegacy()
    )

    def update_model(self) -> Flow360BaseModel:
        model = {
            "absoluteTolerance": self.absolute_tolerance,
            "linearSolver": _try_update(self.linear_solver),
            "equationEvalFrequency": self.equation_eval_frequency
        }

        return HeatEquationSolver.parse_obj(model)


class TransitionModelSolverLegacy(TransitionModelSolver, LegacyModel):
    """:class:`TransitionModelSolverLegacy` class"""

    reconstruction_gradient_limiter: Optional[pd.confloat(ge=0, le=2)] = pd.Field(
        alias="reconstructionGradientLimiter"
    )
    CFL_multiplier: Optional[PositiveFloat] = pd.Field(alias="CFLMultiplier")
    linear_iterations: Optional[PositiveInt] = pd.Field(alias="linearIterations")
    randomizer: Optional[LinearIterationsRandomizer] = pd.Field()
    linear_solver: Optional[LinearSolverLegacy] = pd.Field(
        alias="linearSolver",
        default=LinearSolverLegacy()
    )

    def update_model(self) -> Flow360BaseModel:
        model = {
            "absoluteTolerance": self.absolute_tolerance,
            "relativeTolerance": self.relative_tolerance,
            "modelType": self.model_type,
            "linearSolver": _try_update(self.linear_solver),
            "updateJacobianFrequency": self.update_jacobian_frequency,
            "equationEvalFrequency": self.equation_eval_frequency,
            "maxForceJacUpdatePhysicalSteps": self.max_force_jac_update_physical_steps,
            "orderOfAccuracy": self.order_of_accuracy,
            "turbulenceIntensityPercent": self.turbulence_intensity_percent,
            "Ncrit": self.N_crit
        }

        return HeatEquationSolver.parse_obj(model)


class BETDiskLegacy(BETDisk, LegacyModel):
    """:class:`BETDiskLegacy` class"""

    volume_name: Optional[str] = pd.Field(alias="volumeName")

    def update_model(self):
        pass


class GeometryLegacy(Geometry, LegacyModel):
    ref_area: Optional[float] = pd.Field(alias="refArea", default_factory=lambda: 1.0)
    moment_center: Optional[Coordinate] = pd.Field(alias="momentCenter")
    moment_length: Optional[Coordinate] = pd.Field(alias="momentLength")

    def update_model(self) -> Flow360BaseModel:
        model = {
            "momentCenter": self.moment_center,
            "momentLength": self.moment_length,
            "refArea": self.ref_area
        }
        if self.comments.get("meshUnit") is not None:
            unit = u.unyt_quantity(1, self.comments["meshUnit"])
            model["meshUnit"] = unit
            _try_add_unit(model, "momentCenter", model["meshUnit"])
            _try_add_unit(model, "momentLength", model["meshUnit"])
            _try_add_unit(model, "refArea", model["meshUnit"] ** 2)

        return Geometry.parse_obj(model)


class FreestreamLegacy(LegacyModel):
    """:class: `FreestreamLegacy` class"""
    Reynolds: Optional[PositiveFloat] = pd.Field()
    Mach: Optional[NonNegativeFloat] = pd.Field()
    MachRef: Optional[PositiveFloat] = pd.Field()
    mu_ref: Optional[PositiveFloat] = pd.Field(alias="muRef")
    temperature: PositiveFloat = pd.Field(alias="Temperature")
    density: Optional[PositiveFloat]
    speed: Optional[PositiveFloat]
    alpha: Optional[float] = pd.Field(alias="alphaAngle")
    beta: Optional[float] = pd.Field(alias="betaAngle", default=0)
    turbulent_viscosity_ratio: Optional[NonNegativeFloat] = pd.Field(alias="turbulentViscosityRatio")

    def update_model(self) -> Flow360BaseModel:
        # Apply items from comments
        model = {
            "alphaAngle": self.alpha,
            "betaAngle": self.beta,
            "turbulentViscosityRatio": self.turbulent_viscosity_ratio,
            "temperature": self.temperature,
        }

        pass


class TimeSteppingLegacy(TimeStepping, LegacyModel):
    time_step_size: Optional[Union[Literal["inf"], PositiveFloat]] = pd.Field(
        alias="timeStepSize",
        default="inf"
    )

    def update_model(self) -> Flow360BaseModel:
        model = {
            "CFL": self.CFL,
            "physicalSteps": self.physical_steps,
            "maxPseudoSteps": self.max_pseudo_steps,
            "timeStepSize": self.time_step_size
        }

        if model["timeStepSize"] != "inf" and self.comments.get("timeStepSizeInSeconds") is not None:
            step_unit = u.unyt_quantity(self.comments["timeStepSizeInSeconds"], "s")
            _try_add_unit(model, "timeStepSize", step_unit)

        return TimeStepping.parse_obj(model)


class SlidingInterfaceLegacy(SlidingInterface, LegacyModel):
    """:class:`SlidingInterfaceLegacy` class"""

    omega: Optional[float] = pd.Field()

    # pylint: disable=missing-class-docstring,too-few-public-methods
    class Config(Flow360BaseModel.Config):
        require_one_of = [
            "omega",
            "theta",
            "omega_radians",
            "omega_degrees",
            "theta_radians",
            "theta_degrees",
            "is_dynamic",
        ]

    def update_model(self) -> Flow360BaseModel:
        model = {
            "modelType": "FluidDynamics",
            "referenceFrame": {
                "axis": self.axis,
                "center": self.center * u.m,
            }
        }

        _try_set(model["referenceFrame"], "isDynamic", self.is_dynamic)
        _try_set(model["referenceFrame"], "omega", self.omega)
        _try_set(model["referenceFrame"], "omega", self.omega_radians)
        _try_set(model["referenceFrame"], "omega", self.omega_degrees)
        _try_set(model["referenceFrame"], "thetaRadians", self.theta_radians)
        _try_set(model["referenceFrame"], "thetaDegrees", self.theta_degrees)

        if self.omega_degrees is not None:
            omega = model["referenceFrame"]["omega"] * u.deg / u.s
            _try_set(model["referenceFrame"], "omega", omega)

        if self.comments.get("rpm") is not None:
            omega = self.comments["rpm"] * u.rpm
            _try_set(model["referenceFrame"], "omega", omega)

        return FluidDynamicsVolumeZone.parse_obj(model)


class Flow360ParamsLegacy(Flow360Params, LegacyModel):
    geometry: Optional[GeometryLegacy] = pd.Field()
    freestream: Optional[FreestreamLegacy] = pd.Field()
    time_stepping: Optional[TimeSteppingLegacy] = pd.Field(alias="timeStepping")
    navier_stokes_solver: Optional[NavierStokesSolverLegacy] = pd.Field(alias="navierStokesSolver")
    turbulence_model_solver: Optional[TurbulenceModelSolverLegacy] = pd.Field(alias="turbulenceModelSolver")
    transition_model_solver: Optional[TransitionModelSolverLegacy] = pd.Field(alias="transitionModelSolver")
    heat_equation_solver: Optional[HeatEquationSolverLegacy] = pd.Field(alias="heatEquationSolver")
    bet_disks: Optional[List[BETDiskLegacy]] = pd.Field(alias="BETDisks")
    sliding_interfaces: Optional[List[SlidingInterfaceLegacy]] = pd.Field(alias="slidingInterfaces")
    surface_output: Optional[SurfaceOutputLegacy] = pd.Field(alias="surfaceOutput")
    volume_output: Optional[VolumeOutputLegacy] = pd.Field(alias="volumeOutput")
    slice_output: Optional[SliceOutputLegacy] = pd.Field(alias="sliceOutput")

    def update_model(self) -> Flow360BaseModel:
        model = Flow360Params()
        model.geometry = _try_update(self.geometry)
        model.boundaries = self.boundaries
        model.initial_condition = self.initial_condition
        model.time_stepping = _try_update(self.time_stepping)
        model.navier_stokes_solver = _try_update(self.navier_stokes_solver)
        model.turbulence_model_solver = _try_update(self.turbulence_model_solver)
        model.transition_model_solver = _try_update(self.transition_model_solver)
        model.heat_equation_solver = _try_update(self.heat_equation_solver)
        # model.freestream = _try_update(self.freestream)

        if self.bet_disks is not None:
            bet_disks = []
            for disk in self.bet_disks:
                bet_disks.append(_try_update(disk))
            model.bet_disks = bet_disks

        model.actuator_disks = self.actuator_disks
        model.porous_media = self.porous_media
        model.user_defined_dynamics = self.user_defined_dynamics
        # model.surface_output = _try_update(self.surface_output)
        # model.volume_output = _try_update(self.volume_output)
        # model.slice_output = _try_update(self.slice_output)
        model.iso_surface_output = self.iso_surface_output
        model.monitor_output = self.monitor_output

        if self.sliding_interfaces is not None:
            volume_zones = {}
            for interface in self.sliding_interfaces:
                volume_zone = _try_update(interface)
                volume_zones[interface.volume_name] = volume_zone
            model.volume_zones = VolumeZones(**volume_zones)

        model.aeroacoustic_output = self.aeroacoustic_output

        return model
