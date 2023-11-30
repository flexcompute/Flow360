"""
Private and legacy field definitions (fields that are not
specified in the documentation but can be used internally
during validation)
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
    VolumeZones,
    FreestreamFromMach,
    FreestreamFromVelocity
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
    NoneSolver
)

from flow360.component.flow360_params.unit_system import DimensionedType

from flow360.component.types import (
    NonNegativeInt,
    PositiveInt,
    PositiveFloat,
    Coordinate
)


def _try_add_unit(model, key, unit: DimensionedType):
    if model[key] is not None:
        model[key] *= unit


def _try_set(model, key, value):
    if value is not None:
        model[key] = value


class LegacyModel(Flow360BaseModel, metaclass=ABCMeta):
    comments: Optional[Dict] = pd.Field()

    @abstractmethod
    def update_model(self) -> Flow360BaseModel:
        pass


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


class SurfaceOutputPrivate(SurfaceOutput):
    """:class:`SurfaceOutputPrivate` class"""

    wall_function_metric: Optional[bool] = pd.Field(alias="wallFunctionMetric")
    node_moments_per_unit_area: Optional[bool] = pd.Field(alias="nodeMomentsPerUnitArea")
    residual_sa: Optional[bool] = pd.Field(alias="residualSA")
    coarsen_iterations: Optional[int] = pd.Field(alias="coarsenIterations")


class SurfaceOutputLegacy(SurfaceOutputPrivate, OutputLegacy, LegacyModel):
    """:class:`SurfaceOutputLegacy` class"""

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


class SliceOutputPrivate(SliceOutput):
    """:class:`SliceOutputPrivate` class"""

    coarsen_iterations: Optional[int] = pd.Field(alias="coarsenIterations")


class SliceOutputLegacy(SliceOutputPrivate, OutputLegacy, LegacyModel):
    """:class:`SliceOutputLegacy` class"""

    bet_metrics: Optional[bool] = pd.Field(alias="betMetrics")
    bet_metrics_per_disk: Optional[bool] = pd.Field(alias="betMetricsPerDisk")

    def update_model(self) -> Flow360BaseModel:
        pass


class VolumeOutputPrivate(VolumeOutput):
    """:class:`VolumeOutputPrivate` class"""

    write_single_file: Optional[bool] = pd.Field(alias="writeSingleFile")
    write_distributed_file: Optional[bool] = pd.Field(alias="writeDistributedFile")
    residual_components_sa: Optional[bool] = pd.Field(alias="residualComponentsSA")
    wall_distance_dir: Optional[bool] = pd.Field(alias="wallDistanceDir")
    velocity_relative: Optional[bool] = pd.Field(alias="VelocityRelative")
    debug_transition: Optional[bool] = pd.Field(alias="debugTransition")
    debug_turbulence: Optional[bool] = pd.Field(alias="debugTurbulence")
    debug_navier_stokes: Optional[bool] = pd.Field(alias="debugNavierStokes")


class VolumeOutputLegacy(VolumeOutputPrivate, OutputLegacy, LegacyModel):
    """:class:`VolumeOutputLegacy` class"""

    bet_metrics: Optional[bool] = pd.Field(alias="betMetrics")
    bet_metrics_per_disk: Optional[bool] = pd.Field(alias="betMetricsPerDisk")

    def update_model(self) -> Flow360BaseModel:
        pass


class LinearSolverPrivate(LinearSolver):
    """:class:`LinearSolverPrivate` class"""

    max_level_limit: Optional[NonNegativeInt] = pd.Field(alias="maxLevelLimit")


class PressureCorrectionSolverPrivate(PressureCorrectionSolver):
    """:class:`PressureCorrectionSolverPrivate` class"""

    linear_solver: Optional[LinearSolverPrivate] = pd.Field(
        alias="linearSolver",
        default=LinearSolverPrivate()
    )


class NavierStokesSolverPrivate(NavierStokesSolver):
    """:class:`NavierStokesSolverPrivate` class"""

    viscous_wave_speed_scale: Optional[float] = pd.Field(alias="viscousWaveSpeedScale")
    extra_dissipation: Optional[float] = pd.Field(alias="extraDissipation")
    pressure_correction_solver: Optional[PressureCorrectionSolver] = pd.Field(
        alias="pressureCorrectionSolver"
    )
    numerical_dissipation_factor: Optional[pd.confloat(ge=0.01, le=1)] = pd.Field(
        alias="numericalDissipationFactor"
    )
    randomizer: Optional[LinearIterationsRandomizer] = pd.Field()
    linear_solver_config: Optional[LinearSolverPrivate] = pd.Field(
        alias="linearSolver",
        default=LinearSolverPrivate()
    )


class TurbulenceModelSolverPrivate(TurbulenceModelSolver):
    """:class:`TurbulenceModelSolverPrivate` class"""

    randomizer: Optional[LinearIterationsRandomizer] = pd.Field()
    kappa_MUSCL: Optional[pd.confloat(ge=-1, le=1)] = pd.Field(alias="kappaMUSCL")
    linear_iterations: Optional[PositiveInt] = pd.Field(alias="linearIterations")
    linear_solver: Optional[LinearSolverPrivate] = pd.Field(
        alias="linearSolver",
        default=LinearSolverPrivate()
    )


class TurbulenceModelSolverSSTPrivate(TurbulenceModelSolverPrivate):
    """:class:`TurbulenceModelSolverSSTPrivate` class"""

    model_type: Literal["kOmegaSST"] = pd.Field("kOmegaSST", alias="modelType", const=True)


class TurbulenceModelSolverSAPrivate(TurbulenceModelSolverPrivate):
    """:class:`TurbulenceModelSolverSAPrivate` class"""

    model_type: Literal["SpalartAllmaras"] = pd.Field(
        "SpalartAllmaras", alias="modelType", const=True
    )
    rotation_correction: Optional[bool] = pd.Field(alias="rotationCorrection")


TurbulenceModelSolversPrivate = Union[
    TurbulenceModelSolverSAPrivate,
    TurbulenceModelSolverSSTPrivate,
    NoneSolver
]


class HeatEquationSolverPrivate(HeatEquationSolver):
    """:class:`HeatEquationSolverPrivate` class"""

    order_of_accuracy: Optional[Literal[1, 2]] = pd.Field(alias="orderOfAccuracy")
    model_type: Literal["HeatEquation"] = pd.Field("HeatEquation", alias="modelType")
    CFL_multiplier: Optional[PositiveFloat] = pd.Field(alias="CFLMultiplier")
    update_jacobian_frequency: Optional[PositiveInt] = pd.Field(alias="updateJacobianFrequency")
    max_force_jac_update_physical_steps: Optional[NonNegativeInt] = pd.Field(
        alias="maxForceJacUpdatePhysicalSteps"
    )
    linear_solver: Optional[LinearSolverPrivate] = pd.Field(
        alias="linearSolver",
        default=LinearSolverPrivate()
    )


class TransitionModelSolverPrivate(TransitionModelSolver):
    """:class:`TransitionModelSolverPrivate` class"""

    reconstruction_gradient_limiter: Optional[pd.confloat(ge=0, le=2)] = pd.Field(
        alias="reconstructionGradientLimiter"
    )
    CFL_multiplier: Optional[PositiveFloat] = pd.Field(alias="CFLMultiplier")
    linear_iterations: Optional[PositiveInt] = pd.Field(alias="linearIterations")
    randomizer: Optional[LinearIterationsRandomizer] = pd.Field()


class BETDiskPrivate(BETDisk):
    """:class:`BETDiskPrivate` class"""

    volume_name: Optional[str] = pd.Field(alias="volumeName")


class GeometryLegacy(Geometry, LegacyModel):
    ref_area: Optional[float] = pd.Field(alias="refArea", default_factory=lambda: 1.0)
    moment_center: Optional[Coordinate] = pd.Field(alias="momentCenter")
    moment_length: Optional[Coordinate] = pd.Field(alias="momentLength")

    def update_model(self) -> Flow360BaseModel:
        # Apply items from comments
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

        if self.comments.get("timeStepSizeInSeconds") is not None:
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


class Flow360ParamsPrivate(Flow360Params):
    """
    Flow360 solver parameters (legacy version for back-compatibility only)
    """

    navier_stokes_solver: Optional[NavierStokesSolverPrivate] = pd.Field(alias="navierStokesSolver")
    turbulence_model_solver: Optional[TurbulenceModelSolversPrivate] = pd.Field(
        alias="turbulenceModelSolver", discriminator="model_type"
    )
    transition_model_solver: Optional[TransitionModelSolverPrivate] = pd.Field(
        alias="transitionModelSolver"
    )
    heat_equation_solver: Optional[HeatEquationSolverPrivate] = pd.Field(alias="heatEquationSolver")
    bet_disks: Optional[List[BETDiskPrivate]] = pd.Field(alias="BETDisks")
    sliding_interfaces: Optional[List[SlidingInterfaceLegacy]] = pd.Field(alias="slidingInterfaces")
    surface_output: Optional[SurfaceOutputPrivate] = pd.Field(alias="surfaceOutput")
    volume_output: Optional[VolumeOutputPrivate] = pd.Field(alias="volumeOutput")
    slice_output: Optional[SliceOutputPrivate] = pd.Field(alias="sliceOutput")


class Flow360ParamsLegacy(Flow360ParamsPrivate, LegacyModel):
    geometry: Optional[GeometryLegacy] = pd.Field()
    freestream: Optional[FreestreamLegacy] = pd.Field()
    time_stepping: Optional[TimeSteppingLegacy] = pd.Field(alias="timeStepping")
    sliding_interfaces: Optional[List[SlidingInterfaceLegacy]] = pd.Field(alias="slidingInterfaces")
    surface_output: Optional[SurfaceOutputLegacy] = pd.Field(alias="surfaceOutput")
    volume_output: Optional[VolumeOutputLegacy] = pd.Field(alias="volumeOutput")
    slice_output: Optional[SliceOutputLegacy] = pd.Field(alias="sliceOutput")

    def update_model(self) -> Flow360BaseModel:
        model = Flow360Params()
        model.geometry = self.geometry.update_model()
        model.boundaries = self.boundaries
        model.initial_condition = self.initial_condition
        model.time_stepping = self.time_stepping.update_model()
        model.navier_stokes_solver = self.navier_stokes_solver
        #model.turbulence_model_solver = self.turbulence_model_solver
        #model.transition_model_solver = self.transition_model_solver
        #model.heat_equation_solver = self.heat_equation_solver
        #model.freestream = self.freestream.update_model()
        model.bet_disks = self.bet_disks
        model.actuator_disks = self.actuator_disks
        model.porous_media = self.porous_media
        model.user_defined_dynamics = self.user_defined_dynamics
        #model.surface_output = self.surface_output.update_model()
        #model.volume_output = self.volume_output.update_model()
        #model.slice_output = self.slice_output.update_model()
        model.iso_surface_output = self.iso_surface_output
        model.monitor_output = self.monitor_output

        volume_zones = {}
        for interface in self.sliding_interfaces:
            volume_zone = interface.update_model()
            volume_zones[interface.volume_name] = volume_zone
        model.volume_zones = VolumeZones(**volume_zones)

        model.aeroacoustic_output = self.aeroacoustic_output

        return model
