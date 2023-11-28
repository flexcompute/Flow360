"""
Private and legacy field definitions (fields that are not
specified in the documentation but can be used internally
during validation)
"""
from abc import ABCMeta, abstractmethod

import pydantic as pd
from typing import Optional, Literal, Union, List, Any, Dict

import unyt
from pydantic import Extra, NonNegativeFloat

from flow360 import (
    SurfaceOutput,
    SliceOutput,
    VolumeOutput,
    BETDisk,
    Flow360Params, SlidingInterface, Geometry, TimeStepping
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

from flow360.component.types import (
    NonNegativeInt,
    PositiveInt,
    PositiveFloat
)

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


class SlidingInterfacePrivate(SlidingInterface):
    omega: Optional[float] = pd.Field()
    theta: Optional[float] = pd.Field()

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


class GeometryLegacy(Geometry, LegacyModel):
    def update_model(self) -> Flow360BaseModel:
        # Apply mesh units from comments
        model = {
            "momentCenter": self.moment_center.value,
            "momentLength": self.moment_length.value,
            "refArea": self.ref_area.value
        }
        if self.comments.get("meshUnit") is not None:
            unit = unyt.unyt_quantity(1, self.comments["meshUnit"])
            model["meshUnit"] = unit
            model["momentCenter"] *= model["meshUnit"]
            model["momentLength"] *= model["meshUnit"]
            model["refArea"] *= model["meshUnit"]**2

        return Geometry.parse_obj(model)


class FreestreamLegacy(LegacyModel):
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
        pass


class TimeSteppingLegacy(TimeStepping, LegacyModel):
    def update_model(self) -> Flow360BaseModel:
        pass


class SlidingInterfaceLegacy(SlidingInterfacePrivate, LegacyModel):
    def update_model(self) -> Flow360BaseModel:
        pass


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
    sliding_interfaces: Optional[List[SlidingInterfacePrivate]] = pd.Field(alias="slidingInterfaces")
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
        return model
