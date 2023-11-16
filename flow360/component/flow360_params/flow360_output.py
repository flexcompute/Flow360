"""
Flow360 output parameters models
"""
from abc import ABC
from typing import List, Literal, Optional, Union

import pydantic as pd

from ..types import Coordinate, PositiveInt
from .flow360_fields import (
    CommonFieldVars,
    IsoSurfaceFieldVars,
    SurfaceFieldVars,
    VolumeSliceFieldVars,
)
from .params_base import (
    Flow360BaseModel,
    Flow360SortableBaseModel,
    _self_named_property_validator,
)


class OutputBase(ABC, Flow360BaseModel):
    """
    Base class for common output parameters
    """

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


OutputFormat = Union[Literal["paraview"], Literal["tecplot"], Literal["both"]]


class Surface(Flow360BaseModel):
    """:class:`Surface` class"""

    output_fields: Optional[List[str]] = pd.Field(alias="outputFields")


class _GenericSurfaceWrapper(Flow360BaseModel):
    """:class:`_GenericSurfaceWrapper` class"""

    v: Surface


class Surfaces(Flow360SortableBaseModel):
    """:class:`Surfaces` class"""

    # pylint: disable=no-self-argument
    @pd.root_validator(pre=True)
    def validate_surface(cls, values):
        """
        root validator
        """
        return _self_named_property_validator(
            values, _GenericSurfaceWrapper, msg="is not any of supported surface types."
        )


class SurfaceOutput(OutputBase):
    """:class:`SurfaceOutput` class"""

    output_format: Optional[OutputFormat] = pd.Field(alias="outputFormat")
    animation_frequency: Optional[Union[PositiveInt, Literal[-1]]] = pd.Field(
        alias="animationFrequency"
    )
    animation_frequency_offset: Optional[int] = pd.Field(alias="animationFrequencyOffset")
    animation_frequency_time_average: Optional[Union[PositiveInt, Literal[-1]]] = pd.Field(
        alias="animationFrequencyTimeAverage"
    )
    animation_frequency_time_average_offset: Optional[int] = pd.Field(
        alias="animationFrequencyTimeAverageOffset"
    )
    compute_time_averages: Optional[bool] = pd.Field(alias="computeTimeAverages")
    velocity_relative: Optional[bool] = pd.Field(alias="VelocityRelative")
    surfaces: Optional[Surfaces] = pd.Field()
    write_single_file: Optional[bool] = pd.Field(alias="writeSingleFile")
    primitive_vars: Optional[bool] = pd.Field(alias="primitiveVars")
    start_average_integration_step: Optional[bool] = pd.Field(alias="startAverageIntegrationStep")
    output_fields: Optional[List[Union[CommonFieldVars, SurfaceFieldVars]]] = pd.Field(
        alias="outputFields"
    )

    Cf: Optional[bool] = pd.Field(alias="Cf")
    Cf_vec: Optional[bool] = pd.Field(alias="CfVec")
    Cf_normal: Optional[bool] = pd.Field(alias="CfNormal")
    Cf_tangent: Optional[bool] = pd.Field(alias="CfTangent")
    y_plus: Optional[bool] = pd.Field(alias="yPlus")
    wall_distance: Optional[bool] = pd.Field(alias="wallDistance")
    heat_flux: Optional[bool] = pd.Field(alias="heatFlux")
    node_forces_per_unit_area: Optional[bool] = pd.Field(alias="nodeForcesPerUnitArea")
    node_normals: Optional[bool] = pd.Field(alias="nodeNormals")

    _wall_function_metric: Optional[bool] = pd.Field(alias="wallFunctionMetric")
    _node_moments_per_unit_area: Optional[bool] = pd.Field(alias="nodeMomentsPerUnitArea")
    _residual_sa: Optional[bool] = pd.Field(alias="residualSA")
    _coarsen_iterations: Optional[int] = pd.Field(alias="coarsenIterations")


class Slice(Flow360BaseModel):
    """:class:`NamedSlice` class"""

    slice_normal: Coordinate = pd.Field(alias="sliceNormal")
    slice_origin: Coordinate = pd.Field(alias="sliceOrigin")
    output_fields: Optional[List[str]] = pd.Field(alias="outputFields")


class Slices(Flow360SortableBaseModel):
    """:class:`SelfNamedSlices` class"""

    # pylint: disable=no-self-argument
    @pd.root_validator(pre=True)
    def validate_slice(cls, values):
        """
        root validator
        """
        return _self_named_property_validator(
            values, _GenericSliceWrapper, msg="is not any of supported slice types."
        )


class _GenericSliceWrapper(Flow360BaseModel):
    """:class:`_GenericMonitorWrapper` class"""

    v: Slice


class SliceOutput(OutputBase):
    """:class:`SliceOutput` class"""

    output_format: Optional[OutputFormat] = pd.Field(alias="outputFormat")
    animation_frequency: Optional[Union[PositiveInt, Literal[-1]]] = pd.Field(
        alias="animationFrequency"
    )
    animation_frequency_offset: Optional[int] = pd.Field(alias="animationFrequencyOffset")
    slices: Optional[Slices]
    output_fields: Optional[List[Union[CommonFieldVars, VolumeSliceFieldVars]]] = pd.Field(
        alias="outputFields"
    )

    bet_metrics: Optional[bool] = pd.Field(alias="betMetrics")
    bet_metrics_per_disk: Optional[bool] = pd.Field(alias="betMetricsPerDisk")

    _coarsen_iterations: Optional[int] = pd.Field(alias="coarsenIterations")


class VolumeOutput(OutputBase):
    """:class:`VolumeOutput` class"""

    output_format: Optional[OutputFormat] = pd.Field(alias="outputFormat")
    animation_frequency: Optional[Union[PositiveInt, Literal[-1]]] = pd.Field(
        alias="animationFrequency"
    )
    animation_frequency_offset: Optional[int] = pd.Field(alias="animationFrequencyOffset")
    animation_frequency_time_average: Optional[Union[PositiveInt, Literal[-1]]] = pd.Field(
        alias="animationFrequencyTimeAverage"
    )
    animation_frequency_time_average_offset: Optional[int] = pd.Field(
        alias="animationFrequencyTimeAverageOffset"
    )
    compute_time_averages: Optional[bool] = pd.Field(alias="computeTimeAverages")
    start_average_integration_step: Optional[int] = pd.Field(alias="startAverageIntegrationStep")
    output_fields: Optional[List[Union[CommonFieldVars, VolumeSliceFieldVars]]] = pd.Field(
        alias="outputFields"
    )

    bet_metrics: Optional[bool] = pd.Field(alias="betMetrics")
    bet_metrics_per_disk: Optional[bool] = pd.Field(alias="betMetricsPerDisk")

    _write_single_file: Optional[bool] = pd.Field(alias="writeSingleFile")
    _write_distributed_file: Optional[bool] = pd.Field(alias="writeDistributedFile")
    _residual_components_sa: Optional[bool] = pd.Field(alias="residualComponentsSA")
    _wall_distance_dir: Optional[bool] = pd.Field(alias="wallDistanceDir")
    _velocity_relative: Optional[bool] = pd.Field(alias="VelocityRelative")
    _debug_transition: Optional[bool] = pd.Field(alias="debugTransition")
    _debug_turbulence: Optional[bool] = pd.Field(alias="debugTurbulence")
    _debug_navier_stokes: Optional[bool] = pd.Field(alias="debugNavierStokes")


class MonitorBase(ABC, Flow360BaseModel):
    """:class:`MonitorBase` class"""

    type: Optional[str]


class SurfaceIntegralMonitor(MonitorBase):
    """:class:`SurfaceIntegralMonitor` class"""

    type = pd.Field("surfaceIntegral", const=True)
    surfaces: Optional[List[str]] = pd.Field()
    output_fields: Optional[List[CommonFieldVars]] = pd.Field(alias="outputFields")


class ProbeMonitor(MonitorBase):
    """:class:`ProbeMonitor` class"""

    type = pd.Field("probe", const=True)
    monitor_locations: Optional[List[Coordinate]]
    output_fields: Optional[List[CommonFieldVars]] = pd.Field(alias="outputFields")


MonitorType = Union[SurfaceIntegralMonitor, ProbeMonitor]


class _GenericMonitorWrapper(Flow360BaseModel):
    """:class:`_GenericMonitorWrapper` class"""

    v: MonitorType


class Monitors(Flow360SortableBaseModel):
    """:class:`Monitors` class"""

    # pylint: disable=no-self-argument
    @pd.root_validator(pre=True)
    def validate_monitor(cls, values):
        """
        root validator
        """
        return _self_named_property_validator(
            values, _GenericMonitorWrapper, msg="is not any of supported monitor types."
        )


class MonitorOutput(Flow360BaseModel):
    """:class:`MonitorOutput` class"""

    monitors: Optional[Monitors] = pd.Field()
    output_fields: Optional[List[CommonFieldVars]] = pd.Field(alias="outputFields")


class IsoSurface(Flow360BaseModel):
    """:class:`IsoSurface` class"""

    surface_field: Optional[List[IsoSurfaceFieldVars]] = pd.Field(alias="surfaceField")
    surface_field_magnitude: Optional[float] = pd.Field(alias="surfaceFieldMagnitude")
    output_fields: Optional[List[CommonFieldVars]] = pd.Field(alias="outputFields")


class _GenericIsoSurfaceWrapper(Flow360BaseModel):
    """:class:`_GenericIsoSurfaceWrapper` class"""

    v: IsoSurface


class IsoSurfaces(Flow360SortableBaseModel):
    """:class:`IsoSurfaces` class"""

    # pylint: disable=no-self-argument
    @pd.root_validator(pre=True)
    def validate_monitor(cls, values):
        """
        root validator
        """
        return _self_named_property_validator(
            values, _GenericIsoSurfaceWrapper, msg="is not any of supported surface types."
        )


class IsoSurfaceOutput(Flow360BaseModel):
    """:class:`IsoSurfaceOutput` class"""

    output_format: Optional[OutputFormat] = pd.Field(alias="outputFormat")
    animation_frequency: Optional[Union[PositiveInt, Literal[-1]]] = pd.Field(
        alias="animationFrequency"
    )
    animation_frequency_offset: Optional[int] = pd.Field(alias="animationFrequencyOffset")
    iso_surfaces: Optional[IsoSurfaces] = pd.Field(alias="isoSurfaces")

    _coarsen_iterations: Optional[int] = pd.Field(alias="coarsenIterations")
    _output_fields: Optional[List[CommonFieldVars]] = pd.Field(alias="outputFields")
