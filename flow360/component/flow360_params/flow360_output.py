from abc import ABC
from enum import Enum
from typing import Literal, Union, List, Optional

from .params_base import Flow360BaseModel, Flow360SortableBaseModel, _self_named_property_validator

import pydantic as pd

from ..types import PositiveInt, Coordinate


class OutputFormat(Enum):
    PARAVIEW = "paraview"
    TECPLOT = "tecplot"
    BOTH = "both"


class Surface(Flow360BaseModel):
    output_fields: Optional[List[str]] = pd.Field(alias="outputFields")


class _GenericSurfaceWrapper(Flow360BaseModel):
    v: Surface


class Surfaces(Flow360SortableBaseModel):
    @pd.root_validator(pre=True)
    def validate_monitor(cls, values):
        return _self_named_property_validator(
            values, _GenericSurfaceWrapper, msg="is not any of supported surface types."
        )


class SurfaceOutput(Flow360BaseModel):
    output_format: Optional[OutputFormat] = pd.Field(alias="outputFormat")
    animation_frequency: Optional[Union[PositiveInt, Literal[-1]]] = pd.Field(alias="animationFrequency")
    animation_frequency_offset: Optional[int] = pd.Field(alias="animationFrequencyOffset")
    animation_frequency_time_average: Optional[Union[PositiveInt, Literal[-1]]] = pd.Field(alias="animationFrequencyTimeAverage")
    animation_frequency_time_average_offset: Optional[int] = pd.Field(alias="animationFrequencyTimeAverageOffset")
    primitive_vars: Optional[bool] = pd.Field(alias="primitiveVars")
    cp: Optional[bool] = pd.Field(alias="Cp")
    cf: Optional[bool] = pd.Field(alias="Cf")
    cf_vec: Optional[bool] = pd.Field(alias="CfVec")
    y_plus: Optional[bool] = pd.Field(alias="yPlus")
    wall_function_metric: Optional[bool] = pd.Field(alias="wallFunctionMetric")
    wall_distance: Optional[bool] = pd.Field(alias="wallDistance")
    mach: Optional[bool] = pd.Field(alias="Mach")
    residual_sa: Optional[bool] = pd.Field(alias="residualSA")
    node_forces_per_unit_area: Optional[bool] = pd.Field(alias="nodeForcesPerUnitArea")
    node_moments_per_unit_area: Optional[bool] = pd.Field(alias="nodeMomentsPerUnitArea")
    node_normals: Optional[bool] = pd.Field(alias="nodeNormals")
    cf_normal: Optional[bool] = pd.Field(alias="CfNormal")
    cf_tangent: Optional[bool] = pd.Field(alias="CfTangent")
    heat_flux: Optional[bool] = pd.Field(alias="heatFlux")
    compute_time_averages: Optional[bool] = pd.Field(alias="computeTimeAverages")
    write_single_file: Optional[bool] = pd.Field(alias="writeSingleFile")
    mut_ratio: Optional[bool] = pd.Field(alias="mutRatio")
    residual_navier_stokes: Optional[bool] = pd.Field(alias="residualNavierStokes")
    residual_turbulence: Optional[bool] = pd.Field(alias="residualTurbulence")
    residual_transition: Optional[bool] = pd.Field(alias="residualTransition")
    solution_navier_stokes: Optional[bool] = pd.Field(alias="solutionNavierStokes")
    solution_turbulence: Optional[bool] = pd.Field(alias="solutionTurbulence")
    solution_transition: Optional[bool] = pd.Field(alias="solutionTransition")
    velocity_relative: Optional[bool] = pd.Field(alias="VelocityRelative")
    start_average_integration_step: Optional[bool] = pd.Field(alias="startAverageIntegrationStep")
    output_fields: Optional[List[str]] = pd.Field(alias="outputFields")
    coarsen_iterations: Optional[int] = pd.Field(alias="coarsenIterations")
    surfaces: Optional[Surfaces] = pd.Field()


class SliceBase(ABC, Flow360BaseModel):
    slice_normal: Coordinate = pd.Field(alias="sliceNormal")
    slice_origin: Coordinate = pd.Field(alias="sliceOrigin")


class NamedSlice(SliceBase):
    slice_name: str = pd.Field(alias="sliceName")


class Slice(SliceBase):
    output_fields: Optional[List[str]] = pd.Field(alias="outputFields")


SliceType = Union[Slice, NamedSlice]


class _GenericSliceWrapper(Flow360BaseModel):
    v: SliceType


class Slices(Flow360SortableBaseModel):
    @pd.root_validator(pre=True)
    def validate_monitor(cls, values):
        return _self_named_property_validator(
            values, _GenericSliceWrapper, msg="is not any of supported slice types."
        )


class SliceOutput(Flow360BaseModel):
    output_format: Optional[OutputFormat] = pd.Field(alias="outputFormat")
    animation_frequency: Optional[Union[PositiveInt, Literal[-1]]] = pd.Field(alias="animationFrequency")
    animation_frequency_offset: Optional[int] = pd.Field(alias="animationFrequencyOffset")
    primitive_vars: Optional[bool] = pd.Field(alias="primitiveVars")
    vorticity: Optional[bool] = pd.Field()
    t: Optional[bool] = pd.Field(alias="T")
    s: Optional[bool] = pd.Field()
    cp: Optional[bool] = pd.Field(alias="Cp")
    mut: Optional[bool] = pd.Field()
    mut_ratio: Optional[bool] = pd.Field(alias="mutRatio")
    wall_distance: Optional[bool] = pd.Field(alias="wallDistance")
    residual_turbulence: Optional[bool] = pd.Field(alias="residualTurbulence")
    residual_navier_stokes: Optional[bool] = pd.Field(alias="residualNavierStokes")
    solution_navier_stokes: Optional[bool] = pd.Field(alias="solutionNavierStokes")
    mach: Optional[bool] = pd.Field(alias="Mach")
    grad_w: Optional[bool] = pd.Field(alias="gradW")
    bet_metrics: Optional[bool] = pd.Field(alias="betMetrics")
    bet_metrics_per_disk: Optional[bool] = pd.Field(alias="betMetricsPerDisk")
    q_criterion: Optional[bool] = pd.Field(alias="qCriterion")
    output_fields: Optional[List[str]] = pd.Field(alias="outputFields")
    coarsen_iterations: Optional[int] = pd.Field(alias="coarsenIterations")
    slices: Optional[Slices]


class VolumeOutput(Flow360BaseModel):
    output_format: Optional[OutputFormat] = pd.Field(alias="outputFormat")
    animation_frequency: Optional[Union[PositiveInt, Literal[-1]]] = pd.Field(alias="animationFrequency")
    animation_frequency_offset: Optional[int] = pd.Field(alias="animationFrequencyOffset")
    animation_frequency_time_average: Optional[Union[PositiveInt, Literal[-1]]] = pd.Field(alias="animationFrequencyTimeAverage")
    animation_frequency_time_average_offset: Optional[int] = pd.Field(alias="animationFrequencyTimeAverageOffset")
    compute_time_averages: Optional[bool] = pd.Field(alias="computeTimeAverages")
    write_single_file: Optional[bool] = pd.Field(alias="writeSingleFile")
    write_distributed_file: Optional[bool] = pd.Field(alias="writeDistributedFile")
    start_average_integration_step: Optional[bool] = pd.Field(alias="startAverageIntegrationStep")
    primitive_vars: Optional[bool] = pd.Field(alias="primitiveVars")
    vorticity: Optional[bool] = pd.Field()
    residual_navier_stokes: Optional[bool] = pd.Field(alias="residualNavierStokes")
    residual_turbulence: Optional[bool] = pd.Field(alias="residualTurbulence")
    residual_transition: Optional[bool] = pd.Field(alias="residualTransition")
    residual_components_sa: Optional[bool] = pd.Field(alias="residualComponentsSA")
    solution_navier_stokes: Optional[bool] = pd.Field(alias="solutionNavierStokes")
    solution_turbulence: Optional[bool] = pd.Field(alias="solutionTurbulence")
    solution_transition: Optional[bool] = pd.Field(alias="solutionTransition")
    t: Optional[bool] = pd.Field(alias="T")
    s: Optional[bool] = pd.Field()
    cp: Optional[bool] = pd.Field(alias="Cp")
    mut: Optional[bool] = pd.Field()
    nu_hat: Optional[bool] = pd.Field(alias="nuHat")
    k_omega: Optional[bool] = pd.Field(alias="kOmega")
    mut_ratio: Optional[bool] = pd.Field(alias="mutRatio")
    mach: Optional[bool] = pd.Field(alias="Mach")
    grad_w: Optional[bool] = pd.Field(alias="gradW")
    wall_distance_dir: Optional[bool] = pd.Field(alias="wallDistanceDir")
    wall_distance: Optional[bool] = pd.Field(alias="wallDistance")
    velocity_relative: Optional[bool] = pd.Field(alias="VelocityRelative")
    q_criterion: Optional[bool] = pd.Field(alias="qCriterion")
    debug_transition: Optional[bool] = pd.Field(alias="debugTransition")
    debug_turbulence: Optional[bool] = pd.Field(alias="debugTurbulence")
    debug_navier_stokes: Optional[bool] = pd.Field(alias="debugNavierStokes")
    bet_metrics: Optional[bool] = pd.Field(alias="betMetrics")
    bet_metrics_per_disk: Optional[bool] = pd.Field(alias="betMetricsPerDisk")
    output_fields: Optional[List[str]] = pd.Field(alias="outputFields")


class MonitorBase(ABC, Flow360BaseModel):
    type: Optional[str]


class SurfaceIntegralMonitor(MonitorBase):
    type = pd.Field("surfaceIntegral", const=True)
    surfaces: Optional[List[str]] = pd.Field()
    output_fields: Optional[List[str]] = pd.Field(alias="outputFields")


class ProbeMonitor(MonitorBase):
    type = pd.Field("probe", const=True)
    monitor_locations: Optional[List[Coordinate]]
    output_fields: Optional[List[str]] = pd.Field(alias="outputFields")


MonitorType = Union[SurfaceIntegralMonitor, ProbeMonitor]


class _GenericMonitorWrapper(Flow360BaseModel):
    v: MonitorType


class Monitors(Flow360SortableBaseModel):
    @pd.root_validator(pre=True)
    def validate_monitor(cls, values):
        return _self_named_property_validator(
            values, _GenericMonitorWrapper, msg="is not any of supported monitor types."
        )


class MonitorOutput(Flow360BaseModel):
    output_fields: Optional[List[str]] = pd.Field(alias="outputFields")
    monitors: Optional[Monitors] = pd.Field()


class IsoSurface(Flow360BaseModel):
    surface_field: Optional[List]
    output_fields: Optional[List[str]] = pd.Field(alias="outputFields")


class _GenericIsoSurfaceWrapper(Flow360BaseModel):
    v: Surface


class IsoSurfaces(Flow360SortableBaseModel):
    @pd.root_validator(pre=True)
    def validate_monitor(cls, values):
        return _self_named_property_validator(
            values, _GenericIsoSurfaceWrapper, msg="is not any of supported surface types."
        )


class IsoSurfaceOutput(Flow360BaseModel):
    output_format: Optional[OutputFormat] = pd.Field(alias="outputFormat")
    animation_frequency: Optional[Union[PositiveInt, Literal[-1]]] = pd.Field(alias="animationFrequency")
    animation_frequency_offset: Optional[int] = pd.Field(alias="animationFrequencyOffset")
    output_fields: Optional[List[str]] = pd.Field(alias="outputFields")
    coarsen_iterations: Optional[int] = pd.Field(alias="coarsenIterations")
    iso_surfaces: Optional[IsoSurfaces] = pd.Field(alias="isoSurfaces")

