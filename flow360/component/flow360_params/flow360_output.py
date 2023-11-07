from enum import Enum
from typing import Literal, Union, List, Optional

from .params_base import Flow360BaseModel, Flow360SortableBaseModel, _self_named_property_validator

import pydantic as pd

from ..types import PositiveInt


class OutputFormat(Enum):
    PARAVIEW = "paraview",
    TECPLOT = "tecplot",
    BOTH = "both"


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
    # TODO: Surfaces - how to handle self-named properties with one possible type


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
    # TODO: Slices - self-named properties


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


class MonitorOutput(Flow360BaseModel):
    output_fields: Optional[List[str]] = pd.Field(alias="outputFields")
    # TODO: Monitors - self-named properties


class IsoSurfaceOutput(Flow360BaseModel):
    output_format: Optional[OutputFormat] = pd.Field(alias="outputFormat")
    animation_frequency: Optional[Union[PositiveInt, Literal[-1]]] = pd.Field(alias="animationFrequency")
    animation_frequency_offset: Optional[int] = pd.Field(alias="animationFrequencyOffset")
    output_fields: Optional[List[str]] = pd.Field(alias="outputFields")
    coarsen_iterations: Optional[int] = pd.Field(alias="coarsenIterations")
    # TODO: Isosurfaces - self-named properties

