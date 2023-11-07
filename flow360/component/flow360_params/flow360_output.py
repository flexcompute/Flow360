from enum import Enum
from typing import Literal, Union, List, Optional

from .params_base import Flow360BaseModel

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
    output_fields: List[str] = pd.Field(alias="outputFields")