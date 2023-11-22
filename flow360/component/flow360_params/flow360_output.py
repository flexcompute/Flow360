"""
Flow360 output parameters models
"""
from abc import ABCMeta
from typing import List, Literal, Optional, Union, get_args, Dict

import pydantic as pd

from ..types import Coordinate, PositiveInt
from .flow360_fields import output_names
from .params_base import (
    Flow360BaseModel,
    Flow360SortableBaseModel,
    _self_named_property_validator,
)


OutputFormat = Literal["paraview", "tecplot", "both"]

CommonOutputFields = List[Literal[*tuple(output_names(["common"]))]]
SurfaceOutputFields = List[Literal[*tuple(output_names(["common", "surface"]))]]
SliceOutputFields = List[Literal[*tuple(output_names(["common", "slice"]))]]
VolumeOutputFields = List[Literal[*tuple(output_names(["common", "volume"]))]]
IsoSurfaceOutputField = Literal[*tuple(output_names(["common", "iso_surface"]))]

_common_long = output_names(["common"], False)
_surface_long = output_names(["common", "surface"], False)
_slice_long = output_names(["common", "slice"], False)
_volume_long = output_names(["common", "volume"], False)
_iso_surface_long = output_names(["common", "iso_surface"], False)


def _filter_fields(fields, field_filters):
    return [field for field in fields if field in field_filters]


class AnimationSettings(Flow360BaseModel):
    frequency: Optional[PositiveInt] = pd.Field(alias="frequency")
    frequency_offset: Optional[int] = pd.Field(
        alias="frequencyOffset", displayed="Frequency offset"
    )
    frequency_time_average: Optional[PositiveInt] = pd.Field(
        alias="frequencyTimeAverage", displayed="Frequency time average"
    )
    frequency_time_average_offset: Optional[int] = pd.Field(
        alias="frequencyTimeAverageOffset", displayed="Frequency time average offset"
    )


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


class Surface(Flow360BaseModel):
    """:class:`Surface` class"""

    output_fields: Optional[SurfaceOutputFields] = pd.Field(
        alias="outputFields"
    )

    class Config:
        @staticmethod
        def schema_extra(schema, model):
            fields = schema["properties"]["outputFields"]["items"]["enum"]
            schema["properties"]["outputFields"]["items"]["enum"] = _filter_fields(fields, _surface_long)


class _GenericSurfaceWrapper(Flow360BaseModel):
    """:class:`_GenericSurfaceWrapper` class"""

    v: Surface


class Surfaces(Flow360SortableBaseModel):
    """:class:`Surfaces` class"""

    @classmethod
    def get_subtypes(cls) -> list:
        return [_GenericSurfaceWrapper.__fields__["v"].type_]

    # pylint: disable=no-self-argument
    @pd.root_validator(pre=True)
    def validate_surface(cls, values):
        """
        root validator
        """
        return _self_named_property_validator(
            values, _GenericSurfaceWrapper, msg="is not any of supported surface types."
        )


class SurfaceOutput(Flow360BaseModel):
    """:class:`SurfaceOutput` class"""

    output_format: Optional[OutputFormat] = pd.Field(alias="outputFormat")
    animation: Optional[AnimationSettings] = pd.Field(alias="animation")
    compute_time_averages: Optional[bool] = pd.Field(alias="computeTimeAverages")
    write_single_file: Optional[bool] = pd.Field(alias="writeSingleFile")
    start_average_integration_step: Optional[bool] = pd.Field(alias="startAverageIntegrationStep")
    output_fields: Optional[SurfaceOutputFields] = pd.Field(
        alias="outputFields"
    )
    surfaces: Optional[Surfaces] = pd.Field()

    class Config:
        @staticmethod
        def schema_extra(schema, model):
            fields = schema["properties"]["outputFields"]["items"]["enum"]
            schema["properties"]["outputFields"]["items"]["enum"] = _filter_fields(fields, _surface_long)


class SurfaceOutputPrivate(SurfaceOutput):
    """:class:`SurfaceOutputPrivate` class"""

    wall_function_metric: Optional[bool] = pd.Field(alias="wallFunctionMetric")
    node_moments_per_unit_area: Optional[bool] = pd.Field(alias="nodeMomentsPerUnitArea")
    residual_sa: Optional[bool] = pd.Field(alias="residualSA")
    coarsen_iterations: Optional[int] = pd.Field(alias="coarsenIterations")


class SurfaceOutputLegacy(SurfaceOutputPrivate, OutputLegacy):
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


class Slice(Flow360BaseModel):
    """:class:`NamedSlice` class"""

    slice_normal: Coordinate = pd.Field(alias="sliceNormal")
    slice_origin: Coordinate = pd.Field(alias="sliceOrigin")
    output_fields: Optional[CommonOutputFields] = pd.Field(alias="outputFields")

    class Config:
        @staticmethod
        def schema_extra(schema, model):
            fields = schema["properties"]["outputFields"]["items"]["enum"]
            schema["properties"]["outputFields"]["items"]["enum"] = _filter_fields(fields, _slice_long)


class Slices(Flow360SortableBaseModel):
    """:class:`SelfNamedSlices` class"""

    @classmethod
    def get_subtypes(cls) -> list:
        return [_GenericSliceWrapper.__fields__["v"].type_]

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


class SliceOutput(Flow360BaseModel):
    """:class:`SliceOutput` class"""

    output_format: Optional[OutputFormat] = pd.Field(alias="outputFormat")
    animation: Optional[AnimationSettings] = pd.Field(alias="animation")
    output_fields: Optional[SliceOutputFields] = pd.Field(
        alias="outputFields"
    )
    slices: Optional[Slices]

    class Config:
        @staticmethod
        def schema_extra(schema, model):
            fields = schema["properties"]["outputFields"]["items"]["enum"]
            schema["properties"]["outputFields"]["items"]["enum"] = _filter_fields(fields, _slice_long)


class SliceOutputPrivate(SliceOutput):
    """:class:`SliceOutputPrivate` class"""

    coarsen_iterations: Optional[int] = pd.Field(alias="coarsenIterations")


class SliceOutputLegacy(SliceOutputPrivate, OutputLegacy):
    """:class:`SliceOutputLegacy` class"""

    bet_metrics: Optional[bool] = pd.Field(alias="betMetrics")
    bet_metrics_per_disk: Optional[bool] = pd.Field(alias="betMetricsPerDisk")


class VolumeOutput(Flow360BaseModel):
    """:class:`VolumeOutput` class"""

    output_format: Optional[OutputFormat] = pd.Field(alias="outputFormat")
    animation: Optional[AnimationSettings] = pd.Field(alias="animation")
    compute_time_averages: Optional[bool] = pd.Field(alias="computeTimeAverages")
    start_average_integration_step: Optional[int] = pd.Field(alias="startAverageIntegrationStep")
    output_fields: Optional[VolumeOutputFields] = pd.Field(
        alias="outputFields"
    )

    class Config:
        @staticmethod
        def schema_extra(schema, model):
            fields = schema["properties"]["outputFields"]["items"]["enum"]
            schema["properties"]["outputFields"]["items"]["enum"] = _filter_fields(fields, _volume_long)


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


class VolumeOutputLegacy(VolumeOutputPrivate, OutputLegacy):
    """:class:`VolumeOutputLegacy` class"""

    bet_metrics: Optional[bool] = pd.Field(alias="betMetrics")
    bet_metrics_per_disk: Optional[bool] = pd.Field(alias="betMetricsPerDisk")


class MonitorBase(Flow360BaseModel, metaclass=ABCMeta):
    """:class:`MonitorBase` class"""

    type: Optional[str]


class SurfaceIntegralMonitor(MonitorBase):
    """:class:`SurfaceIntegralMonitor` class"""

    type = pd.Field("surfaceIntegral", const=True)
    surfaces: Optional[List[str]] = pd.Field()
    output_fields: Optional[CommonOutputFields] = pd.Field(alias="outputFields")

    class Config:
        @staticmethod
        def schema_extra(schema, model):
            fields = schema["properties"]["outputFields"]["items"]["enum"]
            schema["properties"]["outputFields"]["items"]["enum"] = _filter_fields(fields, _common_long)


class ProbeMonitor(MonitorBase):
    """:class:`ProbeMonitor` class"""

    type = pd.Field("probe", const=True)
    monitor_locations: Optional[List[Coordinate]] = pd.Field(alias="monitorLocations")
    output_fields: Optional[CommonOutputFields] = pd.Field(alias="outputFields")

    class Config:
        @staticmethod
        def schema_extra(schema, model):
            fields = schema["properties"]["outputFields"]["items"]["enum"]
            schema["properties"]["outputFields"]["items"]["enum"] = _filter_fields(fields, _common_long)


MonitorType = Union[SurfaceIntegralMonitor, ProbeMonitor]


class _GenericMonitorWrapper(Flow360BaseModel):
    """:class:`_GenericMonitorWrapper` class"""

    v: MonitorType


class Monitors(Flow360SortableBaseModel):
    """:class:`Monitors` class"""

    @classmethod
    def get_subtypes(cls) -> list:
        return list(get_args(_GenericMonitorWrapper.__fields__["v"].type_))

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
    output_fields: Optional[CommonOutputFields] = pd.Field(alias="outputFields")

    class Config:
        @staticmethod
        def schema_extra(schema, model):
            fields = schema["properties"]["outputFields"]["items"]["enum"]
            schema["properties"]["outputFields"]["items"]["enum"] = _filter_fields(fields, _common_long)


class IsoSurface(Flow360BaseModel):
    """:class:`IsoSurface` class"""

    surface_field: Optional[IsoSurfaceOutputField] = pd.Field(alias="surfaceField")
    surface_field_magnitude: Optional[float] = pd.Field(alias="surfaceFieldMagnitude")
    output_fields: Optional[CommonOutputFields] = pd.Field(alias="outputFields")

    class Config:
        @staticmethod
        def schema_extra(schema, model):
            fields = schema["properties"]["outputFields"]["items"]["enum"]
            schema["properties"]["outputFields"]["items"]["enum"] = _filter_fields(fields, _common_long)
            fields = schema["properties"]["surfaceField"]["enum"]
            schema["properties"]["surfaceField"]["enum"] = _filter_fields(fields, _iso_surface_long)


class _GenericIsoSurfaceWrapper(Flow360BaseModel):
    """:class:`_GenericIsoSurfaceWrapper` class"""

    v: IsoSurface


class IsoSurfaces(Flow360SortableBaseModel):
    """:class:`IsoSurfaces` class"""

    @classmethod
    def get_subtypes(cls) -> list:
        return [_GenericIsoSurfaceWrapper.__fields__["v"].type_]

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
    animation: Optional[AnimationSettings] = pd.Field(alias="animation")
    iso_surfaces: Optional[IsoSurfaces] = pd.Field(alias="isoSurfaces")


class IsoSurfaceOutputPrivate(IsoSurfaceOutput):
    """:class:`IsoSurfaceOutputPrivate` class"""

    coarsen_iterations: Optional[int] = pd.Field(alias="coarsenIterations")
    output_fields: Optional[CommonOutputFields] = pd.Field(alias="outputFields")

    class Config:
        @staticmethod
        def schema_extra(schema, model):
            fields = schema["properties"]["outputFields"]["items"]["enum"]
            schema["properties"]["outputFields"]["items"]["enum"] = _filter_fields(fields, _common_long)
