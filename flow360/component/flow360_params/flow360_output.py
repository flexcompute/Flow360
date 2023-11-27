"""
Flow360 output parameters models
"""
from abc import ABCMeta
from typing import List, Literal, Optional, Union, get_args

import pydantic as pd
from pydantic import conlist

from ..types import Coordinate, PositiveInt
from .flow360_fields import (
    CommonFieldNames,
    CommonFieldNamesFull,
    IsoSurfaceFieldNames,
    IsoSurfaceFieldNamesFull,
    SurfaceFieldNames,
    SurfaceFieldNamesFull,
    VolumeSliceFieldNames,
    VolumeSliceFieldNamesFull,
    get_field_values,
)
from .params_base import (
    Flow360BaseModel,
    Flow360SortableBaseModel,
    _self_named_property_validator,
)

OutputFormat = Literal["paraview", "tecplot", "both"]

CommonFields = Literal[CommonFieldNames, CommonFieldNamesFull]
SurfaceFields = Literal[SurfaceFieldNames, SurfaceFieldNamesFull]
SliceFields = Literal[VolumeSliceFieldNames, VolumeSliceFieldNamesFull]
VolumeFields = Literal[VolumeSliceFieldNames, VolumeSliceFieldNamesFull]
IsoSurfaceFields = Literal[IsoSurfaceFieldNames, IsoSurfaceFieldNamesFull]

CommonOutputFields = conlist(CommonFields, unique_items=True)
SurfaceOutputFields = conlist(SurfaceFields, unique_items=True)
SliceOutputFields = conlist(SliceFields, unique_items=True)
VolumeOutputFields = conlist(VolumeFields, unique_items=True)
IsoSurfaceOutputField = IsoSurfaceFields


def _filter_fields(fields, literal_filter):
    """Take two literals, filter"""
    values = get_field_values(literal_filter)
    fields[:] = [field for field in fields if field in values]


class AnimationSettings(Flow360BaseModel):
    """:class:`AnimationSettings` class"""

    frequency: Optional[PositiveInt] = pd.Field(alias="frequency")
    frequency_offset: Optional[int] = pd.Field(alias="frequencyOffset")


class AnimationSettingsExtended(AnimationSettings):
    """:class:`AnimationSettingsExtended` class"""

    frequency_time_average: Optional[PositiveInt] = pd.Field(alias="frequencyTimeAverage")
    frequency_time_average_offset: Optional[int] = pd.Field(alias="frequencyTimeAverageOffset")


class AnimatedOutput(pd.BaseModel):
    """:class:`AnimatedOutput` class"""

    animation_frequency: Optional[Union[PositiveInt, Literal[-1]]] = pd.Field(
        alias="animationFrequency"
    )
    animation_frequency_offset: Optional[int] = pd.Field(alias="animationFrequencyOffset")
    animation_settings: Optional[AnimationSettings] = pd.Field(alias="animationSettings")

    def to_solver(self):
        """Convert animation settings (UI representation) to solver representation"""
        if self.animation_settings is not None:
            if self.animation_settings.frequency is not None:
                self.animation_frequency = self.animation_settings.frequency
            else:
                self.animation_frequency = -1

            if self.animation_settings.frequency_offset is not None:
                self.animation_frequency_offset = self.animation_settings.frequency_offset
            else:
                self.animation_frequency_offset = 0


class AnimatedOutputExtended(AnimatedOutput):
    """:class:`AnimatedOutputExtended` class"""

    animation_frequency_time_average: Optional[Union[PositiveInt, Literal[-1]]] = pd.Field(
        alias="animationFrequencyTimeAverage"
    )
    animation_frequency_time_average_offset: Optional[int] = pd.Field(
        alias="animationFrequencyTimeAverageOffset"
    )
    animation_settings: Optional[AnimationSettingsExtended] = pd.Field(alias="animationSettings")

    def to_solver(self):
        if self.animation_settings is not None:
            if self.animation_settings.frequency is not None:
                self.animation_frequency = self.animation_settings.frequency
            else:
                self.animation_frequency = -1

            if self.animation_settings.frequency_offset is not None:
                self.animation_frequency_offset = self.animation_settings.frequency_offset
            else:
                self.animation_frequency_offset = 0

            if self.animation_settings.frequency_time_average is not None:
                self.animation_frequency_time_average = (
                    self.animation_settings.frequency_time_average
                )
            else:
                self.animation_frequency_time_average = -1

            if self.animation_settings.frequency_time_average_offset is not None:
                self.animation_frequency_time_average_offset = (
                    self.animation_settings.frequency_time_average_offset
                )
            else:
                self.animation_frequency_time_average_offset = 0


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
        alias="outputFields", displayed="Output fields"
    )

    # pylint: disable=too-few-public-methods
    class Config(Flow360BaseModel.Config):
        """:class: Model config to cull output field shorthands"""

        # pylint: disable=unused-argument
        @staticmethod
        def schema_extra(schema, model):
            """Remove output field shorthands from schema"""
            _filter_fields(
                schema["properties"]["outputFields"]["items"]["enum"], SurfaceFieldNamesFull
            )


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


class SurfaceOutput(Flow360BaseModel, AnimatedOutputExtended):
    """:class:`SurfaceOutput` class"""

    output_format: Optional[OutputFormat] = pd.Field(alias="outputFormat")
    compute_time_averages: Optional[bool] = pd.Field(alias="computeTimeAverages")
    write_single_file: Optional[bool] = pd.Field(alias="writeSingleFile")
    start_average_integration_step: Optional[bool] = pd.Field(alias="startAverageIntegrationStep")
    output_fields: Optional[SurfaceOutputFields] = pd.Field(alias="outputFields")
    surfaces: Optional[Surfaces] = pd.Field()

    # pylint: disable=too-few-public-methods
    class Config(Flow360BaseModel.Config):
        """:class: Model config to cull output field shorthands"""

        # pylint: disable=unused-argument
        @staticmethod
        def schema_extra(schema, model):
            """Remove output field shorthands from schema"""
            _filter_fields(
                schema["properties"]["outputFields"]["items"]["enum"], SurfaceFieldNamesFull
            )


class SurfaceOutputPrivate(SurfaceOutput):
    """:class:`SurfaceOutputPrivate` class"""

    wall_function_metric: Optional[bool] = pd.Field(alias="wallFunctionMetric")
    node_moments_per_unit_area: Optional[bool] = pd.Field(alias="nodeMomentsPerUnitArea")
    residual_sa: Optional[bool] = pd.Field(alias="residualSA")
    coarsen_iterations: Optional[int] = pd.Field(alias="coarsenIterations")


# pylint: disable=too-many-ancestors
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

    # pylint: disable=too-few-public-methods
    class Config(Flow360BaseModel.Config):
        """:class: Model config to cull output field shorthands"""

        # pylint: disable=unused-argument
        @staticmethod
        def schema_extra(schema, model):
            """Remove output field shorthands from schema"""
            _filter_fields(
                schema["properties"]["outputFields"]["items"]["enum"], VolumeSliceFieldNamesFull
            )


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


class SliceOutput(Flow360BaseModel, AnimatedOutput):
    """:class:`SliceOutput` class"""

    output_format: Optional[OutputFormat] = pd.Field(alias="outputFormat")
    output_fields: Optional[SliceOutputFields] = pd.Field(alias="outputFields")
    slices: Optional[Slices]

    # pylint: disable=too-few-public-methods
    class Config(Flow360BaseModel.Config):
        """:class: Model config to cull output field shorthands"""

        # pylint: disable=unused-argument
        @staticmethod
        def schema_extra(schema, model):
            """Remove output field shorthands from schema"""
            _filter_fields(
                schema["properties"]["outputFields"]["items"]["enum"], VolumeSliceFieldNamesFull
            )


class SliceOutputPrivate(SliceOutput):
    """:class:`SliceOutputPrivate` class"""

    coarsen_iterations: Optional[int] = pd.Field(alias="coarsenIterations")


class SliceOutputLegacy(SliceOutputPrivate, OutputLegacy):
    """:class:`SliceOutputLegacy` class"""

    bet_metrics: Optional[bool] = pd.Field(alias="betMetrics")
    bet_metrics_per_disk: Optional[bool] = pd.Field(alias="betMetricsPerDisk")


class VolumeOutput(Flow360BaseModel, AnimatedOutputExtended):
    """:class:`VolumeOutput` class"""

    output_format: Optional[OutputFormat] = pd.Field(alias="outputFormat")
    compute_time_averages: Optional[bool] = pd.Field(alias="computeTimeAverages")
    start_average_integration_step: Optional[int] = pd.Field(alias="startAverageIntegrationStep")
    output_fields: Optional[VolumeOutputFields] = pd.Field(alias="outputFields")

    # pylint: disable=too-few-public-methods
    class Config(Flow360BaseModel.Config):
        """:class: Model config to cull output field shorthands"""

        # pylint: disable=unused-argument
        @staticmethod
        def schema_extra(schema, model):
            """Remove output field shorthands from schema"""
            _filter_fields(
                schema["properties"]["outputFields"]["items"]["enum"], VolumeSliceFieldNamesFull
            )


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


# pylint: disable=too-many-ancestors
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

    # pylint: disable=too-few-public-methods
    class Config(Flow360BaseModel.Config):
        """:class: Model config to cull output field shorthands"""

        # pylint: disable=unused-argument
        @staticmethod
        def schema_extra(schema, model):
            """Remove output field shorthands from schema"""
            _filter_fields(
                schema["properties"]["outputFields"]["items"]["enum"], CommonFieldNamesFull
            )


class ProbeMonitor(MonitorBase):
    """:class:`ProbeMonitor` class"""

    type = pd.Field("probe", const=True)
    monitor_locations: Optional[List[Coordinate]] = pd.Field(alias="monitorLocations")
    output_fields: Optional[CommonOutputFields] = pd.Field(alias="outputFields")

    # pylint: disable=too-few-public-methods
    class Config(Flow360BaseModel.Config):
        """:class: Model config to cull output field shorthands"""

        # pylint: disable=unused-argument
        @staticmethod
        def schema_extra(schema, model):
            """Remove output field shorthands from schema"""
            _filter_fields(
                schema["properties"]["outputFields"]["items"]["enum"], CommonFieldNamesFull
            )


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

    # pylint: disable=too-few-public-methods
    class Config(Flow360BaseModel.Config):
        """:class: Model config to cull output field shorthands"""

        # pylint: disable=unused-argument
        @staticmethod
        def schema_extra(schema, model):
            """Remove output field shorthands from schema"""
            _filter_fields(
                schema["properties"]["outputFields"]["items"]["enum"], CommonFieldNamesFull
            )


class IsoSurface(Flow360BaseModel):
    """:class:`IsoSurface` class"""

    surface_field: Optional[IsoSurfaceOutputField] = pd.Field(alias="surfaceField")
    surface_field_magnitude: Optional[float] = pd.Field(alias="surfaceFieldMagnitude")
    output_fields: Optional[CommonOutputFields] = pd.Field(alias="outputFields")

    # pylint: disable=too-few-public-methods
    class Config(Flow360BaseModel.Config):
        """:class: Model config to cull output field shorthands"""

        # pylint: disable=unused-argument
        @staticmethod
        def schema_extra(schema, model):
            """Remove output field shorthands from schema"""
            _filter_fields(
                schema["properties"]["outputFields"]["items"]["enum"], CommonFieldNamesFull
            )
            _filter_fields(schema["properties"]["surfaceField"]["enum"], IsoSurfaceFieldNamesFull)


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


class IsoSurfaceOutput(Flow360BaseModel, AnimatedOutput):
    """:class:`IsoSurfaceOutput` class"""

    output_format: Optional[OutputFormat] = pd.Field(alias="outputFormat")
    iso_surfaces: Optional[IsoSurfaces] = pd.Field(alias="isoSurfaces")


class IsoSurfaceOutputPrivate(IsoSurfaceOutput):
    """:class:`IsoSurfaceOutputPrivate` class"""

    coarsen_iterations: Optional[int] = pd.Field(alias="coarsenIterations")
    output_fields: Optional[CommonOutputFields] = pd.Field(alias="outputFields")

    # pylint: disable=too-few-public-methods
    class Config(Flow360BaseModel.Config):
        """:class: Model config to cull output field shorthands"""

        # pylint: disable=unused-argument
        @staticmethod
        def schema_extra(schema, model):
            """Remove output field shorthands from schema"""
            _filter_fields(
                schema["properties"]["outputFields"]["items"]["enum"], CommonFieldNamesFull
            )


class AeroacousticOutput(Flow360BaseModel):
    """:class:`AeroacousticOutput` class for configuring output data about acoustic pressure signals

    Parameters
    ----------
    observers : List[Coordinate]
        List of observer locations at which time history of acoustic pressure signal is stored in aeroacoustic output
        file. The observer locations can be outside the simulation domain, but cannot be inside the solid surfaces of
        the simulation domain.
    animation: AnimationSettings, optional
        Animation settings for the output data

    Returns
    -------
    :class:`AeroacousticOutput`
        An instance of the component class AeroacousticOutput.

    Example
    -------
    >>> aeroacoustics = AeroacousticOutput(observers=[(0, 0, 0), (1, 1, 1)], animation_frequency=1)
    """

    observers: List[Coordinate] = pd.Field()


class AeroacousticOutputPrivate(AeroacousticOutput, AnimatedOutput):
    """:class:`AeroacousticOutputPrivate` class"""

    patch_type: Optional[str] = pd.Field("solid", const=True, alias="patchType")
