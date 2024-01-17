"""
Flow360 output parameters models
"""
from __future__ import annotations

from abc import ABCMeta
from typing import List, Literal, Optional, Union, get_args

import pydantic as pd
from pydantic import conlist

from flow360.flags import Flags

from ..types import Axis, Coordinate, PositiveInt
from .flow360_fields import (
    CommonFieldNames,
    CommonFieldNamesFull,
    IsoSurfaceFieldNames,
    IsoSurfaceFieldNamesFull,
    SliceFieldNames,
    SliceFieldNamesFull,
    SurfaceFieldNames,
    SurfaceFieldNamesFull,
    VolumeFieldNames,
    VolumeFieldNamesFull,
    get_field_values,
    to_short,
)
from .flow360_legacy import LegacyModel, get_output_fields
from .params_base import (
    Flow360BaseModel,
    Flow360SortableBaseModel,
    _self_named_property_validator,
)

OutputFormat = Literal["paraview", "tecplot", "both"]

CommonFields = Literal[CommonFieldNames, CommonFieldNamesFull]
SurfaceFields = Literal[SurfaceFieldNames, SurfaceFieldNamesFull]
SliceFields = Literal[SliceFieldNames, SliceFieldNamesFull]
VolumeFields = Literal[VolumeFieldNames, VolumeFieldNamesFull]
IsoSurfaceFields = Literal[IsoSurfaceFieldNames, IsoSurfaceFieldNamesFull]

CommonOutputFields = conlist(CommonFields, unique_items=True)
SurfaceOutputFields = conlist(SurfaceFields, unique_items=True)
SliceOutputFields = conlist(SliceFields, unique_items=True)
VolumeOutputFields = conlist(VolumeFields, unique_items=True)
IsoSurfaceOutputField = IsoSurfaceFields


def _filter_fields(fields, literal_filter):
    """Take two literals, keep only arguments present in the filter"""
    values = get_field_values(literal_filter)
    fields[:] = [field for field in fields if field in values]


class AnimationSettings(Flow360BaseModel):
    """:class:`AnimationSettings` class"""

    frequency: Optional[Union[PositiveInt, Literal[-1]]] = pd.Field(
        alias="frequency", options=["Animated", "Static"]
    )
    frequency_offset: Optional[int] = pd.Field(alias="frequencyOffset")


class AnimationSettingsExtended(AnimationSettings):
    """:class:`AnimationSettingsExtended` class"""

    frequency_time_average: Optional[Union[PositiveInt, Literal[-1]]] = pd.Field(
        alias="frequencyTimeAverage", options=["Animated", "Static"]
    )
    frequency_time_average_offset: Optional[int] = pd.Field(alias="frequencyTimeAverageOffset")


class AnimatedOutput(pd.BaseModel, metaclass=ABCMeta):
    """:class:`AnimatedOutput` class"""

    animation_frequency: Optional[Union[PositiveInt, Literal[-1]]] = pd.Field(
        alias="animationFrequency", options=["Animated", "Static"]
    )
    animation_frequency_offset: Optional[int] = pd.Field(alias="animationFrequencyOffset")

    # Temporarily disabled until solver-side support for new animation format is introduced
    """
    animation_settings: Optional[AnimationSettings] = pd.Field(alias="animationSettings")

    # pylint: disable=unused-argument
    def to_solver(self, params, **kwargs) -> AnimatedOutput:
        # Convert animation settings (UI representation) to solver representation
        if self.animation_settings is not None:
            if self.animation_settings.frequency is not None:
                self.animation_frequency = self.animation_settings.frequency
            else:
                self.animation_frequency = -1

            if self.animation_settings.frequency_offset is not None:
                self.animation_frequency_offset = self.animation_settings.frequency_offset
            else:
                self.animation_frequency_offset = 0
        solver_animations = self.__class__(
            animation_frequency=self.animation_frequency,
            animation_frequency_offset=self.animation_frequency_offset,
        )
        return solver_animations
    """


class AnimatedOutputExtended(AnimatedOutput, metaclass=ABCMeta):
    """:class:`AnimatedOutputExtended` class"""

    animation_frequency_time_average: Optional[Union[PositiveInt, Literal[-1]]] = pd.Field(
        alias="animationFrequencyTimeAverage", options=["Animated", "Static"]
    )
    animation_frequency_time_average_offset: Optional[int] = pd.Field(
        alias="animationFrequencyTimeAverageOffset"
    )

    # Temporarily disabled until solver-side support for new animation format is introduced
    """
    animation_settings: Optional[AnimationSettingsExtended] = pd.Field(alias="animationSettings")

    # pylint: disable=unused-argument
    def to_solver(self, params, **kwargs) -> AnimatedOutputExtended:
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
        solver_animations = self.__class__(
            animation_frequency=self.animation_frequency,
            animation_frequency_offset=self.animation_frequency_offset,
        )
        return solver_animations"""


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

    # pylint: disable=arguments-differ
    def to_solver(self, params, **kwargs) -> SurfaceOutput:
        solver_values = self._convert_dimensions_to_solver(params, **kwargs)
        fields = solver_values.pop("output_fields")
        fields = [to_short(field) for field in fields]
        return SurfaceOutput(**solver_values, output_fields=fields)


class Slice(Flow360BaseModel):
    """:class:`NamedSlice` class"""

    slice_normal: Axis = pd.Field(alias="sliceNormal")
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
                schema["properties"]["outputFields"]["items"]["enum"], VolumeFieldNamesFull
            )

    # pylint: disable=arguments-differ
    def to_solver(self, params, **kwargs) -> Slice:
        solver_values = self._convert_dimensions_to_solver(params, **kwargs)
        fields = solver_values.pop("output_fields")
        fields = [to_short(field) for field in fields]
        return Slice(**solver_values, output_fields=fields)


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
                schema["properties"]["outputFields"]["items"]["enum"], VolumeFieldNamesFull
            )

    # pylint: disable=arguments-differ
    def to_solver(self, params, **kwargs) -> SliceOutput:
        solver_values = self._convert_dimensions_to_solver(params, **kwargs)
        fields = solver_values.pop("output_fields")
        fields = [to_short(field) for field in fields]
        return SliceOutput(**solver_values, output_fields=fields)


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
                schema["properties"]["outputFields"]["items"]["enum"], VolumeFieldNamesFull
            )

    # pylint: disable=arguments-differ
    def to_solver(self, params, **kwargs) -> VolumeOutput:
        solver_values = self._convert_dimensions_to_solver(params, **kwargs)
        fields = solver_values.pop("output_fields")
        fields = [to_short(field) for field in fields]
        return VolumeOutput(**solver_values, output_fields=fields)


class MonitorBase(Flow360BaseModel, metaclass=ABCMeta):
    """:class:`MonitorBase` class"""

    type: str


class SurfaceIntegralMonitor(MonitorBase):
    """:class:`SurfaceIntegralMonitor` class"""

    type: Literal["surfaceIntegral"] = pd.Field("surfaceIntegral", const=True)
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

    # pylint: disable=arguments-differ
    def to_solver(self, params, **kwargs) -> SurfaceIntegralMonitor:
        solver_values = self._convert_dimensions_to_solver(params, **kwargs)
        fields = solver_values.pop("output_fields")
        fields = [to_short(field) for field in fields]
        return SurfaceIntegralMonitor(**solver_values, output_fields=fields)


class ProbeMonitor(MonitorBase):
    """:class:`ProbeMonitor` class"""

    type: Literal["probe"] = pd.Field("probe", const=True)
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

    # pylint: disable=arguments-differ
    def to_solver(self, params, **kwargs) -> ProbeMonitor:
        solver_values = self._convert_dimensions_to_solver(params, **kwargs)
        fields = solver_values.pop("output_fields")
        fields = [to_short(field) for field in fields]
        return ProbeMonitor(**solver_values, output_fields=fields)


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

    # pylint: disable=arguments-differ
    def to_solver(self, params, **kwargs) -> MonitorOutput:
        solver_values = self._convert_dimensions_to_solver(params, **kwargs)
        fields = solver_values.pop("output_fields")
        fields = [to_short(field) for field in fields]
        return MonitorOutput(**solver_values, output_fields=fields)


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

    # pylint: disable=arguments-differ
    def to_solver(self, params, **kwargs) -> IsoSurface:
        solver_values = self._convert_dimensions_to_solver(params, **kwargs)
        fields = solver_values.pop("output_fields")
        fields = [to_short(field) for field in fields]
        return IsoSurface(**solver_values, output_fields=fields)


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


class AeroacousticOutput(Flow360BaseModel, AnimatedOutput):
    """:class:`AeroacousticOutput` class for configuring output data about acoustic pressure signals

    Parameters
    ----------
    observers : List[Coordinate]
        List of observer locations at which time history of acoustic pressure signal is stored in aeroacoustic output
        file. The observer locations can be outside the simulation domain, but cannot be inside the solid surfaces of
        the simulation domain.
    animation_frequency: Union[PositiveInt, Literal[-1]], optional
        Frame frequency in the animation
    animation_frequency_offset: int, optional
        Animation frequency offset

    Returns
    -------
    :class:`AeroacousticOutput`
        An instance of the component class AeroacousticOutput.

    Example
    -------
    >>> aeroacoustics = AeroacousticOutput(observers=[(0, 0, 0), (1, 1, 1)], animation_frequency=1)
    """

    patch_type: Optional[str] = pd.Field("solid", const=True, alias="patchType")
    observers: List[Coordinate] = pd.Field()
    if Flags.beta_features():
        write_per_surface_output: Optional[bool] = pd.Field(False, alias="writePerSurfaceOutput")


class LegacyOutputFormat(pd.BaseModel, metaclass=ABCMeta):
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


# pylint: disable=too-many-ancestors
class SurfaceOutputLegacy(SurfaceOutput, LegacyOutputFormat, LegacyModel):
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
        fields = get_output_fields(
            self,
            [],
            allowed=get_field_values(CommonFieldNames) + get_field_values(SurfaceFieldNames),
        )

        if self.output_fields is not None:
            fields += self.output_fields

        model = {
            "animationFrequency": self.animation_frequency,
            "animationFrequencyOffset": self.animation_frequency_offset,
            "animationFrequencyTimeAverage": self.animation_frequency_time_average,
            "animationFrequencyTimeAverageOffset": self.animation_frequency_time_average_offset,
            "computeTimeAverages": self.compute_time_averages,
            "outputFormat": self.output_format,
            "outputFields": fields,
            "startAverageIntegrationStep": self.start_average_integration_step,
        }

        return SurfaceOutput.parse_obj(model)


class SliceNamedLegacy(Flow360BaseModel):
    """:class:`SliceNamedLegacy` class"""

    slice_name: str = pd.Field(alias="sliceName")
    slice_normal: Axis = pd.Field(alias="sliceNormal")
    slice_origin: Coordinate = pd.Field(alias="sliceOrigin")
    output_fields: Optional[List[str]] = pd.Field(alias="outputFields")


class SliceOutputLegacy(SliceOutput, LegacyOutputFormat, LegacyModel):
    """:class:`SliceOutputLegacy` class"""

    coarsen_iterations: Optional[int] = pd.Field(alias="coarsenIterations")
    bet_metrics: Optional[bool] = pd.Field(alias="betMetrics")
    bet_metrics_per_disk: Optional[bool] = pd.Field(alias="betMetricsPerDisk")
    slices: Optional[Union[Slices, List[SliceNamedLegacy]]] = pd.Field()

    def update_model(self) -> Flow360BaseModel:
        fields = get_output_fields(
            self, [], allowed=get_field_values(CommonFieldNames) + get_field_values(SliceFieldNames)
        )

        if self.output_fields is not None:
            fields += self.output_fields

        model = {
            "animationFrequency": self.animation_frequency,
            "animationFrequencyOffset": self.animation_frequency_offset,
            "outputFormat": self.output_format,
            "outputFields": fields,
        }

        if (
            isinstance(self.slices, List)
            and len(self.slices) > 0
            and isinstance(self.slices[0], SliceNamedLegacy)
        ):
            slices = {}
            for named_slice in self.slices:
                slices[named_slice.slice_name] = Slice(
                    slice_normal=named_slice.slice_normal,
                    slice_origin=named_slice.slice_origin,
                    output_fields=named_slice.output_fields,
                )
            model["slices"] = Slices(**slices)
        elif isinstance(self.slices, Slices):
            model["slices"] = self.slices

        return SliceOutput.parse_obj(model)


# pylint: disable=too-many-ancestors
class VolumeOutputLegacy(VolumeOutput, LegacyOutputFormat, LegacyModel):
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
        fields = get_output_fields(
            self,
            ["write_single_file", "write_distributed_file"],
            allowed=get_field_values(CommonFieldNames) + get_field_values(VolumeFieldNames),
        )

        if self.output_fields is not None:
            fields += self.output_fields

        model = {
            "animationFrequency": self.animation_frequency,
            "animationFrequencyOffset": self.animation_frequency_offset,
            "animationFrequencyTimeAverage": self.animation_frequency_time_average,
            "animationFrequencyTimeAverageOffset": self.animation_frequency_time_average_offset,
            "computeTimeAverages": self.compute_time_averages,
            "outputFormat": self.output_format,
            "outputFields": fields,
            "startAverageIntegrationStep": self.start_average_integration_step,
        }

        return VolumeOutput.parse_obj(model)


# Legacy models for Flow360 updater, do not expose


class IsoSurfaceOutputLegacy(IsoSurfaceOutput, LegacyModel):
    """:class:`IsoSurfaceOutputLegacy` class"""

    def update_model(self):
        model = {
            "animationSettings": {
                "frequency": self.animation_frequency,
                "frequencyOffset": self.animation_frequency_offset,
            },
            "outputFormat": self.output_format,
            "isoSurfaces": self.iso_surfaces,
        }

        return IsoSurfaceOutput.parse_obj(model)
