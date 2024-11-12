"""
Flow360 output parameters models
"""

from __future__ import annotations

from abc import ABCMeta
from typing import List, Literal, Optional, Union, get_args

import pydantic.v1 as pd
from pydantic.v1 import conlist

from flow360.component.types import Axis, Coordinate
from flow360.component.utils import process_expressions
from flow360.component.v1.flow360_fields import (
    CommonFieldNames,
    IsoSurfaceFieldNames,
    SliceFieldNames,
    SurfaceFieldNames,
    VolumeFieldNames,
    _distribute_shared_output_fields,
    get_field_values,
)
from flow360.component.v1.flow360_legacy import LegacyModel, get_output_fields
from flow360.component.v1.params_base import (
    Flow360BaseModel,
    Flow360SortableBaseModel,
    _self_named_property_validator,
)
from flow360.component.v1.unit_system import Flow360UnitSystem, LengthType

OutputFormat = Literal[
    "paraview", "tecplot", "both", "paraview,tecplot"
]  # Removed "paraview,tecplot" during schema generation

CommonFields = CommonFieldNames
SurfaceFields = SurfaceFieldNames
SliceFields = SliceFieldNames
VolumeFields = VolumeFieldNames
IsoSurfaceFields = IsoSurfaceFieldNames

CommonOutputFields = conlist(CommonFields, unique_items=True)
SurfaceOutputFields = conlist(SurfaceFields, unique_items=True)
SliceOutputFields = conlist(SliceFields, unique_items=True)
VolumeOutputFields = conlist(VolumeFields, unique_items=True)
IsoSurfaceOutputField = IsoSurfaceFields


def _deduplicate_output_fields(solver_values: dict, item_names: str = None):
    duplicate_outputs = [["solutionTurbulence", "kOmega"], ["solutionTurbulence", "nuHat"]]
    for name_pair in duplicate_outputs:
        if (
            name_pair[0] in solver_values["output_fields"]
            and name_pair[1] in solver_values["output_fields"]
        ):
            solver_values["output_fields"].remove(name_pair[1])
        if item_names is not None and solver_values[item_names] is not None:
            for name in solver_values[item_names].names():
                item = solver_values[item_names][name]
                if item.output_fields is None:
                    continue
                if name_pair[0] in item.output_fields and name_pair[1] in item.output_fields:
                    item.output_fields.remove(name_pair[1])


class AnimationSettings(Flow360BaseModel):
    """:class:`AnimationSettings` class"""

    frequency: Optional[Union[pd.PositiveInt, Literal[-1]]] = pd.Field(
        alias="frequency", options=["Animated", "Static"]
    )
    frequency_offset: Optional[int] = pd.Field(alias="frequencyOffset")


class AnimationSettingsExtended(AnimationSettings):
    """:class:`AnimationSettingsExtended` class"""

    frequency_time_average: Optional[Union[pd.PositiveInt, Literal[-1]]] = pd.Field(
        alias="frequencyTimeAverage", options=["Animated", "Static"]
    )
    frequency_time_average_offset: Optional[int] = pd.Field(alias="frequencyTimeAverageOffset")


class AnimatedOutput(pd.BaseModel, metaclass=ABCMeta):
    """:class:`AnimatedOutput` class"""

    output_format: Optional[OutputFormat] = pd.Field(alias="outputFormat", default="paraview")
    animation_frequency: Optional[Union[pd.PositiveInt, Literal[-1]]] = pd.Field(
        alias="animationFrequency", options=["Animated", "Static"], default=-1
    )
    animation_frequency_offset: Optional[int] = pd.Field(
        alias="animationFrequencyOffset", default=0
    )

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


class TimeAverageAnimatedOutput(AnimatedOutput, metaclass=ABCMeta):
    """:class:`TimeAverageAnimatedOutput` class"""

    compute_time_averages: Optional[bool] = pd.Field(alias="computeTimeAverages", default=False)

    animation_frequency_time_average: Optional[Union[pd.PositiveInt, Literal[-1]]] = pd.Field(
        alias="animationFrequencyTimeAverage", options=["Animated", "Static"], default=-1
    )
    animation_frequency_time_average_offset: Optional[int] = pd.Field(
        alias="animationFrequencyTimeAverageOffset", default=0
    )
    start_average_integration_step: Optional[Union[pd.NonNegativeInt, Literal[-1]]] = pd.Field(
        alias="startAverageIntegrationStep", default=-1, options=["From step", "No averaging"]
    )

    # Temporarily disabled until solver-side support for new animation format is introduced
    """
    animation_settings: Optional[AnimationSettingsExtended] = pd.Field(alias="animationSettings")

    # pylint: disable=unused-argument
    def to_solver(self, params, **kwargs) -> TimeAverageAnimatedOutput:
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

    output_fields: Optional[List[Union[SurfaceFields, str]]] = pd.Field(
        alias="outputFields",
        displayed="Output fields",
        default=[],
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


class SurfaceOutput(Flow360BaseModel, TimeAverageAnimatedOutput):
    """:class:`SurfaceOutput` class"""

    write_single_file: Optional[bool] = pd.Field(alias="writeSingleFile", default=False)
    output_fields: Optional[List[Union[SurfaceFields, str]]] = pd.Field(
        alias="outputFields",
        default=[],
    )
    surfaces: Optional[Surfaces] = pd.Field()

    # pylint: disable=arguments-differ
    def to_solver(self, params, **kwargs) -> SurfaceOutput:
        solver_model = super().to_solver(params, **kwargs)
        solver_values = solver_model.__dict__
        _deduplicate_output_fields(solver_values, "surfaces")
        # Add boundaries that are not listed into `surfaces` and applying shared fields.
        boundary_names = []
        for boundary_name in params.boundaries.names():
            boundary = params.boundaries[boundary_name]
            if boundary.name is not None:
                boundary_names.append(boundary.name)
            else:
                boundary_names.append(boundary_name)
        if solver_values["surfaces"] is None:
            solver_values["surfaces"] = Surfaces()
        if solver_values["output_format"] == "both":
            solver_values["output_format"] = "paraview,tecplot"
        for boundary_name in boundary_names:
            if boundary_name not in solver_values["surfaces"].names():
                solver_values["surfaces"][boundary_name] = Surface()
                solver_values["surfaces"][boundary_name].output_fields = []
        _distribute_shared_output_fields(solver_values, "surfaces")

        return SurfaceOutput(**solver_values)


class Slice(Flow360BaseModel):
    """:class:`NamedSlice` class"""

    slice_normal: Axis = pd.Field(alias="sliceNormal")
    slice_origin: LengthType.Point = pd.Field(alias="sliceOrigin")
    output_fields: Optional[List[Union[SliceFields, str]]] = pd.Field(
        alias="outputFields", default=[]
    )

    # pylint: disable=arguments-differ
    def to_solver(self, params, **kwargs) -> Slice:
        solver_values = self._convert_dimensions_to_solver(params, **kwargs)
        fields = solver_values.pop("output_fields")
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

    output_fields: Optional[List[Union[SliceFields, str]]] = pd.Field(
        alias="outputFields", default=[]
    )
    slices: Slices = pd.Field()

    # pylint: disable=arguments-differ
    def to_solver(self, params, **kwargs) -> SliceOutput:
        solver_model = super().to_solver(params, **kwargs)
        solver_values = solver_model.__dict__
        _deduplicate_output_fields(solver_values, "slices")
        if solver_values["output_format"] == "both":
            solver_values["output_format"] = "paraview,tecplot"
        _distribute_shared_output_fields(solver_values, "slices")
        return SliceOutput(**solver_values)


class VolumeOutput(Flow360BaseModel, TimeAverageAnimatedOutput):
    """:class:`VolumeOutput` class"""

    output_fields: Optional[List[Union[VolumeFields, str]]] = pd.Field(
        alias="outputFields",
        default=["primitiveVars", "Cp", "mut", "Mach"],
    )

    # pylint: disable=arguments-differ
    def to_solver(self, params, **kwargs) -> VolumeOutput:
        """
        Add betMetrics and betMetricsPerDisk if used in slices but not in volume.
        """
        solver_model = super().to_solver(params, **kwargs)
        solver_values = solver_model.__dict__
        _deduplicate_output_fields(solver_values)
        fields = solver_values.pop("output_fields")
        if solver_values["output_format"] == "both":
            solver_values["output_format"] = "paraview,tecplot"

        if params.slice_output is not None and params.slice_output.slices is not None:
            slice_output_dict = params.slice_output.__dict__
            _distribute_shared_output_fields(slice_output_dict, "slices")
            for slice_name in slice_output_dict["slices"].names():
                for item_to_add in ["betMetrics", "betMetricsPerDisk"]:
                    if (
                        item_to_add in slice_output_dict["slices"][slice_name].output_fields
                        and item_to_add not in fields
                    ):
                        fields.append(item_to_add)

        return VolumeOutput(**solver_values, output_fields=fields)


class MonitorBase(Flow360BaseModel, metaclass=ABCMeta):
    """:class:`MonitorBase` class"""

    type: str


class SurfaceIntegralMonitor(MonitorBase):
    """:class:`SurfaceIntegralMonitor` class"""

    type: Literal["surfaceIntegral"] = pd.Field("surfaceIntegral", const=True)
    surfaces: List[str] = pd.Field()
    output_fields: Optional[List[Union[CommonFields, str]]] = pd.Field(
        alias="outputFields", default=[]
    )

    # pylint: disable=arguments-differ
    def to_solver(self, params, **kwargs) -> SurfaceIntegralMonitor:
        solver_model = super().to_solver(params, **kwargs)
        solver_values = solver_model.__dict__
        fields = solver_values.pop("output_fields")
        return SurfaceIntegralMonitor(**solver_values, output_fields=fields)


class ProbeMonitor(MonitorBase):
    """:class:`ProbeMonitor` class"""

    type: Literal["probe"] = pd.Field("probe", const=True)
    monitor_locations: List[Coordinate] = pd.Field(alias="monitorLocations")
    output_fields: Optional[List[Union[CommonFields, str]]] = pd.Field(
        alias="outputFields", default=[]
    )

    # pylint: disable=arguments-differ
    def to_solver(self, params, **kwargs) -> ProbeMonitor:
        solver_model = super().to_solver(params, **kwargs)
        solver_values = solver_model.__dict__
        fields = solver_values.pop("output_fields")
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

    monitors: Monitors = pd.Field()
    output_fields: Optional[List[Union[CommonFields, str]]] = pd.Field(
        alias="outputFields", default=[]
    )

    # pylint: disable=arguments-differ
    def to_solver(self, params, **kwargs) -> MonitorOutput:
        solver_model = super().to_solver(params, **kwargs)
        solver_values = solver_model.__dict__
        _deduplicate_output_fields(solver_values, "monitors")
        _distribute_shared_output_fields(solver_values, "monitors")
        return MonitorOutput(**solver_values)


class LegacyMonitor(MonitorBase):
    """:class:`LegacyMonitor` class"""

    monitor_locations: List[Coordinate] = pd.Field(alias="monitorLocations")
    output_fields: Optional[List[Union[CommonFields, str]]] = pd.Field(
        alias="outputFields", default=[]
    )

    # pylint: disable=arguments-differ
    def to_solver(self, params, **kwargs) -> ProbeMonitor:
        solver_model = super().to_solver(params, **kwargs)
        solver_values = solver_model.__dict__
        fields = solver_values.pop("output_fields")
        return ProbeMonitor(**solver_values, output_fields=fields)


LegacyMonitorType = Union[SurfaceIntegralMonitor, ProbeMonitor, LegacyMonitor]


class _GenericLegacyMonitorWrapper(Flow360BaseModel):
    """:class:`_GenericLegacyMonitorWrapper` class"""

    v: LegacyMonitorType


class LegacyMonitors(Flow360SortableBaseModel):
    """:class:`LegacyMonitors` class"""

    @classmethod
    def get_subtypes(cls) -> list:
        return list(get_args(_GenericLegacyMonitorWrapper.__fields__["v"].type_))

    # pylint: disable=no-self-argument
    @pd.root_validator(pre=True)
    def validate_monitor(cls, values):
        """
        root validator
        """
        return _self_named_property_validator(
            values, _GenericLegacyMonitorWrapper, msg="is not any of supported monitor types."
        )


class MonitorOutputLegacy(LegacyModel):
    """:class:`MonitorOutputLegacy` class"""

    monitors: LegacyMonitors = pd.Field()
    output_fields: Optional[List[Union[CommonFields, str]]] = pd.Field(
        alias="outputFields", default=[]
    )

    def update_model(self):
        new_monitors = {}
        # pylint: disable=no-member,unsubscriptable-object
        for monitor_name in self.monitors.names():
            if isinstance(self.monitors[monitor_name], LegacyMonitor):
                self.monitors[monitor_name].type = "probe"
            else:
                new_monitors[monitor_name] = self.monitors[monitor_name]
        model = {
            "monitors": new_monitors,
            "output_fields": self.output_fields,
        }
        return MonitorOutput.parse_obj(model)


class IsoSurface(Flow360BaseModel):
    """:class:`IsoSurface` class"""

    surface_field: Literal[IsoSurfaceFields] = pd.Field(alias="surfaceField")
    surface_field_magnitude: float = pd.Field(alias="surfaceFieldMagnitude")
    output_fields: Optional[List[Union[CommonFields, str]]] = pd.Field(
        alias="outputFields", default=[]
    )

    # pylint: disable=arguments-differ
    def to_solver(self, params, **kwargs) -> IsoSurface:
        solver_model = super().to_solver(params, **kwargs)
        solver_values = solver_model.__dict__
        fields = solver_values.pop("output_fields")
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

    iso_surfaces: IsoSurfaces = pd.Field(alias="isoSurfaces")
    output_fields: Optional[List[Union[CommonFields, str]]] = pd.Field(
        alias="outputFields", default=[]
    )

    # pylint: disable=arguments-differ
    def to_solver(self, params, **kwargs) -> IsoSurfaceOutput:
        solver_model = super().to_solver(params, **kwargs)
        solver_values = solver_model.__dict__
        _deduplicate_output_fields(solver_values, "iso_surfaces")
        if solver_values["output_format"] == "both":
            solver_values["output_format"] = "paraview,tecplot"
        _distribute_shared_output_fields(solver_values, "iso_surfaces")
        return IsoSurfaceOutput(**solver_values)


class AeroacousticOutput(Flow360BaseModel):
    """:class:`AeroacousticOutput` class for configuring output data about acoustic pressure signals

    Parameters
    ----------
    observers : List[Coordinate]
        List of observer locations at which time history of acoustic pressure signal is stored in aeroacoustic output
        file. The observer locations can be outside the simulation domain, but cannot be inside the solid surfaces of
        the simulation domain.

    Returns
    -------
    :class:`AeroacousticOutput`
        An instance of the component class AeroacousticOutput.

    Example
    -------
    >>> aeroacoustics = AeroacousticOutput(observers=[(0, 0, 0), (1, 1, 1)])
    """

    patch_type: Optional[str] = pd.Field("solid", const=True, alias="patchType")
    observers: List[Coordinate] = pd.Field()
    write_per_surface_output: Optional[bool] = pd.Field(False, alias="writePerSurfaceOutput")


class UserDefinedField(Flow360BaseModel):
    """Variable that can be used as output variables"""

    name: str = pd.Field()
    expression: str = pd.Field()

    _processed_expression = pd.validator("expression", allow_reuse=True)(process_expressions)


class UserDefinedFieldLegacy(LegacyModel):
    """Variable that can be used as output variables"""

    name: str = pd.Field()
    expression: str = pd.Field()

    def update_model(self):
        model = {
            "name": self.name,
            "expression": self.expression,
        }

        return UserDefinedField.parse_obj(model)


# Legacy models for Flow360 updater, do not expose


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
    low_mach_preconditioner_sensor: Optional[bool] = pd.Field(alias="lowMachPreconditionerSensor")


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
            fields = list(set(fields + self.output_fields))

        model = {
            "animationFrequency": self.animation_frequency,
            "animationFrequencyOffset": self.animation_frequency_offset,
            "animationFrequencyTimeAverage": self.animation_frequency_time_average,
            "animationFrequencyTimeAverageOffset": self.animation_frequency_time_average_offset,
            "computeTimeAverages": self.compute_time_averages,
            "outputFormat": self.output_format,
            "outputFields": fields,
            "startAverageIntegrationStep": self.start_average_integration_step,
            "surfaces": self.surfaces,
            "writeSingleFile": self.write_single_file,
        }

        return SurfaceOutput.parse_obj(model)


class SliceNamedLegacy(Flow360BaseModel):
    """:class:`SliceNamedLegacy` class"""

    slice_name: str = pd.Field(alias="sliceName")
    slice_normal: Axis = pd.Field(alias="sliceNormal")
    slice_origin: Coordinate = pd.Field(alias="sliceOrigin")
    output_fields: Optional[List[str]] = pd.Field([], alias="outputFields")


class SliceOutputLegacy(SliceOutput, LegacyOutputFormat, LegacyModel):
    """:class:`SliceOutputLegacy` class"""

    coarsen_iterations: Optional[int] = pd.Field(alias="coarsenIterations")
    bet_metrics: Optional[bool] = pd.Field(alias="betMetrics")
    bet_metrics_per_disk: Optional[bool] = pd.Field(alias="betMetricsPerDisk")
    slices: Optional[Union[Slices, List[SliceNamedLegacy]]] = pd.Field({})

    def __init__(self, *args, **kwargs):
        with Flow360UnitSystem(verbose=False):
            super().__init__(*args, **kwargs)

    def update_model(self) -> Flow360BaseModel:
        fields = get_output_fields(
            self, [], allowed=get_field_values(CommonFieldNames) + get_field_values(SliceFieldNames)
        )

        if self.output_fields is not None:
            fields = list(set(fields + self.output_fields))

        model = {
            "animationFrequency": self.animation_frequency,
            "animationFrequencyOffset": self.animation_frequency_offset,
            "outputFormat": self.output_format,
            "outputFields": fields,
        }

        if (
            isinstance(self.slices, List)
            and len(self.slices) > 0
            # pylint: disable=unsubscriptable-object
            and isinstance(self.slices[0], SliceNamedLegacy)
        ):
            slices = {}
            # pylint: disable=not-an-iterable
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
            fields = list(set(fields + self.output_fields))

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


class IsoSurfaceOutputLegacy(IsoSurfaceOutput, LegacyModel):
    """:class:`IsoSurfaceOutputLegacy` class"""

    iso_surfaces: Optional[IsoSurfaces] = pd.Field({}, alias="isoSurfaces")

    def update_model(self):
        fields = get_output_fields(
            self,
            [],
            allowed=get_field_values(CommonFieldNames),
        )
        if self.output_fields is not None:
            fields = list(set(fields + self.output_fields))

        model = {
            "animationFrequency": self.animation_frequency,
            "animationFrequencyOffset": self.animation_frequency_offset,
            "outputFormat": self.output_format,
            "isoSurfaces": self.iso_surfaces,
            "outputFields": fields,
        }

        return IsoSurfaceOutput.parse_obj(model)
