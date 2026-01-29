"""Mostly the same as Flow360Param counterparts.
Caveats:
1. Check if we support non-average and average output specified at the same time in solver.
(Yes but they share the same output_fields)
2. We do not support multiple output frequencies/file format for the same type of output.
"""

# pylint: disable=too-many-lines
import re
from typing import Annotated, List, Literal, Optional, Union, get_args

import pydantic as pd
from typing_extensions import deprecated

import flow360.component.simulation.units as u
from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.framework.entity_base import EntityList
from flow360.component.simulation.framework.entity_utils import generate_uuid
from flow360.component.simulation.framework.expressions import StringExpression
from flow360.component.simulation.framework.param_utils import serialize_model_obj_to_id
from flow360.component.simulation.framework.unique_list import UniqueItemList
from flow360.component.simulation.models.surface_models import Wall
from flow360.component.simulation.models.volume_models import (
    ActuatorDisk,
    BETDisk,
    PorousMedium,
)
from flow360.component.simulation.outputs.output_entities import (
    Isosurface,
    Point,
    PointArray,
    PointArray2D,
    Slice,
)
from flow360.component.simulation.outputs.output_fields import (
    AllFieldNames,
    CommonFieldNames,
    ForceOutputCoefficientNames,
    InvalidOutputFieldsForLiquid,
    SliceFieldNames,
    SurfaceFieldNames,
    VolumeFieldNames,
    get_field_values,
)
from flow360.component.simulation.outputs.render_config import (
    Camera,
    Environment,
    FieldMaterial,
    Lighting,
    PBRMaterial,
    SceneTransform,
)
from flow360.component.simulation.primitives import (
    GhostCircularPlane,
    GhostSphere,
    GhostSurface,
    ImportedSurface,
    MirroredSurface,
    Surface,
    WindTunnelGhostSurface,
)
from flow360.component.simulation.unit_system import LengthType, TimeType
from flow360.component.simulation.user_code.core.types import (
    Expression,
    UserVariable,
    solver_variable_to_user_variable,
)
from flow360.component.simulation.validation.validation_context import (
    ALL,
    CASE,
    ParamsValidationInfo,
    TimeSteppingType,
    add_validation_warning,
    contextual_field_validator,
    contextual_model_validator,
    get_validation_levels,
)
from flow360.component.simulation.validation.validation_utils import (
    validate_entity_list_surface_existence,
    validate_improper_surface_field_usage_for_imported_surface,
)
from flow360.component.types import Axis

# Invalid characters for Linux filenames: / is path separator, \0 is null terminator
_INVALID_FILENAME_CHARS_PATTERN = re.compile(r"[/\0]")


def _validate_filename_string(value: str) -> str:
    """
    Validate that a string is a valid Linux filename.

    Args:
        value: The string to validate

    Returns:
        The validated string

    Raises:
        ValueError: If the string is not a valid filename

    Notes:
        - Disallows forward slash (/) - path separator
        - Disallows null byte (\\0)
        - Disallows empty strings
        - Disallows reserved names (. and ..)
    """
    if not value:
        raise ValueError("Filename cannot be empty")

    # Check for reserved names
    if value in (".", ".."):
        raise ValueError(f"Filename cannot be '{value}' (reserved name)")

    # Check for invalid characters
    invalid_chars = _INVALID_FILENAME_CHARS_PATTERN.findall(value)
    if invalid_chars:
        # Show unique invalid characters found
        unique_chars = sorted(set(invalid_chars))
        char_display = ", ".join(repr(c) for c in unique_chars)
        raise ValueError(
            f"Filename contains invalid characters: {char_display}. "
            f"Linux filenames cannot contain '/' or null bytes. "
            f"Got: '{value}'"
        )

    return value


# Type alias for a validated filename string
FileNameString = Annotated[
    str,
    pd.AfterValidator(_validate_filename_string),
]


ForceOutputModelType = Annotated[
    Union[Wall, BETDisk, ActuatorDisk, PorousMedium],
    pd.Field(discriminator="type"),
]


@deprecated("The `UserDefinedField` class is deprecated! Use `UserVariable` instead.")
class UserDefinedField(Flow360BaseModel):
    """

    Defines additional fields that can be used as output variables.


    Example
    -------

    - Compute :code:`Mach` using :class:`UserDefinedField`
      (Showcase use, already supported in :ref:`Output Fields <UniversalVariablesV2>`):

    >>> fl.UserDefinedField(
    ...     name="Mach_UDF",
    ...     expression="double Mach = sqrt(primitiveVars[1] * primitiveVars[1] + "
    ...     + "primitiveVars[2] * primitiveVars[2] + primitiveVars[3] * primitiveVars[3])"
    ...     + " / sqrt(gamma * primitiveVars[4] / primitiveVars[0]);",
    ... )


    - Compute :code:`PressureForce` using :class:`UserDefinedField`:

    >>> fl.UserDefinedField(
    ...     name="PressureForce",
    ...     expression="double prel = primitiveVars[4] - pressureFreestream; "
    ...     + "PressureForce[0] = prel * nodeNormals[0]; "
    ...     + "PressureForce[1] = prel * nodeNormals[1]; "
    ...     + "PressureForce[2] = prel * nodeNormals[2];",
    ... )

    ====

    """

    type_name: Literal["UserDefinedField"] = pd.Field("UserDefinedField", frozen=True)
    name: str = pd.Field(description="The name of the output field.")
    expression: StringExpression = pd.Field(
        description="The mathematical expression for the field."
    )

    @pd.field_validator("name", mode="after")
    @classmethod
    def _check_redefined_user_defined_fields(cls, value):
        current_levels = get_validation_levels() if get_validation_levels() else []
        if all(level not in current_levels for level in (ALL, CASE)):
            return value
        defined_field_names = get_field_values(AllFieldNames)
        if value in defined_field_names:
            raise ValueError(
                f"User defined field variable name: {value} conflicts with pre-defined field names."
                " Please consider renaming this user defined field variable."
            )
        return value

    @contextual_model_validator(mode="after")
    def _deprecation_warning(self):
        add_validation_warning(
            "The `UserDefinedField` class is deprecated! Please use `UserVariable` instead "
            "which provides the same functionality but with better interface."
        )
        return self


class MovingStatistic(Flow360BaseModel):
    """

    :class:`MovingStatistic` class for moving statistic settings in
    :class:`ProbeOutput`, :class:`SurfaceProbeOutput`,
    :class:`SurfaceIntegralOutput` and :class:`ForceOutput`.

    Notes
    -----
    - The window size is defined by the number of data points recorded in the output.
    - For steady simulations, the solver typically outputs a data point once every **10 pseudo steps**.
      This means a :py:attr:`moving_window_size` = 10 would cover 100 pseudo steps.
      Thus, the :py:attr:`start_step` value is automatically rounded up to
      the nearest multiple of 10 for steady simulations.
    - For unsteady simulations, the solver outputs a data point for **every physical step**.
      A :py:attr:`moving_window_size` = 10 would cover 10 physical steps.
    - When :py:attr:`method` is set to "standard_deviation", the standard deviation is computed as a
      **sample standard deviation** normalized by :math:`n-1` (Bessel's correction), where :math:`n`
      is the number of data points in the moving window.
    - When :py:attr:`method` is set to "range", the difference between the maximum and minimum values of
      the monitored field in the moving window is computed.

    Example
    -------

    Define a moving statistic to compute the standard deviation in a moving window of
    10 data points, with the initial 100 steps skipped.

    >>> fl.MovingStatistic(
    ...     moving_window_size=10,
    ...     method="standard_deviation",
    ...     start_step=100,
    ... )

    ====
    """

    moving_window_size: pd.StrictInt = pd.Field(
        10,
        ge=2,
        description="The size of the moving window in data points over which the "
        "statistic is calculated. Must be greater than or equal to 2.",
    )
    method: Literal["mean", "min", "max", "standard_deviation", "range"] = pd.Field(
        "mean", description="The statistical method to apply to the data within the moving window."
    )
    start_step: pd.NonNegativeInt = pd.Field(
        0,
        description="The number of steps (pseudo or physical) to skip at the beginning of the "
        "simulation before the moving statistics calculation starts. For steady "
        "simulations, this value is automatically rounded up to the nearest multiple of 10, "
        "as the solver outputs data every 10 pseudo steps.",
    )
    type_name: Literal["MovingStatistic"] = pd.Field("MovingStatistic", frozen=True)


class _OutputBase(Flow360BaseModel):
    output_fields: UniqueItemList[str] = pd.Field()
    private_attribute_id: str = pd.Field(default_factory=generate_uuid, frozen=True)

    @pd.field_validator("output_fields", mode="after")
    @classmethod
    def _validate_improper_surface_field_usage(cls, value: UniqueItemList):
        if any(
            output_type in cls.__name__
            for output_type in [
                "SurfaceProbeOutput",
                "SurfaceOutput",
                "SurfaceSliceOutput",
                "SurfaceIntegralOutput",
            ]
        ):
            return value
        for output_item in value.items:
            if not isinstance(output_item, UserVariable) or not isinstance(
                output_item.value, Expression
            ):
                continue
            surface_solver_variable_names = output_item.value.solver_variable_names(
                recursive=True, variable_type="Surface"
            )
            if len(surface_solver_variable_names) > 0:
                raise ValueError(
                    f"Variable `{output_item}` cannot be used in `{cls.__name__}` "
                    + "since it contains Surface solver variable(s): "
                    + f"{', '.join(sorted(surface_solver_variable_names))}.",
                )
        return value

    @contextual_field_validator("output_fields", mode="after")
    @classmethod
    def _validate_non_liquid_output_fields(
        cls, value: UniqueItemList, param_info: ParamsValidationInfo
    ):
        if param_info.using_liquid_as_material is False:
            return value
        for output_item in value.items:
            if output_item in get_args(InvalidOutputFieldsForLiquid):
                raise ValueError(
                    f"Output field {output_item} cannot be selected when using liquid as simulation material."
                )
        return value

    @pd.field_validator("output_fields", mode="before")
    @classmethod
    def _convert_solver_variables_as_user_variables(cls, value):
        # Handle both dict/list (deserialization) and UniqueItemList (python object)
        # If input is a dict (from deserialization so no SolverVariable expected)
        if isinstance(value, dict):
            return value
        # If input is a list (from Python mode)
        if isinstance(value, list):
            return [solver_variable_to_user_variable(item) for item in value]
        # If input is a UniqueItemList (python object)
        if hasattr(value, "items") and isinstance(value.items, list):
            value.items = [solver_variable_to_user_variable(item) for item in value.items]
            return value
        return value


class _AnimationSettings(Flow360BaseModel):
    """
    Controls how frequently the output files are generated.
    """

    frequency: Union[pd.PositiveInt, Literal[-1]] = pd.Field(
        default=-1,
        description="Frequency (in number of physical time steps) at which output is saved. "
        + "-1 is at end of simulation. Important for child cases - this parameter refers to the "
        + "**global** time step, which gets transferred from the parent case. Example: if the parent "
        + "case finished at time_step=174, the child case will start from time_step=175. If "
        + "frequency=100 (child case), the output will be saved at time steps 200 (25 time steps of "
        + "the child simulation), 300 (125 time steps of the child simulation), etc. "
        + "This setting is NOT applicable for steady cases.",
    )
    frequency_offset: int = pd.Field(
        default=0,
        ge=0,
        description="Offset (in number of physical time steps) at which output is started to be saved."
        + " 0 is at beginning of simulation. Important for child cases - this parameter refers to the "
        + "**global** time step, which gets transferred from the parent case (see `frequency` "
        + "parameter for an example). Example: if an output has a frequency of 100 and a "
        + "frequency_offset of 10, the output will be saved at **global** time step 10, 110, 210, "
        + "etc. This setting is NOT applicable for steady cases.",
    )

    @contextual_field_validator("frequency", "frequency_offset", mode="after")
    @classmethod
    def disable_frequency_settings_in_steady_simulation(
        cls, value, info: pd.ValidationInfo, param_info: ParamsValidationInfo
    ):
        """Disable frequency settings in a steady simulation"""
        if param_info.time_stepping != TimeSteppingType.STEADY:
            return value
        # pylint: disable=unsubscriptable-object
        if value != cls.model_fields[info.field_name].default:
            raise ValueError(
                f"Output {info.field_name} cannot be specified in a steady simulation."
            )
        return value


class _AnimationAndFileFormatSettings(_AnimationSettings):
    """
    Controls how frequently the output files are generated and the file format.
    """

    output_format: Literal["paraview", "tecplot", "both"] = pd.Field(
        default="paraview", description=":code:`paraview`, :code:`tecplot` or :code:`both`."
    )


class SurfaceOutput(_AnimationAndFileFormatSettings, _OutputBase):
    """

    :class:`SurfaceOutput` class for surface output settings.

    Example
    -------

    - Define :class:`SurfaceOutput` on all surfaces of the geometry
      using naming pattern :code:`"*"`.

      >>> fl.SurfaceOutput(
      ...     entities=[geometry['*']],,
      ...     output_format="paraview",
      ...     output_fields=["vorticity", "T"],
      ... )

    - Define :class:`SurfaceOutput` on the selected surfaces of the volume_mesh
      using name pattern :code:`"fluid/inflow*"`.

      >>> fl.SurfaceOutput(
      ...     entities=[volume_mesh["fluid/inflow*"]],,
      ...     output_format="paraview",
      ...     output_fields=["vorticity", "T"],
      ... )

    ====
    """

    # pylint: disable=fixme
    # TODO: entities is None --> use all surfaces. This is not implemented yet.

    name: Optional[str] = pd.Field("Surface output", description="Name of the `SurfaceOutput`.")
    entities: EntityList[  # pylint: disable=duplicate-code
        Surface,
        MirroredSurface,
        GhostSurface,
        WindTunnelGhostSurface,
        GhostCircularPlane,
        GhostSphere,
        ImportedSurface,
    ] = pd.Field(
        alias="surfaces",
        description="List of boundaries where output is generated.",
    )
    write_single_file: bool = pd.Field(
        default=False,
        description="Enable writing all surface outputs into a single file instead of one file per surface."
        + "This option currently only supports Tecplot output format."
        + "Will choose the value of the last instance of this option of the same output type "
        + "(:class:`SurfaceOutput` or :class:`TimeAverageSurfaceOutput`) in the output list.",
    )
    output_fields: UniqueItemList[Union[SurfaceFieldNames, str, UserVariable]] = pd.Field(
        description="List of output variables. Including :ref:`universal output variables<UniversalVariablesV2>`,"
        + " :ref:`variables specific to SurfaceOutput<SurfaceSpecificVariablesV2>` and :class:`UserDefinedField`."
    )
    output_type: Literal["SurfaceOutput"] = pd.Field("SurfaceOutput", frozen=True)

    @contextual_field_validator("entities", mode="after")
    @classmethod
    def ensure_surface_existence(cls, value, param_info: ParamsValidationInfo):
        """Ensure all boundaries will be present after mesher"""
        return validate_entity_list_surface_existence(value, param_info)

    @contextual_model_validator(mode="after")
    def validate_imported_surface_output_fields(self, param_info: ParamsValidationInfo):
        """Validate output fields when using imported surfaces"""
        expanded_entities = param_info.expand_entity_list(self.entities)
        validate_improper_surface_field_usage_for_imported_surface(
            expanded_entities, self.output_fields
        )
        return self


class TimeAverageSurfaceOutput(SurfaceOutput):
    """
    :class:`TimeAverageSurfaceOutput` class for time average surface output settings.

    Example
    -------

    Calculate the average value starting from the :math:`4^{th}` physical step.
    The results are output every 10 physical step starting from the :math:`14^{th}` physical step
    (14, 24, 34 etc.).

    >>> fl.TimeAverageSurfaceOutput(
    ...     output_format="paraview",
    ...     output_fields=["primitiveVars"],
    ...     entities=[
    ...         volume_mesh["VOLUME/LEFT"],
    ...         volume_mesh["VOLUME/RIGHT"],
    ...     ],
    ...     start_step=4,
    ...     frequency=10,
    ...     frequency_offset=14,
    ... )

    ====
    """

    name: Optional[str] = pd.Field(
        "Time average surface output", description="Name of the `TimeAverageSurfaceOutput`."
    )

    start_step: Union[pd.NonNegativeInt, Literal[-1]] = pd.Field(
        default=-1,
        description="Physical time step to start calculating averaging. Important for child cases "
        + "- this parameter refers to the **global** time step, which gets transferred from the "
        + "parent case (see `frequency` parameter for an example).",
    )
    output_type: Literal["TimeAverageSurfaceOutput"] = pd.Field(
        "TimeAverageSurfaceOutput", frozen=True
    )


class VolumeOutput(_AnimationAndFileFormatSettings, _OutputBase):
    """
    :class:`VolumeOutput` class for volume output settings.

    Example
    -------

    >>> fl.VolumeOutput(
    ...     output_format="paraview",
    ...     output_fields=["Mach", "vorticity", "T"],
    ... )

    ====
    """

    name: Optional[str] = pd.Field("Volume output", description="Name of the `VolumeOutput`.")
    output_fields: UniqueItemList[Union[VolumeFieldNames, str, UserVariable]] = pd.Field(
        description="List of output variables. Including :ref:`universal output variables<UniversalVariablesV2>`,"
        " :ref:`variables specific to VolumeOutput<VolumeAndSliceSpecificVariablesV2>`"
        " and :class:`UserDefinedField`."
    )
    output_type: Literal["VolumeOutput"] = pd.Field("VolumeOutput", frozen=True)


class TimeAverageVolumeOutput(VolumeOutput):
    """
    :class:`TimeAverageVolumeOutput` class for time average volume output settings.

    Example
    -------

    Calculate the average value starting from the :math:`4^{th}` physical step.
    The results are output every 10 physical step starting from the :math:`14^{th}` physical step
    (14, 24, 34 etc.).

    >>> fl.TimeAverageVolumeOutput(
    ...     output_format="paraview",
    ...     output_fields=["primitiveVars"],
    ...     start_step=4,
    ...     frequency=10,
    ...     frequency_offset=14,
    ... )

    ====
    """

    name: Optional[str] = pd.Field(
        "Time average volume output", description="Name of the `TimeAverageVolumeOutput`."
    )
    start_step: Union[pd.NonNegativeInt, Literal[-1]] = pd.Field(
        default=-1,
        description="Physical time step to start calculating averaging. Important for child cases "
        + "- this parameter refers to the **global** time step, which gets transferred from the "
        + "parent case (see `frequency` parameter for an example).",
    )
    output_type: Literal["TimeAverageVolumeOutput"] = pd.Field(
        "TimeAverageVolumeOutput", frozen=True
    )


class SliceOutput(_AnimationAndFileFormatSettings, _OutputBase):
    """
    :class:`SliceOutput` class for slice output settings.

    Example
    -------

    >>> fl.SliceOutput(
    ...     slices=[
    ...         fl.Slice(
    ...             name="Slice_1",
    ...             normal=(0, 1, 0),
    ...             origin=(0, 0.56, 0)*fl.u.m
    ...         ),
    ...     ],
    ...     output_format="paraview",
    ...     output_fields=["vorticity", "T"],
    ... )

    ====
    """

    name: Optional[str] = pd.Field("Slice output", description="Name of the `SliceOutput`.")
    entities: EntityList[Slice] = pd.Field(
        alias="slices",
        description="List of output :class:`~flow360.Slice` entities.",
    )
    output_fields: UniqueItemList[Union[SliceFieldNames, str, UserVariable]] = pd.Field(
        description="List of output variables. Including :ref:`universal output variables<UniversalVariablesV2>`,"
        " :ref:`variables specific to SliceOutput<VolumeAndSliceSpecificVariablesV2>`"
        " and :class:`UserDefinedField`."
    )
    output_type: Literal["SliceOutput"] = pd.Field("SliceOutput", frozen=True)


class TimeAverageSliceOutput(SliceOutput):
    """

    :class:`TimeAverageSliceOutput` class for time average slice output settings.

    Example
    -------

    Calculate the average value starting from the :math:`4^{th}` physical step.
    The results are output every 10 physical step starting from the :math:`14^{th}` physical step
    (14, 24, 34 etc.).

    >>> fl.TimeAverageSliceOutput(
    ...     entities=[
    ...         fl.Slice(name="Slice_1",
    ...             origin=(0, 0, 0) * fl.u.m,
    ...             normal=(0, 0, 1),
    ...         )
    ...     ],
    ...     output_fields=["s", "T"],
    ...     start_step=4,
    ...     frequency=10,
    ...     frequency_offset=14,
    ... )

    ====
    """

    name: Optional[str] = pd.Field(
        "Time average slice output", description="Name of the `TimeAverageSliceOutput`."
    )
    start_step: Union[pd.NonNegativeInt, Literal[-1]] = pd.Field(
        default=-1,
        description="Physical time step to start calculating averaging. Important for child cases "
        + "- this parameter refers to the **global** time step, which gets transferred from the "
        + "parent case (see `frequency` parameter for an example).",
    )
    output_type: Literal["TimeAverageSliceOutput"] = pd.Field("TimeAverageSliceOutput", frozen=True)


class IsosurfaceOutput(_AnimationAndFileFormatSettings, _OutputBase):
    """

    :class:`IsosurfaceOutput` class for isosurface output settings.

    Example
    -------

    Define the :class:`IsosurfaceOutput` of :code:`qcriterion` on two isosurfaces:

    - :code:`Isosurface_T_0.1` is the :class:`Isosurface` with its temperature equals
      to 1.5 non-dimensional temperature;
    - :code:`Isosurface_p_0.5` is the :class:`Isosurface` with its pressure equals
      to 0.5 non-dimensional pressure.

    >>> fl.IsosurfaceOutput(
    ...     isosurfaces=[
    ...         fl.Isosurface(
    ...             name="Isosurface_T_0.1",
    ...             iso_value=0.1,
    ...             field="T",
    ...         ),
    ...         fl.Isosurface(
    ...             name="Isosurface_p_0.5",
    ...             iso_value=0.5,
    ...             field="p",
    ...         ),
    ...     ],
    ...     output_fields=["qcriterion"],
    ... )

    ====
    """

    name: Optional[str] = pd.Field(
        "Isosurface output", description="Name of the `IsosurfaceOutput`."
    )
    entities: UniqueItemList[Isosurface] = pd.Field(
        alias="isosurfaces",
        description="List of :class:`~flow360.Isosurface` entities.",
    )
    output_fields: UniqueItemList[Union[CommonFieldNames, str, UserVariable]] = pd.Field(
        description="List of output variables. Including "
        ":ref:`universal output variables<UniversalVariablesV2>` and :class:`UserDefinedField`."
    )
    output_type: Literal["IsosurfaceOutput"] = pd.Field("IsosurfaceOutput", frozen=True)

    def preprocess(
        self,
        *,
        params=None,
        exclude: List[str] = None,
        required_by: List[str] = None,
        flow360_unit_system=None,
    ) -> Flow360BaseModel:
        exclude_isosurface_output = exclude + ["iso_value"]
        return super().preprocess(
            params=params,
            exclude=exclude_isosurface_output,
            required_by=required_by,
            flow360_unit_system=flow360_unit_system,
        )


class TimeAverageIsosurfaceOutput(IsosurfaceOutput):
    """

    :class:`TimeAverageIsosurfaceOutput` class for isosurface output settings.

    Example
    -------

    Define the :class:`TimeAverageIsosurfaceOutput` of :code:`qcriterion` on two isosurfaces:

    - :code:`TimeAverageIsosurface_T_0.1` is the :class:`Isosurface` with its temperature equals
      to 1.5 non-dimensional temperature;
    - :code:`TimeAverageIsosurface_p_0.5` is the :class:`Isosurface` with its pressure equals
      to 0.5 non-dimensional pressure.

    >>> fl.TimeAverageIsosurfaceOutput(
    ...     isosurfaces=[
    ...         fl.Isosurface(
    ...             name="TimeAverageIsosurface_T_0.1",
    ...             iso_value=0.1,
    ...             field="T",
    ...         ),
    ...         fl.Isosurface(
    ...             name="TimeAverageIsosurface_p_0.5",
    ...             iso_value=0.5,
    ...             field="p",
    ...         ),
    ...     ],
    ...     output_fields=["qcriterion"],
    ... )

    ====
    """

    name: Optional[str] = pd.Field(
        "Time Average Isosurface output", description="Name of `TimeAverageIsosurfaceOutput`."
    )
    start_step: Union[pd.NonNegativeInt, Literal[-1]] = pd.Field(
        default=-1,
        description="Physical time step to start calculating averaging. Important for child cases "
        + "- this parameter refers to the **global** time step, which gets transferred from the "
        + "parent case (see `frequency` parameter for an example).",
    )
    output_type: Literal["TimeAverageIsosurfaceOutput"] = pd.Field(
        "TimeAverageIsosurfaceOutput", frozen=True
    )


class SurfaceIntegralOutput(_OutputBase):
    """

    :class:`SurfaceIntegralOutput` class for surface integral output settings.

    Note
    ----
    :class:`SurfaceIntegralOutput` can only be used with :class:`UserDefinedField`.
    See :doc:`User Defined Postprocessing Tutorial </python_api/example_library/notebooks/hinge_torques>`
    for more details about how to set up :class:`UserDefinedField`.

    Example
    -------
    Define :class:`SurfaceIntegralOutput` of :code:`PressureForce`.

    >>> fl.SurfaceIntegralOutput(
    ...     name="surface_integral",
    ...     output_fields=["PressureForce"],
    ...     entities=[volume_mesh["wing1"], volume_mesh["wing2"]],
    ... )

    ====
    """

    name: FileNameString = pd.Field(
        "Surface integral output",
        description="Name of integral. Must be a valid Linux filename (no slashes or null bytes).",
    )
    entities: EntityList[  # pylint: disable=duplicate-code
        Surface,
        MirroredSurface,
        GhostSurface,
        WindTunnelGhostSurface,
        GhostCircularPlane,
        GhostSphere,
        ImportedSurface,
    ] = pd.Field(
        alias="surfaces",
        description="List of boundaries where the surface integral will be calculated.",
    )
    output_fields: UniqueItemList[Union[str, UserVariable]] = pd.Field(
        description="List of output variables, only the :class:`UserDefinedField` is allowed."
    )
    moving_statistic: Optional[MovingStatistic] = pd.Field(
        None, description="When specified, report moving statistics of the fields instead."
    )
    output_type: Literal["SurfaceIntegralOutput"] = pd.Field("SurfaceIntegralOutput", frozen=True)

    @contextual_field_validator("entities", mode="after")
    @classmethod
    def ensure_surface_existence(cls, value, param_info: ParamsValidationInfo):
        """Ensure all boundaries will be present after mesher"""
        return validate_entity_list_surface_existence(value, param_info)

    @contextual_field_validator("entities", mode="after")
    @classmethod
    def allow_only_simulation_surfaces_or_imported_surfaces(
        cls, value, param_info: ParamsValidationInfo
    ):
        """Support only simulation surfaces or imported surfaces in each SurfaceIntegralOutput"""
        expanded = param_info.expand_entity_list(value)
        has_imported = isinstance(expanded[0], ImportedSurface)
        for entity in expanded[1:]:
            if has_imported != isinstance(entity, ImportedSurface):
                raise ValueError(
                    "Imported and simulation surfaces cannot be used together in the same SurfaceIntegralOutput."
                    " Please assign them to separate outputs."
                )
        return value

    @contextual_model_validator(mode="after")
    def validate_imported_surface_output_fields(self, param_info: ParamsValidationInfo):
        """Validate output fields when using imported surfaces"""
        expanded_entities = param_info.expand_entity_list(self.entities)
        validate_improper_surface_field_usage_for_imported_surface(
            expanded_entities, self.output_fields
        )
        return self


class ForceOutput(_OutputBase):
    """
    :class:`ForceOutput` class for setting total force output of specific surfaces.

    Example
    -------

    Define :class:`ForceOutput` to output total CL and CD on multiple wing surfaces and a BET disk.

    >>> wall = fl.Wall(name = 'wing', surfaces=[volume_mesh['1'], volume_mesh["wing2"]])
    >>> bet_disk = fl.BETDisk(...)
    >>> fl.ForceOutput(
    ...     name="force_output",
    ...     models=[wall, bet_disk],
    ...     output_fields=["CL", "CD"]
    ... )

    ====
    """

    name: str = pd.Field("Force output", description="Name of the force output.")
    output_fields: UniqueItemList[ForceOutputCoefficientNames] = pd.Field(
        description="List of force coefficients. Including CL, CD, CFx, CFy, CFz, CMx, CMy, CMz. "
        "For surface forces, their SkinFriction/Pressure is also supported, such as CLSkinFriction and CLPressure."
    )
    models: List[Union[ForceOutputModelType, str]] = pd.Field(
        description="List of surface/volume models (or model ids) whose force contribution will be calculated.",
    )
    moving_statistic: Optional[MovingStatistic] = pd.Field(
        None, description="When specified, report moving statistics of the fields instead."
    )
    output_type: Literal["ForceOutput"] = pd.Field("ForceOutput", frozen=True)

    @pd.field_validator("models", mode="after")
    @classmethod
    def _convert_model_obj_to_id(cls, value):
        """Validate duplicate models and convert model object to id"""
        model_ids = set()
        for model in value:
            model_id = (
                model if isinstance(model, str) else serialize_model_obj_to_id(model_obj=model)
            )
            if model_id in model_ids:
                raise ValueError("Duplicate models are not allowed in the same `ForceOutput`.")
            model_ids.add(model_id)
        return list(model_ids)

    @contextual_field_validator("models", mode="after", required_context=["physics_model_dict"])
    @classmethod
    def _check_model_exist_in_model_list(cls, value, param_info: ParamsValidationInfo):
        """Ensure all models exist in SimulationParams' model list."""
        for model_id in value:
            model_obj = param_info.physics_model_dict.get(model_id)
            if model_obj is None:
                raise ValueError("The model does not exist in simulation params' models list.")

        return value

    @contextual_field_validator("models", mode="after", required_context=["physics_model_dict"])
    @classmethod
    def _check_output_fields_with_volume_models_specified(
        cls, value, info: pd.ValidationInfo, param_info: ParamsValidationInfo
    ):
        """Ensure the output field exists when volume models are specified."""

        model_objs = [param_info.physics_model_dict.get(model_id) for model_id in value]

        if all(isinstance(model, Wall) for model in model_objs):
            return value
        output_fields = info.data.get("output_fields", None)
        if all(
            field in ["CL", "CD", "CFx", "CFy", "CFz", "CMx", "CMy", "CMz"]
            for field in output_fields.items
        ):
            return value
        raise ValueError(
            "When ActuatorDisk/BETDisk/PorousMedium is specified, "
            "only CL, CD, CFx, CFy, CFz, CMx, CMy, CMz can be set as output_fields."
        )


class RenderOutputGroup(Flow360BaseModel):
    """

    :class:`RenderOutputGroup` for defining a render output group - i.e. a set of
    entities sharing a common material (display options) settings.

    Example
    -------
    Define two :class:`RenderOutputGroup` objects, one assigning all boundaries of the
    uploaded geometry to a flat metallic material, and another assigning a slice and an
    isosurface to a material which will display a scalar field on the surface of the
    entity.

    >>> fl.RenderOutputGroup(
    ...     surfaces=geometry["*"],
    ...     material=fl.PBRMaterial.metal(shine=0.8)
    ... ),
    ... fl.RenderOutputGroup(
    ...     slices=[
    ...         fl.Slice(name="Example slice", normal=(0, 1, 0), origin=(0, 0, 0))
    ...     ],
    ...     isosurfaces=[
    ...         fl.Isosurface(name="Example isosurface", iso_value=0.1, field="T")
    ...     ],
    ...     material=fl.FieldMaterial.rainbow(field="T", min_value=0, max_value=1, alpha=0.4)
    ... )
    ====

    """

    surfaces: Optional[EntityList[Surface, MirroredSurface]] = pd.Field(
        None, description="List of of :class:`~flow360.Surface` entities."
    )
    slices: Optional[EntityList[Slice]] = pd.Field(
        None, description="List of of :class:`~flow360.Slice` entities."
    )
    isosurfaces: Optional[UniqueItemList[Isosurface]] = pd.Field(
        None, description="List of :class:`~flow360.Isosurface` entities."
    )
    material: Union[PBRMaterial, FieldMaterial] = pd.Field(
        description="Materials settings (color, surface field, roughness etc..) to be applied to the entire group"
    )

    @contextual_field_validator("surfaces", mode="after")
    @classmethod
    def ensure_surface_existence(cls, value, param_info: ParamsValidationInfo):
        """Ensure all boundaries will be present after mesher"""
        return validate_entity_list_surface_existence(value, param_info)

    @contextual_model_validator(mode="after")
    def check_not_empty(self, param_info: ParamsValidationInfo):
        """Verify the render group has at least one entity assigned to it"""
        expanded_surfaces = (
            param_info.expand_entity_list(self.surfaces) if self.surfaces is not None else None
        )
        if not expanded_surfaces and not self.slices and not self.isosurfaces:
            raise ValueError(
                "Render group should include at least one entity (surface, slice or isosurface)"
            )
        return self


class RenderOutput(_AnimationSettings):
    """

    :class:`RenderOutput` class for backend rendered output settings.

    Example
    -------

    Define the :class:`RenderOutput` that outputs a basic image - boundaries and a Y-slice:

    >>> fl.RenderOutput(
    ...     name="Example render",
    ...     groups=[
    ...         fl.RenderOutputGroup(
    ...             surfaces=geometry["*"],
    ...             material=fl.render.PBRMaterial.metal(shine=0.8)
    ...         ),
    ...         fl.RenderOutputGroup(
    ...             slices=[
    ...                 fl.Slice(name="Example slice", normal=(0, 1, 0), origin=(0, 0, 0))
    ...             ],
    ...             material=fl.render.FieldMaterial.rainbow(field="T", min_value=0, max_value=1, alpha=0.4)
    ...         )
    ...     ],
    ...     camera=fl.render.Camera.orthographic(scale=5, view=fl.Viewpoint.TOP + fl.Viewpoint.LEFT)
    ... )
    ====
    """

    name: str = pd.Field("Render output", description="Name of the `RenderOutput`.")
    groups: List[RenderOutputGroup] = pd.Field([])
    camera: Camera = pd.Field(description="Camera settings", default_factory=Camera.orthographic)
    lighting: Lighting = pd.Field(description="Lighting settings", default_factory=Lighting.default)
    environment: Environment = pd.Field(
        description="Environment settings", default_factory=Environment.simple
    )
    transform: Optional[SceneTransform] = pd.Field(
        None, description="Optional model transform to apply to all entities"
    )
    output_type: Literal["RenderOutput"] = pd.Field("RenderOutput", frozen=True)
    private_attribute_id: str = pd.Field(default_factory=generate_uuid, frozen=True)

    @pd.field_validator("groups", mode="after")
    @classmethod
    def check_has_output_groups(cls, value):
        """Verify the render output has at least one group to render"""
        if len(value) < 1:
            raise ValueError("Render output requires at least one output group to be defined")
        return value


class ProbeOutput(_OutputBase):
    """
    :class:`ProbeOutput` class for setting output data probed at monitor points in the voulume of the domain.
    Regardless of the motion of the mesh, the points retain their positions in the
    global reference frame during the simulation.

    Example
    -------

    Define :class:`ProbeOutput` on multiple specific monitor points and monitor points along the line.

    - :code:`Point_1` and :code:`Point_2` are two specific points we want to monitor in this probe output group.
    - :code:`Line_1` is from (1,0,0) * fl.u.m to (1.5,0,0) * fl.u.m and has 6 monitor points.
    - :code:`Line_2` is from (-1,0,0) * fl.u.m to (-1.5,0,0) * fl.u.m and has 3 monitor points,
      namely, (-1,0,0) * fl.u.m, (-1.25,0,0) * fl.u.m and (-1.5,0,0) * fl.u.m.

    >>> fl.ProbeOutput(
    ...     name="probe_group_points_and_lines",
    ...     entities=[
    ...         fl.Point(
    ...             name="Point_1",
    ...             location=(0.0, 1.5, 0.0) * fl.u.m,
    ...         ),
    ...         fl.Point(
    ...             name="Point_2",
    ...             location=(0.0, -1.5, 0.0) * fl.u.m,
    ...         ),
    ...         fl.PointArray(
    ...             name="Line_1",
    ...             start=(1.0, 0.0, 0.0) * fl.u.m,
    ...             end=(1.5, 0.0, 0.0) * fl.u.m,
    ...             number_of_points=6,
    ...         ),
    ...         fl.PointArray(
    ...             name="Line_2",
    ...             start=(-1.0, 0.0, 0.0) * fl.u.m,
    ...             end=(-1.5, 0.0, 0.0) * fl.u.m,
    ...             number_of_points=3,
    ...         ),
    ...     ],
    ...     output_fields=["primitiveVars"],
    ... )

    ====
    """

    name: str = pd.Field("Probe output", description="Name of the monitor group.")
    entities: EntityList[Point, PointArray] = pd.Field(
        alias="probe_points",
        description="List of monitored :class:`~flow360.Point`/"
        + ":class:`~flow360.PointArray` entities belonging to this "
        + "monitor group. :class:`~flow360.PointArray` is used to "
        + "define monitored points along a line.",
    )
    output_fields: UniqueItemList[Union[CommonFieldNames, str, UserVariable]] = pd.Field(
        description="List of output fields. Including :ref:`universal output variables<UniversalVariablesV2>`"
        " and :class:`UserDefinedField`."
    )
    moving_statistic: Optional[MovingStatistic] = pd.Field(
        None, description="When specified, report moving statistics of the fields instead."
    )
    output_type: Literal["ProbeOutput"] = pd.Field("ProbeOutput", frozen=True)


class SurfaceProbeOutput(_OutputBase):
    """
    :class:`SurfaceProbeOutput` class for setting surface output data probed at monitor points.
    The specified monitor point will be projected to the :py:attr:`~SurfaceProbeOutput.target_surfaces`
    closest to the point. The probed results on the projected point will be dumped.
    The projection is executed at the start of the simulation. If the surface that the point was
    projected to is moving (mesh motion), the point moves with it (it remains stationary
    in the reference frame of the target surface).

    Example
    -------

    Define :class:`SurfaceProbeOutput` on the :code:`geometry["wall"]` surface
    with multiple specific monitor points and monitor points along the line.

    - :code:`Point_1` and :code:`Point_2` are two specific points we want to monitor in this probe output group.
    - :code:`Line_surface` is from (1,0,0) * fl.u.m to (1,0,-10) * fl.u.m and has 11 monitor points,
      including both starting and end points.

    >>> fl.SurfaceProbeOutput(
    ...     name="surface_probe_group_points",
    ...     entities=[
    ...         fl.Point(
    ...             name="Point_1",
    ...             location=(0.0, 1.5, 0.0) * fl.u.m,
    ...         ),
    ...         fl.Point(
    ...             name="Point_2",
    ...             location=(0.0, -1.5, 0.0) * fl.u.m,
    ...         ),
    ...         fl.PointArray(
    ...             name="Line_surface",
    ...             start=(1.0, 0.0, 0.0) * fl.u.m,
    ...             end=(1.0, 0.0, -10.0) * fl.u.m,
    ...             number_of_points=11,
    ...         ),
    ...     ],
    ...     target_surfaces=[
    ...         geometry["wall"],
    ...     ],
    ...     output_fields=["heatFlux", "T"],
    ... )

    ====
    """

    name: str = pd.Field("Surface probe output", description="Name of the surface monitor group.")
    entities: EntityList[Point, PointArray] = pd.Field(
        alias="probe_points",
        description="List of monitored :class:`~flow360.Point`/"
        + ":class:`~flow360.PointArray` entities belonging to this "
        + "surface monitor group. :class:`~flow360.PointArray` "
        + "is used to define monitored points along a line.",
    )
    # Maybe add preprocess for this and by default add all Surfaces?
    target_surfaces: EntityList[Surface, MirroredSurface, WindTunnelGhostSurface] = pd.Field(
        description="List of :class:`~flow360.component.simulation.primitives.Surface` "
        + "entities belonging to this monitor group."
    )

    output_fields: UniqueItemList[Union[SurfaceFieldNames, str, UserVariable]] = pd.Field(
        description="List of output variables. Including :ref:`universal output variables<UniversalVariablesV2>`,"
        " :ref:`variables specific to SurfaceOutput<SurfaceSpecificVariablesV2>` and :class:`UserDefinedField`."
    )
    moving_statistic: Optional[MovingStatistic] = pd.Field(
        None, description="When specified, report moving statistics of the fields instead."
    )
    output_type: Literal["SurfaceProbeOutput"] = pd.Field("SurfaceProbeOutput", frozen=True)

    @contextual_field_validator("target_surfaces", mode="after")
    @classmethod
    def ensure_surface_existence(cls, value, param_info: ParamsValidationInfo):
        """Ensure all boundaries will be present after mesher"""
        return validate_entity_list_surface_existence(value, param_info)


class SurfaceSliceOutput(_AnimationAndFileFormatSettings, _OutputBase):
    """
    Surface slice settings.
    """

    name: str = pd.Field("Surface slice output", description="Name of the `SurfaceSliceOutput`.")
    entities: EntityList[Slice] = pd.Field(
        alias="slices", description="List of :class:`Slice` entities."
    )
    # Maybe add preprocess for this and by default add all Surfaces?
    target_surfaces: EntityList[Surface, MirroredSurface, WindTunnelGhostSurface] = pd.Field(
        description="List of :class:`Surface` entities on which the slice will cut through."
    )

    output_format: Literal["paraview"] = pd.Field(default="paraview")

    output_fields: UniqueItemList[Union[SurfaceFieldNames, str, UserVariable]] = pd.Field(
        description="List of output variables. Including :ref:`universal output variables<UniversalVariablesV2>`,"
        " :ref:`variables specific to SurfaceOutput<SurfaceSpecificVariablesV2>` and :class:`UserDefinedField`."
    )
    output_type: Literal["SurfaceSliceOutput"] = pd.Field("SurfaceSliceOutput", frozen=True)

    @contextual_field_validator("target_surfaces", mode="after")
    @classmethod
    def ensure_surface_existence(cls, value, param_info: ParamsValidationInfo):
        """Ensure all boundaries will be present after mesher"""
        return validate_entity_list_surface_existence(value, param_info)


class TimeAverageProbeOutput(ProbeOutput):
    """
    :class:`TimeAverageProbeOutput` class for time average probe monitor output settings.
    Regardless of the motion of the mesh, the points retain their positions in the
    global reference frame during the simulation.

    Example
    -------

    - Calculate the average value on multiple monitor points starting from the :math:`4^{th}` physical step.
      The results are output every 10 physical step starting from the :math:`14^{th}` physical step
      (14, 24, 34 etc.).

      >>> fl.TimeAverageProbeOutput(
      ...     name="time_average_probe_group_points",
      ...     entities=[
      ...         fl.Point(
      ...             name="Point_1",
      ...             location=(0.0, 1.5, 0.0) * fl.u.m,
      ...         ),
      ...         fl.Point(
      ...             name="Point_2",
      ...             location=(0.0, -1.5, 0.0) * fl.u.m,
      ...         ),
      ...     ],
      ...     output_fields=["primitiveVars", "Mach"],
      ...     start_step=4,
      ...     frequency=10,
      ...     frequency_offset=14,
      ... )

    - Calculate the average value on multiple monitor points starting from the :math:`4^{th}` physical step.
      The results are output every 10 physical step starting from the :math:`14^{th}` physical step
      (14, 24, 34 etc.).

      - :code:`Line_1` is from (1,0,0) * fl.u.m to (1.5,0,0) * fl.u.m and has 6 monitor points.
      - :code:`Line_2` is from (-1,0,0) * fl.u.m to (-1.5,0,0) * fl.u.m and has 3 monitor points,
        namely, (-1,0,0) * fl.u.m, (-1.25,0,0) * fl.u.m and (-1.5,0,0) * fl.u.m.

      >>> fl.TimeAverageProbeOutput(
      ...     name="time_average_probe_group_points",
      ...     entities=[
      ...         fl.PointArray(
      ...             name="Line_1",
      ...             start=(1.0, 0.0, 0.0) * fl.u.m,
      ...             end=(1.5, 0.0, 0.0) * fl.u.m,
      ...             number_of_points=6,
      ...         ),
      ...         fl.PointArray(
      ...             name="Line_2",
      ...             start=(-1.0, 0.0, 0.0) * fl.u.m,
      ...             end=(-1.5, 0.0, 0.0) * fl.u.m,
      ...             number_of_points=3,
      ...         ),
      ...     ],
      ...     output_fields=["primitiveVars", "Mach"],
      ...     start_step=4,
      ...     frequency=10,
      ...     frequency_offset=14,
      ... )

    ====

    """

    name: Optional[str] = pd.Field(
        "Time average probe output", description="Name of the `TimeAverageProbeOutput`."
    )
    # pylint: disable=abstract-method
    frequency: Union[pd.PositiveInt, Literal[-1]] = pd.Field(
        default=1,
        description="Frequency (in number of physical time steps) at which output is saved. "
        + "-1 is at end of simulation. Important for child cases - this parameter refers to the "
        + "**global** time step, which gets transferred from the parent case. Example: if the parent "
        + "case finished at time_step=174, the child case will start from time_step=175. If "
        + "frequency=100 (child case), the output will be saved at time steps 200 (25 time steps of "
        + "the child simulation), 300 (125 time steps of the child simulation), etc. "
        + "This setting is NOT applicable for steady cases.",
    )
    frequency_offset: int = pd.Field(
        default=0,
        ge=0,
        description="Offset (in number of physical time steps) at which output is started to be saved."
        + " 0 is at beginning of simulation. Important for child cases - this parameter refers to the "
        + "**global** time step, which gets transferred from the parent case (see `frequency` "
        + "parameter for an example). Example: if an output has a frequency of 100 and a "
        + "frequency_offset of 10, the output will be saved at **global** time step 10, 110, 210, "
        + "etc. This setting is NOT applicable for steady cases.",
    )
    start_step: Union[pd.NonNegativeInt, Literal[-1]] = pd.Field(
        default=-1,
        description="Physical time step to start calculating averaging. Important for child cases "
        + "- this parameter refers to the **global** time step, which gets transferred from the "
        + "parent case (see `frequency` parameter for an example).",
    )
    output_type: Literal["TimeAverageProbeOutput"] = pd.Field("TimeAverageProbeOutput", frozen=True)


class TimeAverageSurfaceProbeOutput(SurfaceProbeOutput):
    """
    :class:`TimeAverageSurfaceProbeOutput` class for time average surface probe monitor output settings.
    The specified monitor point will be projected to the :py:attr:`~TimeAverageSurfaceProbeOutput.target_surfaces`
    closest to the point. The probed results on the projected point will be dumped.
    The projection is executed at the start of the simulation. If the surface that the point was
    projected to is moving (mesh motion), the point moves with it (it remains stationary
    in the reference frame of the target surface).

    Example
    -------

    - Calculate the average value on the :code:`geometry["surface1"]` and :code:`geometry["surface2"]` surfaces
      with multiple monitor points. The average is computed starting from the :math:`4^{th}` physical step.
      The results are output every 10 physical step starting from the :math:`14^{th}` physical step
      (14, 24, 34 etc.).

      >>> TimeAverageSurfaceProbeOutput(
      ...     name="time_average_surface_probe_group_points",
      ...     entities=[
      ...         Point(name="Point_1", location=[1, 1.02, 0.03] * fl.u.cm),
      ...         Point(name="Point_2", location=[2, 1.01, 0.03] * fl.u.m),
      ...         Point(name="Point_3", location=[3, 1.02, 0.03] * fl.u.m),
      ...     ],
      ...     target_surfaces=[
      ...         Surface(name="Surface_1", geometry["surface1"]),
      ...         Surface(name="Surface_2", geometry["surface2"]),
      ...     ],
      ...     output_fields=["Mach", "primitiveVars", "yPlus"],
      ...     start_step=4,
      ...     frequency=10,
      ...     frequency_offset=14,
      ... )

    - Calculate the average value on the :code:`geometry["surface1"]` and :code:`geometry["surface2"]` surfaces
      with multiple monitor lines. The average is computed starting from the :math:`4^{th}` physical step.
      The results are output every 10 physical step starting from the :math:`14^{th}` physical step
      (14, 24, 34 etc.).

      - :code:`Line_1` is from (1,0,0) * fl.u.m to (1.5,0,0) * fl.u.m and has 6 monitor points.
      - :code:`Line_2` is from (-1,0,0) * fl.u.m to (-1.5,0,0) * fl.u.m and has 3 monitor points,
        namely, (-1,0,0) * fl.u.m, (-1.25,0,0) * fl.u.m and (-1.5,0,0) * fl.u.m.

      >>> TimeAverageSurfaceProbeOutput(
      ...     name="time_average_surface_probe_group_points",
      ...     entities=[
      ...         fl.PointArray(
      ...             name="Line_1",
      ...             start=(1.0, 0.0, 0.0) * fl.u.m,
      ...             end=(1.5, 0.0, 0.0) * fl.u.m,
      ...             number_of_points=6,
      ...         ),
      ...         fl.PointArray(
      ...             name="Line_2",
      ...             start=(-1.0, 0.0, 0.0) * fl.u.m,
      ...             end=(-1.5, 0.0, 0.0) * fl.u.m,
      ...             number_of_points=3,
      ...         ),
      ...     ],
      ...     target_surfaces=[
      ...         Surface(name="Surface_1", geometry["surface1"]),
      ...         Surface(name="Surface_2", geometry["surface2"]),
      ...     ],
      ...     output_fields=["Mach", "primitiveVars", "yPlus"],
      ...     start_step=4,
      ...     frequency=10,
      ...     frequency_offset=14,
      ... )

    ====
    """

    name: Optional[str] = pd.Field(
        "Time average surface probe output",
        description="Name of the `TimeAverageSurfaceProbeOutput`.",
    )
    # pylint: disable=abstract-method
    frequency: Union[pd.PositiveInt, Literal[-1]] = pd.Field(
        default=1,
        description="Frequency (in number of physical time steps) at which output is saved. "
        + "-1 is at end of simulation. Important for child cases - this parameter refers to the "
        + "**global** time step, which gets transferred from the parent case. Example: if the parent "
        + "case finished at time_step=174, the child case will start from time_step=175. If "
        + "frequency=100 (child case), the output will be saved at time steps 200 (25 time steps of "
        + "the child simulation), 300 (125 time steps of the child simulation), etc. "
        + "This setting is NOT applicable for steady cases.",
    )
    frequency_offset: int = pd.Field(
        default=0,
        ge=0,
        description="Offset (in number of physical time steps) at which output is started to be saved."
        + " 0 is at beginning of simulation. Important for child cases - this parameter refers to the "
        + "**global** time step, which gets transferred from the parent case (see `frequency` "
        + "parameter for an example). Example: if an output has a frequency of 100 and a "
        + "frequency_offset of 10, the output will be saved at **global** time step 10, 110, 210, "
        + "etc. This setting is NOT applicable for steady cases.",
    )
    start_step: Union[pd.NonNegativeInt, Literal[-1]] = pd.Field(
        default=-1,
        description="Physical time step to start calculating averaging. Important for child cases "
        + "- this parameter refers to the **global** time step, which gets transferred from the "
        + "parent case (see `frequency` parameter for an example).",
    )
    output_type: Literal["TimeAverageSurfaceProbeOutput"] = pd.Field(
        "TimeAverageSurfaceProbeOutput", frozen=True
    )


class Observer(Flow360BaseModel):
    """
    :class:`Observer` class for setting up the :py:attr:`AeroAcousticOutput.observers`.

    Example
    -------

    >>> fl.Observer(position=[1, 2, 3] * fl.u.m, group_name="1")

    ====
    """

    # pylint: disable=no-member
    position: LengthType.Point = pd.Field(
        description="Position at which time history of acoustic pressure signal "
        + "is stored in aeroacoustic output file. The observer position can be outside the simulation domain, "
        + "but cannot be on or inside the solid surfaces of the simulation domain."
    )
    group_name: str = pd.Field(
        description="Name of the group to which the observer will be assigned "
        + "for postprocessing purposes in Flow360 web client."
    )
    private_attribute_expand: Optional[bool] = pd.Field(None)


class AeroAcousticOutput(Flow360BaseModel):
    """

    :class:`AeroAcousticOutput` class for aeroacoustic output settings.

    Example
    -------

    >>> fl.AeroAcousticOutput(
    ...     observers=[
    ...         fl.Observer(position=[1.0, 0.0, 1.75] * fl.u.m, group_name="1"),
    ...         fl.Observer(position=[0.2, 0.3, 1.725] * fl.u.m, group_name="1"),
    ...     ],
    ... )

    If using permeable surfaces:

    >>> fl.AeroAcousticOutput(
    ...     observers=[
    ...         fl.Observer(position=[1.0, 0.0, 1.75] * fl.u.m, group_name="1"),
    ...         fl.Observer(position=[0.2, 0.3, 1.725] * fl.u.m, group_name="1"),
    ...     ],
    ...     patch_type="permeable",
    ...     permeable_surfaces=[volume_mesh["inner/interface*"]]
    ... )

    ====
    """

    name: Optional[str] = pd.Field(
        "Aeroacoustic output", description="Name of the `AeroAcousticOutput`."
    )
    patch_type: Literal["solid", "permeable"] = pd.Field(
        default="solid",
        description="Type of aeroacoustic simulation to "
        + "perform. `solid` uses solid walls to compute the "
        + "aeroacoustic solution. `permeable` uses surfaces "
        + "embedded in the volumetric domain as aeroacoustic solver "
        + "input.",
    )
    permeable_surfaces: Optional[
        EntityList[Surface, GhostSurface, GhostCircularPlane, GhostSphere, WindTunnelGhostSurface]
    ] = pd.Field(
        None, description="List of permeable surfaces. Left empty if `patch_type` is solid"
    )
    # pylint: disable=no-member
    observers: List[Observer] = pd.Field(
        description="A List of :class:`Observer` objects specifying each observer's position and group name."
    )
    write_per_surface_output: bool = pd.Field(
        False,
        description="Enable writing of aeroacoustic results on a per-surface basis, "
        + "in addition to results for all wall surfaces combined.",
    )
    observer_time_step_size: Optional[TimeType.Positive] = pd.Field(
        None,
        description="Time step size for aeroacoustic output. "
        + "A valid value is smaller than or equal to the time step size of the CFD simulation. "
        + "Defaults to time step size of CFD.",
    )
    aeroacoustic_solver_start_time: TimeType.NonNegative = pd.Field(
        0 * u.s,
        description="Time to start the aeroacoustic solver. "
        + "Signals emitted after this start time at the source surfaces are included in the output.",
    )
    force_clean_start: bool = pd.Field(
        False, description="Force a clean start when an aeroacoustic case is forked."
    )

    output_type: Literal["AeroAcousticOutput"] = pd.Field("AeroAcousticOutput", frozen=True)
    private_attribute_id: str = pd.Field(default_factory=generate_uuid, frozen=True)

    @pd.field_validator("observers", mode="after")
    @classmethod
    def validate_observer_has_same_unit(cls, input_value):
        """
        All observer location should have the same length unit.
        This is because UI has single toggle for all coordinates.
        """
        unit_set = {}
        for observer in input_value:
            unit_set[observer.position.units] = None
            if len(unit_set.keys()) > 1:
                raise ValueError(
                    "All observer locations should have the same unit."
                    f" But now it has both `{list(unit_set.keys())[0]}` and `{list(unit_set.keys())[1]}`."
                )
        return input_value

    @pd.model_validator(mode="after")
    def check_consistent_patch_type_and_permeable_surfaces(self):
        """Check if permeable_surfaces is None when patch_type is solid."""
        if self.patch_type == "solid" and self.permeable_surfaces is not None:
            raise ValueError("`permeable_surfaces` cannot be specified when `patch_type` is solid.")
        if self.patch_type == "permeable" and self.permeable_surfaces is None:
            raise ValueError("`permeable_surfaces` cannot be empty when `patch_type` is permeable.")

        return self

    @contextual_field_validator("permeable_surfaces", mode="after")
    @classmethod
    def ensure_surface_existence(cls, value, param_info: ParamsValidationInfo):
        """Ensure all boundaries will be present after mesher"""
        return validate_entity_list_surface_existence(value, param_info)


class StreamlineOutput(_OutputBase):
    """
    :class:`StreamlineOutput` class for calculating streamlines.
    Stramtraces are computed upwind and downwind, and may originate from a single point,
    from a line, or from points evenly distributed across a parallelogram.

    Example
    -------

    Define a :class:`StreamlineOutput` with streaptraces originating from points,
    lines (:class:`~flow360.PointArray`), and parallelograms (:class:`~flow360.PointArray2D`).

    - :code:`Point_1` and :code:`Point_2` are two specific points we want to track the streamlines.
    - :code:`Line_streamline` is from (1,0,0) * fl.u.m to (1,0,-10) * fl.u.m and has 11 points,
      including both starting and end points.
    - :code:`Parallelogram_streamline` is a parallelogram in 3D space with an origin at (1.0, 0.0, 0.0), a u-axis
      orientation of (0, 2.0, 2.0) with 11 points in the u direction, and a v-axis orientation of (0, 1.0, 0)
      with 20 points along the v direction.

    >>> fl.StreamlineOutput(
    ...     entities=[
    ...         fl.Point(
    ...             name="Point_1",
    ...             location=(0.0, 1.5, 0.0) * fl.u.m,
    ...         ),
    ...         fl.Point(
    ...             name="Point_2",
    ...             location=(0.0, -1.5, 0.0) * fl.u.m,
    ...         ),
    ...         fl.PointArray(
    ...             name="Line_streamline",
    ...             start=(1.0, 0.0, 0.0) * fl.u.m,
    ...             end=(1.0, 0.0, -10.0) * fl.u.m,
    ...             number_of_points=11,
    ...         ),
    ...         fl.PointArray2D(
    ...             name="Parallelogram_streamline",
    ...             origin=(1.0, 0.0, 0.0) * fl.u.m,
    ...             u_axis_vector=(0, 2.0, 2.0) * fl.u.m,
    ...             v_axis_vector=(0, 1.0, 0) * fl.u.m,
    ...             u_number_of_points=11,
    ...             v_number_of_points=20
    ...         )
    ...     ],
    ...     output_fields = [fl.solution.pressure, fl.solution.velocity],
    ... )

    ====
    """

    name: Optional[str] = pd.Field(
        "Streamline output", description="Name of the `StreamlineOutput`."
    )
    entities: EntityList[Point, PointArray, PointArray2D] = pd.Field(
        alias="streamline_points",
        description="List of monitored :class:`~flow360.Point`/"
        + ":class:`~flow360.PointArray`/:class:`~flow360.PointArray2D` "
        + "entities belonging to this "
        + "streamline group. :class:`~flow360.PointArray` "
        + "is used to define streamline originating along a line. "
        + ":class:`~flow360.PointArray2D` "
        + "is used to define streamline originating from a parallelogram.",
    )
    output_fields: Optional[UniqueItemList[UserVariable]] = pd.Field(
        [],
        description="List of output variables. Vector-valued fields will be colored by their magnitude.",
    )
    output_type: Literal["StreamlineOutput"] = pd.Field("StreamlineOutput", frozen=True)
    private_attribute_id: str = pd.Field(default_factory=generate_uuid, frozen=True)


class TimeAverageStreamlineOutput(StreamlineOutput):
    """
    :class:`StreamlineOutput` class for calculating time-averaged streamlines.
    Stramtraces are computed upwind and downwind, and may originate from a single point,
    from a line, or from points evenly distributed across a parallelogram.

    Example
    -------

    Define a :class:`TimeAverageStreamlineOutput` with streaptraces originating from points,
    lines (:class:`~flow360.PointArray`), and parallelograms (:class:`~flow360.PointArray2D`).

    - :code:`Point_1` and :code:`Point_2` are two specific points we want to track the streamlines.
    - :code:`Line_streamline` is from (1,0,0) * fl.u.m to (1,0,-10) * fl.u.m and has 11 points,
      including both starting and end points.
    - :code:`Parallelogram_streamline` is a parallelogram in 3D space with an origin at (1.0, 0.0, 0.0), a u-axis
      orientation of (0, 2.0, 2.0) with 11 points in the u direction, and a v-axis orientation of (0, 1.0, 0)
      with 20 points along the v direction.

    >>> fl.TimeAverageStreamlineOutput(
    ...     entities=[
    ...         fl.Point(
    ...             name="Point_1",
    ...             location=(0.0, 1.5, 0.0) * fl.u.m,
    ...         ),
    ...         fl.Point(
    ...             name="Point_2",
    ...             location=(0.0, -1.5, 0.0) * fl.u.m,
    ...         ),
    ...         fl.PointArray(
    ...             name="Line_streamline",
    ...             start=(1.0, 0.0, 0.0) * fl.u.m,
    ...             end=(1.0, 0.0, -10.0) * fl.u.m,
    ...             number_of_points=11,
    ...         ),
    ...         fl.PointArray2D(
    ...             name="Parallelogram_streamline",
    ...             origin=(1.0, 0.0, 0.0) * fl.u.m,
    ...             u_axis_vector=(0, 2.0, 2.0) * fl.u.m,
    ...             v_axis_vector=(0, 1.0, 0) * fl.u.m,
    ...             u_number_of_points=11,
    ...             v_number_of_points=20
    ...         )
    ...     ]
    ... )

    ====
    """

    name: Optional[str] = pd.Field(
        "Time-average Streamline output", description="Name of the `TimeAverageStreamlineOutput`."
    )

    start_step: Union[pd.NonNegativeInt, Literal[-1]] = pd.Field(
        default=-1,
        description="Physical time step to start calculating averaging. Important for child cases "
        + "- this parameter refers to the **global** time step, which gets transferred from the "
        + "parent case (see `frequency` parameter for an example).",
    )

    output_type: Literal["TimeAverageStreamlineOutput"] = pd.Field(
        "TimeAverageStreamlineOutput", frozen=True
    )


class ForceDistributionOutput(Flow360BaseModel):
    """
    :class:`ForceDistributionOutput` class for customized force and moment distribution output.
    Axis-aligned components are output for force and moment coefficients at the end of the simulation.

    Example
    -------

    Basic usage with default settings (all wall surfaces):

    >>> fl.ForceDistributionOutput(
    ...     name="spanwise",
    ...     distribution_direction=[0.1, 0.9, 0.0],
    ... )

    Specifying specific surfaces to include in the force integration (useful for automotive cases
    to exclude road/floor surfaces):

    >>> fl.ForceDistributionOutput(
    ...     name="vehicle_x_distribution",
    ...     distribution_direction=[1.0, 0.0, 0.0],
    ...     entities=[volume_mesh["vehicle_body"], volume_mesh["wheels"]],
    ...     number_of_segments=500,
    ... )

    ====
    """

    name: str = pd.Field(description="Name of the `ForceDistributionOutput`.")
    distribution_direction: Axis = pd.Field(
        description="Direction of the force distribution output."
    )
    distribution_type: Literal["incremental", "cumulative"] = pd.Field(
        "incremental", description="Type of the distribution."
    )
    entities: Optional[EntityList[Surface, MirroredSurface]] = pd.Field(
        None,
        alias="surfaces",
        description="List of surfaces to include in the force integration. "
        "If not specified, all wall surfaces are included. "
        "This is useful for automotive cases to exclude road/floor surfaces.",
    )
    number_of_segments: pd.PositiveInt = pd.Field(
        300,
        description="Number of segments (bins) to use along the distribution direction. "
        "Default is 300 segments. "
        "Increasing this value provides higher resolution in the force distribution plot.",
    )
    output_type: Literal["ForceDistributionOutput"] = pd.Field(
        "ForceDistributionOutput", frozen=True
    )

    @contextual_field_validator("entities", mode="after")
    @classmethod
    def ensure_surface_existence(cls, value, param_info: ParamsValidationInfo):
        """Ensure all boundaries will be present after mesher"""
        return validate_entity_list_surface_existence(value, param_info)

    @contextual_model_validator(mode="after")
    def ensure_surfaces_have_wall_bc(self, param_info: ParamsValidationInfo):
        """Ensure all specified surfaces have Wall boundary conditions assigned."""
        if self.entities is None:
            return self

        # Skip validation if physics_model_dict is not yet available
        if param_info.physics_model_dict is None:
            return self

        # Collect all surfaces that have Wall boundary conditions
        wall_surface_names = set()
        for model in param_info.physics_model_dict.values():
            if isinstance(model, Wall) and model.entities is not None:
                expanded_entities = param_info.expand_entity_list(model.entities)
                for entity in expanded_entities:
                    if hasattr(entity, "full_name"):
                        wall_surface_names.add(entity.full_name)
                    elif hasattr(entity, "name"):
                        wall_surface_names.add(entity.name)

        # Check that all specified surfaces have Wall BC
        expanded_entities = param_info.expand_entity_list(self.entities)
        non_wall_surfaces = []
        for entity in expanded_entities:
            entity_name = entity.full_name if hasattr(entity, "full_name") else entity.name
            if entity_name not in wall_surface_names:
                non_wall_surfaces.append(entity_name)

        if non_wall_surfaces:
            raise ValueError(
                f"The following surfaces do not have Wall boundary conditions assigned: "
                f"{non_wall_surfaces}. Force distribution output can only be computed on "
                f"surfaces with Wall boundary conditions."
            )

        return self


class TimeAverageForceDistributionOutput(ForceDistributionOutput):
    """
    :class:`TimeAverageForceDistributionOutput` class for time-averaged customized force and moment distribution output.
    Axis-aligned components are output for force and moment coefficients at the end of the simulation.

    Example
    -------

    Calculate the average value starting from the :math:`4^{th}` physical step.

    >>> fl.TimeAverageForceDistributionOutput(
    ...     name="spanwise",
    ...     distribution_direction=[0.1, 0.9, 0.0],
    ...     start_step=4,
    ... )

    Specifying specific surfaces to include in the force integration (useful for automotive cases
    to exclude road/floor surfaces):

    >>> fl.TimeAverageForceDistributionOutput(
    ...     name="vehicle_x_distribution",
    ...     distribution_direction=[1.0, 0.0, 0.0],
    ...     entities=[volume_mesh["vehicle_body"], volume_mesh["wheels"]],
    ...     number_of_segments=500,
    ...     start_step=100,
    ... )

    ====
    """

    name: str = pd.Field(
        "Time average force distribution output",
        description="Name of the `TimeAverageForceDistributionOutput`.",
    )
    start_step: Union[pd.NonNegativeInt, Literal[-1]] = pd.Field(
        default=-1,
        description="Physical time step to start calculating averaging. Important for child cases "
        + "- this parameter refers to the **global** time step, which gets transferred from the "
        + "parent case (see `frequency` parameter for an example).",
    )
    output_type: Literal["TimeAverageForceDistributionOutput"] = pd.Field(
        "TimeAverageForceDistributionOutput", frozen=True
    )


OutputTypes = Annotated[
    Union[
        SurfaceOutput,
        TimeAverageSurfaceOutput,
        VolumeOutput,
        TimeAverageVolumeOutput,
        SliceOutput,
        TimeAverageSliceOutput,
        IsosurfaceOutput,
        TimeAverageIsosurfaceOutput,
        SurfaceIntegralOutput,
        ProbeOutput,
        SurfaceProbeOutput,
        SurfaceSliceOutput,
        TimeAverageProbeOutput,
        TimeAverageSurfaceProbeOutput,
        AeroAcousticOutput,
        StreamlineOutput,
        TimeAverageStreamlineOutput,
        ForceDistributionOutput,
        TimeAverageForceDistributionOutput,
        ForceOutput,
        RenderOutput,
    ],
    pd.Field(discriminator="output_type"),
]

TimeAverageOutputTypes = (
    TimeAverageSurfaceOutput,
    TimeAverageVolumeOutput,
    TimeAverageSliceOutput,
    TimeAverageIsosurfaceOutput,
    TimeAverageProbeOutput,
    TimeAverageSurfaceProbeOutput,
    TimeAverageStreamlineOutput,
    TimeAverageForceDistributionOutput,
)

MonitorOutputType = Annotated[
    Union[ForceOutput, SurfaceIntegralOutput, ProbeOutput, SurfaceProbeOutput],
    pd.Field(discriminator="output_type"),
]
