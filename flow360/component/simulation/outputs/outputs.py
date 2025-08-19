"""Mostly the same as Flow360Param counterparts.
Caveats:
1. Check if we support non-average and average output specified at the same time in solver.
(Yes but they share the same output_fields)
2. We do not support multiple output frequencies/file format for the same type of output.
"""

# pylint: disable=too-many-lines
from typing import Annotated, List, Literal, Optional, Union, get_args

import pydantic as pd

from flow360.component.simulation.framework.base_model import (
    Flow360BaseModel,
    RegistryLookup,
)
from flow360.component.simulation.framework.entity_base import EntityList
from flow360.component.simulation.framework.expressions import StringExpression
from flow360.component.simulation.framework.unique_list import UniqueItemList
from flow360.component.simulation.models.surface_models import EntityListAllowingGhost
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
    InvalidOutputFieldsForLiquid,
    SliceFieldNames,
    SurfaceFieldNames,
    VolumeFieldNames,
    get_field_values,
)
from flow360.component.simulation.primitives import (
    GhostCircularPlane,
    GhostSphere,
    GhostSurface,
    ImportedSurface,
    Surface,
)
from flow360.component.simulation.unit_system import LengthType
from flow360.component.simulation.user_code.core.types import (
    Expression,
    UserVariable,
    solver_variable_to_user_variable,
)
from flow360.component.simulation.validation.validation_context import (
    ALL,
    CASE,
    get_validation_info,
    get_validation_levels,
)
from flow360.component.simulation.validation.validation_utils import (
    check_deleted_surface_in_entity_list,
)


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


class _OutputBase(Flow360BaseModel):
    output_fields: UniqueItemList[str] = pd.Field()

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

    @pd.field_validator("output_fields", mode="after")
    @classmethod
    def _validate_non_liquid_output_fields(cls, value: UniqueItemList):
        validation_info = get_validation_info()
        if validation_info is None or validation_info.using_liquid_as_material is False:
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


class _AnimationSettings(_OutputBase):
    """
    Controls how frequently the output files are generated.
    """

    frequency: int = pd.Field(
        default=-1,
        ge=-1,
        description="Frequency (in number of physical time steps) at which output is saved. "
        + "-1 is at end of simulation.",
    )
    frequency_offset: int = pd.Field(
        default=0,
        ge=0,
        description="Offset (in number of physical time steps) at which output animation is started."
        + " 0 is at beginning of simulation.",
    )


class _AnimationAndFileFormatSettings(_AnimationSettings):
    """
    Controls how frequently the output files are generated and the file format.
    """

    output_format: Literal["paraview", "tecplot", "both"] = pd.Field(
        default="paraview", description=":code:`paraview`, :code:`tecplot` or :code:`both`."
    )


class SurfaceOutput(_AnimationAndFileFormatSettings):
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
    entities: EntityListAllowingGhost[Surface, GhostSurface, GhostCircularPlane, GhostSphere] = (
        pd.Field(
            alias="surfaces",
            description="List of boundaries where output is generated.",
        )
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

    @pd.field_validator("entities", mode="after")
    @classmethod
    def ensure_surface_existence(cls, value):
        """Ensure all boundaries will be present after mesher"""
        return check_deleted_surface_in_entity_list(value)


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
        default=-1, description="Physical time step to start calculating averaging."
    )
    output_type: Literal["TimeAverageSurfaceOutput"] = pd.Field(
        "TimeAverageSurfaceOutput", frozen=True
    )


class VolumeOutput(_AnimationAndFileFormatSettings):
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
        default=-1, description="Physical time step to start calculating averaging."
    )
    output_type: Literal["TimeAverageVolumeOutput"] = pd.Field(
        "TimeAverageVolumeOutput", frozen=True
    )


class SliceOutput(_AnimationAndFileFormatSettings):
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
        default=-1, description="Physical time step to start calculating averaging."
    )
    output_type: Literal["TimeAverageSliceOutput"] = pd.Field("TimeAverageSliceOutput", frozen=True)


class IsosurfaceOutput(_AnimationAndFileFormatSettings):
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
        registry_lookup: RegistryLookup = None,
    ) -> Flow360BaseModel:
        exclude_isosurface_output = exclude + ["iso_value"]
        return super().preprocess(
            params=params,
            exclude=exclude_isosurface_output,
            required_by=required_by,
            registry_lookup=registry_lookup,
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
        default=-1, description="Physical time step to start calculating averaging."
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
    See :ref:`User Defined Postprocessing Tutorial <UserDefinedPostprocessing>` for more details
    about how to set up :class:`UserDefinedField`.

    Example
    -------
    Define :class:`SurfaceIntegralOutput` of :code:`PressureForce` as set up in this
    :ref:`User Defined Postprocessing Tutorial Case <UDFSurfIntegral>`.

    >>> fl.SurfaceIntegralOutput(
    ...     name="surface_integral",
    ...     output_fields=["PressureForce"],
    ...     entities=[volume_mesh["wing1"], volume_mesh["wing2"]],
    ... )

    ====
    """

    name: str = pd.Field("Surface integral output", description="Name of integral.")
    entities: EntityListAllowingGhost[Surface, GhostSurface, GhostCircularPlane, GhostSphere] = (
        pd.Field(
            alias="surfaces",
            description="List of boundaries where the surface integral will be calculated.",
        )
    )
    output_fields: UniqueItemList[Union[str, UserVariable]] = pd.Field(
        description="List of output variables, only the :class:`UserDefinedField` is allowed."
    )
    output_type: Literal["SurfaceIntegralOutput"] = pd.Field("SurfaceIntegralOutput", frozen=True)

    @pd.field_validator("entities", mode="after")
    @classmethod
    def ensure_surface_existence(cls, value):
        """Ensure all boundaries will be present after mesher"""
        return check_deleted_surface_in_entity_list(value)


class ProbeOutput(_OutputBase):
    """
    :class:`ProbeOutput` class for setting output data probed at monitor points.

    Example
    -------

    Define :class:`ProbeOutput` on multiple specific monitor points and monitor points along the line.

    - :code:`Point_1` and :code:`Point_2` are two specific points we want to monitor in this probe output group.
    - :code:`Line_1` is from (1,0,0) * fl.u,m to (1.5,0,0) * fl.u,m and has 6 monitor points.
    - :code:`Line_2` is from (-1,0,0) * fl.u,m to (-1.5,0,0) * fl.u,m and has 3 monitor points,
      namely, (-1,0,0) * fl.u,m, (-1.25,0,0) * fl.u,m and (-1.5,0,0) * fl.u,m.

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
    output_type: Literal["ProbeOutput"] = pd.Field("ProbeOutput", frozen=True)


class SurfaceProbeOutput(_OutputBase):
    """
    :class:`SurfaceProbeOutput` class for setting surface output data probed at monitor points.
    The specified monitor point will be projected to the :py:attr:`~SurfaceProbeOutput.target_surfaces`
    closest to the point. The probed results on the projected point will be dumped.

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
    target_surfaces: EntityList[Surface] = pd.Field(
        description="List of :class:`~flow360.component.simulation.primitives.Surface` "
        + "entities belonging to this monitor group."
    )

    output_fields: UniqueItemList[Union[SurfaceFieldNames, str, UserVariable]] = pd.Field(
        description="List of output variables. Including :ref:`universal output variables<UniversalVariablesV2>`,"
        " :ref:`variables specific to SurfaceOutput<SurfaceSpecificVariablesV2>` and :class:`UserDefinedField`."
    )
    output_type: Literal["SurfaceProbeOutput"] = pd.Field("SurfaceProbeOutput", frozen=True)

    @pd.field_validator("target_surfaces", mode="after")
    @classmethod
    def ensure_surface_existence(cls, value):
        """Ensure all boundaries will be present after mesher"""
        return check_deleted_surface_in_entity_list(value)


class SurfaceSliceOutput(_AnimationAndFileFormatSettings):
    """
    Surface slice settings.
    """

    name: str = pd.Field("Surface slice output", description="Name of the `SurfaceSliceOutput`.")
    entities: EntityList[Slice] = pd.Field(
        alias="slices", description="List of :class:`Slice` entities."
    )
    # Maybe add preprocess for this and by default add all Surfaces?
    target_surfaces: EntityList[Surface] = pd.Field(
        description="List of :class:`Surface` entities on which the slice will cut through."
    )

    output_format: Literal["paraview"] = pd.Field(default="paraview")

    output_fields: UniqueItemList[Union[SurfaceFieldNames, str, UserVariable]] = pd.Field(
        description="List of output variables. Including :ref:`universal output variables<UniversalVariablesV2>`,"
        " :ref:`variables specific to SurfaceOutput<SurfaceSpecificVariablesV2>` and :class:`UserDefinedField`."
    )
    output_type: Literal["SurfaceSliceOutput"] = pd.Field("SurfaceSliceOutput", frozen=True)

    @pd.field_validator("target_surfaces", mode="after")
    @classmethod
    def ensure_surface_existence(cls, value):
        """Ensure all boundaries will be present after mesher"""
        return check_deleted_surface_in_entity_list(value)


class TimeAverageProbeOutput(ProbeOutput):
    """
    :class:`TimeAverageProbeOutput` class for time average probe monitor output settings.

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

      - :code:`Line_1` is from (1,0,0) * fl.u,m to (1.5,0,0) * fl.u,m and has 6 monitor points.
      - :code:`Line_2` is from (-1,0,0) * fl.u,m to (-1.5,0,0) * fl.u,m and has 3 monitor points,
        namely, (-1,0,0) * fl.u,m, (-1.25,0,0) * fl.u,m and (-1.5,0,0) * fl.u,m.

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
    frequency: int = pd.Field(
        default=1,
        ge=-1,
        description="Frequency (in number of physical time steps) at which output is saved. "
        + "-1 is at end of simulation.",
    )
    frequency_offset: int = pd.Field(
        default=0,
        ge=0,
        description="Offset (in number of physical time steps) at which output animation is started."
        + " 0 is at beginning of simulation.",
    )
    start_step: Union[pd.NonNegativeInt, Literal[-1]] = pd.Field(
        default=-1, description="Physical time step to start calculating averaging"
    )
    output_type: Literal["TimeAverageProbeOutput"] = pd.Field("TimeAverageProbeOutput", frozen=True)


class TimeAverageSurfaceProbeOutput(SurfaceProbeOutput):
    """
    :class:`TimeAverageSurfaceProbeOutput` class for time average surface probe monitor output settings.
    The specified monitor point will be projected to the :py:attr:`~TimeAverageSurfaceProbeOutput.target_surfaces`
    closest to the point. The probed results on the projected point will be dumped.

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

      - :code:`Line_1` is from (1,0,0) * fl.u,m to (1.5,0,0) * fl.u,m and has 6 monitor points.
      - :code:`Line_2` is from (-1,0,0) * fl.u,m to (-1.5,0,0) * fl.u,m and has 3 monitor points,
        namely, (-1,0,0) * fl.u,m, (-1.25,0,0) * fl.u,m and (-1.5,0,0) * fl.u,m.

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
    frequency: int = pd.Field(
        default=1,
        ge=-1,
        description="Frequency (in number of physical time steps) at which output is saved. "
        + "-1 is at end of simulation.",
    )
    frequency_offset: int = pd.Field(
        default=0,
        ge=0,
        description="Offset (in number of physical time steps) at which output animation is started."
        + " 0 is at beginning of simulation.",
    )
    start_step: Union[pd.NonNegativeInt, Literal[-1]] = pd.Field(
        default=-1, description="Physical time step to start calculating averaging"
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
        EntityListAllowingGhost[Surface, GhostSurface, GhostCircularPlane, GhostSphere]
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
    output_type: Literal["AeroAcousticOutput"] = pd.Field("AeroAcousticOutput", frozen=True)

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

    @pd.field_validator("permeable_surfaces", mode="after")
    @classmethod
    def ensure_surface_existence(cls, value):
        """Ensure all boundaries will be present after mesher"""
        if value is None:
            return value
        return check_deleted_surface_in_entity_list(value)


class StreamlineOutput(Flow360BaseModel):
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
    ...     ]
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
    output_type: Literal["StreamlineOutput"] = pd.Field("StreamlineOutput", frozen=True)


class ImportedSurfaceOutput(_AnimationAndFileFormatSettings):
    """
    :class:`ImportedSurfaceOutput` class for generating interpolated output on imported surfaces.

    Example
    -------
    >>> fl.ImportedSurfaceOutput(
    ...     name="Jet_cross_sections_output",
    ...     entities=[
    ...         geometry.imported_surfaces["*"],
    ...     ],
    ...     output_fields=[
    ...         fl.solution.Cp,
    ...     ]
    ... )

    ====
    """

    name: Optional[str] = pd.Field(
        "Imported surface output", description="Name of the `ImportedSurfaceOutput`."
    )
    entities: EntityList[ImportedSurface] = pd.Field(
        alias="surfaces",
        description="List of imported surfaces where output is generated.",
    )
    output_fields: UniqueItemList[UserVariable] = pd.Field(description="List of output variables.")
    output_type: Literal["ImportedSurfaceOutput"] = pd.Field("ImportedSurfaceOutput", frozen=True)


class TimeAverageImportedSurfaceOutput(ImportedSurfaceOutput):
    """
    :class:`TimeAverageImportedSurfaceOutput` class for generating **time-averaged**
    output on imported surfaces.

    Similar to :class:`ImportedSurfaceOutput`, this output type records user-specified
    variables on imported geometry surfaces, but instead of instantaneous values,
    it computes averages over a specified range of physical time steps.

    Example
    -------
    >>> fl.TimeAverageImportedSurfaceOutput(
    ...     name="Jet_cross_sections_output",
    ...     entities=[
    ...         geometry.imported_surfaces["*"],
    ...     ],
    ...     output_fields=[
    ...         fl.solution.Cp,
    ...     ],
    ...     start_step=2000
    ... )

    ====
    """

    name: Optional[str] = pd.Field(
        "Time average imported surface output",
        description="Name of the `TimeAverageImportedSurfaceOutput`.",
    )
    start_step: Union[pd.NonNegativeInt, Literal[-1]] = pd.Field(
        default=-1, description="Physical time step to start calculating averaging"
    )
    output_type: Literal["TimeAverageImportedSurfaceOutput"] = pd.Field(
        "TimeAverageImportedSurfaceOutput", frozen=True
    )


class ImportedSurfaceIntegralOutput(_OutputBase):
    """
    :class:`ImportedSurfaceIntegralOutput` class for computing integrals of
    user-specified variables over imported surfaces.
    Integrals are computed for each of the individual surfaces.

    Example
    -------
    Define a :class:`ImportedSurfaceIntegralOutput` to compute the integrated
    mass flow rate across an imported cross-section plane
    placed downstream of a nozzle. These planes are provided only for
    post-processing and are not part of the simulated mesh boundaries.

    >>> fl.ImportedSurfaceIntegralOutput(
    ...     name="Nozzle_exit_planes_integrals",
    ...     entities=[
    ...         geometry.imported_surfaces["*"],
    ...     ],
    ...     output_fields=[
    ...         fl.UserVariable(
    ...             name="MassFlowRate",
    ...             value=fl.solution.density
    ...             * fl.math.dot(fl.solution.velocity, fl.solution.node_unit_normal)
    ...         ),
    ...     ]
    ... )

    ====
    """

    name: str = pd.Field("Imported surface integral output", description="Name of integral.")
    entities: EntityList[ImportedSurface] = pd.Field(
        alias="surfaces",
        description="List of boundaries where the surface integral will be calculated.",
    )
    output_fields: UniqueItemList[UserVariable] = pd.Field(description="List of output variables.")
    output_type: Literal["ImportedSurfaceIntegralOutput"] = pd.Field(
        "ImportedSurfaceIntegralOutput", frozen=True
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
        ImportedSurfaceOutput,
        TimeAverageImportedSurfaceOutput,
        ImportedSurfaceIntegralOutput,
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
    TimeAverageImportedSurfaceOutput,
)
