"""Mostly the same as Flow360Param counterparts.
Caveats:
1. Check if we support non-average and average output specified at the same time in solver.
(Yes but they share the same output_fields)
2. We do not support multiple output frequencies/file format for the same type of output.
"""

from typing import Annotated, List, Literal, Optional, Union

import pydantic as pd

from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.framework.entity_base import EntityList
from flow360.component.simulation.framework.expressions import StringExpression
from flow360.component.simulation.framework.unique_list import UniqueItemList
from flow360.component.simulation.outputs.output_entities import (
    Isosurface,
    Point,
    PointArray,
    Slice,
)
from flow360.component.simulation.outputs.output_fields import (
    CommonFieldNames,
    SliceFieldNames,
    SurfaceFieldNames,
    VolumeFieldNames,
)
from flow360.component.simulation.primitives import GhostSurface, Surface
from flow360.component.simulation.unit_system import LengthType


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


class _AnimationSettings(Flow360BaseModel):
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
    entities: EntityList[Surface, GhostSurface] = pd.Field(
        alias="surfaces",
        description="List of output :class:`~flow360.Surface`/"
        + ":class:`~flow360.GhostSurface` entities. ",
    )
    write_single_file: bool = pd.Field(
        default=False,
        description="Enable writing all surface outputs into a single file instead of one file per surface."
        + "This option currently only supports Tecplot output format."
        + "Will choose the value of the last instance of this option of the same output type "
        + "(:class:`SurfaceOutput` or :class:`TimeAverageSurfaceOutput`) in the output list.",
    )
    output_fields: UniqueItemList[Union[SurfaceFieldNames, str]] = pd.Field(
        description="List of output variables. Including :ref:`universal output variables<UniversalVariablesV2>`,"
        + " :ref:`variables specific to SurfaceOutput<SurfaceSpecificVariablesV2>` and :class:`UserDefinedField`."
    )
    output_type: Literal["SurfaceOutput"] = pd.Field("SurfaceOutput", frozen=True)


class TimeAverageSurfaceOutput(SurfaceOutput):
    """
    :class:`TimeAverageSurfaceOutput` class for time average surface output settings.

    Example
    -------

    Calculate the average value starting from the :math:`4^{th}` physical step.
    The results are output every 10 physical step starting from the :math:`14^{th}` physical step
    (14, 24, 34 etc.).

    >>> fl.TimeAverageSurfaceOutput(
    ...     output_format=["primitiveVars"],
    ...     output_fields=restart_output_fields,
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
    output_fields: UniqueItemList[Union[VolumeFieldNames, str]] = pd.Field(
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
    output_fields: UniqueItemList[Union[SliceFieldNames, str]] = pd.Field(
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
    output_fields: UniqueItemList[Union[CommonFieldNames, str]] = pd.Field(
        description="List of output variables. Including "
        ":ref:`universal output variables<UniversalVariablesV2>` and :class:`UserDefinedField`."
    )
    output_type: Literal["IsosurfaceOutput"] = pd.Field("IsosurfaceOutput", frozen=True)


class SurfaceIntegralOutput(Flow360BaseModel):
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

    name: str = pd.Field(description="Name of integral.")
    entities: EntityList[Surface, GhostSurface] = pd.Field(
        alias="surfaces",
        description="List of :class:`~flow360.component.simulation.primitives.Surface`/"
        + ":class:`~flow360.component.simulation.primitives.GhostSurface` entities on which "
        + "the surface integral will be calculated.",
    )
    output_fields: UniqueItemList[str] = pd.Field(
        description="List of output variables, only the :class:`UserDefinedField` is allowed."
    )
    output_type: Literal["SurfaceIntegralOutput"] = pd.Field("SurfaceIntegralOutput", frozen=True)


class ProbeOutput(Flow360BaseModel):
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

    name: str = pd.Field(description="Name of the monitor group.")
    entities: EntityList[Point, PointArray] = pd.Field(
        alias="probe_points",
        description="List of monitored :class:`~flow360.Point`/"
        + ":class:`~flow360.PointArray` entities belonging to this "
        + "monitor group. :class:`~flow360.PointArray` is used to "
        + "define monitored points along a line.",
    )
    output_fields: UniqueItemList[Union[CommonFieldNames, str]] = pd.Field(
        description="List of output fields. Including :ref:`universal output variables<UniversalVariablesV2>`"
        " and :class:`UserDefinedField`."
    )
    output_type: Literal["ProbeOutput"] = pd.Field("ProbeOutput", frozen=True)

    @classmethod
    def load_point_location_from_file(cls, file_path: str):
        """Load probe point locations from a file. (Not implemented yet)"""
        raise NotImplementedError("Not implemented yet.")


class SurfaceProbeOutput(Flow360BaseModel):
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

    name: str = pd.Field(description="Name of the surface monitor group.")
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

    output_fields: UniqueItemList[Union[SurfaceFieldNames, str]] = pd.Field(
        description="List of output variables. Including :ref:`universal output variables<UniversalVariablesV2>`,"
        " :ref:`variables specific to SurfaceOutput<SurfaceSpecificVariablesV2>` and :class:`UserDefinedField`."
    )
    output_type: Literal["SurfaceProbeOutput"] = pd.Field("SurfaceProbeOutput", frozen=True)


class SurfaceSliceOutput(_AnimationAndFileFormatSettings):
    """
    Surface slice settings.
    """

    name: str = pd.Field(description="Name of the `SurfaceSliceOutput`.")
    entities: EntityList[Slice] = pd.Field(
        alias="slices", description="List of :class:`Slice` entities."
    )
    # Maybe add preprocess for this and by default add all Surfaces?
    target_surfaces: EntityList[Surface] = pd.Field(
        description="List of :class:`Surface` entities on which the slice will cut through."
    )

    output_format: Literal["paraview"] = pd.Field(default="paraview")

    output_fields: UniqueItemList[Union[SurfaceFieldNames, str]] = pd.Field(
        description="List of output variables. Including :ref:`universal output variables<UniversalVariablesV2>`,"
        " :ref:`variables specific to SurfaceOutput<SurfaceSpecificVariablesV2>` and :class:`UserDefinedField`."
    )
    output_type: Literal["SurfaceSliceOutput"] = pd.Field("SurfaceSliceOutput", frozen=True)


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

    # pylint: disable=abstract-method
    frequency: int = pd.Field(default=1, ge=1)
    frequency_offset: int = pd.Field(default=0, ge=0)
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
      ...         Point(name="Point_1", location=[1, 1.02, 0.03] * u.cm),
      ...         Point(name="Point_2", location=[2, 1.01, 0.03] * u.m),
      ...         Point(name="Point_3", location=[3, 1.02, 0.03] * u.m),
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

    # pylint: disable=abstract-method
    frequency: int = pd.Field(default=1, ge=1)
    frequency_offset: int = pd.Field(default=0, ge=0)
    start_step: Union[pd.NonNegativeInt, Literal[-1]] = pd.Field(
        default=-1, description="Physical time step to start calculating averaging"
    )
    output_type: Literal["TimeAverageSurfaceProbeOutput"] = pd.Field(
        "TimeAverageSurfaceProbeOutput", frozen=True
    )


class AeroAcousticOutput(Flow360BaseModel):
    """

    :class:`AeroAcousticOutput` class for aeroacoustic output settings.

    Example
    -------

    >>> fl.AeroAcousticOutput(
    ...     observers=[
    ...         [0.0, 0.0, 1.75] * fl.u.m,
    ...         [0.0, 0.3, 1.725] * fl.u.m,
    ...     ],
    ... )

    ====
    """

    name: Optional[str] = pd.Field(
        "Aeroacoustic output", description="Name of the `AeroAcousticOutput`."
    )
    patch_type: Literal["solid"] = pd.Field("solid", frozen=True)
    # pylint: disable=no-member
    observers: List[LengthType.Point] = pd.Field(
        description="List of observer locations at which time history of acoustic pressure signal "
        + "is stored in aeroacoustic output file. The observer locations can be outside the simulation domain, "
        + "but cannot be on or inside the solid surfaces of the simulation domain."
    )
    write_per_surface_output: bool = pd.Field(
        False,
        description="Enable writing of aeroacoustic results on a per-surface basis, "
        + "in addition to results for all wall surfaces combined.",
    )
    output_type: Literal["AeroAcousticOutput"] = pd.Field("AeroAcousticOutput", frozen=True)


OutputTypes = Annotated[
    Union[
        SurfaceOutput,
        TimeAverageSurfaceOutput,
        VolumeOutput,
        TimeAverageVolumeOutput,
        SliceOutput,
        TimeAverageSliceOutput,
        IsosurfaceOutput,
        SurfaceIntegralOutput,
        ProbeOutput,
        SurfaceProbeOutput,
        SurfaceSliceOutput,
        TimeAverageProbeOutput,
        TimeAverageSurfaceProbeOutput,
        AeroAcousticOutput,
    ],
    pd.Field(discriminator="output_type"),
]
