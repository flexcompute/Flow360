"""Mostly the same as Flow360Param counterparts.
Caveats:
1. Check if we support non-average and average output specified at the same time in solver.
(Yes but they share the same output_fields)
2. We do not support mulitple output frequencies/file format for the same type of output.
"""

from typing import Annotated, List, Literal, Optional, Union

import pydantic as pd

from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.framework.entity_base import EntityList
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
from flow360.component.simulation.validation.validation_output import (
    _check_unique_probe_type,
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
        default="paraview", description='"paraview", "tecplot" or "both".'
    )


class SurfaceOutput(_AnimationAndFileFormatSettings):
    """:class:`SurfaceOutput` class for surface output settings."""

    # pylint: disable=fixme
    # TODO: entities is None --> use all surfaces. This is not implemented yet.

    name: Optional[str] = pd.Field(None, description="Name of the `SurfaceOutput`.")
    entities: EntityList[Surface, GhostSurface] = pd.Field(
        alias="surfaces",
        description="List of output :class:`~flow360.component.simulation.primitives.Surface`/"
        + ":class:`~flow360.component.simulation.primitives.GhostSurface` entities. "
        + "These surface names have to be the patch name in the grid file or the alias name specified in case JSON.",
    )
    write_single_file: bool = pd.Field(
        default=False,
        description="Enable writing all surface outputs into a single file instead of one file per surface."
        + "This option currently only supports Tecplot output format."
        + "Will choose the value of the last instance of this option of the same output type"
        + "(SurfaceOutput or TimeAverageSurfaceOutput) in the `output` list.",
    )
    output_fields: UniqueItemList[SurfaceFieldNames] = pd.Field(
        description="List of output variables. Including :ref:`universal output variables<UniversalVariables>` "
        + "and :ref:`variables specific to surfaceOutput<SurfaceSpecificVariables>`. "
    )
    output_type: Literal["SurfaceOutput"] = pd.Field("SurfaceOutput", frozen=True)


class TimeAverageSurfaceOutput(SurfaceOutput):
    """
    :class:`TimeAverageSurfaceOutput` class for time average surface output settings.
    Caveats:
    Solver side only accept exactly the same set of output_fields (is shared) between
    VolumeOutput and TimeAverageVolumeOutput.

    Notes
    -----
        Old `computeTimeAverages` can be infered when user is explicitly using for
        example `TimeAverageSurfaceOutput`.
    """

    start_step: Union[pd.NonNegativeInt, Literal[-1]] = pd.Field(
        default=-1, description="Physical time step to start calculating averaging"
    )
    output_type: Literal["TimeAverageSurfaceOutput"] = pd.Field(
        "TimeAverageSurfaceOutput", frozen=True
    )


class VolumeOutput(_AnimationAndFileFormatSettings):
    """:class:`VolumeOutput` class for volume output settings."""

    name: Optional[str] = pd.Field(None, description="Name of the `VolumeOutput`.")
    output_fields: UniqueItemList[VolumeFieldNames] = pd.Field(
        description="List of output variables. Including :ref:`universal output variables<UniversalVariables>`, "
        + "and :ref:`variables specific to volumeOutput<VolumeAndSliceSpecificVariables>`."
    )
    output_type: Literal["VolumeOutput"] = pd.Field("VolumeOutput", frozen=True)


class TimeAverageVolumeOutput(VolumeOutput):
    """
    :class:`TimeAverageVolumeOutput` class for time average volume output settings.
    Caveats:
    Solver side only accept exactly the same set of output_fields (is shared)
    between VolumeOutput and TimeAverageVolumeOutput.
    Also let's not worry about allowing entities here as it is not supported by solver anyway.

    Notes
    -----
        Old `computeTimeAverages` can be infered when user is explicitly using for example
        `TimeAverageSurfaceOutput`.
    """

    start_step: Union[pd.NonNegativeInt, Literal[-1]] = pd.Field(
        default=-1, description="Physical time step to start calculating averaging."
    )
    output_type: Literal["TimeAverageVolumeOutput"] = pd.Field(
        "TimeAverageVolumeOutput", frozen=True
    )


class SliceOutput(_AnimationAndFileFormatSettings):
    """:class:`SliceOutput` class for slice output settings."""

    name: Optional[str] = pd.Field(None, description="Name of the `SliceOutput`.")
    entities: EntityList[Slice] = pd.Field(
        alias="slices",
        description="List of output :class:`~flow360.component.simulation.outputs.output_entities.Slice` entities.",
    )
    output_fields: UniqueItemList[SliceFieldNames] = pd.Field(
        description="List of output variables. Including :ref:`universal output variables<UniversalVariables>` "
        + "and :ref:`variables specific to sliceOutput<VolumeAndSliceSpecificVariables>`. "
    )
    output_type: Literal["SliceOutput"] = pd.Field("SliceOutput", frozen=True)


class TimeAverageSliceOutput(SliceOutput):
    """:class:`TimeAverageSliceOutput` class for time average slice output settings."""

    start_step: Union[pd.NonNegativeInt, Literal[-1]] = pd.Field(
        default=-1, description="Physical time step to start calculating averaging"
    )
    output_type: Literal["TimeAverageSliceOutput"] = pd.Field("TimeAverageSliceOutput", frozen=True)


class IsosurfaceOutput(_AnimationAndFileFormatSettings):
    """:class:`IsosurfaceOutput` class for isosurface output settings."""

    name: Optional[str] = pd.Field(None, description="Name of the `IsosurfaceOutput`.")
    entities: UniqueItemList[Isosurface] = pd.Field(
        alias="isosurfaces",
        description="List of :class:`~flow360.component.simulation.outputs.output_entities.Isosurface` entities.",
    )
    output_fields: UniqueItemList[CommonFieldNames] = pd.Field(
        description=" Isosurface field variable to be written. One of :code:`p`, :code:`rho`, "
        + ":code:`Mach`, :code:`qcriterion`, :code:`s`, :code:`T`, :code:`Cp`, :code:`mut`, :code:`nuHat`."
    )
    output_type: Literal["IsosurfaceOutput"] = pd.Field("IsosurfaceOutput", frozen=True)


class SurfaceIntegralOutput(Flow360BaseModel):
    """:class:`SurfaceIntegralOutput` class for surface integral output settings."""

    name: str = pd.Field()

    entities: EntityList[Surface, GhostSurface] = pd.Field(
        alias="surfaces",
        description="List of :class:`~flow360.component.simulation.primitives.Surface`/"
        + ":class:`~flow360.component.simulation.primitives.GhostSurface` entities on which "
        + "the surface integral will be calculated.",
    )
    output_fields: UniqueItemList[CommonFieldNames] = pd.Field(
        description="List of output fields which will be added to all monitors within the monitor group,"
        + " see universal output variables."
    )
    output_type: Literal["SurfaceIntegralOutput"] = pd.Field("SurfaceIntegralOutput", frozen=True)


class ProbeOutput(Flow360BaseModel):
    """:class:`ProbeOutput` class for probe monitor output settings."""

    name: str = pd.Field(description="Name of the monitor group.")
    entities: EntityList[Point, PointArray] = pd.Field(
        alias="probe_points",
        description="List of monitored :class:`~flow360.component.simulation.outputs.output_entities.Point`/"
        + ":class:`~flow360.component.simulation.outputs.output_entities.PointArray` entities belonging to this "
        + "monitor group. :class:`~flow360.component.simulation.outputs.output_entities.PointArray` is used to "
        + "define monitored points along a line.",
    )
    output_fields: UniqueItemList[CommonFieldNames] = pd.Field(
        description="List of output fields which will be added to all monitors within the monitor group,"
        + " see :ref:`universal output variables<UniversalVariables>`"
    )
    output_type: Literal["ProbeOutput"] = pd.Field("ProbeOutput", frozen=True)

    def load_point_location_from_file(self, file_path: str):
        """Load probe point locations from a file."""
        raise NotImplementedError("Not implemented yet.")

    @pd.field_validator("entities", mode="after")
    @classmethod
    def check_unique_probe_type(cls, value):
        """Check to ensure every entity has the same type"""
        return _check_unique_probe_type(value, "ProbeOutput")


class SurfaceProbeOutput(Flow360BaseModel):
    """
    :class:`SurfaceProbeOutput` class for surface probe monitor output settings.
    The monitor location will be projected to the surface closest to the point.
    """

    name: str = pd.Field(description="Name of the surface monitor group.")
    entities: EntityList[Point, PointArray] = pd.Field(
        alias="probe_points",
        description="List of monitored :class:`~flow360.component.simulation.outputs.output_entities.Point`/"
        + ":class:`~flow360.component.simulation.outputs.output_entities.PointArray` entities belonging to this "
        + "surface monitor group. :class:`~flow360.component.simulation.outputs.output_entities.PointArray` "
        + "is used to define monitored points along a line.",
    )
    # Maybe add preprocess for this and by default add all Surfaces?
    target_surfaces: EntityList[Surface] = pd.Field(
        description="List of :class:`~flow360.component.simulation.primitives.Surface` "
        + "entities belonging to this monitor group."
    )

    output_fields: UniqueItemList[SurfaceFieldNames] = pd.Field()
    output_type: Literal["SurfaceProbeOutput"] = pd.Field("SurfaceProbeOutput", frozen=True)

    @pd.field_validator("entities", mode="after")
    @classmethod
    def check_unique_probe_type(cls, value):
        """Check to ensure every entity has the same type"""
        return _check_unique_probe_type(value, "SurfaceProbeOutput")


class AeroAcousticOutput(Flow360BaseModel):
    """:class:`AeroAcousticOutput` class for aeroacoustic output settings."""

    name: Optional[str] = pd.Field(None, description="Name of the `AeroAcousticOutput`.")
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


class UserDefinedFields(Flow360BaseModel):
    """Ignore this for now"""


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
        AeroAcousticOutput,
    ],
    pd.Field(discriminator="output_type"),
]
