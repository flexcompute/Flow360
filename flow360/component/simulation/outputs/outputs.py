"""Mostly the same as Flow360Param counterparts.
Caveats:
1. Check if we support non-average and average output specified at the same time in solver.
(Yes but they share the same output_fields)
2. We do not support mulitple output frequencies/file format for the same type of output.
"""

from typing import Annotated, List, Literal, Optional, Union

import pydantic as pd

from flow360.component.flow360_params.flow360_fields import (
    CommonFieldNames,
    SliceFieldNames,
    SurfaceFieldNames,
    VolumeFieldNames,
)
from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.framework.entity_base import EntityList
from flow360.component.simulation.framework.unique_list import (
    UniqueAliasedStringList,
    UniqueItemList,
)
from flow360.component.simulation.outputs.output_entities import (
    Isosurface,
    Point,
    Slice,
)
from flow360.component.simulation.primitives import GhostSurface, Surface
from flow360.component.simulation.unit_system import LengthType


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
    """Surface output settings."""

    # pylint: disable=fixme
    # TODO: entities is None --> use all surfaces. This is not implemented yet.
    name: Optional[str] = pd.Field(None, description="Name of the `SurfaceOutput`")
    entities: Optional[EntityList[Surface, GhostSurface]] = pd.Field(
        None,
        alias="surfaces",
        description="List of output surfaces. The name of the surface is used as the key. "
        + "These surface names have to be the patch name in the grid file or the alias name specified in case JSON.",
    )
    write_single_file: bool = pd.Field(
        default=False,
        description="Enable writing all surface outputs into a single file instead of one file per surface."
        + "This option currently only supports Tecplot output format."
        + "Will choose the value of the last instance of this option of the same output type "
        + "(SurfaceOutput or TimeAverageSurfaceOutput) in the `output` list.",
    )
    output_fields: UniqueAliasedStringList[SurfaceFieldNames] = pd.Field(
        description="List of output variables. Including :ref:`universal output variables<UniversalVariables>` "
        + "and :ref:`variables specific to surfaceOutput<SurfaceSpecificVariables>`. "
        + ":code:`outputFields` specified under :code:`surfaceOutput` will be added to all surfaces."
    )
    output_type: Literal["SurfaceOutput"] = pd.Field("SurfaceOutput", frozen=True)


class TimeAverageSurfaceOutput(SurfaceOutput):
    """
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
    """Volume output settings."""

    name: Optional[str] = pd.Field(None, description="Name of the `VolumeOutput`")
    output_fields: UniqueAliasedStringList[VolumeFieldNames] = pd.Field(
        description="List of output variables. Including :ref:`universal output variables<UniversalVariables>`, "
        + "and :ref:`variables specific to volumeOutput<VolumeAndSliceSpecificVariables>`."
    )
    output_type: Literal["VolumeOutput"] = pd.Field("VolumeOutput", frozen=True)


class TimeAverageVolumeOutput(VolumeOutput):
    """
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
    """Slice output settings."""

    name: Optional[str] = pd.Field(None, description="Name of the `SliceOutput`")
    entities: Optional[EntityList[Slice]] = pd.Field(
        None, alias="slices", description="List of output slice entities."
    )
    output_fields: UniqueAliasedStringList[SliceFieldNames] = pd.Field(
        description="List of output variables. Including :ref:`universal output variables<UniversalVariables>` "
        + "and :ref:`variables specific to sliceOutput<VolumeAndSliceSpecificVariables>`. "
        + ":code:`outputFields` specified under :code:`sliceOutput` will be added to all slices."
    )
    output_type: Literal["SliceOutput"] = pd.Field("SliceOutput", frozen=True)


class IsosurfaceOutput(_AnimationAndFileFormatSettings):
    """Isosurface output settings."""

    name: Optional[str] = pd.Field(None, description="Name of the `IsosurfaceOutput`")
    entities: Optional[UniqueItemList[Isosurface]] = pd.Field(
        None, alias="isosurfaces", description="List of isosurface entities."
    )
    output_fields: UniqueAliasedStringList[CommonFieldNames] = pd.Field(
        description=" Isosurface field variable to be written. One of :code:`p`, :code:`rho`, "
        + ":code:`Mach`, :code:`qcriterion`, :code:`s`, :code:`T`, :code:`Cp`, :code:`mut`, :code:`nuHat`."
    )
    output_type: Literal["IsosurfaceOutput"] = pd.Field("IsosurfaceOutput", frozen=True)


class SurfaceIntegralOutput(Flow360BaseModel):
    """Surface integral output settings."""

    name: str = pd.Field()
    entities: Optional[EntityList[Surface, GhostSurface]] = pd.Field(
        None,
        alias="surfaces",
        description="List of surface entities on which the surface integral will be calculated.",
    )
    output_fields: UniqueAliasedStringList[CommonFieldNames] = pd.Field(
        description="List of output fields which will be added to all monitors within the monitor group,"
        + " see universal output variables."
    )
    output_type: Literal["SurfaceIntegralOutput"] = pd.Field("SurfaceIntegralOutput", frozen=True)


class ProbeOutput(Flow360BaseModel):
    """Probe monitor output settings."""

    name: str = pd.Field(description="Name of the monitor group")
    entities: Optional[EntityList[Point]] = pd.Field(
        None,
        alias="probe_points",
        description="List of monitored point entities belonging to this monitor group.",
    )
    output_fields: UniqueAliasedStringList[CommonFieldNames] = pd.Field(
        description="List of output fields which will be added to all monitors within the monitor group,"
        + " see :ref:`universal output variables<UniversalVariables>`"
    )
    output_type: Literal["ProbeOutput"] = pd.Field("ProbeOutput", frozen=True)

    def load_point_location_from_file(self, file_path: str):
        """Load probe point locations from a file."""
        raise NotImplementedError("Not implemented yet.")


class AeroAcousticOutput(Flow360BaseModel):
    """AeroAcoustic output settings."""

    name: Optional[str] = pd.Field(None, description="Name of the `AeroAcousticOutput`")
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
        IsosurfaceOutput,
        SurfaceIntegralOutput,
        ProbeOutput,
        AeroAcousticOutput,
    ],
    pd.Field(discriminator="output_type"),
]
