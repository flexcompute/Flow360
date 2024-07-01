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
    Probe,
    Slice,
    SurfaceList,
)
from flow360.component.simulation.primitives import Surface
from flow360.component.simulation.unit_system import LengthType


class _AnimationSettings(Flow360BaseModel):
    """
    Controls how frequently the output files are generated.
    """

    frequency: int = pd.Field(
        default=-1,
        ge=-1,
        description="""Frequency (in number of physical time steps) at which output is saved.
        -1 is at end of simulation.""",
    )
    frequency_offset: int = pd.Field(
        default=0,
        ge=0,
        description="""Offset (in number of physical time steps) at which output animation is started.
        0 is at beginning of simulation.""",
    )


class _AnimationAndFileFormatSettings(_AnimationSettings):
    """
    Controls how frequently the output files are generated and the file format.
    """

    output_format: Literal["paraview", "tecplot", "both"] = pd.Field(default="paraview")


class SurfaceOutput(_AnimationAndFileFormatSettings):
    """Surface output settings."""

    # pylint: disable=fixme
    # TODO: entities is None --> use all surfaces. This is not implemented yet.
    name: Optional[str] = pd.Field(None)
    entities: Optional[EntityList[Surface]] = pd.Field(None, alias="surfaces")
    write_single_file: bool = pd.Field(
        default=False,
        description="""Enable writing all surface outputs into a single file instead of one file per surface.
        This option currently only supports Tecplot output format.
        Will choose the value of the last instance of this option of the same output type
        (SurfaceOutput or TimeAverageSurfaceOutput) in the `output` list.""",
    )
    output_fields: UniqueAliasedStringList[SurfaceFieldNames] = pd.Field()
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

    name: Optional[str] = pd.Field(None)
    output_fields: UniqueAliasedStringList[VolumeFieldNames] = pd.Field()
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
        default=-1, description="Physical time step to start calculating averaging"
    )
    output_type: Literal["TimeAverageVolumeOutput"] = pd.Field(
        "TimeAverageVolumeOutput", frozen=True
    )


class SliceOutput(_AnimationAndFileFormatSettings):
    """Slice output settings."""

    name: Optional[str] = pd.Field(None)
    entities: Optional[UniqueItemList[Slice]] = pd.Field(None, alias="slices")
    output_fields: UniqueAliasedStringList[SliceFieldNames] = pd.Field()
    output_type: Literal["SliceOutput"] = pd.Field("SliceOutput", frozen=True)


class IsosurfaceOutput(_AnimationAndFileFormatSettings):
    """Isosurface output settings."""

    name: Optional[str] = pd.Field(None)
    entities: Optional[UniqueItemList[Isosurface]] = pd.Field(None, alias="isosurfaces")
    output_fields: UniqueAliasedStringList[CommonFieldNames] = pd.Field()
    output_type: Literal["IsosurfaceOutput"] = pd.Field("IsosurfaceOutput", frozen=True)


class SurfaceIntegralOutput(_AnimationSettings):
    """Surface integral output settings."""

    name: Optional[str] = pd.Field(None)
    entities: Optional[UniqueItemList[SurfaceList]] = pd.Field(None, alias="monitors")
    output_fields: UniqueAliasedStringList[CommonFieldNames] = pd.Field()
    output_type: Literal["SurfaceIntegralOutput"] = pd.Field("SurfaceIntegralOutput", frozen=True)


class ProbeOutput(_AnimationSettings):
    """Probe monitor output settings."""

    name: Optional[str] = pd.Field(None)
    entities: Optional[UniqueItemList[Probe]] = pd.Field(None, alias="probes")
    output_fields: UniqueAliasedStringList[CommonFieldNames] = pd.Field()
    output_type: Literal["ProbeOutput"] = pd.Field("ProbeOutput", frozen=True)


class AeroAcousticOutput(Flow360BaseModel):
    """AeroAcoustic output settings."""

    name: Optional[str] = pd.Field(None)
    patch_type: Literal["solid"] = pd.Field("solid", frozen=True)
    # pylint: disable=no-member
    observers: List[LengthType.Point] = pd.Field()
    write_per_surface_output: bool = pd.Field(False)
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
