from typing import List, Literal, Optional, Tuple, Union

import pydantic as pd

from flow360.component.flow360_params.flow360_fields import (
    CommonFields,
    SliceFields,
    SurfaceFields,
    VolumeFields,
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

"""Mostly the same as Flow360Param counterparts.
Caveats:
1. Check if we support non-average and average output specified at the same time in solver. (Yes but they share the same output_fields)
2. We do not support mulitple output frequencies/file format for the same type of output.
"""


class _AnimationSettings(Flow360BaseModel):
    """
    Controls how frequently the output files are generated.
    """

    frequency: int = pd.Field(
        default=-1,
        ge=-1,
        description="Frequency (in number of physical time steps) at which output is saved. -1 is at end of simulation.",
    )
    frequency_offset: int = pd.Field(
        default=0,
        ge=0,
        description="Offset (in number of physical time steps) at which output animation is started. 0 is at beginning of simulation.",
    )


class _AnimationAndFileFormatSettings(_AnimationSettings):
    """
    Controls how frequently the output files are generated and the file format.
    """

    output_format: Literal["paraview", "tecplot", "both"] = pd.Field(default="paraview")


class SurfaceOutput(_AnimationAndFileFormatSettings):
    entities: EntityList[Surface] = pd.Field(alias="surfaces")
    write_single_file: bool = pd.Field(
        default=False,
        description="Enable writing all surface outputs into a single file instead of one file per surface. This option currently only supports Tecplot output format. Will choose the value of the last instance of this option of the same output type (SurfaceOutput or TimeAverageSurfaceOutput) in the `output` list.",
    )
    output_fields: UniqueAliasedStringList[SurfaceFields] = pd.Field()


class TimeAverageSurfaceOutput(SurfaceOutput):
    """
    Caveats:
    Solver side only accept exactly the same set of output_fields (is shared) between VolumeOutput and TimeAverageVolumeOutput.

    Notes
    -----
        Old `computeTimeAverages` can be infered when user is explicitly using for example `TimeAverageSurfaceOutput`.
    """

    start_step: Union[pd.NonNegativeInt, Literal[-1]] = pd.Field(
        default=-1, description="Physical time step to start calculating averaging"
    )


class VolumeOutput(_AnimationAndFileFormatSettings):
    output_fields: UniqueAliasedStringList[VolumeFields] = pd.Field()


class TimeAverageVolumeOutput(VolumeOutput):
    """
    Caveats:
    Solver side only accept exactly the same set of output_fields (is shared) between VolumeOutput and TimeAverageVolumeOutput.
    Also let's not worry about allowing entities here as it is not supported by solver anyway.

    Notes
    -----
        Old `computeTimeAverages` can be infered when user is explicitly using for example `TimeAverageSurfaceOutput`.
    """

    start_step: Union[pd.NonNegativeInt, Literal[-1]] = pd.Field(
        default=-1, description="Physical time step to start calculating averaging"
    )


class SliceOutput(_AnimationAndFileFormatSettings):
    entities: UniqueItemList[Slice] = pd.Field(alias="slices")
    output_fields: UniqueAliasedStringList[SliceFields] = pd.Field()


class IsosurfaceOutput(_AnimationAndFileFormatSettings):
    entities: UniqueItemList[Isosurface] = pd.Field(alias="isosurfaces")
    output_fields: UniqueAliasedStringList[CommonFields] = pd.Field()


class SurfaceIntegralOutput(_AnimationSettings):
    entities: UniqueItemList[SurfaceList] = pd.Field(alias="monitors")
    output_fields: UniqueAliasedStringList[CommonFields] = pd.Field()


class ProbeOutput(_AnimationSettings):
    entities: UniqueItemList[Probe] = pd.Field(alias="probes")
    output_fields: UniqueAliasedStringList[CommonFields] = pd.Field()


class AeroAcousticOutput(Flow360BaseModel):
    patch_type: str = pd.Field("solid", frozen=True)
    observers: List[LengthType.Point] = pd.Field()
    write_per_surface_output: bool = pd.Field(False)


class UserDefinedFields(Flow360BaseModel):
    """Ignore this for now"""

    pass


OutputTypes = Union[
    SurfaceOutput,
    TimeAverageSurfaceOutput,
    VolumeOutput,
    TimeAverageVolumeOutput,
    SliceOutput,
    IsosurfaceOutput,
    SurfaceIntegralOutput,
    ProbeOutput,
    AeroAcousticOutput,
]
