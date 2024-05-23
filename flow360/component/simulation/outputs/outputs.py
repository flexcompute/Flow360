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
    UniqueAliasedItemList,
    UniqueItemList,
)
from flow360.component.simulation.outputs.output_entities import (
    Isosurface,
    Probe,
    Slice,
    SurfaceList,
)
from flow360.component.simulation.primitives import Surface

"""Mostly the same as Flow360Param counterparts.
Caveats:
1. Check if we support non-average and average output specified at the same time in solver. (Yes but they share the same output_fields)
2. We do not support mulitple output frequencies for the same type of output.
"""


class _AnimationSettings(Flow360BaseModel):
    """
    Controls how frequently the output files are generated.
    """

    frequency: Optional[int] = pd.Field(
        default=-1,
        ge=-1,
        description="Frequency (in number of physical time steps) at which output is saved. -1 is at end of simulation.",
    )
    frequency_offset: Optional[int] = pd.Field(
        default=0,
        description="Offset (in number of physical time steps) at which output animation is started. 0 is at beginning of simulation.",
    )


class _TimeAverageAdditionalAnimationSettings(Flow360BaseModel):
    """
    Additional controls when using time-averaged output.

    Notes
    -----
        Old `computeTimeAverages` can be infered when user is explicitly using for example `TimeAverageSurfaceOutput`.
    """

    start_step: Optional[Union[pd.NonNegativeInt, Literal[-1]]] = pd.Field(
        default=-1, description="Physical time step to start calculating averaging"
    )


class SurfaceOutput(_AnimationSettings):
    entities: EntityList[Surface] = pd.Field(alias="surfaces")
    write_single_file: Optional[bool] = pd.Field(
        default=False,
        description="Enable writing all surface outputs into a single file instead of one file per surface. This option currently only supports Tecplot output format. Will choose the value of the last instance of this option of the same output type (SurfaceOutput or TimeAverageSurfaceOutput) in the `output` list.",
    )
    output_fields: UniqueAliasedItemList[SurfaceFields] = pd.Field()


class TimeAverageSurfaceOutput(SurfaceOutput, _TimeAverageAdditionalAnimationSettings):
    """
    Caveats:
    Solver side only accept exactly the same set of output_fields (is shared) between VolumeOutput and TimeAverageVolumeOutput.
    """

    pass


class VolumeOutput(_AnimationSettings):
    output_fields: UniqueAliasedItemList[VolumeFields] = pd.Field()


class TimeAverageVolumeOutput(VolumeOutput, _TimeAverageAdditionalAnimationSettings):
    """
    Caveats:
    Solver side only accept exactly the same set of output_fields (is shared) between VolumeOutput and TimeAverageVolumeOutput.
    Also let's not worry about allowing entities here as it is not supported by solver anyway.
    """

    pass


class SliceOutput(_AnimationSettings):
    slices: UniqueItemList[Slice] = pd.Field()
    output_fields: UniqueAliasedItemList[SliceFields] = pd.Field()


class IsosurfaceOutput(_AnimationSettings):
    isosurfaces: UniqueItemList[Isosurface] = pd.Field()
    output_fields: UniqueAliasedItemList[CommonFields] = pd.Field()


class SurfaceIntegralOutput(_AnimationSettings):
    monitors: UniqueItemList[SurfaceList] = pd.Field()
    output_fields: UniqueAliasedItemList[CommonFields] = pd.Field()


class ProbeOutput(_AnimationSettings):
    probes: UniqueItemList[Probe] = pd.Field()
    output_fields: UniqueAliasedItemList[CommonFields] = pd.Field()


class AeroAcousticOutput(Flow360BaseModel):
    patch_type: Optional[str] = pd.Field("solid", frozen=True)
    observers: List[Tuple[float, float, float]] = pd.Field()
    write_per_surface_output: Optional[bool] = pd.Field(False)


class UserDefinedFields(Flow360BaseModel):
    """Ignore this for now"""

    pass


OutputTypes = Union[
    SurfaceOutput,
    VolumeOutput,
    SliceOutput,
    IsosurfaceOutput,
    SurfaceIntegralOutput,
    ProbeOutput,
    AeroAcousticOutput,
]
