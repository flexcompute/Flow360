from typing import Union
from flow360.component.volume_mesh import VolumeMesh, VolumeMeshDraft
from flow360.exceptions import Flow360ConfigError
import pydantic as pd


class VolumeMeshHandler:
    _managed_volume_mesher: Union[VolumeMesh, VolumeMeshDraft] = pd.Field()

    def construct(cls, input: Union[VolumeMesh, VolumeMeshDraft]):
        if isinstance(input, VolumeMesh):
            return input
        elif isinstance(input, VolumeMeshDraft):
            return input
        else:
            raise Flow360ConfigError(
                f"Wrong data passed to VolumeMesh. Got {input.__class__.__name__}."
            )
