import abc
from enum import Enum
from typing import Dict, List, Optional, Union

import pydantic as pd

from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.unit_system import AngleType, LengthType
from flow360.component.types import Color, Vector


class StaticCamera(Flow360BaseModel):
    position: LengthType.Point = pd.Field(description="Position of the camera in the scene")
    target: LengthType.Point = pd.Field(description="Target point of the camera")
    up: Optional[Vector] = pd.Field(
        default=(0, 1, 0), description="Up vector, if not specified assume Y+"
    )


class Keyframe(Flow360BaseModel):
    time: pd.confloat(ge=0) = pd.Field(0)
    view: StaticCamera = pd.Field()


class AnimatedCamera(Flow360BaseModel):
    keyframes: List[Keyframe] = pd.Field([])


AllCameraTypes = Union[StaticCamera, AnimatedCamera]


class OrthographicProjection(Flow360BaseModel):
    type: str = pd.Field(default="orthographic", frozen=True)
    width: LengthType = pd.Field()
    near: LengthType = pd.Field()
    far: LengthType = pd.Field()


class PerspectiveProjection(Flow360BaseModel):
    type: str = pd.Field(default="perspective", frozen=True)
    fov: AngleType = pd.Field()
    near: LengthType = pd.Field()
    far: LengthType = pd.Field()


class RenderCameraConfig(Flow360BaseModel):
    view: AllCameraTypes = pd.Field()
    projection: Union[OrthographicProjection, PerspectiveProjection] = pd.Field()


class AmbientLight(Flow360BaseModel):
    intensity: float = pd.Field()
    color: Color = pd.Field()


class DirectionalLight(Flow360BaseModel):
    intensity: float = pd.Field()
    color: Color = pd.Field()
    direction: Vector = pd.Field()


class RenderLightingConfig(Flow360BaseModel):
    directional: DirectionalLight = pd.Field()
    ambient: Optional[AmbientLight] = pd.Field(None)


class RenderBackgroundBase(Flow360BaseModel, metaclass=abc.ABCMeta):
    type: str = pd.Field(default="", frozen=True)


class SolidBackground(RenderBackgroundBase):
    type: str = pd.Field(default="solid", frozen=True)
    color: Color = pd.Field()


class SkyboxTexture(str, Enum):
    SKY = "sky"


class SkyboxBackground(RenderBackgroundBase):
    type: str = pd.Field(default="skybox", frozen=True)
    texture: SkyboxTexture = pd.Field(SkyboxTexture.SKY)


AllBackgroundTypes = Union[SolidBackground, SkyboxBackground]


class RenderEnvironmentConfig(Flow360BaseModel):
    background: AllBackgroundTypes = pd.Field()


class RenderMaterialBase(Flow360BaseModel, metaclass=abc.ABCMeta):
    type: str = pd.Field(default="", frozen=True)


class PBRMaterial(RenderMaterialBase):
    color: Color = pd.Field(default=[255, 255, 255])
    alpha: float = pd.Field(default=1)
    roughness: float = pd.Field(default=0.5)
    f0: Vector = pd.Field(default=(0.03, 0.03, 0.03))
    type: str = pd.Field(default="pbr", frozen=True)


class ColorKey(Flow360BaseModel):
    color: Color = pd.Field(default=[255, 255, 255])
    value: pd.confloat(ge=0, le=1) = pd.Field(default=0.5)


class FieldMaterial(RenderMaterialBase):
    alpha: float = pd.Field(default=1)
    output_field: str = pd.Field(default="")
    min: float = pd.Field(default=0)
    max: float = pd.Field(default=0)
    colormap: List[ColorKey] = pd.Field()
    type: str = pd.Field(default="field", frozen=True)


AllMaterialTypes = Union[PBRMaterial, FieldMaterial]


class RenderMaterialConfig(Flow360BaseModel):
    materials: List[AllMaterialTypes] = pd.Field([])
    mappings: Dict[str, int] = pd.Field({})


class Transform(Flow360BaseModel):
    translation: LengthType.Point = pd.Field(default=[0, 0, 0])
    rotation: AngleType.Vector = pd.Field(default=[0, 0, 0])
    scale: Vector = pd.Field(default=[1, 1, 1])
