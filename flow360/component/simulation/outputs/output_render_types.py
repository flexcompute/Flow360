import abc
from enum import Enum
from typing import Dict, List, Optional, Union

import pydantic as pd

import flow360.component.simulation.units as u
from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.unit_system import AngleType, LengthType
from flow360.component.types import Color, Vector



class StaticCamera(Flow360BaseModel):
    position: LengthType.Point = pd.Field(description="Position of the camera in the scene")
    target: LengthType.Point = pd.Field(description="Target point of the camera")
    up: Optional[Vector] = pd.Field(default=(0, 1, 0), description="Up vector, if not specified assume Y+")


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

    @classmethod
    def orthographic(cls, x=1, y=1, z=1, scale=1):
        return RenderCameraConfig(
            view=StaticCamera(
                position=(x * scale, y * scale, z * scale) * u.m,
                target=(0, 0, 0) * u.m
            ),
            projection=OrthographicProjection(
                width=scale * u.m,
                near=0.01 * u.m,
                far=50 * scale * u.m
            )
        )
    
    @classmethod
    def perspective(cls, x=1, y=1, z=1, scale=1):
        return RenderCameraConfig(
            view=StaticCamera(
                position=(x * scale, y * scale, z * scale) * u.m,
                target=(0, 0, 0) * u.m
            ),
            projection=PerspectiveProjection(
                fov=60 * u.deg,
                near=0.01 * u.m,
                far=50 * scale * u.m
            )
        )


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

    @classmethod
    def default(cls):
        return RenderLightingConfig(
            ambient=AmbientLight(
                intensity=0.5,
                color=(255, 255, 255)
            ),
            directional=DirectionalLight(
                intensity=1.5,
                color=(255, 255, 255),
                direction=(-1.0, -1.0, -1.0)
            )
        )


class RenderBackgroundBase(Flow360BaseModel, metaclass=abc.ABCMeta):
    type: str = pd.Field(default="", frozen=True)
        

class SolidBackground(RenderBackgroundBase):
    type: str = pd.Field(default="solid", frozen=True)
    color: Color = pd.Field()


class SkyboxTexture(str, Enum):
    SKY = "sky"
    GRADIENT = "gradient"


class SkyboxBackground(RenderBackgroundBase):
    type: str = pd.Field(default="skybox", frozen=True)
    texture: SkyboxTexture = pd.Field(SkyboxTexture.SKY)


AllBackgroundTypes = Union[SolidBackground, SkyboxBackground]


class RenderEnvironmentConfig(Flow360BaseModel):
    background: AllBackgroundTypes = pd.Field()

    @classmethod
    def simple(cls):
        return RenderEnvironmentConfig(
            background=SolidBackground(
                color=(207, 226, 230)
            )
        )

    @classmethod
    def sky(cls):
        return RenderEnvironmentConfig(
            background=SkyboxBackground(
                texture=SkyboxTexture.SKY
            )
        )

    @classmethod
    def gradient(cls):
        return RenderEnvironmentConfig(
            background=SkyboxBackground(
                texture=SkyboxTexture.GRADIENT
            )
        )


class RenderMaterialBase(Flow360BaseModel, metaclass=abc.ABCMeta):
    type: str = pd.Field(default="", frozen=True)


class PBRMaterial(RenderMaterialBase):
    color: Color = pd.Field(default=[255, 255, 255])
    alpha: float = pd.Field(default=1)
    roughness: float = pd.Field(default=0.5)
    f0: Vector = pd.Field(default=(0.03, 0.03, 0.03))
    type: str = pd.Field(default="pbr", frozen=True)

    @classmethod
    def metal(cls, shine=0.5, alpha=1.0):
        return PBRMaterial(
            color=(255, 255, 255),
            alpha=alpha,
            roughness=1 - shine,
            f0=(0.56, 0.56, 0.56)
        )

    @classmethod
    def plastic(cls, shine=0.5, alpha=1.0):
        return PBRMaterial(
            color=(255, 255, 255),
            alpha=alpha,
            roughness=1 - shine,
            f0=(0.03, 0.03, 0.03)
        )


class ColorKey(Flow360BaseModel):
    color: Color = pd.Field(default=[255, 255, 255])
    value: pd.confloat(ge=0, le=1) = pd.Field(default=0.5)


class FieldMaterial(RenderMaterialBase):
    alpha: float = pd.Field(default=1)
    output_field: str = pd.Field(default="")
    min: float = pd.Field(default=0)
    max: float = pd.Field(default=1)
    colormap: List[ColorKey] = pd.Field()
    type: str = pd.Field(default="field", frozen=True)

    @classmethod
    def greyscale(cls, field, min=0, max=1, alpha=1):
        return FieldMaterial(
            alpha=alpha,
            output_field=field,
            min=min,
            max=max,
            colormap = [
                ColorKey(color=(0, 0, 0), value=0),
                ColorKey(color=(255, 255, 255), value=1.0)
            ]   
        )

    @classmethod
    def hot_cold(cls, field, min=0, max=1, alpha=1):
        return FieldMaterial(
            alpha=alpha,
            output_field=field,
            min=min,
            max=max,
            colormap = [
                ColorKey(color=(0, 0, 255), value=0),
                ColorKey(color=(255, 255, 255), value=0.5),
                ColorKey(color=(255, 0, 0), value=1.0)
            ]   
        )
    
    @classmethod
    def rainbow(cls, field, min=0, max=1, alpha=1):
        return FieldMaterial(
            alpha=alpha,
            output_field=field,
            min=min,
            max=max,
            colormap = [
                ColorKey(color=(0, 0, 255), value=0),
                ColorKey(color=(0, 255, 255), value=0.25),
                ColorKey(color=(0, 255, 0), value=0.5),
                ColorKey(color=(255, 255, 0), value=0.75),
                ColorKey(color=(255, 0, 0), value=1.0)
            ]   
        )


AllMaterialTypes = Union[PBRMaterial, FieldMaterial]
    

class Transform(Flow360BaseModel):
    translation: LengthType.Point = pd.Field(default=[0, 0, 0])
    rotation: AngleType.Vector = pd.Field(default=[0, 0, 0])
    scale: Vector = pd.Field(default=[1, 1, 1])
