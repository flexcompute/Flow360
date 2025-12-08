import abc
from enum import Enum
from typing import Dict, List, Optional, Union

import pydantic as pd
import colorsys

import flow360.component.simulation.units as u
from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.unit_system import AngleType, LengthType
from flow360.component.types import Color, Vector


class StaticCamera(Flow360BaseModel):
    position: LengthType.Point = pd.Field(description="Position of the camera in the scene")
    target: LengthType.Point = pd.Field(description="Target point of the camera")
    up: Optional[Vector] = pd.Field(default=(0, 0, 1), description="Up vector, if not specified assume Z+")


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


class View(Enum):
    FRONT=(-1, 0, 0)
    BACK=(1, 0, 0)
    RIGHT=(0, -1, 0)
    LEFT=(0, 1, 0)
    TOP=(0, 0, 1)
    BOTTOM=(0, 0, -1)

    def __getitem__(self, idx):
        return self.value[idx]

    def __add__(self, other):
        if isinstance(other, View):
            a = self.value
            b = other.value
            return tuple(x + y for x, y in zip(a, b))
        return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)


class RenderCameraConfig(Flow360BaseModel):
    view: AllCameraTypes = pd.Field()
    projection: Union[OrthographicProjection, PerspectiveProjection] = pd.Field()

    @classmethod
    def orthographic(cls, x=0, y=0, z=0, scale=1, view=None):
        if view is None:
            view = View.FRONT + View.RIGHT + View.TOP
        
        up = (0, 0, 1)

        if view == View.TOP or view == View.BOTTOM:
            up = (0, 1, 0)

        return RenderCameraConfig(
            view=StaticCamera(
                position=(x + view[0] * scale, y + view[1] * scale, z + view[2] * scale) * u.m,
                target=(x, y, z),
                up=up
            ),
            projection=OrthographicProjection(
                width=scale * u.m,
                near=0.01 * u.m,
                far=50 * scale * u.m
            )
        )
    
    @classmethod
    def perspective(cls, x=0, y=0, z=0, scale=1, view=None):
        if view is None:
            view = View.FRONT + View.RIGHT + View.TOP

        up = (0, 0, 1)

        if view == View.TOP or view == View.BOTTOM:
            up = (0, 1, 0)

        return RenderCameraConfig(
            view=StaticCamera(
                position=(x + view[0] * scale, y + view[1] * scale, z + view[2] * scale) * u.m,
                target=(x, y, z) * u.m,
                up=up
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
                intensity=0.4,
                color=(255, 255, 255)
            ),
            directional=DirectionalLight(
                intensity=1.0,
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
    def rainbow(cls, field, min=0, max=1, alpha=1):
        def _rainbow_rgb(t):
            h = (((((1 - t) * 2) / 3) % 1) + 1) % 1
            r, g, b = colorsys.hsv_to_rgb(h, 1.0, 1.0)
            return (int(round(r*255)), int(round(g*255)), int(round(b*255)))

        colormap = []
        for i in range(20):
            t = i / (20 - 1)
            colormap.append(ColorKey(color=_rainbow_rgb(t), value=t))

        # Approximated from TS rainbowGradient sampling
        return FieldMaterial(
            alpha=alpha,
            output_field=field,
            min=min,
            max=max,
            colormap=colormap
        )

    @classmethod
    def orizon(cls, field, min=0, max=1, alpha=1):
        def _orizon_rgb(t):
            h = 0.7 * t + 0.025
            r, g, b = colorsys.hsv_to_rgb(h % 1.0, 0.9, 1.0)
            return (int(round(r*255)), int(round(g*255)), int(round(b*255)))

        colormap = []
        for i in range(20):
            t = i / (20 - 1)
            colormap.append(ColorKey(color=_orizon_rgb(t), value=t))

        # Approximated from TS orizonGradient sampling
        return FieldMaterial(
            alpha=alpha,
            output_field=field,
            min=min,
            max=max,
            colormap=colormap
        )

    @classmethod
    def viridis(cls, field, min=0, max=1, alpha=1):
        return FieldMaterial(
            alpha=alpha,
            output_field=field,
            min=min,
            max=max,
            colormap=[
                ColorKey(color=(68, 1, 84), value=0.0),
                ColorKey(color=(65, 68, 135), value=0.2),
                ColorKey(color=(42, 120, 142), value=0.4),
                ColorKey(color=(34, 168, 132), value=0.6),
                ColorKey(color=(122, 209, 81), value=0.8),
                ColorKey(color=(253, 231, 37), value=1.0)
            ]
        )

    @classmethod
    def magma(cls, field, min=0, max=1, alpha=1):
        return FieldMaterial(
            alpha=alpha,
            output_field=field,
            min=min,
            max=max,
            colormap=[
                ColorKey(color=(  0,   0,   4), value=0.0),
                ColorKey(color=( 86,  20, 125), value=0.25),
                ColorKey(color=(192,  58, 118), value=0.5),
                ColorKey(color=(253, 154, 106), value=0.75),
                ColorKey(color=(252, 253, 191), value=1.0)
            ]
        )

    @classmethod
    def airflow(cls, field, min=0, max=1, alpha=1):
        return FieldMaterial(
            alpha=alpha,
            output_field=field,
            min=min,
            max=max,
            colormap=[
                ColorKey(color=(  0, 100,  60), value=0.0),
                ColorKey(color=( 97, 178, 156), value=0.14),
                ColorKey(color=(123, 189, 240), value=0.28),
                ColorKey(color=(241, 241, 240), value=0.42),
                ColorKey(color=(254, 216, 139), value=0.57),
                ColorKey(color=(247, 139, 141), value=0.71),
                ColorKey(color=(252, 122,  76), value=0.85),
                ColorKey(color=(176,  90, 249), value=1.0)
            ]
        )


AnyMaterial = Union[PBRMaterial, FieldMaterial]
    

class RenderSceneTransform(Flow360BaseModel):
    translation: LengthType.Point = pd.Field(default=[0, 0, 0])
    rotation: AngleType.Vector = pd.Field(default=[0, 0, 0])
    scale: Vector = pd.Field(default=[1, 1, 1])
