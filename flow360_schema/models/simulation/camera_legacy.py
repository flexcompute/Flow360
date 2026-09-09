"""Deprecated render-camera schema (pre-viewpoint).

This module preserves the old ``RenderOutput`` camera model — an explicit
``view`` (``StaticView`` / ``AnimatedView``) plus a ``projection``
(orthographic / perspective) — so render configs authored before the viewpoint
``Camera`` (see :mod:`flow360_schema.models.simulation.camera`) still load and
translate unchanged. It is retained for one release for backwards compatibility
and will be removed; new code should use :class:`~flow360_schema.models.simulation.camera.Camera`.
"""

import warnings
from typing import Literal

import pydantic as pd
import unyt as u

from flow360_schema.framework.base_model import Flow360BaseModel
from flow360_schema.framework.entity.geometric_types import Vector
from flow360_schema.framework.physical_dimensions import Angle, Length, Time
from flow360_schema.models.simulation.camera import Viewpoint

__all__ = [
    "StaticView",
    "Keyframe",
    "AnimatedView",
    "OrthographicProjection",
    "PerspectiveProjection",
    "LegacyCamera",
]


class StaticView(Flow360BaseModel):
    """A fixed camera pose (position, target, up)."""

    type_name: Literal["StaticView"] = pd.Field("StaticView", frozen=True)
    position: Length.Vector3 = pd.Field(description="Position of the camera in the scene")
    target: Length.Vector3 = pd.Field(description="Target point of the camera")
    up: Vector | None = pd.Field(default=(0, 0, 1), description="Up vector, if not specified assume Z+")


class Keyframe(Flow360BaseModel):
    """A timestamped camera pose for animated rendering."""

    type_name: Literal["Keyframe"] = pd.Field("Keyframe", frozen=True)
    time: Time.Float64 = pd.Field(0, ge=0, description="Timestamp at which the keyframe should be reached")
    view: StaticView = pd.Field(description="Camera parameters at this keyframe")


class AnimatedView(Flow360BaseModel):
    """A sequence of camera keyframes to create motion."""

    type_name: Literal["AnimatedView"] = pd.Field("AnimatedView", frozen=True)
    keyframes: list[Keyframe] = pd.Field(
        [], description="List of keyframes between which the animated camera interpolates"
    )

    @pd.field_validator("keyframes", mode="after")
    @classmethod
    def check_has_keyframes_and_sort(cls, value):
        """First keyframe must be at time 0; keyframes are returned sorted by time."""
        if len(value) < 1:
            raise ValueError("Animated camera requires at least one keyframe to be defined")
        value = sorted(value, key=lambda v: v.time)
        if value[0].time != 0:
            raise ValueError("The first keyframe needs to be defined at time = 0 (the starting camera position)")
        return value


class OrthographicProjection(Flow360BaseModel):
    """Orthographic projection (frustum width + near/far)."""

    type_name: Literal["OrthographicProjection"] = pd.Field("OrthographicProjection", frozen=True)
    width: Length.Float64 = pd.Field(description="Width of the camera frustum in world units")
    near: Length.Float64 = pd.Field(description="Near clipping plane in world units")
    far: Length.Float64 = pd.Field(description="Far clipping plane in world units")


class PerspectiveProjection(Flow360BaseModel):
    """Perspective projection (fov + near/far)."""

    type_name: Literal["PerspectiveProjection"] = pd.Field("PerspectiveProjection", frozen=True)
    fov: Angle.Float64 = pd.Field(description="Field of view of the camera (angle)")
    near: Length.Float64 = pd.Field(description="Near clipping plane in world units")
    far: Length.Float64 = pd.Field(description="Far clipping plane in world units")


class LegacyCamera(Flow360BaseModel):
    """
    Deprecated render camera (explicit view + projection).

    Retained for one release so pre-viewpoint render configs load and translate
    unchanged. Use :class:`~flow360_schema.models.simulation.camera.Camera` instead.
    """

    type_name: Literal["Camera"] = pd.Field("Camera", frozen=True)
    view: StaticView | AnimatedView = pd.Field(
        discriminator="type_name", description="View settings (position, target)"
    )
    projection: OrthographicProjection | PerspectiveProjection = pd.Field(
        discriminator="type_name",
        description="Projection settings (FOV / width, near/far clipping planes)",
    )

    @pd.model_validator(mode="after")
    def _warn_deprecated(self):
        warnings.warn(
            "The explicit view/projection render camera is deprecated and will be removed next "
            "release; use the viewpoint Camera (flow360_schema.models.simulation.camera.Camera).",
            DeprecationWarning,
            stacklevel=2,
        )
        return self

    @classmethod
    def orthographic(cls, position=(0, 0, 0), scale=1, view=None):
        """Create an orthographic (deprecated) camera from a canonical view direction."""
        if view is None:
            view = Viewpoint.FRONT + Viewpoint.RIGHT + Viewpoint.TOP
        up = (0, 1, 0) if view in (Viewpoint.TOP, Viewpoint.BOTTOM) else (0, 0, 1)
        x, y, z = position
        return cls(
            view=StaticView(
                position=(x + view[0] * scale, y + view[1] * scale, z + view[2] * scale) * u.m,
                target=(x, y, z),
                up=up,
            ),
            projection=OrthographicProjection(width=scale * u.m, near=0.01 * u.m, far=50 * scale * u.m),
        )

    @classmethod
    def perspective(cls, position=(0, 0, 0), scale=1, view=None):
        """Create a perspective (deprecated) camera from a canonical view direction."""
        if view is None:
            view = Viewpoint.FRONT + Viewpoint.RIGHT + Viewpoint.TOP
        up = (0, 1, 0) if view in (Viewpoint.TOP, Viewpoint.BOTTOM) else (0, 0, 1)
        x, y, z = position
        return cls(
            view=StaticView(
                position=(x + view[0] * scale, y + view[1] * scale, z + view[2] * scale) * u.m,
                target=(x, y, z) * u.m,
                up=up,
            ),
            projection=PerspectiveProjection(fov=60 * u.deg, near=0.01 * u.m, far=50 * scale * u.m),
        )
