"""Shared camera schema for renders and reports.

Defines a single :class:`Camera` — a viewpoint camera (eye *direction* plus framing,
in model units) used directly by both ``RenderOutput.camera`` and the report camera,
and the :class:`Viewpoint` enum of canonical view directions.

The absolute camera is resolved downstream from the scene bounding box: ``position``
is a direction on the unit sphere about ``look_at`` (world position =
``look_at + 4 * bounding_sphere_radius * normalize(position)``); ``look_at`` defaults
to the scene bounding-box center; ``dimension`` (measured along ``dimension_dir``)
sets the orthographic frustum extent, or fits the scene when ``None``.

Perspective projection is a future addition; the viewpoint currently resolves to an
orthographic camera.
"""

from enum import Enum
from typing import Literal

import pydantic as pd

from flow360_schema.framework.base_model import Flow360BaseModel


class Camera(Flow360BaseModel):
    """
    A viewpoint camera shared by :class:`RenderOutput` and the report — an eye
    *direction* plus framing, expressed in plain floats measured in model
    (geometry / mesh) units.

    Example
    -------
    >>> Camera(position=(-1, -1, 1), dimension=2.5)
    >>> Camera(position=Viewpoint.TOP + Viewpoint.LEFT)  # frames the scene
    """

    position: tuple[float, float, float] | None = pd.Field(
        (-1, -1, 1),
        description="Eye position as a DIRECTION from look_at on the unit sphere (not an absolute "
        "position); the absolute position is resolved against the scene bounding box downstream. "
        "Model units.",
    )
    up: tuple[float, float, float] | None = pd.Field((0, 0, 1), description="Up vector, if not specified assume Z+")
    look_at: tuple[float, float, float] | None = pd.Field(
        None,
        description="Look-at target; if None, defaults to the scene bounding-box center. Model units.",
    )
    pan_target: tuple[float, float, float] | None = pd.Field(
        None,
        description="Viewport pan-center; if None, defaults to look_at. On a render this recenters the "
        "frustum (off-center) so pan_target lands at the viewport center. Model units.",
    )
    dimension_dir: Literal["width", "height", "diagonal"] | None = pd.Field(
        "diagonal",
        alias="dimensionDirection",
        description="Which viewport extent 'dimension' measures. Defaults to 'diagonal' to match the "
        "webUI viewer. 'height'/'diagonal' are resolved to frustum width using the render output "
        "aspect ratio; 'width' needs no aspect",
    )
    dimension: float | None = pd.Field(
        None,
        alias="dimensionSizeModelUnits",
        description="Framing size in model units measured along dimension_dir; sets the orthographic "
        "frustum extent. If None, the framing fits the scene bounding box.",
    )
    type: Literal["Camera"] = pd.Field("Camera", frozen=True)

    @pd.model_validator(mode="before")
    @classmethod
    def _accept_type_name_alias(cls, data):
        """Accept ``type_name`` as an input alias for the ``type`` discriminator.

        RenderOutput/simulation.json (and the schema-wide convention, incl. LegacyCamera) key
        this discriminator as ``type_name``, while the report/UVF-JSON camera hierarchy uses
        ``type``. A field-level alias can't be used because ``type`` is the discriminator of the
        report ``Chart3D.camera`` union and pydantic forbids aliased discriminators, so normalize
        here instead. Serialization stays ``type`` (report JSON unchanged).
        """
        if isinstance(data, dict) and "type_name" in data:
            data = {**data}
            data.setdefault("type", data.pop("type_name"))
        return data


class Viewpoint(Enum):
    """
    :class:`Viewpoint` provides predefined canonical view directions, usable as a
    :class:`Camera` ``position``.

    Example
    -------
    >>> Viewpoint.FRONT.value
    (-1, 0, 0)

    >>> Viewpoint.FRONT + Viewpoint.TOP
    (-1, 0, 1)
    """

    FRONT = (-1, 0, 0)
    BACK = (1, 0, 0)
    RIGHT = (0, -1, 0)
    LEFT = (0, 1, 0)
    TOP = (0, 0, 1)
    BOTTOM = (0, 0, -1)

    def __getitem__(self, idx):
        return self.value[idx]

    def __add__(self, other):
        if isinstance(other, Viewpoint):
            b = other.value
        elif isinstance(other, tuple):
            b = other
        else:
            return NotImplemented

        a = self.value
        return tuple(x + y for x, y in zip(a, b, strict=False))

    def __radd__(self, other):
        if isinstance(other, tuple):
            a = other
        elif isinstance(other, Viewpoint):
            a = other.value
        else:
            return NotImplemented

        b = self.value
        return tuple(x + y for x, y in zip(a, b, strict=False))
