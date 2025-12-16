"""Defines render types for setting up a RenderOutput object"""

import abc
import colorsys
from enum import Enum
from typing import List, Literal, Optional, Union

import pydantic as pd

import flow360.component.simulation.units as u
from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.outputs.output_fields import CommonFieldNames
from flow360.component.simulation.unit_system import AngleType, LengthType, TimeType
from flow360.component.simulation.user_code.core.types import (
    Expression,
    UnytQuantity,
    UserVariable,
    ValueOrExpression,
    get_input_value_dimensions,
    get_input_value_length,
    infer_units_by_unit_system,
    is_variable_with_unit_system_as_units,
    solver_variable_to_user_variable,
)
from flow360.component.simulation.user_code.core.utils import is_runtime_expression
from flow360.component.types import Axis, Color, Vector


class StaticView(Flow360BaseModel):
    """
    :class:`StaticView` defines a fixed camera in the scene with a position, target, and up-direction.

    Example
    -------
    Define a simple static camera positioned at (1, 1, 1) looking at the origin:

    >>> cam = StaticView(
    ...     position=(1, 1, 1),
    ...     target=(0, 0, 0),
    ...     up=(0, 0, 1),
    ... )
    """

    type_name: Literal["StaticView"] = pd.Field("StaticView", frozen=True)
    # pylint: disable=no-member
    position: LengthType.Point = pd.Field(description="Position of the camera in the scene")
    # pylint: disable=no-member
    target: LengthType.Point = pd.Field(description="Target point of the camera")
    up: Optional[Vector] = pd.Field(
        default=(0, 0, 1), description="Up vector, if not specified assume Z+"
    )


class Keyframe(Flow360BaseModel):
    """
    :class:`Keyframe` represents a timestamped camera pose for animated rendering.

    Example
    -------
    >>> Keyframe(
    ...     time=0.5,
    ...     view=StaticView(position=(2, 0, 1), target=(0, 0, 0))
    ... )
    """

    type_name: Literal["Keyframe"] = pd.Field("Keyframe", frozen=True)
    time: TimeType = pd.Field(
        0, ge=0, description="Timestamp at which the keyframe should be reached"
    )
    view: StaticView = pd.Field(description="Camera parameters at this keyframe")


class AnimatedView(Flow360BaseModel):
    """
    :class:`AnimatedView` defines a sequence of camera keyframes to create motion.

    Example
    -------
    >>> AnimatedView(
    ...     keyframes=[
    ...         Keyframe(time=0, view=StaticView(position=(2,0,0), target=(0,0,0))),
    ...         Keyframe(time=1, view=StaticView(position=(0,2,0), target=(0,0,0))),
    ...     ]
    ... )
    """

    type_name: Literal["AnimatedView"] = pd.Field("AnimatedView", frozen=True)
    keyframes: List[Keyframe] = pd.Field(
        [], description="List of keyframes between which the animated camera interpolates"
    )

    @pd.field_validator("keyframes", mode="after")
    @classmethod
    def check_has_keyframes_and_sort(cls, value):
        """Check if the view has any keyframes assigned, the
        first frame is at time 0 and the frames are sorted"""
        if len(value) < 1:
            raise ValueError("Animated camera requires at least one keyframe to be defined")

        value = sorted(value, key=lambda v: v.time)

        if value[0].time != 0:
            raise ValueError(
                "The first keyframe needs to be defined at time = 0 (the starting camera position)"
            )


class OrthographicProjection(Flow360BaseModel):
    """
    :class:`OrthographicProjection` defines an orthographic camera projection model.

    Example
    -------
    >>> OrthographicProjection(
    ...     width=1.0 * u.m,
    ...     near=0.01 * u.m,
    ...     far=10 * u.m,
    ... )
    """

    type_name: Literal["OrthographicProjection"] = pd.Field("OrthographicProjection", frozen=True)
    width: LengthType = pd.Field(description="Width of the camera frustum in world units")
    near: LengthType = pd.Field(
        description="Near clipping plane in world units, pixels closer to the camera than this value are culled"
    )
    far: LengthType = pd.Field(
        description="Far clipping plane in world units, pixels further from the camera than this value are culled"
    )


class PerspectiveProjection(Flow360BaseModel):
    """
    :class:`PerspectiveProjection` defines a perspective camera projection.

    Example
    -------
    >>> PerspectiveProjection(
    ...     fov=60 * u.deg,
    ...     near=0.01 * u.m,
    ...     far=50 * u.m,
    ... )
    """

    type_name: Literal["PerspectiveProjection"] = pd.Field("PerspectiveProjection", frozen=True)
    fov: AngleType = pd.Field(description="Field of view of the camera (angle)")
    near: LengthType = pd.Field(
        description="Near clipping plane in world units, pixels closer to the camera than this value are culled"
    )
    far: LengthType = pd.Field(
        description="Far clipping plane in world units, pixels further from the camera than this value are culled"
    )


class Viewpoint(Enum):
    """
    :class:`View` provides predefined canonical view directions.

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
        return tuple(x + y for x, y in zip(a, b))

    def __radd__(self, other):
        if isinstance(other, tuple):
            a = other
        elif isinstance(other, Viewpoint):
            a = other.value
        else:
            return NotImplemented

        b = self.value
        return tuple(x + y for x, y in zip(a, b))


class Camera(Flow360BaseModel):
    """
    :class:`Camera` configures the camera and projection used for rendering.

    Example
    -------
    >>> Camera.perspective(
    ...     x=1, y=1, z=1, scale=2, view=Viewpoint.FRONT
    ... )
    """

    type_name: Literal["Camera"] = pd.Field("Camera", frozen=True)
    view: Union[StaticView, AnimatedView] = pd.Field(
        discriminator="type_name", description="View settings (position, target)"
    )
    projection: Union[OrthographicProjection, PerspectiveProjection] = pd.Field(
        discriminator="type_name",
        description="Projection settings (FOV / width, near/far clipping planes)",
    )

    @classmethod
    def orthographic(cls, position=(0, 0, 0), scale=1, view=None):
        """
        Create an orthographic camera configuration.

        Example
        -------
        >>> Camera.orthographic(
        ...     position=(0, 0, 0), scale=1.5, view=Viewpoint.TOP
        ... )
        """
        if view is None:
            view = Viewpoint.FRONT + Viewpoint.RIGHT + Viewpoint.TOP

        up = (0, 0, 1)

        if view in (Viewpoint.TOP, Viewpoint.BOTTOM):
            up = (0, 1, 0)

        x = position[0]
        y = position[1]
        z = position[2]

        return Camera(
            view=StaticView(
                # pylint: disable=no-member
                position=(x + view[0] * scale, y + view[1] * scale, z + view[2] * scale) * u.m,
                target=(x, y, z),
                up=up,
            ),
            projection=OrthographicProjection(
                # pylint: disable=no-member
                width=scale * u.m,
                near=0.01 * u.m,
                far=50 * scale * u.m,
            ),
        )

    @classmethod
    def perspective(cls, position=(0, 0, 0), scale=1, view=None):
        """
        Create a perspective camera configuration.

        Example
        -------
        >>> Camera.perspective(
        ...     position=(0, 0, 0), scale=3, view=Viewpoint.LEFT
        ... )
        """
        if view is None:
            view = Viewpoint.FRONT + Viewpoint.RIGHT + Viewpoint.TOP

        up = (0, 0, 1)

        if view in (Viewpoint.TOP, Viewpoint.BOTTOM):
            up = (0, 1, 0)

        x = position[0]
        y = position[1]
        z = position[2]

        return Camera(
            view=StaticView(
                # pylint: disable=no-member
                position=(x + view[0] * scale, y + view[1] * scale, z + view[2] * scale) * u.m,
                # pylint: disable=no-member
                target=(x, y, z) * u.m,
                up=up,
            ),
            # pylint: disable=no-member
            projection=PerspectiveProjection(fov=60 * u.deg, near=0.01 * u.m, far=50 * scale * u.m),
        )


class AmbientLight(Flow360BaseModel):
    """
    :class:`AmbientLight` controls uniform ambient lighting in the scene.

    Example
    -------
    >>> AmbientLight(
    ...     intensity=0.4,
    ...     color=(255, 255, 255)
    ... )
    """

    type_name: Literal["AmbientLight"] = pd.Field("AmbientLight", frozen=True)
    intensity: float = pd.Field(ge=0, description="Light intensity multiplier")
    color: Color = pd.Field(description="Color of the ambient light")


class DirectionalLight(Flow360BaseModel):
    """
    :class:`DirectionalLight` defines a directional light source with intensity and color.

    Example
    -------
    >>> DirectionalLight(
    ...     intensity=1.0,
    ...     color=(255, 255, 255),
    ...     direction=(-1, -1, -1),
    ... )
    """

    type_name: Literal["DirectionalLight"] = pd.Field("DirectionalLight", frozen=True)
    intensity: float = pd.Field(ge=0, description="Light intensity multiplier")
    color: Color = pd.Field(description="Color of the directional light beam")
    direction: Axis = pd.Field(
        description="The direction of the light beam (all beams are parallel)"
    )


class Lighting(Flow360BaseModel):
    """
    :class:`Lighting` defines ambient and directional lighting for rendering.

    Example
    -------
    >>> Lighting.default()
    """

    type_name: Literal["Lighting"] = pd.Field("Lighting", frozen=True)
    directional: DirectionalLight = pd.Field(
        description="Directional component of the light (falls from a single direction)"
    )
    ambient: Optional[AmbientLight] = pd.Field(
        description="Ambient component of the light (applied from all directions equally)"
    )

    @classmethod
    def default(cls):
        """
        Returns the default lighting configuration.

        Example
        -------
        >>> light = Lighting.default()
        """
        return Lighting(
            ambient=AmbientLight(intensity=0.4, color=(255, 255, 255)),
            directional=DirectionalLight(
                intensity=1.0, color=(255, 255, 255), direction=(-1.0, -1.0, -1.0)
            ),
        )


class BackgroundBase(Flow360BaseModel, metaclass=abc.ABCMeta):
    """
    :class:`RenderBackgroundBase` is an abstract base class for all background types.
    """

    type_name: str = pd.Field(default="", frozen=True)


class SolidBackground(BackgroundBase):
    """
    :class:`SolidBackground` defines a single-color background.

    Example
    -------
    >>> SolidBackground(color=(200, 200, 255))
    """

    type_name: Literal["SolidBackground"] = pd.Field("SolidBackground", frozen=True)
    color: Color = pd.Field(description="Flat background color")


class SkyboxTexture(str, Enum):
    """
    :class:`SkyboxTexture` specifies available skybox texture presets.

    Example
    -------
    >>> SkyboxTexture.SKY.value
    'sky'
    """

    SKY = "sky"
    GRADIENT = "gradient"


class SkyboxBackground(BackgroundBase):
    """
    :class:`SkyboxBackground` defines a skybox background using a sky or gradient texture.

    Example
    -------
    >>> SkyboxBackground(texture=SkyboxTexture.SKY)
    """

    type_name: Literal["SkyboxBackground"] = pd.Field("SkyboxBackground", frozen=True)
    texture: SkyboxTexture = pd.Field(
        SkyboxTexture.SKY, description="Cubemap texture applied to the skybox"
    )


class Environment(Flow360BaseModel):
    """
    :class:`Environment` configures the background environment for rendering.

    Example
    -------
    >>> Environment.simple()
    """

    type_name: Literal["Environment"] = pd.Field("Environment", frozen=True)
    background: Union[SolidBackground, SkyboxBackground] = pd.Field(
        discriminator="type_name", description="Background image, solid or textured"
    )

    @classmethod
    def simple(cls):
        """
        Create a render environment with a solid background.

        Example
        -------
        >>> Environment.simple()
        """
        return Environment(background=SolidBackground(color=(207, 226, 230)))

    @classmethod
    def sky(cls):
        """
        Create a render environment using a sky texture.

        Example
        -------
        >>> Environment.sky()
        """
        return Environment(background=SkyboxBackground(texture=SkyboxTexture.SKY))

    @classmethod
    def gradient(cls):
        """
        Create a render environment using a gradient skybox.

        Example
        -------
        >>> Environment.gradient()
        """
        return Environment(background=SkyboxBackground(texture=SkyboxTexture.GRADIENT))


class MaterialBase(Flow360BaseModel, metaclass=abc.ABCMeta):
    """
    :class:`MaterialBase` is an abstract base class for material definitions used during rendering.
    """

    type_name: str = pd.Field("", frozen=True)


class PBRMaterial(MaterialBase):
    """
    :class:`PBRMaterial` defines a physically based rendering (PBR) material.

    Example
    -------
    >>> PBRMaterial(color=(180, 180, 255), roughness=0.3)
    """

    type_name: Literal["PBRMaterial"] = pd.Field("PBRMaterial", frozen=True)
    color: Color = pd.Field(
        default=[255, 255, 255], description="Basic diffuse color of the material (base color)"
    )
    alpha: float = pd.Field(
        default=1,
        ge=0,
        le=1,
        description="The transparency of the material 1 is fully opaque, 0 is fully transparent",
    )
    roughness: float = pd.Field(
        default=0.5,
        ge=0,
        le=1,
        description="Material roughness, controls the fuzziness of reflections",
    )
    f0: Vector = pd.Field(
        default=(0.03, 0.03, 0.03),
        description="Fresnel reflection coeff. at 0 incidence angle, controls reflectivity",
    )

    @classmethod
    def metal(cls, shine=0.5, alpha=1.0):
        """
        Create a metallic PBR material.

        Example
        -------
        >>> PBRMaterial.metal(shine=0.8)
        """
        return PBRMaterial(
            color=(255, 255, 255), alpha=alpha, roughness=1 - shine, f0=(0.56, 0.56, 0.56)
        )

    @classmethod
    def plastic(cls, shine=0.5, alpha=1.0):
        """
        Create a plastic PBR material.

        Example
        -------
        >>> PBRMaterial.plastic(shine=0.2)
        """
        return PBRMaterial(
            color=(255, 255, 255), alpha=alpha, roughness=1 - shine, f0=(0.03, 0.03, 0.03)
        )


class FieldMaterial(MaterialBase):
    """
    :class:`FieldMaterial` maps scalar field values to colors for flow visualization.

    Example
    -------
    >>> FieldMaterial.rainbow(field="pressure", min_value=0, max_value=100000)
    """

    type_name: Literal["FieldMaterial"] = pd.Field("FieldMaterial", frozen=True)
    alpha: float = pd.Field(
        default=1,
        ge=0,
        le=1,
        description="The transparency of the material 1 is fully opaque, 0 is fully transparent",
    )
    output_field: Union[CommonFieldNames, str, UserVariable] = pd.Field(
        description="Scalar field applied to the surface via the colormap"
    )
    min: ValueOrExpression[Union[UnytQuantity, float]] = pd.Field(
        description="Reference min value (in solver units) representing the left boundary of the colormap"
    )
    max: ValueOrExpression[Union[UnytQuantity, float]] = pd.Field(
        description="Reference max value (in solver units) representing the right boundary of the colormap"
    )
    colormap: List[Color] = pd.Field(
        description="List of key colors distributed evenly across the gradient, defines value to color mappings"
    )

    @pd.field_validator("output_field", mode="before")
    @classmethod
    def _preprocess_expression_and_solver_variable(cls, value):
        if isinstance(value, Expression):
            raise ValueError(
                f"Expression ({value}) cannot be directly used as output field, "
                "please define a UserVariable first."
            )
        return solver_variable_to_user_variable(value)

    @pd.field_validator("output_field", mode="after")
    @classmethod
    def check_expression_length(cls, v):
        """Ensure the output field is a scalar."""
        if isinstance(v, UserVariable) and len(v) != 0:
            raise ValueError(f"The output field ({v}) must be defined with a scalar variable.")
        return v

    @pd.field_validator("output_field", mode="after")
    @classmethod
    def check_runtime_expression(cls, v):
        """Ensure the output field is a runtime expression but not a constant value."""
        if isinstance(v, UserVariable):
            if not isinstance(v.value, Expression):
                raise ValueError(f"The output field ({v}) cannot be a constant value.")
            try:
                result = v.value.evaluate(raise_on_non_evaluable=False, force_evaluate=True)
            except Exception as err:
                raise ValueError(
                    f"expression evaluation failed for the output field: {err}"
                ) from err
            if not is_runtime_expression(result):
                raise ValueError(f"The output field ({v}) cannot be a constant value.")
        return v

    @pd.field_validator("min", "max", mode="before")
    @classmethod
    def _preprocess_range_with_unit_system(cls, value, info: pd.ValidationInfo):
        if is_variable_with_unit_system_as_units(value):
            return value
        if info.data.get("field") is None:
            # `field` validation failed.
            raise ValueError(
                "The output field is invalid and therefore unit inference is not possible."
            )
        units = value["units"]
        field = info.data["field"]
        value_dimensions = get_input_value_dimensions(value=field)
        value = infer_units_by_unit_system(
            value=value, value_dimensions=value_dimensions, unit_system=units
        )
        return value

    @pd.field_validator("min", "max", mode="after")
    @classmethod
    def check_range_single_value(cls, v):
        """Ensure the min/max range is a single value."""
        if get_input_value_length(v) == 0:
            return v
        raise ValueError(f"The min/max range ({v}) must be a scalar.")

    @pd.field_validator("min", "max", mode="after")
    @classmethod
    def check_range_dimensions(cls, v, info: pd.ValidationInfo):
        """Ensure the min/max range has the same dimensions as the field."""
        field = info.data.get("output_field", None)
        if not isinstance(field, UserVariable):
            return v
        range_dimensions = get_input_value_dimensions(value=v)
        if range_dimensions is None:
            return v
        field_dimensions = get_input_value_dimensions(value=field)
        if field_dimensions != range_dimensions:
            raise ValueError(
                f"The min/max range ({v}, dimensions:{range_dimensions}) should have the same dimensions as "
                f"the output field ({field}, dimensions: {field_dimensions})."
            )
        return v

    @pd.field_validator("min", "max", mode="after")
    @classmethod
    def check_iso_value_for_string_field(cls, v, info: pd.ValidationInfo):
        """Ensure the iso_value is float when string field is used."""

        field = info.data.get("output_field", None)
        if isinstance(field, str) and not isinstance(v, float):
            raise ValueError(
                f"The output field ({field}) specified by string "
                "can only be used with a nondimensional min/max range."
            )
        return v

    @classmethod
    def rainbow(cls, field, min_value, max_value, alpha=1):
        """
        Create a rainbow-style colormap for scalar fields.

        Example
        -------
        >>> FieldMaterial.rainbow("velocity_magnitude")
        """

        def _rainbow_rgb(t):
            h = (((((1 - t) * 2) / 3) % 1) + 1) % 1
            r, g, b = colorsys.hsv_to_rgb(h, 1.0, 1.0)
            return (int(round(r * 255)), int(round(g * 255)), int(round(b * 255)))

        colormap = []
        for i in range(20):
            t = i / (20 - 1)
            colormap.append(_rainbow_rgb(t))

        # Approximated from TS rainbowGradient sampling
        return FieldMaterial(
            alpha=alpha, output_field=field, min=min_value, max=max_value, colormap=colormap
        )

    @classmethod
    def orizon(cls, field, min_value, max_value, alpha=1):
        """
        Create an Orizon-style (blue–orange) colormap.

        Example
        -------
        >>> FieldMaterial.orizon("temperature")
        """

        def _orizon_rgb(t):
            h = 0.7 * t + 0.025
            r, g, b = colorsys.hsv_to_rgb(h % 1.0, 0.9, 1.0)
            return (int(round(r * 255)), int(round(g * 255)), int(round(b * 255)))

        colormap = []
        for i in range(20):
            t = i / (20 - 1)
            colormap.append(_orizon_rgb(t))

        # Approximated from TS orizonGradient sampling
        return FieldMaterial(
            alpha=alpha, output_field=field, min=min_value, max=max_value, colormap=colormap
        )

    @classmethod
    def viridis(cls, field, min_value, max_value, alpha=1):
        """
        Create a Viridis colormap.

        Example
        -------
        >>> FieldMaterial.viridis("vorticity")
        """
        return FieldMaterial(
            alpha=alpha,
            output_field=field,
            min=min_value,
            max=max_value,
            colormap=[
                (68, 1, 84),
                (65, 68, 135),
                (42, 120, 142),
                (34, 168, 132),
                (122, 209, 81),
                (253, 231, 37),
            ],
        )

    @classmethod
    def magma(cls, field, min_value, max_value, alpha=1):
        """
        Create a Magma colormap.

        Example
        -------
        >>> FieldMaterial.magma("density")
        """
        return FieldMaterial(
            alpha=alpha,
            output_field=field,
            min=min_value,
            max=max_value,
            colormap=[
                (0, 0, 4),
                (86, 20, 125),
                (192, 58, 118),
                (253, 154, 106),
                (252, 253, 191),
            ],
        )

    @classmethod
    def airflow(cls, field, min_value, max_value, alpha=1):
        """
        Create an Airflow-style visualization colormap.

        Example
        -------
        >>> FieldMaterial.airflow("pressure_coefficient")
        """
        return FieldMaterial(
            alpha=alpha,
            output_field=field,
            min=min_value,
            max=max_value,
            colormap=[
                (0, 100, 60),
                (97, 178, 156),
                (123, 189, 240),
                (241, 241, 240),
                (254, 216, 139),
                (247, 139, 141),
                (252, 122, 76),
                (176, 90, 249),
            ],
        )


class SceneTransform(Flow360BaseModel):
    """
    :class:`SceneTransform` applies translation, rotation, and scaling to renderable objects.

    This may be

    Example
    -------
    >>> SceneTransform(
    ...     translation=(1, 0, 0) * u.m,
    ...     rotation=(0, 0, 90) * u.deg,
    ...     scale=(1, 2, 1),
    ... )
    """

    type_name: Literal["SceneTransform"] = pd.Field("SceneTransform", frozen=True)
    # pylint: disable=no-member
    translation: LengthType.Point = pd.Field(
        (0, 0, 0) * u.m, description="Translation applied to all scene objects"
    )
    # pylint: disable=no-member
    rotation: AngleType.Vector = pd.Field(
        (0, 0, 0) * u.deg, description="Rotation applied to all scene objects (Euler XYZ)"
    )
    scale: Vector = pd.Field((1, 1, 1), description="Scaling applied to all scene objects")
