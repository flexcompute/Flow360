"""
Legacy field definitions and updaters (fields that are not
specified in the documentation but can be used internally
during validation, most legacy classes can be updated to
the current standard via the update_model method)
"""

from abc import ABCMeta, abstractmethod
from typing import Dict, Literal, Optional

import pydantic.v1 as pd

from flow360.component.v1.params_base import DeprecatedAlias, Flow360BaseModel
from flow360.component.v1.unit_system import DimensionedType


class LegacyModel(Flow360BaseModel, metaclass=ABCMeta):
    """:class: `LegacyModel` is used by legacy classes to"""

    comments: Optional[Dict] = pd.Field()

    @abstractmethod
    def update_model(self):
        """Update the legacy model to the up-to-date version"""


class LinearSolverLegacy(LegacyModel):
    """:class:`LinearSolverLegacy` class"""

    max_level_limit: Optional[pd.NonNegativeInt] = pd.Field(alias="maxLevelLimit")
    # pylint: disable=duplicate-code
    max_iterations: Optional[pd.PositiveInt] = pd.Field(alias="maxIterations", default=50)
    absolute_tolerance: Optional[pd.PositiveFloat] = pd.Field(alias="absoluteTolerance")
    relative_tolerance: Optional[pd.PositiveFloat] = pd.Field(alias="relativeTolerance")

    # pylint: disable=missing-class-docstring,too-few-public-methods
    class Config(Flow360BaseModel.Config):
        deprecated_aliases = [DeprecatedAlias(name="absolute_tolerance", deprecated="tolerance")]

    def update_model(self):
        model = {
            "absoluteTolerance": self.absolute_tolerance,
            "relativeTolerance": self.relative_tolerance,
            "maxIterations": self.max_iterations,
        }

        return model


class FreestreamInitialConditionLegacy(LegacyModel):
    """:class:`FreestreamInitialConditionLegacy` class"""

    type: Literal["freestream"] = pd.Field("freestream", const=True)

    def update_model(self):
        return None


def try_add_unit(model, key, unit: DimensionedType):
    """Add unit to an existing updater field"""
    if model[key] is not None:
        model[key] *= unit


def try_set(model, key, value):
    """Set existing updater field if it exists in the legacy model"""
    if value is not None and model.get(key) is None:
        model[key] = value


def try_add_discriminator(model, path, discriminators, parsed_cls):
    """Try finding the first valid discriminator for the object, throw if not found"""
    path = path.split("/")

    target = model

    for item in path[:-1]:
        target = model[item]

    key = path[-1]

    for discriminator in discriminators:
        target[key] = discriminator
        try:
            parsed_cls.parse_obj(model)
            return model
        except pd.ValidationError:
            pass

    raise ValueError(f"Cannot infer discriminator for {model}, tried: {discriminators}")


def try_update(field: Optional[LegacyModel]):
    """Try running updater on the field if it exists"""
    if field is not None:
        if isinstance(field, LegacyModel):
            return field.update_model()
        if isinstance(field, list):
            return [try_update(item) for item in field]
    return None


def get_output_fields(instance: Flow360BaseModel, exclude, allowed=None):
    """Retrieve all output fields of a legacy output instance"""
    fields = []
    for key, value in instance.__fields__.items():
        if value.type_ == bool and value.alias not in exclude and getattr(instance, key) is True:
            if allowed is not None and value.alias in allowed:
                fields.append(value.alias)
    return fields


def set_linear_solver_config_if_none(linear_sovler_config, values):
    """Use to 'linear_solver' if linear_solver_config is not present and default if both are none"""
    if linear_sovler_config is None:
        if values.get("linear_solver") is not None:
            linear_sovler_config = values.get("linear_solver").dict()
        else:
            linear_sovler_config = LinearSolverLegacy().dict()
    return linear_sovler_config
