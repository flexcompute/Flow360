"""
validation utility functions
"""

from functools import wraps
from typing import get_args

from flow360.component.simulation.entity_info import DraftEntityTypes
from flow360.component.simulation.primitives import (
    Surface,
    _SurfaceEntityBase,
    _VolumeEntityBase,
)
from flow360.component.simulation.validation.validation_context import (
    get_validation_info,
)


def _validator_append_instance_name(func):
    """
    If the validation throw ValueError (expected), append the instance name to the error message.
    """

    def _get_prepend_message(type_object, instance_name):
        if instance_name is None:
            prepend_message = "In one of the " + type_object.__name__
        else:
            prepend_message = f"{type_object.__name__} with name '{instance_name}'"
        return prepend_message

    @wraps(func)
    def wrapper(*args, **kwargs):
        prepend_message = None
        # for field validator
        if len(args) == 3:
            model_class = args[0]
            validation_info = args[2]
            if validation_info is not None:
                name = validation_info.data.get("name", None)
                prepend_message = _get_prepend_message(model_class, name)
            else:
                raise NotImplementedError(
                    "[Internal] Make sure your field_validator has validationInfo in the args or"
                    " this wrapper is used with a field_validator!!"
                )
        # for model validator
        elif len(args) == 1:
            instance = args[0]
            if "name" not in instance.model_dump():
                raise NotImplementedError("[Internal] Make sure the model has name field.")
            prepend_message = _get_prepend_message(type(instance), instance.name)
        else:
            raise NotImplementedError(
                f"[Internal] {_validator_append_instance_name.__qualname__} decorator only supports 1 or 3 arguments."
            )

        try:
            result = func(*args, **kwargs)  # Call the original function
            return result
        except ValueError as e:
            raise ValueError(f"{prepend_message}: {str(e)}") from e

    return wrapper


def check_deleted_surface_in_entity_list(value):
    """
    Check if any boundary is meant to be deleted
    value--> EntityList
    """
    validation_info = get_validation_info()
    if (
        validation_info is None
        or validation_info.auto_farfield_method is None
        or validation_info.is_beta_mesher is True
    ):
        # validation not necessary now.
        return value

    # - Check if the surfaces are deleted.
    for surface in value.stored_entities:
        if isinstance(
            surface, Surface
        ) and surface._will_be_deleted_by_mesher(  # pylint:disable=protected-access
            validation_info.auto_farfield_method
        ):
            raise ValueError(
                f"Boundary `{surface.name}` will likely be deleted after mesh generation. Therefore it cannot be used."
            )

    return value


def check_deleted_surface_pair(value):
    """
    Check if any boundary is meant to be deleted
    value--> SurfacePair
    """
    validation_info = get_validation_info()
    if validation_info is None or validation_info.auto_farfield_method is None:
        # validation not necessary now.
        return value

    # - Check if the surfaces are deleted.
    for surface in value.pair:
        if surface._will_be_deleted_by_mesher(  # pylint:disable=protected-access
            validation_info.auto_farfield_method
        ):
            raise ValueError(
                f"Boundary `{surface.name}` will likely be deleted after mesh generation. Therefore it cannot be used."
            )

    return value


class EntityUsageMap:  # pylint:disable=too-few-public-methods
    """
    A customized dict to store the entity name and its usage.
    {"$EntityID": [$UsedInWhatModel]}
    """

    def __init__(self):
        self.dict_entity = {"Surface": {}, "Volume": {}}

    @classmethod
    def _get_entity_key(cls, entity) -> str:
        """
        Get unique identifier for the entity.
        """
        draft_entity_types = get_args(get_args(DraftEntityTypes)[0])
        if isinstance(entity, draft_entity_types):
            return entity.private_attribute_id
        return entity.name

    def add_entity_usage(self, entity, model_type):
        """
        Add the entity usage to the dictionary.
        """
        entity_type = None
        if isinstance(entity, _SurfaceEntityBase):
            entity_type = "Surface"
        elif isinstance(entity, _VolumeEntityBase):
            entity_type = "Volume"
        else:
            raise ValueError(
                f"[Internal Error] Entity `{entity.name}` in the {model_type} model "
                f"cannot be registered as a valid Surface or Volume entity."
            )
        entity_key = self._get_entity_key(entity=entity)
        entity_log = self.dict_entity[entity_type].get(
            entity_key, {"entity_name": entity.name, "model_list": []}
        )
        entity_log["model_list"].append(model_type)
        self.dict_entity[entity_type][entity_key] = entity_log
