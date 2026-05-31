"""EntityBase: base class for dynamic entity types."""

import hashlib
from abc import ABCMeta
from typing import Any

import pydantic as pd

from flow360_schema.framework.base_model import Flow360BaseModel


class EntityBase(Flow360BaseModel, metaclass=ABCMeta):
    """
    Base class for dynamic entity types.

    Attributes:
        private_attribute_entity_type_name (str):
            A string representing the specific type of the entity.
            This should be set in subclasses to differentiate between entity types.

        name (str):
            The name of the entity instance, used for identification and retrieval.
    """

    private_attribute_entity_type_name: str = "Invalid"
    private_attribute_id: str | None = pd.Field(
        # TODO: This should not have default value. Everyone is supposed to set it.
        None,
        frozen=True,
        description="Unique identifier for the entity. Used by front end to track entities and enable auto update etc.",
    )

    name: str = pd.Field(frozen=True)

    # Whether the entity is dirty and needs to be re-hashed
    _dirty: bool = pd.PrivateAttr(True)
    # Cached hash of the entity
    _hash_cache: str | None = pd.PrivateAttr(None)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Validate required class-level attributes at subclass creation time.

        This avoids per-instance checks and catches misconfigured subclasses early.

        Rules:
        - If a subclass explicitly defines `private_attribute_entity_type_name` in its own
          class body, it must also be non-"Invalid". Intermediate abstract bases that do not
          set an entity type are allowed.
        """
        super().__init_subclass__(**kwargs)
        if cls is EntityBase:
            return

        # Only enforce entity type when the subclass explicitly sets it.
        if "private_attribute_entity_type_name" not in cls.__dict__:
            return

        # entity_type remains a Pydantic field
        def _resolve_field_default(field_name: str) -> Any:
            for base in cls.__mro__:
                if field_name in getattr(base, "__dict__", {}):
                    raw_value = base.__dict__[field_name]
                    return getattr(raw_value, "default", raw_value)
            model_fields = getattr(cls, "model_fields", None)
            if isinstance(model_fields, dict) and field_name in model_fields:
                field_info = model_fields[field_name]
                return getattr(field_info, "default", None)
            return None

        type_value = _resolve_field_default("private_attribute_entity_type_name")
        if type_value is None or type_value == "Invalid":
            raise NotImplementedError(
                f"private_attribute_entity_type_name is not defined in the entity class: {cls.__name__}."
            )

    def copy(self, update: dict[str, Any], **kwargs: Any) -> "EntityBase":  # type: ignore[override]
        """
        Creates a copy of the entity with compulsory updates.

        Parameters:
            update: A dictionary containing the updated attributes to apply to the copied entity.
            **kwargs: Additional arguments to pass to the copy constructor.

        Returns:
            A copy of the entity with the specified updates.
        """
        if "name" not in update or update["name"] == self.name:
            raise ValueError(
                "Copying an entity requires a new name to be specified. "
                "Please provide a new name in the update dictionary."
            )
        return super().copy(update=update, **kwargs)  # type: ignore[return-value]

    def __eq__(self, other: object) -> bool:
        """Defines the equality comparison for entities to support usage in UniqueItemList."""
        if isinstance(other, EntityBase):
            return (self.name + "-" + self.__class__.__name__) == (other.name + "-" + other.__class__.__name__)
        return False

    @property
    def entity_type(self) -> str:
        """returns the entity class name."""
        return self.private_attribute_entity_type_name

    @entity_type.setter
    def entity_type(self, value: str) -> None:
        raise AttributeError("Cannot modify the name of entity class.")

    def __str__(self) -> str:
        return "\n".join([f"        {attr}: {value}" for attr, value in self.__dict__.items()])

    def _recompute_hash(self) -> str:
        new_hash = hashlib.sha256(self.model_dump_json().encode("utf-8")).hexdigest()
        # Can further speed up 10% by using `object.__setattr__`
        self._hash_cache = new_hash
        self._dirty = False
        return new_hash

    def _get_hash(self) -> str:
        """hash generator to identify if two entities are the same"""
        # Can further speed up 10% by using `object.__getattribute__`
        dirty = self._dirty
        cache = self._hash_cache
        if dirty or cache is None:
            return self._recompute_hash()
        return cache

    def __setattr__(self, name: str, value: Any) -> None:
        """
        [Large model performance]
        Wrapping the __setattr__ to mark the entity as dirty when the attribute is not private
        This enables caching the hash of the entity to avoid re-calculating the hash when the entity is not changed.
        """

        super().__setattr__(name, value)
        if not name.startswith("_") and not self._dirty:
            # Not using self to avoid invoking
            # Can further speed up 10% by using `object.__setattr__`
            self._dirty = True

    @property
    def id(self) -> str:
        """Returns private_attribute_id of the entity."""
        return self.private_attribute_id  # type: ignore[return-value]

    def _manual_assignment_validation(self, _: Any) -> "EntityBase":
        """
        Pre-expansion contextual validation for the entity.
        This handles validation for the entity manually assigned.
        """
        return self

    def _per_entity_type_validation(self, _: Any) -> "EntityBase":
        """Contextual validation with validation logic bond with the specific entity type."""
        return self
