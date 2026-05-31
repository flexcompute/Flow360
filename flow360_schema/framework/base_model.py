"""Flow360BaseModel: core base model for all Flow360 schema models."""

from __future__ import annotations

import hashlib
import json
import logging
from functools import lru_cache
from typing import Any, Literal, get_args, get_origin

import pydantic as pd
from pydantic._internal._decorators import Decorator, FieldValidatorDecoratorInfo
from pydantic.aliases import AliasChoices
from pydantic_core import InitErrorDetails, PydanticUndefined

from flow360_schema.exceptions import Flow360FileError

from .base_model_utils import (
    _preprocess_any_model,
    _preprocess_nested,
    need_conversion,
)
from .validation.context import ALL, get_validation_levels

logger = logging.getLogger(__name__)

_HASH_TAMPER_WARNING = (
    "The file was manually edited. It is recommended to not edit config files manually but to use:\n"
    ">>> params = Flow360Params(filename)\n"
    ">>> # edit your params here\n"
    ">>> params.to_file(filename)\n"
)

DISCRIMINATOR_NAMES = [
    "type",
    "type_name",
    "refinement_type",
    "output_type",
    "private_attribute_entity_type_name",
]


def get_field_aliases(model_cls: type[pd.BaseModel], field_name: str | None = None) -> tuple[str, ...]:
    """Return all supported input aliases for a field name."""
    if field_name is None:
        return ()
    field_info = model_cls.model_fields.get(field_name)
    if field_info is None:
        return ()

    aliases: list[str] = []

    def _append_alias(alias: str) -> None:
        if alias not in aliases:
            aliases.append(alias)

    if isinstance(field_info.alias, str):
        _append_alias(field_info.alias)

    if isinstance(field_info.validation_alias, str):
        _append_alias(field_info.validation_alias)
    elif isinstance(field_info.validation_alias, AliasChoices):
        for choice in field_info.validation_alias.choices:
            if isinstance(choice, str):
                _append_alias(choice)

    return tuple(aliases)


def get_field_alias(model_cls: type[pd.BaseModel], field_name: str | None = None) -> str | None:
    """Return one supported input alias for a field name, if available."""
    aliases = get_field_aliases(model_cls, field_name=field_name)
    if len(aliases) == 0:
        return None
    return aliases[0]


def snake_to_camel(string: str) -> str:
    """Convert a snake_case string to camelCase.

    Parameters
    ----------
    string : str
        The input string in snake_case format.

    Returns
    -------
    str
        The converted string in camelCase format.

    Raises
    ------
    ValueError
        If the input contains empty split components after ``_`` split, such as
        leading/trailing/consecutive underscores.

    Example
    -------
    >>> snake_to_camel("example_snake_case")
    'exampleSnakeCase'
    """
    components = string.split("_")
    if any(component == "" for component in components):
        raise ValueError(f"Invalid field name '{string}': leading/trailing/consecutive underscores are not allowed.")

    camel_case_string = components[0]
    for component in components[1:]:
        camel_case_string += component[0].upper() + component[1:]
    return camel_case_string


class Flow360BaseModel(pd.BaseModel):
    """Base class for all Flow360 schema models.

    Provides:
    - Strict validation config (extra=forbid, camelCase serialization, etc.)
    - Frozen field enforcement via __setattr__
    - Field alias / field context helpers used by validation and mixins
    - Conditional field validation based on ValidationContext
    - Error enhancement with relevant_for context
    """

    model_config = pd.ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        frozen=False,
        populate_by_name=True,
        validate_assignment=True,
        validate_default=True,
        # TODO: Move camelCase serialization out of the base model and into
        # explicit translator/request export helpers. The primary wire format is
        # simulation.json, which should stay snake_case by default.
        alias_generator=pd.AliasGenerator(
            serialization_alias=snake_to_camel,
        ),
    )

    @classmethod
    def _enrich_validation_error(cls, error: pd.ValidationError) -> pd.ValidationError:
        validation_errors = error.errors()
        enriched = False
        for i, validation_error in enumerate(validation_errors):
            ctx = validation_error.get("ctx")
            if not isinstance(ctx, dict) or ctx.get("relevant_for") is None:
                loc_tuple = tuple(validation_error.get("loc", ()))
                rf = cls._infer_relevant_for_cached(loc_tuple)
                if rf is not None:
                    new_ctx = {} if not isinstance(ctx, dict) else dict(ctx)
                    new_ctx["relevant_for"] = list(rf)
                    validation_errors[i]["ctx"] = new_ctx
                    enriched = True
        if not enriched:
            return error
        return pd.ValidationError.from_exception_data(
            title=cls.__name__,
            line_errors=validation_errors,  # type: ignore[arg-type]  # Pydantic internal API type mismatch, safe at runtime
        )

    @pd.model_validator(mode="wrap")
    @classmethod
    def enrich_validation_error(cls, data: Any, handler: pd.ValidatorFunctionWrapHandler) -> Any:
        try:
            return handler(data)
        except pd.ValidationError as error:
            enriched_error = cls._enrich_validation_error(error)
            if enriched_error is error:
                raise
            raise enriched_error from None

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        need_to_rebuild = cls._handle_conditional_validators()
        if need_to_rebuild is True:
            cls.model_rebuild(force=True)
        super().__pydantic_init_subclass__(**kwargs)

    @classmethod
    def deserialize(cls, data: dict[str, Any], **kwargs: Any) -> Flow360BaseModel:
        """Validate a dict that contains already-serialized (JSON-round-tripped) data.

        Wraps ``model_validate`` in a ``DeserializationContext`` so that bare
        numeric values are interpreted as SI and no spurious unit-context
        warnings are emitted.  Use this instead of ``model_validate`` whenever
        the input comes from JSON, ``model_dump()``, or any other serialized
        source.
        """
        from .validation.context import DeserializationContext

        with DeserializationContext():
            return cls.model_validate(data, **kwargs)

    def __setattr__(self, name: str, value: object) -> None:
        if name in self.__class__.model_fields:
            is_frozen = self.__class__.model_fields[name].frozen
            if is_frozen is not None and is_frozen is True:
                raise ValueError(f"Cannot modify immutable/frozen fields: {name}")
        super().__setattr__(name, value)

    def _force_set_attr(self, name: str, value: Any) -> None:
        """Internal only: bypass frozen-field and assignment-validation checks."""
        object.__setattr__(self, name, value)
        if not name.startswith("_") and name in self.__class__.model_fields:
            self.__pydantic_fields_set__.add(name)
        private_attributes = getattr(self, "__pydantic_private__", None)
        if not name.startswith("_") and isinstance(private_attributes, dict) and "_dirty" in private_attributes:
            self.__dict__.pop("_dirty", None)
            private_attributes["_dirty"] = True

    # -- SDK methods: copy, help, file I/O, hash, preprocess -------------------

    def copy(self, update: dict[str, Any] | None = None, **kwargs: Any) -> Flow360BaseModel:
        """Deep-copy a Flow360BaseModel."""
        if "deep" in kwargs and kwargs["deep"] is False:
            raise ValueError("Can't do shallow copy of component, set `deep=True` in copy().")
        kwargs.pop("deep", None)
        from .validation.context import DeserializationContext, unit_system_manager

        new_copy = pd.BaseModel.model_copy(self, update=update, deep=True, **kwargs)
        data = new_copy.model_dump(exclude={"private_attribute_id"})
        with unit_system_manager.suspended(), DeserializationContext():
            return self.model_validate(data)

    def help(self, methods: bool = False) -> None:
        """Print fields and methods of a Flow360BaseModel using rich."""
        try:
            import rich  # noqa: PLC0415

            rich.inspect(self, methods=methods)
        except ImportError:
            print(repr(self))

    @classmethod
    def from_file(cls, filename: str) -> Flow360BaseModel:
        """Load a Flow360BaseModel from a .json file."""
        model_dict = cls._handle_file(filename=filename)
        return cls.deserialize(model_dict)

    def to_file(self, filename: str, **kwargs: Any) -> None:
        """Export Flow360BaseModel instance to a .json file."""
        if not filename.endswith(".json"):
            raise Flow360FileError(f"File must be *.json type, given {filename}")
        self._to_json(filename=filename, **kwargs)

    @classmethod
    def _dict_from_file(cls, filename: str) -> dict[str, Any]:
        """Load a dictionary from a .json file."""
        if not filename.endswith(".json"):
            raise Flow360FileError(f"File must be *.json type, given {filename}")
        model_dict = cls._dict_from_json(filename=filename)
        model_dict = cls._handle_dict_with_hash(model_dict)
        return model_dict

    @classmethod
    def _dict_from_json(cls, filename: str) -> dict[str, Any]:
        """Load dictionary from a .json file."""
        with open(filename, encoding="utf-8") as fh:
            result: dict[str, Any] = json.load(fh)
            return result

    def _to_json(self, filename: str, **kwargs: Any) -> None:
        """Export to .json file."""
        # TODO: _to_json hard-codes exclude_none=True while also accepting **kwargs,
        # which can cause duplicate kwarg TypeError. Candidate for removal — use
        # model_dump_json + json.dump directly at call sites.
        json_string = self.model_dump_json(exclude_none=True, **kwargs)
        model_dict = json.loads(json_string)
        if self.model_config.get("include_hash", False) is True:
            model_dict["hash"] = self._calculate_hash(model_dict)
        with open(filename, "w+", encoding="utf-8") as fh:
            json.dump(model_dict, fh, indent=4, sort_keys=True)

    @classmethod
    def _handle_dict(cls, **kwargs: Any) -> dict[str, Any]:
        """Handle dictionary input for the model."""
        return cls._handle_dict_with_hash(kwargs)

    @classmethod
    def _handle_file(cls, filename: str | None = None, **kwargs: Any) -> dict[str, Any]:
        """Handle file input for the model."""
        if filename is not None:
            return cls._dict_from_file(filename=filename)
        return kwargs

    @classmethod
    def _handle_dict_with_hash(cls, model_dict: dict[str, Any]) -> dict[str, Any]:
        """Pop hash from dict, warn if it doesn't match (file was tampered with)."""
        hash_from_input = model_dict.pop("hash", None)
        if hash_from_input is not None and hash_from_input != cls._calculate_hash(model_dict):
            logger.warning(_HASH_TAMPER_WARNING)
        return model_dict

    @classmethod
    def _calculate_hash(cls, model_dict: dict[str, Any]) -> str:
        """Calculate SHA-256 hash of a model dict, ignoring private_attribute_id."""

        def _strip_ids(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {k: _strip_ids(v) for k, v in obj.items() if k != "private_attribute_id"}
            if isinstance(obj, list):
                return [_strip_ids(item) for item in obj]
            return obj

        cleaned = _strip_ids(model_dict)
        hasher = hashlib.sha256()
        hasher.update(json.dumps(cleaned, sort_keys=True).encode("utf-8"))
        return hasher.hexdigest()

    def _nondimensionalization(
        self,
        *,
        exclude: list[str] | None = None,
        required_by: list[str] | None = None,
        flow360_unit_system: Any = None,
    ) -> dict[str, Any]:
        """Convert dimensioned fields on the current model to base units."""
        if exclude is None:
            exclude = []
        solver_values = {}
        for property_name, value in self.__dict__.items():
            if isinstance(value, Flow360BaseModel) or property_name in exclude:
                solver_values[property_name] = value
            elif need_conversion(value):
                solver_values[property_name] = value.in_base(flow360_unit_system)
            else:
                solver_values[property_name] = value
        return solver_values

    def preprocess(
        self,
        *,
        params: Any = None,
        exclude: list[str] | None = None,
        required_by: list[str] | None = None,
        flow360_unit_system: Any = None,
    ) -> Flow360BaseModel:
        """Convert all dimensioned fields to flow360 base unit system, recursively."""
        if exclude is None:
            exclude = []
        if required_by is None:
            required_by = []

        solver_values = self._nondimensionalization(
            exclude=exclude,
            required_by=required_by,
            flow360_unit_system=flow360_unit_system,
        )
        for property_name, value in self.__dict__.items():
            if property_name in exclude:
                continue
            loc_name = property_name
            field = self.__class__.model_fields.get(property_name)
            if field is not None and field.alias is not None:
                loc_name = field.alias
            if isinstance(value, Flow360BaseModel):
                solver_values[property_name] = _preprocess_any_model(
                    value,
                    params=params,
                    required_by=[*required_by, loc_name],
                    exclude=exclude,
                    flow360_unit_system=flow360_unit_system,
                )
            elif isinstance(value, (list, dict)):
                solver_values[property_name] = _preprocess_nested(
                    value, [*required_by, loc_name], params, exclude, flow360_unit_system
                )

        return self.__class__(**solver_values)

    @classmethod
    def _get_field_alias(cls, field_name: str | None = None) -> str | None:
        """Return the Pydantic alias for *field_name*, or ``None`` if unset."""
        return get_field_alias(cls, field_name=field_name)

    @classmethod
    def _get_field_context(cls, info: pd.fields.FieldInfo | object, context_key: str) -> object:
        """Extract *context_key* from a field's ``json_schema_extra``."""
        if info.field_name is not None:  # type: ignore[union-attr]
            field_info = cls.model_fields[info.field_name]  # type: ignore[union-attr]
            if isinstance(field_info.json_schema_extra, dict):
                return field_info.json_schema_extra.get(context_key)
        return None

    # -- Conditional field validation ------------------------------------------

    @classmethod
    def _handle_conditional_validators(cls) -> bool:
        """Dynamically apply field validators excluding discriminator fields."""
        validators = [
            ("before", "validate_conditionally_required_field"),
        ]
        fields_to_validate: list[str] = []
        need_to_rebuild = False

        for field_name, field in cls.model_fields.items():
            if get_origin(field.annotation) == Literal and field_name in DISCRIMINATOR_NAMES:
                need_to_rebuild = True
                continue
            fields_to_validate.append(field_name)

        if need_to_rebuild is True:
            for mode, method in validators:
                info = FieldValidatorDecoratorInfo(
                    fields=tuple(fields_to_validate),
                    mode=mode,  # type: ignore[arg-type]  # mode is str but guaranteed to be valid Literal value
                    check_fields=None,
                    # Sentinel "no override" — let each field's declared type drive its JSON Schema.
                    # `Any` would collapse type/bounds/format on every covered field's schema output.
                    json_schema_input_type=PydanticUndefined,
                )
                deco = Decorator.build(cls, cls_var_name=method, info=info, shim=None)
                cls.__pydantic_decorators__.field_validators[method] = deco
        return need_to_rebuild

    @pd.field_validator("*", mode="before")
    @classmethod
    def validate_conditionally_required_field(cls, value: Any, info: Any) -> Any:
        """Check conditionally required fields depending on validation context."""
        validation_levels = get_validation_levels()
        if validation_levels is None:
            return value

        conditionally_required = cls._get_field_context(info, "conditionally_required")
        relevant_for = cls._get_field_context(info, "relevant_for")

        all_relevant_levels = tuple(relevant_for + [ALL]) if isinstance(relevant_for, list) else (relevant_for, ALL)

        if (
            conditionally_required is True
            and any(lvl in all_relevant_levels for lvl in validation_levels)
            and value is None
        ):
            raise pd.ValidationError.from_exception_data(
                "validation error",
                [
                    InitErrorDetails(
                        type="missing",
                        input=value,  # Provide the actual input that failed validation
                    )
                ],
            )

        return value

    # -- relevant_for inference ------------------------------------------------

    @classmethod
    @lru_cache(maxsize=4096)
    def _infer_relevant_for_cached(cls, loc: tuple[str, ...]) -> tuple[str, ...] | None:
        """Infer relevant_for along the loc path starting at this model class."""
        model: type = cls
        last_relevant = None
        for seg in loc:
            if not (isinstance(model, type) and issubclass(model, Flow360BaseModel)):
                break
            fields = getattr(model, "model_fields", None)
            if not isinstance(seg, str) or not fields or seg not in fields:
                break
            field_info = fields[seg]
            extra = getattr(field_info, "json_schema_extra", None)
            if isinstance(extra, dict):
                rf = extra.get("relevant_for")
                if rf is not None:
                    last_relevant = rf

            next_model = cls._first_model_type_from(field_info)
            if next_model is None:
                break
            model = next_model

        if last_relevant is None:
            return None
        if isinstance(last_relevant, list):
            return tuple(last_relevant)
        return (last_relevant,)

    @staticmethod
    def _first_model_type_from(field_info: Any) -> type | None:
        """Extract first Flow360BaseModel subclass from a field's annotation."""
        annotation = getattr(field_info, "annotation", None)
        return Flow360BaseModel._extract_model_type(annotation)

    @staticmethod
    def _extract_model_type(tp: Any) -> type | None:
        if tp is None:
            return None
        if isinstance(tp, type):
            try:
                if issubclass(tp, Flow360BaseModel):
                    return tp
            except TypeError:
                return None
            return None
        origin = get_origin(tp)
        if origin is None:
            return None
        # typing.Annotated
        if str(origin) == "typing.Annotated":
            args = get_args(tp)
            if args:
                return Flow360BaseModel._extract_model_type(args[0])
            return None
        # Optional/Union
        if origin is Literal:
            return None
        if str(origin) == "typing.Union":
            for arg in get_args(tp):
                mt = Flow360BaseModel._extract_model_type(arg)
                if mt is not None:
                    return mt
            return None
        # Containers: List[T], Dict[K,V], Tuple[...]
        args = get_args(tp)
        if not args:
            return None
        dict_types = (dict,)

        start_index = 1 if origin in dict_types and len(args) == 2 else 0
        for arg in args[start_index:]:
            mt = Flow360BaseModel._extract_model_type(arg)
            if mt is not None:
                return mt
        return None
