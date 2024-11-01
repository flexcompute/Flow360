"""
Flow360 solver parameters
"""

# pylint: disable=too-many-lines
from __future__ import annotations

import hashlib
import json
import re
from abc import ABCMeta, abstractmethod
from copy import deepcopy
from typing import Any, Dict, List, Optional, Type

import numpy as np
import pydantic.v1 as pd
import rich
import unyt
import yaml
from pydantic.v1 import BaseModel
from pydantic.v1.fields import ModelField
from typing_extensions import Literal

from flow360.component.types import COMMENTS, TYPE_TAG_STR
from flow360.component.v1.conversions import need_conversion, require, unit_converter
from flow360.component.v1.unit_system import DimensionedType, is_flow360_unit
from flow360.error_messages import do_not_modify_file_manually_msg
from flow360.exceptions import Flow360FileError, Flow360ValidationError
from flow360.log import log


def json_dumps(value, *args, **kwargs):
    """custom json dump with sort_keys=True"""
    kwargs["sort_keys"] = True  # Set sort_keys within kwargs
    return json.dumps(value, *args, **kwargs)


# pylint: disable=invalid-name
def params_generic_validator(value, ExpectedParamsType):
    """generic validator for params comming from webAPI

    Parameters
    ----------
    value : dict
        value to validate
    ExpectedParamsType : SurfaceMeshingParams, Flow360MeshParams
        expected type of params
    """
    params = value
    if isinstance(value, str):
        try:
            params = json.loads(value)
        except json.decoder.JSONDecodeError:
            return None
    try:
        ExpectedParamsType(**params)
    except Flow360ValidationError:
        return None
    except pd.ValidationError:
        return None
    except TypeError:
        return None

    return params


def _self_named_property_validator(values: dict, validator: Type[BaseModel], msg: str = "") -> dict:
    """When model uses custom, user defined property names, for example boundary names.

    Parameters
    ----------
    values : dict
        pydantic root validator values
    validator : BaseModel
        pydantic BaseModel with field 'v' for validating entry of this class properites
    msg : str, optional
        Additional message to be displayed on error, by default ""

    Returns
    -------
    dict
        values to be passed to next pydantic root validator

    Raises
    ------
    ValueError
        When validation fails
    """
    for key, v in values.items():
        # skip validation for comments and internal _type
        if key in (COMMENTS, TYPE_TAG_STR):
            continue
        try:
            values[key] = validator(v=v).v
        except pd.ValidationError as exc:
            raise exc
        except Exception as exc:
            raise ValueError(f"{v} (type={type(v)}) {msg}") from exc
    return values


class DeprecatedAlias(BaseModel):
    """
    Wrapper for handling deprecated field aliases
    """

    name: str
    deprecated: str


class Conflicts(BaseModel):
    """
    Wrapper for handling fields that cannot be specified simultaneously
    """

    field1: str
    field2: str


def encode_ndarray(x):
    """
    encoder for ndarray
    """
    if x.size == 1:
        return float(x)
    return tuple(x.tolist())


def dimensioned_type_serializer(x):
    """
    encoder for dimensioned type (unyt_quantity, unyt_array, DimensionedType)
    """
    return {"value": x.value, "units": str(x.units)}


_json_encoders_map = {
    unyt.unyt_array: dimensioned_type_serializer,
    DimensionedType: dimensioned_type_serializer,
    unyt.Unit: str,
    np.ndarray: encode_ndarray,
}


def _flow360_solver_dimensioned_type_serializer(x):
    """
    encoder for dimensioned type (unyt_quantity, unyt_array, DimensionedType)
    """
    if not is_flow360_unit(x):
        raise ValueError(
            f"Value {x} is not in flow360 unit system and should not be directly exported to flow360 solver json."
        )
    return x.value


_flow360_solver_json_encoder_map = {
    unyt.unyt_array: _flow360_solver_dimensioned_type_serializer,
    DimensionedType: _flow360_solver_dimensioned_type_serializer,
    np.ndarray: encode_ndarray,
}


def _flow360_solver_json_encoder(obj):
    for custom_type, encoder in _flow360_solver_json_encoder_map.items():
        if isinstance(obj, custom_type):
            return encoder(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def flow360_json_encoder(obj):
    """
    flow360 json encoder. Removes {value:, units:} formatting and returns value directly.
    """
    try:
        return json.JSONEncoder().default(obj)
    except TypeError:
        return _flow360_solver_json_encoder(obj)


def _schema_optional_toggle_name(name):
    return f"_add{name[0].upper() + name[1:]}"


# pylint: disable=too-many-public-methods
class Flow360BaseModel(BaseModel):
    """Base pydantic model that all Flow360 components inherit from.
    Defines configuration for handling data structures
    as well as methods for imporing, exporting, and hashing Flow360 objects.
    For more details on pydantic base models, see:
    `Pydantic Models <https://pydantic-docs.helpmanual.io/usage/models/>`_
    """

    _will_export_to_flow360: bool = pd.PrivateAttr(False)

    def __init__(self, filename: str = None, **kwargs):
        model_dict = self._init_handle_file(filename=filename, **kwargs)
        super().__init__(**model_dict)

    @classmethod
    def _init_handle_dict(cls, **kwargs):
        model_dict = kwargs
        model_dict = cls._init_handle_hash(model_dict)
        return model_dict

    @classmethod
    def _init_handle_file(cls, filename: str = None, **kwargs):
        if filename is not None:
            return cls.dict_from_file(filename=filename)
        return kwargs

    def __init_subclass__(cls) -> None:
        """Things that are done to each of the models."""

        cls.add_type_field()
        cls.generate_docstring()

    class Config:  # pylint: disable=too-few-public-methods
        """Sets config for all :class:`Flow360BaseModel` objects.

        Configuration Options
        ---------------------
        allow_population_by_field_name : bool = True
            Allow properties to stand in for fields(?).
        arbitrary_types_allowed : bool = True
            Allow types like numpy arrays.
        extra : str = 'forbid'
            Forbid extra kwargs not specified in model.
        json_encoders : Dict[type, Callable]
            Defines how to encode type in json file.
        validate_all : bool = True
            Validate default values just to be safe.
        validate_assignment : bool
            Re-validate after re-assignment of field in model.
        """

        arbitrary_types_allowed = True
        validate_all = True
        extra = "forbid"
        validate_assignment = True
        allow_population_by_field_name = True
        json_encoders = _json_encoders_map
        allow_mutation = True
        copy_on_model_validation = "none"
        underscore_attrs_are_private = True
        require_one_of: Optional[List[str]] = []
        allow_but_remove: Optional[List[str]] = []
        deprecated_aliases: Optional[List[DeprecatedAlias]] = []
        conflicting_fields: Optional[List[Conflicts]] = []
        include_hash: bool = False
        include_defaults_in_schema: bool = True
        exclude_on_flow360_export: Optional[Any] = None

    def __setattr__(self, name, value):
        if name in self.__fields__:
            is_mutable = self.__fields__[name].field_info.extra.get("mutable")
            if is_mutable is not None and is_mutable is False:
                raise ValueError(f"Cannot modify immutable fields: {name}")
        super().__setattr__(name, value)

    # pylint: disable=unused-argument
    @classmethod
    def __modify_schema__(cls, field_schema, field):
        field_schema.update(cls.schema())
        if cls.SchemaConfig.displayed is not None:
            field_schema["displayed"] = cls.SchemaConfig.displayed

    # pylint: disable=no-self-argument
    @pd.root_validator(pre=True)
    def one_of(cls, values):
        """
        root validator for require one of
        """
        if cls.Config.require_one_of:
            set_values = [key for key, v in values.items() if v is not None]
            aliases = [cls._get_field_alias(field_name=name) for name in cls.Config.require_one_of]
            aliases = [item for item in aliases if item is not None]
            intersection = list(set(set_values) & set(cls.Config.require_one_of + aliases))
            if len(intersection) == 0:
                raise ValueError(f"One of {cls.Config.require_one_of} is required.")
        return values

    # pylint: disable=no-self-argument
    @pd.root_validator(pre=True)
    def allow_but_remove(cls, values):
        """
        root validator for allow_but_remove, e.g., legacy properties that are no longer in use
        """
        if cls.Config.allow_but_remove:
            for field in cls.Config.allow_but_remove:
                values.pop(field, None)
        return values

    # pylint: disable=no-self-argument
    @pd.root_validator(pre=True)
    def handle_deprecated_aliases(cls, values):
        """
        root validator to handle deprecated aliases
        """
        if cls.Config.deprecated_aliases:
            for deprecated_alias in cls.Config.deprecated_aliases:
                values = cls._handle_depracated_alias(values, deprecated_alias)
        return values

    # pylint: disable=no-self-argument
    @pd.root_validator(pre=True)
    def handle_conflicting_fields(cls, values):
        """
        root validator to handle deprecated aliases
        """
        if cls.Config.conflicting_fields:
            for conflicting_field in cls.Config.conflicting_fields:
                values = cls._handle_conflicting_fields(values, conflicting_field)
        return values

    @classmethod
    def _handle_conflicting_fields(cls, values, conflicting_field: Conflicts = None):
        conflicting_field1_value = values.get(conflicting_field.field1, None)
        conflicting_field2_value = values.get(conflicting_field.field2, None)

        if conflicting_field1_value is None:
            field1_alias = cls._get_field_alias(field_name=conflicting_field.field1)
            conflicting_field1_value = values.get(field1_alias, None)

        if conflicting_field2_value is None:
            field2_alias = cls._get_field_alias(field_name=conflicting_field.field2)
            conflicting_field2_value = values.get(field2_alias, None)

        if conflicting_field1_value is not None and conflicting_field2_value is not None:
            raise ValueError(
                f"{conflicting_field.field1} and {conflicting_field.field2} cannot be specified at the same time."
            )

        return values

    @classmethod
    def _get_field_alias(cls, field_name: str = None):
        if field_name is not None:
            alias = [field.alias for field in cls.__fields__.values() if field.name == field_name]
            if len(alias) > 0:
                return alias[0]
        return None

    @classmethod
    def _handle_depracated_alias(cls, values, deprecated_alias: DeprecatedAlias = None):
        deprecated_value = values.get(deprecated_alias.deprecated, None)
        alias = cls._get_field_alias(field_name=deprecated_alias.name)
        actual_value = values.get(deprecated_alias.name, values.get(alias, None))

        if deprecated_value is not None:
            if actual_value is not None and actual_value != deprecated_value:
                allowed = [deprecated_alias.deprecated, deprecated_alias.name]
                raise ValueError(f"Only one of {allowed} can be used.")
            if actual_value is None:
                if alias and alias != deprecated_alias.name:
                    log.warning(
                        f'"{deprecated_alias.deprecated}" is deprecated. '
                        f'Use "{deprecated_alias.name}" OR "{alias}" instead.'
                    )
                else:
                    log.warning(
                        f'"{deprecated_alias.deprecated}" is deprecated. '
                        f'Use "{deprecated_alias.name}" instead.'
                    )
                values[deprecated_alias.name] = deprecated_value
            values.pop(deprecated_alias.deprecated)

        return values

    # pylint: disable=too-few-public-methods
    class SchemaConfig:
        """Sets JSON schema generation config for :class:`Flow360BaseModel` objects.

        Configuration Options
        ---------------------
        field_order : List[str]
            Ordering of schema fields in the output JSON dictionary
        optional_objects : List[str]
            Fields for which optional schema structure is generated
        exclude_fields : List[str]
            Fields to be excluded from the output schema (slash-separated paths to the field in nested objects)
        widgets : Dict[str, (str, str)]
            Widget mappings for the UI schema
        displayed : Optional[str]
            Title override for the class name in the schema root
        root_property : Optional[str]
            Title
        """

        field_order = []
        optional_objects = []
        exclude_fields = []
        field_properties = {}
        root_property = None
        swap_fields = {}
        displayed = None

    @classmethod
    def _schema_fix_single_allof(cls, dictionary):
        if not isinstance(dictionary, dict):
            raise ValueError("Input must be a dictionary")

        for key, value in list(dictionary.items()):
            if key == "allOf" and len(value) == 1 and isinstance(value[0], dict):
                for allOfKey, allOfValue in list(value[0].items()):
                    dictionary[allOfKey] = allOfValue
                del dictionary["allOf"]
            elif isinstance(value, dict):
                cls._schema_fix_single_allof(value)

        return dictionary

    @classmethod
    def _schema_camel_to_space(cls, name: str):
        if len(name) > 0:
            if name[0] == "_":
                name = name[1:]
            if name[0].isupper():
                name = name[0].lower() + name[1:]
        name = re.sub("(.)([A-Z][a-z]+)", r"\1 \2", name)
        name = re.sub("([a-z0-9])([A-Z])", r"\1 \2", name).lower()
        name = name.capitalize()
        return name

    @classmethod
    def _schema_format_titles(cls, dictionary, name=None):
        if not isinstance(dictionary, dict):
            raise ValueError("Input must be a dictionary")

        for key, value in list(dictionary.items()):
            if isinstance(value, dict):
                cls._schema_format_titles(value, name=key)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        cls._schema_format_titles(item)
            elif key == "title" and dictionary.get("displayed") is None:
                if name is None:
                    dictionary["title"] = cls._schema_camel_to_space(value)
                else:
                    dictionary["title"] = cls._schema_camel_to_space(name)

        return dictionary

    @classmethod
    def _schema_remove(cls, schema, key_path: List[str]):
        if not isinstance(schema, dict):
            raise ValueError("Input must be a dictionary")

        if schema.get(key_path[0]) is not None:
            if len(key_path) == 1:
                del schema[key_path[0]]
            elif len(key_path) > 1:
                value = schema[key_path[0]]
                if isinstance(value, Dict):
                    cls._schema_remove(value, key_path[1:])
                elif isinstance(value, List):
                    for item in value:
                        if isinstance(item, dict):
                            cls._schema_remove(item, key_path[1:])

        return schema

    @classmethod
    def _schema_remove_all(cls, schema, key_to_remove: str):
        """Removes all occurences of a"""
        if not isinstance(schema, dict):
            raise ValueError("Input must be a dictionary")

        for key, value in list(schema.items()):
            if key == key_to_remove:
                del schema[key]
            elif isinstance(value, dict):
                cls._schema_remove_all(value, key_to_remove)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        cls._schema_remove_all(item, key_to_remove)

        return schema

    @classmethod
    def _schema_swap_key(cls, dictionary, key_to_replace, replacement_key):
        if not isinstance(dictionary, dict):
            raise ValueError("Input must be a dictionary")

        for key, value in list(dictionary.items()):
            if key == replacement_key and dictionary.get(key_to_replace) is not None:
                dictionary[key_to_replace] = dictionary[replacement_key]
                del dictionary[replacement_key]
            elif isinstance(value, dict):
                cls._schema_swap_key(value, key_to_replace, replacement_key)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        cls._schema_swap_key(item, key_to_replace, replacement_key)

        return dictionary

    @classmethod
    def _schema_apply_optional_objects(cls, schema: dict, key):
        field = schema["properties"].get(key)

        if field is not None:
            schema["properties"].pop(key)

            toggle_name = _schema_optional_toggle_name(key)

            schema["properties"][toggle_name] = {
                "title": "",
                "displayed": cls._schema_camel_to_space(key),
                "type": "boolean",
                "default": False,
            }

            displayed = schema.get("displayed")

            if displayed is not None:
                schema["properties"][toggle_name]["displayed"] = displayed

            if schema.get("dependencies") is None:
                schema["dependencies"] = {}

            schema["dependencies"][toggle_name] = {
                "oneOf": [
                    {
                        "type": "object",
                        "properties": {
                            toggle_name: {"default": True, "const": True, "type": "boolean"},
                            key: field,
                        },
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {
                            toggle_name: {"default": False, "const": False, "type": "boolean"}
                        },
                        "additionalProperties": False,
                    },
                ]
            }

    @classmethod
    def _schema_generate_optional(cls, schema: dict, path):
        if not isinstance(schema, dict):
            raise ValueError("Input must be a dictionary")

        if len(path) == 2:
            if path[0] != "properties":
                raise ValueError(
                    "Path must point to a property, ending with (...)/properties/fieldName"
                )
            cls._schema_apply_optional_objects(schema, path[1])
        else:
            key = path[0]
            value = schema.get(key)
            if value is not None:
                if isinstance(value, Dict):
                    cls._schema_generate_optional(value, path[1:])
                elif isinstance(value, List):
                    for item in value:
                        if isinstance(item, dict):
                            cls._schema_generate_optional(item, path[1:])

    @classmethod
    def _schema_apply_option_names(cls, dictionary):
        if not isinstance(dictionary, dict):
            raise ValueError("Input must be a dictionary")

        for key, value in list(dictionary.items()):
            options = dictionary.get("anyOf")
            if key == "options" and options is not None:
                if len(value) != len(options):
                    raise ValueError(
                        f"Tried applying options for {value}, but lengths "
                        f"of anyOf items and options do not match"
                    )
                for option, title in zip(options, value):
                    option["title"] = title
            elif isinstance(value, dict):
                cls._schema_apply_option_names(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        cls._schema_apply_option_names(item)

    @classmethod
    def _schema_fix_single_value_enum(cls, dictionary):
        for key, value in list(dictionary.items()):
            if key == "enum" and len(value) == 1:
                default = value[0]
                del dictionary[key]
                dictionary["const"] = default
            elif isinstance(value, dict):
                cls._schema_fix_single_value_enum(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        cls._schema_fix_single_value_enum(item)

    @classmethod
    def _schema_clean(cls, schema):
        cls._schema_remove_all(schema, "description")
        cls._schema_remove_all(schema, "_type")
        cls._schema_remove_all(schema, "comments")
        cls._schema_remove_all(schema, "options")

    @classmethod
    def _schema_apply_root_property(cls, schema):
        if cls.SchemaConfig.root_property is not None:
            current = schema
            path = cls.SchemaConfig.root_property.split("/")
            for item in path:
                current = current[item]
            schema[path[-1]] = current
            del schema["properties"]
            del schema["required"]

    @classmethod
    def flow360_schema(cls):
        """Generate a schema json string for the flow360 model"""
        schema = cls.schema()
        if cls.SchemaConfig.displayed is not None:
            schema["displayed"] = cls.SchemaConfig.displayed
        for item in cls.SchemaConfig.exclude_fields:
            cls._schema_remove(schema, item.split("/"))
        cls._schema_format_titles(schema)
        cls._schema_apply_option_names(schema)
        cls._schema_apply_root_property(schema)
        cls._schema_fix_single_allof(schema)
        cls._schema_fix_single_value_enum(schema)
        for item in cls.SchemaConfig.optional_objects:
            cls._schema_generate_optional(schema, item.split("/"))
        if cls.SchemaConfig.swap_fields is not None:
            for key, value in cls.SchemaConfig.swap_fields.items():
                value["title"] = schema["properties"][key]["title"]
                displayed = schema["properties"][key].get("displayed")
                if displayed is not None:
                    value["displayed"] = displayed
                schema["properties"][key] = value
        cls._schema_swap_key(schema, "title", "displayed")
        cls._schema_clean(schema)
        return schema

    @classmethod
    def flow360_ui_schema(cls):
        """Generate a UI schema json string for the flow360 model"""
        order = cls.SchemaConfig.field_order
        optionals = cls.SchemaConfig.optional_objects
        properties = cls.SchemaConfig.field_properties
        schema = {}

        # pylint: disable=consider-using-enumerate
        for i in range(0, len(order)):
            name = order[i].split("/")[:-1]
            if name in optionals:
                order.insert(i, _schema_optional_toggle_name(name))

        if len(order) > 0:
            schema["ui:order"] = order
        if len(properties) > 0:
            for key, value in properties.items():
                path = key.split("/")

                target = schema
                for item in path:
                    if target.get(item) is None:
                        target[item] = {}
                    target = target[item]

                target[f"ui:{value[0]}"] = value[1]
        schema["ui:submitButtonOptions"] = {"norender": True}
        schema["ui:options"] = {"orderable": False, "addable": False, "removable": False}
        return schema

    def _convert_dimensions_to_solver(
        self,
        params,
        exclude: List[str] = None,
        required_by: List[str] = None,
        extra: List[Any] = None,
    ) -> dict:
        solver_values = {}
        self_dict = deepcopy(self.__dict__)

        if exclude is None:
            exclude = []

        if required_by is None:
            required_by = []

        if extra is not None:
            for extra_item in extra:
                require(extra_item.dependency_list, required_by, params)
                self_dict[extra_item.name] = extra_item.value_factory()

        for property_name, value in self_dict.items():
            if property_name in [COMMENTS, TYPE_TAG_STR] + exclude:
                continue
            loc_name = property_name
            field = self.__fields__.get(property_name)
            if field is not None and field.alias is not None:
                loc_name = field.alias
            if need_conversion(value):
                log.debug(f"   -> need conversion for: {property_name} = {value}")
                flow360_conv_system = unit_converter(
                    value.units.dimensions,
                    params=params,
                    required_by=[*required_by, loc_name],
                )
                # pylint: disable=no-member
                value.units.registry = flow360_conv_system.registry
                solver_values[property_name] = value.in_base(unit_system="flow360_v1")
                log.debug(f"      converted to: {solver_values[property_name]}")
            else:
                solver_values[property_name] = value

        return solver_values

    def to_solver(
        self, params, exclude: List[str] = None, required_by: List[str] = None
    ) -> Flow360BaseModel:
        """
        Loops through all fields, for Flow360BaseModel runs .to_solver() recusrively. For dimensioned value performs

        unit conversion to flow360_base system.

        Parameters
        ----------
        params : Flow360Params
            Full config definition as Flow360Params.

        exclude: List[str] (optional)
            List of fields to ignore on returned model.

        required_by: List[str] (optional)
            Path to property which requires conversion.

        Returns
        -------
        caller class
            returns caller class with units all in flow360 base unit system
        """

        if exclude is None:
            exclude = []

        if required_by is None:
            required_by = []

        solver_values = self._convert_dimensions_to_solver(params, exclude, required_by)
        for property_name, value in self.__dict__.items():
            if property_name in [COMMENTS, TYPE_TAG_STR] + exclude:
                continue
            loc_name = property_name
            field = self.__fields__.get(property_name)
            if field is not None and field.alias is not None:
                loc_name = field.alias
            if isinstance(value, Flow360BaseModel):
                solver_values[property_name] = value.to_solver(
                    params, required_by=[*required_by, loc_name]
                )
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, Flow360BaseModel):
                        solver_values[property_name][i] = item.to_solver(
                            params, required_by=[*required_by, loc_name, f"{i}"]
                        )

        return self.__class__(**solver_values)

    def copy(self, update=None, **kwargs) -> Flow360BaseModel:
        """Copy a Flow360BaseModel.  With ``deep=True`` as default."""
        if "deep" in kwargs and kwargs["deep"] is False:
            raise ValueError("Can't do shallow copy of component, set `deep=True` in copy().")
        kwargs.update({"deep": True})
        new_copy = BaseModel.copy(self, update=update, **kwargs)
        data = new_copy.dict()
        return self.validate(data)

    def help(self, methods: bool = False) -> None:
        """Prints message describing the fields and methods of a :class:`Flow360BaseModel`.

        Parameters
        ----------
        methods : bool = False
            Whether to also print out information about object's methods.

        Example
        -------
        >>> solver_params.help(methods=True) # doctest: +SKIP
        """
        rich.inspect(self, methods=methods)

    @classmethod
    def from_file(cls, filename: str) -> Flow360BaseModel:
        """Loads a :class:`Flow360BaseModel` from .json, or .yaml file.

        Parameters
        ----------
        filename : str
            Full path to the .yaml or .json file to load the :class:`Flow360BaseModel` from.

        Returns
        -------
        :class:`Flow360BaseModel`
            An instance of the component class calling `load`.

        Example
        -------
        >>> simulation = Simulation.from_file(filename='folder/sim.json') # doctest: +SKIP
        """
        return cls(filename=filename)

    @classmethod
    def dict_from_file(cls, filename: str) -> dict:
        """Loads a dictionary containing the model from a .json or .yaml file.

        Parameters
        ----------
        filename : str
            Full path to the .yaml or .json file to load the :class:`Flow360BaseModel` from.

        Returns
        -------
        dict
            A dictionary containing the model.

        Example
        -------
        >>> params = Flow360Params.from_file(filename='folder/flow360.json') # doctest: +SKIP
        """

        if ".json" in filename:
            model_dict = cls._dict_from_json(filename=filename)
        elif ".yaml" in filename:
            model_dict = cls._dict_from_yaml(filename=filename)

        else:
            raise Flow360FileError(f"File must be .json, or .yaml, type, given {filename}")

        model_dict = cls._init_handle_hash(model_dict)
        return model_dict

    def to_file(self, filename: str) -> None:
        """Exports :class:`Flow360BaseModel` instance to .json or .yaml file

        Parameters
        ----------
        filename : str
            Full path to the .json or .yaml or file to save the :class:`Flow360BaseModel` to.

        Example
        -------
        >>> params.to_file(filename='folder/flow360.json') # doctest: +SKIP
        """

        if ".json" in filename:
            return self.to_json(filename=filename)
        if ".yaml" in filename:
            return self.to_yaml(filename=filename)

        raise Flow360FileError(f"File must be .json, or .yaml, type, given {filename}")

    @classmethod
    def from_json(cls, filename: str, **parse_obj_kwargs) -> Flow360BaseModel:
        """Load a :class:`Flow360BaseModel` from .json file.

        Parameters
        ----------
        filename : str
            Full path to the .json file to load the :class:`Flow360BaseModel` from.

        Returns
        -------
        :class:`Flow360BaseModel`
            An instance of the component class calling `load`.
        **parse_obj_kwargs
            Keyword arguments passed to pydantic's ``parse_obj`` method.

        Example
        -------
        >>> params = Flow360Params.from_json(filename='folder/flow360.json') # doctest: +SKIP
        """
        model_dict = cls.dict_from_file(filename=filename)
        return cls.parse_obj(model_dict, **parse_obj_kwargs)

    @classmethod
    def _dict_from_json(cls, filename: str) -> dict:
        """Load dictionary of the model from a .json file.

        Parameters
        ----------
        filename : str
            Full path to the .json file to load the :class:`Flow360BaseModel` from.

        Returns
        -------
        dict
            A dictionary containing the model.

        Example
        -------
        >>> params_dict = Flow360Params.dict_from_json(filename='folder/flow360.json') # doctest: +SKIP
        """
        with open(filename, encoding="utf-8") as json_fhandle:
            model_dict = json.load(json_fhandle)
        return model_dict

    def to_json(self, filename: str) -> None:
        """Exports :class:`Flow360BaseModel` instance to .json file

        Parameters
        ----------
        filename : str
            Full path to the .json file to save the :class:`Flow360BaseModel` to.

        Example
        -------
        >>> params.to_json(filename='folder/flow360.json') # doctest: +SKIP
        """
        json_string = self.json()
        model_dict = json.loads(json_string)
        if self.Config.include_hash:
            model_dict["hash"] = self._calculate_hash(model_dict)
        with open(filename, "w+", encoding="utf-8") as file_handle:
            json.dump(model_dict, file_handle, indent=4)

    @classmethod
    def from_yaml(cls, filename: str, **parse_obj_kwargs) -> Flow360BaseModel:
        """Loads :class:`Flow360BaseModel` from .yaml file.

        Parameters
        ----------
        filename : str
            Full path to the .yaml file to load the :class:`Flow360BaseModel` from.
        **parse_obj_kwargs
            Keyword arguments passed to pydantic's ``parse_obj`` method.

        Returns
        -------
        :class:`Flow360BaseModel`
            An instance of the component class calling `from_yaml`.

        Example
        -------
        >>> params = Flow360Params.from_yaml(filename='folder/flow360.yaml') # doctest: +SKIP
        """
        model_dict = cls.dict_from_file(filename=filename)
        return cls.parse_obj(model_dict, **parse_obj_kwargs)

    @classmethod
    def _dict_from_yaml(cls, filename: str) -> dict:
        """Load dictionary of the model from a .yaml file.

        Parameters
        ----------
        filename : str
            Full path to the .yaml file to load the :class:`Flow360BaseModel` from.

        Returns
        -------
        dict
            A dictionary containing the model.

        Example
        -------
        >>> params_dict = Flow360Params.dict_from_yaml(filename='folder/flow360.yaml') # doctest: +SKIP
        """
        with open(filename, encoding="utf-8") as yaml_in:
            model_dict = yaml.safe_load(yaml_in)
        return model_dict

    def to_yaml(self, filename: str) -> None:
        """Exports :class:`Flow360BaseModel` instance to .yaml file.

        Parameters
        ----------
        filename : str
            Full path to the .yaml file to save the :class:`Flow360BaseModel` to.

        Example
        -------
        >>> params.to_yaml(filename='folder/flow360.yaml') # doctest: +SKIP
        """
        json_string = self.json()
        model_dict = json.loads(json_string)
        if self.Config.include_hash:
            model_dict["hash"] = self._calculate_hash(model_dict)
        with open(filename, "w+", encoding="utf-8") as file_handle:
            yaml.dump(model_dict, file_handle, indent=4)

    def _handle_export_exclude(self, exclude):
        if exclude:
            exclude = {TYPE_TAG_STR, *exclude}
        else:
            exclude = {TYPE_TAG_STR}

        if self._will_export_to_flow360 is True:
            if self.Config.exclude_on_flow360_export:
                exclude = {*exclude, *self.Config.exclude_on_flow360_export}
            self.Config.will_export = False

        return exclude

    def set_will_export_to_flow360(self, flag: bool):
        """Recursivly sets flag will_export_to_flow360

        Parameters
        ----------
        flag : bool
            set to true before exporting to flow360 json
        """
        self._will_export_to_flow360 = flag
        for value in self.__dict__.values():
            if isinstance(value, Flow360BaseModel):
                value.set_will_export_to_flow360(flag)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, Flow360BaseModel):
                        item.set_will_export_to_flow360(flag)

    def dict(self, *args, exclude=None, **kwargs) -> dict:
        """Returns dict representation of the model.

        Parameters
        ----------

        *args
            Arguments passed to pydantic's ``dict`` method.

        **kwargs
            Keyword arguments passed to pydantic's ``dict`` method.

        Returns
        -------
        dict
            A formatted dict.

        Example
        -------
        >>> params.dict() # doctest: +SKIP
        """

        exclude = self._handle_export_exclude(exclude)
        return super().dict(*args, exclude=exclude, **kwargs)

    def json(self, *args, exclude=None, **kwargs):
        """Returns json representation of the model.

        Parameters
        ----------

        *args
            Arguments passed to pydantic's ``json`` method.

        **kwargs
            Keyword arguments passed to pydantic's ``json`` method.

        Returns
        -------
        json
            A formatted json.
            Sets default vaules by_alias=True, exclude_none=True

        Example
        -------
        >>> params.json() # doctest: +SKIP
        """

        exclude = self._handle_export_exclude(exclude)
        return super().json(*args, by_alias=True, exclude_none=True, exclude=exclude, **kwargs)

    @classmethod
    def _init_handle_hash(cls, model_dict):
        hash_from_input = model_dict.pop("hash", None)
        if hash_from_input is not None:
            if hash_from_input != cls._calculate_hash(model_dict):
                log.warning(do_not_modify_file_manually_msg)
        return model_dict

    @classmethod
    def _calculate_hash(cls, model_dict):
        hasher = hashlib.sha256()
        json_string = json.dumps(model_dict, sort_keys=True)
        hasher.update(json_string.encode("utf-8"))
        return hasher.hexdigest()

    # pylint: disable=unnecessary-dunder-call
    def append(self, params: Flow360BaseModel, overwrite: bool = False):
        """append parametrs to the model

        Parameters
        ----------
        params : Flow360BaseModel
            Flow360BaseModel parameters to be appended
        overwrite : bool, optional
            Whether to overwrite if key exists, by default False
        """
        additional_config = params.dict(exclude_unset=True, exclude_none=True)
        for key, value in additional_config.items():
            if self.__getattribute__(key) and not overwrite:
                log.warning(
                    f'"{key}" already exist in the original model, skipping. Use overwrite=True to overwrite values.'
                )
                continue
            is_mutable = self.__fields__[key].field_info.extra.get("mutable")
            if is_mutable is None or is_mutable is True:
                self.__setattr__(key, value)

    @classmethod
    def add_type_field(cls) -> None:
        """Automatically place "type" field with model name in the model field dictionary."""

        value = cls.__name__
        annotation = Literal[value]

        tag_field = ModelField.infer(
            name=TYPE_TAG_STR,
            value=value,
            annotation=annotation,
            class_validators=None,
            config=cls.__config__,
        )
        cls.__fields__[TYPE_TAG_STR] = tag_field

    @classmethod
    def generate_docstring(cls) -> str:
        """Generates a docstring for a Flow360 model and saves it to the __doc__ of the class."""

        # store the docstring in here
        doc = ""

        # if the model already has a docstring, get the first lines and save the rest
        original_docstrings = []
        if cls.__doc__:
            original_docstrings = cls.__doc__.split("\n\n")
            class_description = original_docstrings.pop(0)
            doc += class_description
        original_docstrings = "\n\n".join(original_docstrings)

        # create the list of parameters (arguments) for the model
        doc += "\n\n    Parameters\n    ----------\n"
        for field_name, field in cls.__fields__.items():
            # ignore the type tag
            if field_name == TYPE_TAG_STR:
                continue

            # get data type
            data_type = field._type_display()  # pylint:disable=protected-access

            # get default values
            default_val = field.get_default()
            if "=" in str(default_val):
                # handle cases where default values are pydantic models
                default_val = f"{default_val.__class__.__name__}({default_val})"
                default_val = (", ").join(default_val.split(" "))

            # make first line: name : type = default
            default_str = "" if field.required else f" = {default_val}"
            doc += f"    {field_name} : {data_type}{default_str}\n"

            # get field metadata
            field_info = field.field_info

            # add units (if present)
            units = field_info.extra.get("units")
            if units is not None:
                doc += "        "
                if isinstance(units, (tuple, list)):
                    unitstr = "("
                    for unit in units:
                        unitstr += str(unit)
                        unitstr += ", "
                    unitstr = unitstr[:-2]
                    unitstr += ")"
                else:
                    unitstr = units
                doc += f"[units = {unitstr}].  "

            # add description
            description_str = field_info.description
            if description_str is not None:
                doc += f"{description_str}\n"

        # add in remaining things in the docs
        if original_docstrings:
            doc += "\n"
            doc += original_docstrings

        doc += "\n"
        cls.__doc__ = doc


class Flow360SortableBaseModel(Flow360BaseModel, metaclass=ABCMeta):
    """:class:`Flow360SortableBaseModel` class for setting up parameters by names, eg. boundary names"""

    def __getitem__(self, item):
        """to support [] access"""
        return getattr(self, item)

    def __setitem__(self, key, value):
        """to support [] assignment"""
        super().__setattr__(key, value)

    def names(self) -> List[str]:
        """return names of all instances"""
        return [k for k, _ in self if k not in [COMMENTS, TYPE_TAG_STR]]

    @classmethod
    def _collect_all_definitions(cls, dictionary, collected):
        if not isinstance(dictionary, dict):
            raise ValueError("Input must be a dictionary")

        for key, value in list(dictionary.items()):
            if key == "definitions":
                collected.update(value)
                del dictionary[key]
            elif isinstance(value, dict):
                cls._collect_all_definitions(value, collected)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        cls._collect_all_definitions(item, collected)

    @classmethod
    @abstractmethod
    def get_subtypes(cls) -> list:
        """retrieve allowed types of this self-named property"""

    @classmethod
    def flow360_schema(cls):
        root_schema = {"additionalProperties": {}}

        models = cls.get_subtypes()

        if len(models) == 1:
            schema = models[0].flow360_schema()
            root_schema["additionalProperties"] = schema
        else:
            root_schema["additionalProperties"]["oneOf"] = []
            for model in models:
                schema = model.flow360_schema()
                root_schema["additionalProperties"]["oneOf"].append(schema)

        if cls.SchemaConfig.displayed is not None:
            root_schema["displayed"] = cls.SchemaConfig.displayed
        for item in cls.SchemaConfig.exclude_fields:
            cls._schema_remove(root_schema, item.split("/"))
        for item in cls.SchemaConfig.optional_objects:
            cls._schema_generate_optional(root_schema, item.split("/"))
        if cls.SchemaConfig.swap_fields is not None:
            for key, value in cls.SchemaConfig.swap_fields.items():
                value["title"] = root_schema["properties"][key]["title"]
                displayed = root_schema["properties"][key].get("displayed")
                if displayed is not None:
                    value["displayed"] = displayed
                root_schema["properties"][key] = value
        cls._schema_swap_key(root_schema, "title", "displayed")
        cls._schema_clean(root_schema)

        definitions = {}

        cls._collect_all_definitions(root_schema, definitions)

        if definitions:
            root_schema["definitions"] = definitions

        root_schema["type"] = "object"

        return root_schema

    # pylint: disable=missing-class-docstring,too-few-public-methods
    class Config(Flow360BaseModel.Config):
        extra = "allow"
        json_dumps = json_dumps
