"""Flow360BaseModel class definition."""

from __future__ import annotations

import hashlib
import json
from itertools import chain
from typing import Any, List, Literal, get_origin

import pydantic as pd
import rich
import yaml
from pydantic import ConfigDict
from pydantic._internal._decorators import Decorator, FieldValidatorDecoratorInfo
from pydantic_core import InitErrorDetails
from unyt import unyt_quantity

from flow360.component.simulation.conversion import (
    need_conversion,
    require,
    unit_converter,
)
from flow360.component.simulation.unit_system import LengthType
from flow360.component.simulation.validation import validation_context
from flow360.component.types import COMMENTS, TYPE_TAG_STR
from flow360.error_messages import do_not_modify_file_manually_msg
from flow360.exceptions import Flow360FileError
from flow360.log import log

DISCRIMINATOR_NAMES = [
    "type",
    "type_name",
    "refinement_type",
    "output_type",
    "private_attribute_entity_type_name",
]


def snake_to_camel(string: str) -> str:
    """
    Convert a snake_case string to camelCase.

    This function takes a snake_case string as input and converts it to camelCase.
    It splits the input string by underscores, capitalizes the first letter of
    each subsequent component (after the first one), and joins them together.

    Parameters:
    string (str): The input string in snake_case format.

    Returns:
    str: The converted string in camelCase format.

    Example:
    >>> snake_to_camel("example_snake_case")
    'exampleSnakeCase'
    """
    components = string.split("_")

    camel_case_string = components[0]

    for component in components[1:]:
        camel_case_string += component[0].upper() + component[1:]

    return camel_case_string


class Conflicts(pd.BaseModel):
    """
    Wrapper for handling fields that cannot be specified simultaneously
    """

    field1: str
    field2: str


class Flow360BaseModel(pd.BaseModel):
    """Base pydantic (V2) model that all Flow360 components inherit from.
    Defines configuration for handling data structures
    as well as methods for imporing, exporting, and hashing Flow360 objects.
    For more details on pydantic base models, see:
    `Pydantic Models <https://pydantic-docs.helpmanual.io/usage/models/>`
    """

    def __init__(self, filename: str = None, **kwargs):
        model_dict = self._handle_file(filename=filename, **kwargs)
        super().__init__(**model_dict)

    @classmethod
    def _handle_dict(cls, **kwargs):
        """Handle dictionary input for the model."""
        model_dict = kwargs
        model_dict = cls._handle_dict_with_hash(model_dict)
        return model_dict

    @classmethod
    def _handle_file(cls, filename: str = None, **kwargs):
        """Handle file input for the model.

        Parameters
        ----------
        filename : str
            Full path to the .json or .yaml file to load the :class:`Flow360BaseModel` from.
        **kwargs
            Keyword arguments to be passed to the model."""
        if filename is not None:
            return cls._dict_from_file(filename=filename)
        return kwargs

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs) -> None:
        """Things that are done to each of the models."""
        need_to_rebuild = cls._handle_wrap_validators()
        if need_to_rebuild is True:
            cls.model_rebuild(force=True)
        super().__pydantic_init_subclass__(**kwargs)  # Correct use of super

    model_config = ConfigDict(
        ##:: Pydantic kwargs
        arbitrary_types_allowed=True,  # ?
        extra="forbid",
        frozen=False,
        populate_by_name=True,
        validate_assignment=True,
        validate_default=True,
        ##:: Custom keys
        require_one_of=[],
        allow_but_remove=[],
        conflicting_fields=[],
        include_hash=False,
        include_defaults_in_schema=True,
        # pylint: disable=fixme
        # TODO: Remove alias_generator since it is only for translator
        alias_generator=pd.AliasGenerator(
            serialization_alias=snake_to_camel,
        ),
    )

    def __setattr__(self, name, value):
        if name in self.model_fields:
            is_frozen = self.model_fields[name].frozen
            if is_frozen is not None and is_frozen is True:
                raise ValueError(f"Cannot modify immutable/frozen fields: {name}")
        super().__setattr__(name, value)

    @pd.model_validator(mode="before")
    @classmethod
    def one_of(cls, values):
        """
        root validator for require one of
        """
        if cls.model_config["require_one_of"]:
            set_values = [key for key, v in values.items() if v is not None]
            aliases = [
                cls._get_field_alias(field_name=name) for name in cls.model_config["require_one_of"]
            ]
            aliases = [item for item in aliases if item is not None]
            intersection = list(set(set_values) & set(cls.model_config["require_one_of"] + aliases))
            if len(intersection) == 0:
                raise ValueError(f"One of {cls.model_config['require_one_of']} is required.")
        return values

    # pylint: disable=no-self-argument
    # pylint: disable=duplicate-code
    @pd.model_validator(mode="before")
    @classmethod
    def handle_conflicting_fields(cls, values):
        """
        root validator to handle deprecated aliases and fields
        which cannot be simultaneously defined in the model
        """
        if cls.model_config["conflicting_fields"]:
            for conflicting_field in cls.model_config["conflicting_fields"]:
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
            alias = [
                info.alias
                for name, info in cls.model_fields.items()
                if name == field_name and info.alias is not None
            ]
            if len(alias) > 0:
                return alias[0]
        return None

    @classmethod
    def _get_field_context(cls, info, context_key):
        if info.field_name is not None:
            field_info = cls.model_fields[info.field_name]
            if isinstance(field_info.json_schema_extra, dict):
                return field_info.json_schema_extra.get(context_key)

        return None

    @classmethod
    def _handle_wrap_validators(cls):
        """
        Applies `wrap` and `before` validators to selected fields while excluding discriminator fields.

        **Purpose**:
        - `wrap` validators cannot be applied to discriminator fields (e.g., `Literal` types like 'type'),
        as they cause Pydantic conflicts during validation.
        - This method manually assigns validators only to non-discriminator fields to avoid errors and
        ensure correct validation flow.

        **How it works**:
        - Iterates over model fields, excluding discriminator fields.
        - Applies validators dynamically to the remaining fields to ensure compatibility.

        """

        validators = [
            ("wrap", "populate_ctx_to_error_messages"),
            ("before", "validate_conditionally_required_field"),
        ]
        fields_to_validate = []
        need_to_rebuild = False

        for field_name, field in cls.model_fields.items():
            # Ignore discriminator validators
            # pylint: disable=comparison-with-callable
            if get_origin(field.annotation) == Literal and field_name in DISCRIMINATOR_NAMES:
                need_to_rebuild = True
                continue

            fields_to_validate.append(field_name)

        if need_to_rebuild is True:
            for mode, method in validators:
                info = FieldValidatorDecoratorInfo(
                    fields=tuple(fields_to_validate), mode=mode, check_fields=None
                )
                deco = Decorator.build(cls, cls_var_name=method, info=info, shim=None)
                cls.__pydantic_decorators__.field_validators[method] = deco
        return need_to_rebuild

    @pd.field_validator("*", mode="before")
    @classmethod
    def validate_conditionally_required_field(cls, value, info):
        """
        this validator checks for conditionally required fields depending on context
        """
        validation_levels = validation_context.get_validation_levels()
        if validation_levels is None:
            return value

        conditionally_required = cls._get_field_context(info, "conditionally_required")
        relevant_for = cls._get_field_context(info, "relevant_for")
        if (
            conditionally_required is True
            and any(lvl in (relevant_for, validation_context.ALL) for lvl in validation_levels)
            and value is None
        ):
            raise pd.ValidationError.from_exception_data(
                "validation error", [InitErrorDetails(type="missing")]
            )

        return value

    @pd.field_validator("*", mode="wrap")
    @classmethod
    def populate_ctx_to_error_messages(cls, values, handler, info) -> Any:
        """
        this validator populates ctx messages of fields tagged with "relevant_for" context
        it will populate to all child messages
        """
        try:
            return handler(values)
        except pd.ValidationError as e:
            validation_errors = e.errors()
            relevant_for = cls._get_field_context(info, "relevant_for")
            if relevant_for is not None:
                for i, error in enumerate(validation_errors):
                    ctx = error.get("ctx", {})
                    if ctx.get("relevant_for") is None:
                        ctx["relevant_for"] = relevant_for
                    validation_errors[i]["ctx"] = ctx
            raise pd.ValidationError.from_exception_data(
                title=cls.__class__.__name__, line_errors=validation_errors
            )

    # Note: to_solver architecture will be reworked in favor of splitting the models between
    # the user-side and solver-side models (see models.py and models_avl.py for reference
    # in the design360 repo)
    #
    # for now the to_solver functionality is removed, although some of the logic
    # (recursive definition) will probably carry over.

    def copy(self, update=None, **kwargs) -> Flow360BaseModel:
        """Copy a Flow360BaseModel.  With ``deep=True`` as default."""
        if "deep" in kwargs and kwargs["deep"] is False:
            raise ValueError("Can't do shallow copy of component, set `deep=True` in copy().")
        new_copy = pd.BaseModel.model_copy(self, update=update, deep=True, **kwargs)
        data = new_copy.model_dump()
        return self.model_validate(data)

    def help(self, methods: bool = False) -> None:
        """Prints message describing the fields and methods of a :class:`Flow360BaseModel`.

        Parameters
        ----------
        methods : bool = False
            Whether to also print out information about object's methods.

        Example
        -------
        >>> params.help(methods=True) # doctest: +SKIP
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
        >>> params = Flow360BaseModel.from_file(filename='folder/sim.json') # doctest: +SKIP
        """
        return cls(filename=filename)

    @classmethod
    def _dict_from_file(cls, filename: str) -> dict:
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
        >>> params = Flow360BaseModel.from_file(filename='folder/flow360.json') # doctest: +SKIP
        """

        if ".json" in filename:
            model_dict = cls._dict_from_json(filename=filename)
        elif ".yaml" in filename:
            model_dict = cls._dict_from_yaml(filename=filename)
        else:
            raise Flow360FileError(f"File must be *.json or *.yaml type, given {filename}")

        model_dict = cls._handle_dict_with_hash(model_dict)
        return model_dict

    def to_file(self, filename: str, **kwargs) -> None:
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
            return self._to_json(filename=filename, **kwargs)
        if ".yaml" in filename:
            return self._to_yaml(filename=filename, **kwargs)

        raise Flow360FileError(f"File must be .json, or .yaml, type, given {filename}")

    @classmethod
    def _from_json(cls, filename: str, **parse_obj_kwargs) -> Flow360BaseModel:
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
        >>> params = Flow360BaseModel._from_json(filename='folder/flow360.json') # doctest: +SKIP
        """
        model_dict = cls._dict_from_file(filename=filename)
        return cls.model_validate(model_dict, **parse_obj_kwargs)

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
        >>> params_dict = Flow360BaseModel.dict_from_json(filename='folder/flow360.json') # doctest: +SKIP
        """
        with open(filename, "r", encoding="utf-8") as json_fhandle:
            model_dict = json.load(json_fhandle)
        return model_dict

    def _to_json(self, filename: str, **kwargs) -> None:
        """Exports :class:`Flow360BaseModel` instance to .json file

        Parameters
        ----------
        filename : str
            Full path to the .json file to save the :class:`Flow360BaseModel` to.

        Example
        -------
        >>> params._to_json(filename='folder/flow360.json') # doctest: +SKIP
        """
        json_string = self.model_dump_json(**kwargs)
        model_dict = json.loads(json_string)
        if self.model_config["include_hash"] is True:
            model_dict["hash"] = self._calculate_hash(model_dict)
        with open(filename, "w+", encoding="utf-8") as file_handle:
            json.dump(model_dict, file_handle, indent=4, sort_keys=True)

    @classmethod
    def _from_yaml(cls, filename: str, **parse_obj_kwargs) -> Flow360BaseModel:
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
        >>> params = Flow360BaseModel._from_yaml(filename='folder/flow360.yaml') # doctest: +SKIP
        """
        model_dict = cls._dict_from_file(filename=filename)
        return cls.model_validate(model_dict, **parse_obj_kwargs)

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
        >>> params_dict = Flow360BaseModel.dict_from_yaml(filename='folder/flow360.yaml') # doctest: +SKIP
        """
        with open(filename, "r", encoding="utf-8") as yaml_in:
            model_dict = yaml.safe_load(yaml_in)
        return model_dict

    def _to_yaml(self, filename: str, **kwargs) -> None:
        """Exports :class:`Flow360BaseModel` instance to .yaml file.

        Parameters
        ----------
        filename : str
            Full path to the .yaml file to save the :class:`Flow360BaseModel` to.

        Example
        -------
        >>> params._to_yaml(filename='folder/flow360.yaml') # doctest: +SKIP
        """
        json_string = self.model_dump_json(**kwargs)
        model_dict = json.loads(json_string)
        if self.model_config["include_hash"]:
            model_dict["hash"] = self._calculate_hash(model_dict)
        with open(filename, "w+", encoding="utf-8") as file_handle:
            yaml.dump(model_dict, file_handle, indent=4, sort_keys=True)

    @classmethod
    def _handle_dict_with_hash(cls, model_dict):
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

    # pylint: disable=too-many-arguments, too-many-locals, too-many-branches
    def _convert_dimensions_to_solver(
        self,
        params,
        mesh_unit: LengthType.Positive = None,
        exclude: List[str] = None,
        required_by: List[str] = None,
        extra: List[Any] = None,
    ) -> dict:
        solver_values = {}
        self_dict = self.__dict__

        if exclude is None:
            exclude = []

        if required_by is None:
            required_by = []

        additional_fields = {}
        if extra is not None:
            for extra_item in extra:
                # Note: we should not be expecting extra field for SimulationParam?
                require(extra_item.dependency_list, required_by, params)
                additional_fields[extra_item.name] = extra_item.value_factory()

        assert mesh_unit is not None

        for property_name, value in chain(self_dict.items(), additional_fields.items()):
            if property_name in [COMMENTS, TYPE_TAG_STR]:
                continue
            loc_name = property_name
            field = self.model_fields.get(property_name)
            if field is not None and field.alias is not None:
                loc_name = field.alias
            if need_conversion(value) and property_name not in exclude:
                log.debug(f"   -> need conversion for: {property_name} = {value}")
                flow360_conv_system = unit_converter(
                    value.units.dimensions,
                    mesh_unit,
                    params=params,
                    required_by=[*required_by, loc_name],
                )
                # pylint: disable=no-member
                value.units.registry = flow360_conv_system.registry
                solver_values[property_name] = value.in_base(unit_system="flow360_v2")
                log.debug(f"      converted to: {solver_values[property_name]}")
            elif isinstance(value, list) and property_name not in exclude:
                new_value = []
                for item in value:
                    if need_conversion(item):
                        flow360_conv_system = unit_converter(
                            item.units.dimensions,
                            mesh_unit,
                            params=params,
                            required_by=[*required_by, loc_name],
                        )
                        # pylint: disable=no-member
                        item.units.registry = flow360_conv_system.registry
                        new_value.append(item.in_base(unit_system="flow360_v2"))
                    else:
                        new_value.append(item)
                solver_values[property_name] = new_value
            else:
                solver_values[property_name] = value

        return solver_values

    def preprocess(
        self,
        params,
        mesh_unit=None,
        exclude: List[str] = None,
        required_by: List[str] = None,
    ) -> Flow360BaseModel:
        """
        Loops through all fields, for Flow360BaseModel runs .preprocess() recusrively. For dimensioned value performs

        unit conversion to flow360_base system.

        Parameters
        ----------
        params : SimulationParams
            Full config definition as Flow360Params.

        exclude: List[str] (optional)
            List of fields to not convert to solver dimensions.

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

        solver_values = self._convert_dimensions_to_solver(params, mesh_unit, exclude, required_by)
        for property_name, value in self.__dict__.items():
            if property_name in [COMMENTS, TYPE_TAG_STR] + exclude:
                continue
            loc_name = property_name
            field = self.model_fields.get(property_name)
            if field is not None and field.alias is not None:
                loc_name = field.alias
            if isinstance(value, Flow360BaseModel):
                solver_values[property_name] = value.preprocess(
                    params,
                    mesh_unit=mesh_unit,
                    required_by=[*required_by, loc_name],
                    exclude=exclude,
                )
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, Flow360BaseModel):
                        solver_values[property_name][i] = item.preprocess(
                            params,
                            mesh_unit=mesh_unit,
                            required_by=[*required_by, loc_name, f"{i}"],
                            exclude=exclude,
                        )
                    elif isinstance(item, unyt_quantity):
                        solver_values[property_name][i] = item.in_base(unit_system="flow360_v2")

        return self.__class__(**solver_values)
