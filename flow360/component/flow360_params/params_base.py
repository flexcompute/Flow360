"""
Flow360 solver parameters
"""
from __future__ import annotations

import json
from functools import wraps
from typing import Any, List, Optional

import numpy as np
import pydantic as pd
import rich
import yaml
from pydantic import BaseModel
from pydantic.fields import ModelField
from typing_extensions import Literal

from ...exceptions import ConfigError, FileError, ValidationError
from ...log import log
from ..types import COMMENTS, TYPE_TAG_STR


def json_dumps(value, *args, **kwargs):
    """custom json dump with sort_keys=True"""
    return json.dumps(value, sort_keys=True, *args, **kwargs)


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
    except ValidationError:
        return None
    except pd.ValidationError:
        return None
    except TypeError:
        return None

    return params


def export_to_flow360(func):
    """wraps to_flow360_json() function to set correct

    Parameters
    ----------
    func : function
        to_flow360_json() function which exports JSON format for flow360

    Returns
    -------
    func
        wrapper
    """

    @wraps(func)
    def wrapper_func(*args, **kwargs):
        args[0].Config.will_export = True
        try:
            value = func(*args, **kwargs)
        except:
            args[0].Config.will_export = False
            raise
        return value

    return wrapper_func


def _self_named_property_validator(values: dict, validator: BaseModel, msg: str = "") -> dict:
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
    ValidationError
        When validation fails
    """
    for key, v in values.items():
        # allow for comments
        if key == COMMENTS:
            continue
        try:
            values[key] = validator(v=v).v
        except Exception as exc:
            raise ValidationError(f"{v} (type={type(v)}) {msg}") from exc
    return values


class DeprecatedAlias(BaseModel):
    name: str
    deprecated: str


class Flow360BaseModel(BaseModel):
    """Base pydantic model that all Flow360 components inherit from.
    Defines configuration for handling data structures
    as well as methods for imporing, exporting, and hashing Flow360 objects.
    For more details on pydantic base models, see:
    `Pydantic Models <https://pydantic-docs.helpmanual.io/usage/models/>`_
    """

    # comments is allowed property at every level
    comments: Optional[Any] = pd.Field()

    def __init__(self, filename: str = None, **kwargs):
        if filename:
            obj = self.from_file(filename=filename)
            super().__init__(**obj.dict())
        else:
            super().__init__(**kwargs)

        self.Config.will_export = False

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
        json_encoders = {
            np.ndarray: lambda x: tuple(x.tolist()),
        }
        allow_mutation = True
        copy_on_model_validation = "none"
        underscore_attrs_are_private = True
        exclude_on_flow360_export: Optional[Any] = None
        will_export: Optional[bool] = False
        require_one_of: Optional[List[str]] = []
        allow_but_remove: Optional[List[str]] = []
        deprecated_aliases: Optional[List[DeprecatedAlias]] = []

    # pylint: disable=no-self-argument
    @pd.root_validator()
    def one_of(cls, values):
        """root validator for require one of"""
        if cls.Config.require_one_of:
            set_values = [key for key, v in values.items() if v is not None]
            intersection = list(set(set_values) & set(cls.Config.require_one_of))
            if len(intersection) == 0:
                raise ConfigError(f"One of {cls.Config.require_one_of} is required.")
        return values

    # pylint: disable=no-self-argument
    @pd.root_validator(pre=True)
    def allow_but_remove(cls, values):
        """root validator for allow_but_remove, e.g., legacy properties that are no longer in use"""
        if cls.Config.allow_but_remove:
            for field in cls.Config.allow_but_remove:
                values.pop(field, None)
        return values

    # pylint: disable=no-self-argument
    @pd.root_validator(pre=True)
    def handle_depracated_aliases(cls, values):
        """
        root validator to handle deprecated aliases
        """
        if cls.Config.deprecated_aliases:
            for deprecated_alias in cls.Config.deprecated_aliases:
                values = cls._handle_depracated_alias(values, deprecated_alias)
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
                raise ValidationError(f"Only one of {allowed} can be used.")
            if actual_value is None:
                log.warning(
                    f'"{deprecated_alias.deprecated}" is deprecated. Use "{deprecated_alias.name}" instead.'
                )
                values[deprecated_alias.name] = deprecated_value
            values.pop(deprecated_alias.deprecated)

        return values

    def copy(self, update=None, **kwargs) -> Flow360BaseModel:
        """Copy a Flow360BaseModel.  With ``deep=True`` as default."""
        if "deep" in kwargs and kwargs["deep"] is False:
            raise ValueError("Can't do shallow copy of component, set `deep=True` in copy().")
        kwargs.update({"deep": True})
        new_copy = BaseModel.copy(self, update=update, **kwargs)
        return self.validate(new_copy.dict())

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
    def from_file(cls, filename: str, **parse_obj_kwargs) -> Flow360BaseModel:
        """Loads a :class:`Flow360BaseModel` from .json, or .yaml file.

        Parameters
        ----------
        filename : str
            Full path to the .yaml or .json file to load the :class:`Flow360BaseModel` from.
        **parse_obj_kwargs
            Keyword arguments passed to either pydantic's ``parse_obj`` function when loading model.

        Returns
        -------
        :class:`Flow360BaseModel`
            An instance of the component class calling `load`.

        Example
        -------
        >>> simulation = Simulation.from_file(filename='folder/sim.json') # doctest: +SKIP
        """
        model_dict = cls.dict_from_file(filename=filename)
        return cls.parse_obj(model_dict, **parse_obj_kwargs)

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
            return cls.dict_from_json(filename=filename)
        if ".yaml" in filename:
            return cls.dict_from_yaml(filename=filename)

        raise FileError(f"File must be .json, or .yaml, type, given {filename}")

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

        raise FileError(f"File must be .json, or .yaml, type, given {filename}")

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
        model_dict = cls.dict_from_json(filename=filename)
        return cls.parse_obj(model_dict, **parse_obj_kwargs)

    @classmethod
    def dict_from_json(cls, filename: str) -> dict:
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
        with open(filename, "r", encoding="utf-8") as json_fhandle:
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
        json_string = self.json(indent=4)
        with open(filename, "w", encoding="utf-8") as file_handle:
            file_handle.write(json_string)

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
        model_dict = cls.dict_from_yaml(filename=filename)
        return cls.parse_obj(model_dict, **parse_obj_kwargs)

    @classmethod
    def dict_from_yaml(cls, filename: str) -> dict:
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
        with open(filename, "r", encoding="utf-8") as yaml_in:
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
        with open(filename, "w+", encoding="utf-8") as file_handle:
            yaml.dump(model_dict, file_handle, indent=4)

    def _handle_export_exclude(self, exclude):
        if exclude:
            exclude = {TYPE_TAG_STR, *exclude}
        else:
            exclude = {TYPE_TAG_STR}

        if self.Config.will_export:
            if self.Config.exclude_on_flow360_export:
                exclude = {*exclude, *self.Config.exclude_on_flow360_export}
            self.Config.will_export = False
        return exclude

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

    @export_to_flow360
    def to_flow360_json(self, return_json: bool = True):
        """Generate a JSON representation of the model, as required by Flow360

        Parameters
        ----------
        return_json : bool, optional
            whether to return value or return None, by default True

        Returns
        -------
        json
            If return_json==True, returns JSON representation of the model.

        Example
        -------
        >>> params.to_flow360_json() # doctest: +SKIP
        """
        if return_json:
            return self.json()
        return None

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
        """Generates a docstring for a Flow360 mode and saves it to the __doc__ of the class."""

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
            doc += "        "

            # add units (if present)
            units = field_info.extra.get("units")
            if units is not None:
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


class Flow360SortableBaseModel(Flow360BaseModel):
    """:class:`Flow360SortableBaseModel` class for setting up parameters by names, eg. boundary names"""

    # pylint: disable=missing-class-docstring,too-few-public-methods
    class Config(Flow360BaseModel.Config):
        extra = "allow"
        json_dumps = json_dumps
