from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any, List, Literal

import pydantic as pd
import rich
import yaml
from pydantic import ConfigDict

import flow360.component.simulation.units as u
from flow360.component.simulation.conversion import (
    need_conversion,
    require,
    unit_converter,
)
from flow360.component.types import COMMENTS, TYPE_TAG_STR
from flow360.error_messages import do_not_modify_file_manually_msg
from flow360.exceptions import Flow360FileError
from flow360.log import log


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
        keys_to_remove = []
        for property_name in model_dict.keys():
            if property_name == TYPE_TAG_STR:
                keys_to_remove.append(property_name)
        for key in keys_to_remove:
            model_dict.pop(key)
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
        super().__pydantic_init_subclass__(**kwargs)  # Correct use of super
        cls._generate_docstring()

    @pd.computed_field
    def _type(self) -> str:
        return self.__class__.__name__

    """Sets config for all :class:`Flow360BaseModel` objects.

        Custom Configuration Options
        ---------------------
        require_one_of: Optional[List[str]] = []
        conflicting_fields: Optional[List[Conflicts]] = []
        include_hash: bool = False
        include_defaults_in_schema: bool = True
    """
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
    )

    def __setattr__(self, name, value):
        if name in self.model_fields:
            is_frozen = self.model_fields[name].frozen
            if is_frozen is not None and is_frozen is True:
                raise ValueError(f"Cannot modify immutable/frozen fields: {name}")
        super().__setattr__(name, value)

    @pd.model_validator(mode="before")
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
    @pd.model_validator(mode="before")
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
        >>> params = Flow360BaseModel.from_json(filename='folder/flow360.json') # doctest: +SKIP
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
        json_string = self.model_dump_json()
        model_dict = json.loads(json_string)
        if self.model_config["include_hash"] is True:
            model_dict["hash"] = self._calculate_hash(model_dict)
        with open(filename, "w+", encoding="utf-8") as file_handle:
            json.dump(model_dict, file_handle, indent=4, sort_keys=True)

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
        >>> params = Flow360BaseModel.from_yaml(filename='folder/flow360.yaml') # doctest: +SKIP
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
        json_string = self.model_dump_json()
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

    def append(self, params: Flow360BaseModel, overwrite: bool = False):
        """append parametrs to the model

        Parameters
        ----------
        params : Flow360BaseModel
            Flow360BaseModel parameters to be appended
        overwrite : bool, optional
            Whether to overwrite if key exists, by default False
        """
        additional_config = params.model_dump(exclude_unset=True, exclude_none=True)
        for key, value in additional_config.items():
            if self.__getattribute__(key) and not overwrite:
                log.warning(
                    f'"{key}" already exist in the original model, skipping. Use overwrite=True to overwrite values.'
                )
                continue
            is_frozen = self.model_fields[key].frozen
            if is_frozen is None or is_frozen is False:
                self.__setattr__(key, value)

    @classmethod
    def _generate_docstring(cls) -> str:
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
        for field_name, field in cls.model_fields.items():
            # ignore the type tag
            if field_name == TYPE_TAG_STR:
                continue

            # get data type
            data_type = field.annotation

            # get default values
            default_val = field.get_default()
            if "=" in str(default_val):
                # handle cases where default values are pydantic models
                default_val = f"{default_val.__class__.__name__}({default_val})"
                default_val = (", ").join(default_val.split(" "))

            # make first line: name : type = default
            default_str = "" if field.is_required() else f" = {default_val}"
            doc += f"    {field_name} : {data_type}{default_str}\n"

            # get field metadata
            doc += "        "

            # add units (if present)
            units = None
            if field.json_schema_extra is not None:
                units = field.json_schema_extra.get("units")
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
            description_str = field.description
            if description_str is not None:
                doc += f"{description_str}\n"

        # add in remaining things in the docs
        if original_docstrings:
            doc += "\n"
            doc += original_docstrings

        doc += "\n"
        cls.__doc__ = doc

    def _convert_dimensions_to_solver(
        self,
        params,
        mesh_unit: u.unyt_quantity = None,
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
                # Note: we should not be expecting extra field for SimulationParam?
                require(extra_item.dependency_list, required_by, params)
                self_dict[extra_item.name] = extra_item.value_factory()

        assert mesh_unit is not None

        for property_name, value in self_dict.items():
            if property_name in [COMMENTS, TYPE_TAG_STR] + exclude:
                continue
            loc_name = property_name
            field = self.model_fields.get(property_name)
            if field is not None and field.alias is not None:
                loc_name = field.alias

            if need_conversion(value):
                log.debug(f"   -> need conversion for: {property_name} = {value}")
                flow360_conv_system = unit_converter(
                    value.units.dimensions,
                    mesh_unit,
                    params=params,
                    required_by=[*required_by, loc_name],
                )
                # pylint: disable=no-member
                value.units.registry = flow360_conv_system.registry
                solver_values[property_name] = value.in_base(unit_system="flow360")
                log.debug(f"      converted to: {solver_values[property_name]}")
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
                    params, mesh_unit=mesh_unit, required_by=[*required_by, loc_name]
                )
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, Flow360BaseModel):
                        solver_values[property_name][i] = item.preprocess(
                            params,
                            mesh_unit=mesh_unit,
                            required_by=[*required_by, loc_name, f"{i}"],
                        )

        return self.__class__(**solver_values)
