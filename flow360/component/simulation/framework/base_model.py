"""Flow360BaseModel class definition."""

from __future__ import annotations

import hashlib
import json
from itertools import chain
from typing import List

import pydantic as pd
import rich
import unyt as u
import yaml
from flow360_schemas.framework.base_model import Flow360BaseModel as _SchemaBaseModel

from flow360.component.simulation.conversion import need_conversion
from flow360.error_messages import do_not_modify_file_manually_msg
from flow360.exceptions import Flow360FileError
from flow360.log import log


def _preprocess_nested_list(value, required_by, params, exclude, flow360_unit_system):
    new_list = []
    for i, item in enumerate(value):
        # Extend the 'required_by' path with the current index.
        new_required_by = required_by + [f"{i}"]
        if isinstance(item, list):
            # Recursively process nested lists.
            new_list.append(
                _preprocess_nested_list(item, new_required_by, params, exclude, flow360_unit_system)
            )
        elif isinstance(item, Flow360BaseModel):
            # Process Flow360BaseModel instances.
            new_list.append(
                item.preprocess(
                    params=params,
                    required_by=new_required_by,
                    exclude=exclude,
                    flow360_unit_system=flow360_unit_system,
                )
            )
        elif need_conversion(item):
            # Convert nested dimensioned values to base unit system
            new_list.append(item.in_base(flow360_unit_system))
        else:
            # Return item unchanged if it doesn't need processing.
            new_list.append(item)
    return new_list


class Flow360BaseModel(_SchemaBaseModel):
    """Base pydantic (V2) model that all Flow360 components inherit from.
    Extends the schema-layer Flow360BaseModel with SDK features:
    file I/O, hash tracking, unit conversion (preprocess), and rich help output.
    """

    def __init__(self, filename: str = None, **kwargs):
        model_dict = self._handle_file(filename=filename, **kwargs)
        super().__init__(**model_dict)

    # -- SDK-only methods: dict / file handling --

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

    def copy(self, update=None, **kwargs) -> Flow360BaseModel:
        """Copy a Flow360BaseModel.  With ``deep=True`` as default."""
        if "deep" in kwargs and kwargs["deep"] is False:
            raise ValueError("Can't do shallow copy of component, set `deep=True` in copy().")
        new_copy = pd.BaseModel.model_copy(self, update=update, deep=True, **kwargs)
        data = new_copy.model_dump(exclude={"private_attribute_id"})
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
        json_string = self.model_dump_json(exclude_none=True, **kwargs)
        model_dict = json.loads(json_string)
        if self.model_config["include_hash"] is True:
            model_dict["hash"] = self._calculate_hash(model_dict)
        with open(filename, "w+", encoding="utf-8") as file_handle:
            json.dump(model_dict, file_handle, indent=4, sort_keys=True)

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
        json_string = self.model_dump_json(exclude_none=True, **kwargs)
        model_dict = json.loads(json_string)
        if self.model_config["include_hash"]:
            model_dict["hash"] = self._calculate_hash(model_dict)
        with open(filename, "w+", encoding="utf-8") as file_handle:
            yaml.dump(model_dict, file_handle, indent=4, sort_keys=True)

    @classmethod
    def _handle_dict_with_hash(cls, model_dict):
        """
        Handle dictionary input for the model.
        1. Pop the hash.
        2. Check file manipulation.
        """
        hash_from_input = model_dict.pop("hash", None)
        if hash_from_input is not None:
            if hash_from_input != cls._calculate_hash(model_dict):
                log.warning(do_not_modify_file_manually_msg)
        return model_dict

    @classmethod
    def _calculate_hash(cls, model_dict):
        def remove_private_attribute_id(obj):
            """
            Recursively remove all 'private_attribute_id' keys from the data structure.
            This ensures hash consistency when private_attribute_id contains UUID4 values
            that change between runs.
            """
            if isinstance(obj, dict):
                # Create new dict excluding 'private_attribute_id' keys
                return {
                    key: remove_private_attribute_id(value)
                    for key, value in obj.items()
                    if key != "private_attribute_id"
                }
            if isinstance(obj, list):
                # Recursively process list elements
                return [remove_private_attribute_id(item) for item in obj]
            # Return other types as-is (maintains reference for immutable objects)
            return obj

        # Remove private_attribute_id before calculating hash
        cleaned_dict = remove_private_attribute_id(model_dict)
        hasher = hashlib.sha256()
        json_string = json.dumps(cleaned_dict, sort_keys=True)
        hasher.update(json_string.encode("utf-8"))
        return hasher.hexdigest()

    # pylint: disable=too-many-arguments, too-many-locals, too-many-branches
    def _nondimensionalization(
        self,
        *,
        exclude: List[str] = None,
        required_by: List[str] = None,
        flow360_unit_system: u.UnitSystem = None,
    ) -> dict:
        solver_values = {}
        self_dict = self.__dict__

        if exclude is None:
            exclude = []

        if required_by is None:
            required_by = []

        additional_fields = {}

        for property_name, value in chain(self_dict.items(), additional_fields.items()):
            if need_conversion(value) and property_name not in exclude:
                solver_values[property_name] = value.in_base(flow360_unit_system)
            else:
                solver_values[property_name] = value

        return solver_values

    def preprocess(
        self,
        *,
        params=None,
        exclude: List[str] = None,
        required_by: List[str] = None,
        flow360_unit_system: u.UnitSystem = None,
    ) -> Flow360BaseModel:
        """
        Loops through all fields, for Flow360BaseModel runs .preprocess() recursively. For dimensioned value performs

        unit conversion to flow360_base system.

        Parameters
        ----------
        params : SimulationParams
            Full config definition as Flow360Params.

        mesh_unit: LengthType.Positive
            The length represented by 1 unit length in the mesh.

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
                solver_values[property_name] = value.preprocess(
                    params=params,
                    required_by=[*required_by, loc_name],
                    exclude=exclude,
                    flow360_unit_system=flow360_unit_system,
                )
            elif isinstance(value, list):
                # Use the helper to handle nested lists.
                solver_values[property_name] = _preprocess_nested_list(
                    value, [loc_name], params, exclude, flow360_unit_system
                )

        return self.__class__(**solver_values)
