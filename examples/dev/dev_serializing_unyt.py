from __future__ import annotations

from typing import List, Annotated, Literal

import pydantic as pd
import unyt as u
from pydantic import PlainSerializer, PlainValidator
from pydantic_core import core_schema


def _has_dimensions(quant, dim):
    try:
        arg_dim = quant.units.dimensions
    except AttributeError:
        arg_dim = u.dimensionless
    return arg_dim == dim


def _unit_object_parser(value, unyt_types: List[type]):
    if isinstance(value, dict) and "units" in value:
        if "value" in value:
            for unyt_type in unyt_types:
                try:
                    return unyt_type(value["value"], value["units"])
                except u.exceptions.UnitParseError:
                    pass
        else:
            raise TypeError(
                f"Dimensioned type instance {value} expects a 'value' field which was not given"
            )
    return value


def _is_unit_validator(value):
    if isinstance(value, str):
        try:
            value = u.Unit(value)
        except u.exceptions.UnitParseError as err:
            raise TypeError(str(err)) from err
    return value


def _unit_array_validator(value, dim):
    if not _has_dimensions(value, dim):
        if any(_has_dimensions(item, dim) for item in value):
            raise TypeError(
                f"arg '{value}' has unit provided per component, "
                "instead provide dimension for entire array."
            )
    return value


def validate(value):
    """
    Validator for unyt value
    """

    value = _unit_object_parser(value, [u.unyt_quantity])
    value = _is_unit_validator(value)

    if isinstance(value, u.Unit):
        return 1.0 * value

    return value


def serialize(value):
    """
    Serializer for unyt value
    """

    return {"value": str(value["data"].value), "units": str(value["data"].units)}


class _DimensionedField(pd.BaseModel):
    data: u.unyt_quantity = pd.Field()
    dim_type: Literal["length"] = pd.Field()

    @classmethod
    def validate(cls, value):
        """
        Validator for value
        """

        try:
            value = _unit_object_parser(value, [u.unyt_quantity])
        except TypeError as err:
            raise pd.ValidationError.from_exception_data('validation error', [])

        if isinstance(value, u.Unit):
            return {"data": 1.0 * value}

        return {"data": value}

    @classmethod
    def __get_pydantic_core_schema__(cls, *args, **kwargs) -> pd.CoreSchema:
        return core_schema.no_info_plain_validator_function(cls.validate)


DimensionedField = Annotated[
    _DimensionedField,
    PlainSerializer(serialize),
]


class _OtherField(_DimensionedField):
    dim_type: Literal["time"] = pd.Field()


OtherField = Annotated[
    _OtherField,
    PlainSerializer(serialize),
]


class DataModel(pd.BaseModel):
    dimensioned_field: OtherField = pd.Field()


data = DataModel(dimensioned_field=1*u.m)


data_json = data.model_dump_json(indent=2)

print(data_json)