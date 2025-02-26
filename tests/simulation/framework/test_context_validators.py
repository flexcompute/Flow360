from typing import Annotated, Literal, Optional, Union

import pydantic as pd

import flow360.component.simulation.units as u
from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.validation import validation_context
from flow360.log import set_logging_level

set_logging_level("DEBUG")


class Model(Flow360BaseModel):
    a: str
    b: Optional[int] = validation_context.ConditionalField(context=validation_context.SURFACE_MESH)
    c: Optional[str] = validation_context.ConditionalField(context=validation_context.VOLUME_MESH)
    d: Optional[float] = validation_context.ConditionalField(context=validation_context.CASE)
    e: Optional[float] = validation_context.ContextField(context=validation_context.CASE)


class ModelA(Flow360BaseModel):
    type: Literal["A"]


class ModelB(Flow360BaseModel):
    type: Literal["B"]


class BaseModel(Flow360BaseModel):
    m: Model
    c: Optional[Model] = validation_context.ConditionalField(context=validation_context.CASE)
    d: Model = validation_context.ContextField(context=validation_context.CASE)
    e: Union[ModelA, ModelB] = validation_context.ContextField(
        discriminator="type", context=validation_context.CASE
    )


# this data is missing m="Model" fields and is missing d field
test_data1 = dict(m=dict())
# this data is correct
test_data2 = dict(
    m=dict(a="f", b=1, c="d", d=1.2),
    c=dict(a="f", b=1, c="d", d=1.2),
    d=dict(a="f", b=1, c="d", d=1.2),
    e=dict(type="B"),
)
# this data has incorrect type for d->a (no context defined for a but defined for d) and d->c
test_data3 = dict(
    m=dict(a="f", b=1, c="d", d=1.2),
    c=dict(a="f", b=1, c="d", d=1.2),
    d=dict(a=1, c=1),
    e=dict(type="B"),
)


def _test_with_given_context_and_data(context, data: dict, expected_errors):
    try:
        with validation_context.ValidationContext(context):
            BaseModel(**data)
    except pd.ValidationError as err:
        errors = err.errors()
    assert len(errors) == len(expected_errors)
    for err, exp_err in zip(errors, expected_errors):
        assert err["loc"] == exp_err["loc"]
        assert err["type"] == err["type"]
        if "ctx" in exp_err.keys():
            assert err["ctx"]["relevant_for"] == exp_err["ctx"]["relevant_for"]


def test_no_context_validate():
    excpected_errors = [
        {"loc": ("m", "a"), "type": "missing"},
        {"loc": ("d",), "type": "model_type", "ctx": {"relevant_for": ["Case"]}},
        {"loc": ("e",), "type": "model_attributes_type", "ctx": {"relevant_for": ["Case"]}},
    ]
    _test_with_given_context_and_data(None, test_data1, excpected_errors)


def test_with_sm_context_validate():
    excpected_errors = [
        {"loc": ("m", "a"), "type": "missing"},
        {"loc": ("m", "b"), "type": "missing", "ctx": {"relevant_for": ["SurfaceMesh"]}},
        {"loc": ("d",), "type": "model_type", "ctx": {"relevant_for": ["Case"]}},
        {"loc": ("e",), "type": "model_attributes_type", "ctx": {"relevant_for": ["Case"]}},
    ]

    _test_with_given_context_and_data(validation_context.SURFACE_MESH, test_data1, excpected_errors)


def test_with_vm_context_validate():
    excpected_errors = [
        {"loc": ("m", "a"), "type": "missing"},
        {"loc": ("m", "c"), "type": "missing", "ctx": {"relevant_for": ["VolumeMesh"]}},
        {"loc": ("d",), "type": "model_type", "ctx": {"relevant_for": ["Case"]}},
        {"loc": ("e",), "type": "model_attributes_type", "ctx": {"relevant_for": ["Case"]}},
    ]

    _test_with_given_context_and_data(validation_context.VOLUME_MESH, test_data1, excpected_errors)


def test_with_case_context_validate():
    excpected_errors = [
        {"loc": ("m", "a"), "type": "missing"},
        {"loc": ("m", "d"), "type": "missing", "ctx": {"relevant_for": ["Case"]}},
        {"loc": ("c",), "type": "missing", "ctx": {"relevant_for": ["Case"]}},
        {"loc": ("d",), "type": "model_type", "ctx": {"relevant_for": ["Case"]}},
        {"loc": ("e",), "type": "model_attributes_type", "ctx": {"relevant_for": ["Case"]}},
    ]

    _test_with_given_context_and_data(validation_context.CASE, test_data1, excpected_errors)


def test_with_all_context_validate():
    excpected_errors = [
        {"loc": ("m", "a"), "type": "missing"},
        {"loc": ("m", "b"), "type": "missing", "ctx": {"relevant_for": ["SurfaceMesh"]}},
        {"loc": ("m", "c"), "type": "missing", "ctx": {"relevant_for": ["VolumeMesh"]}},
        {"loc": ("m", "d"), "type": "missing", "ctx": {"relevant_for": ["Case"]}},
        {"loc": ("c",), "type": "missing", "ctx": {"relevant_for": ["Case"]}},
        {"loc": ("d",), "type": "model_type", "ctx": {"relevant_for": ["Case"]}},
        {"loc": ("e",), "type": "model_attributes_type", "ctx": {"relevant_for": ["Case"]}},
    ]

    _test_with_given_context_and_data(validation_context.ALL, test_data1, excpected_errors)


def test_with_sm_and_vm_context_validate():
    excpected_errors = [
        {"loc": ("m", "a"), "type": "missing"},
        {"loc": ("m", "b"), "type": "missing", "ctx": {"relevant_for": ["SurfaceMesh"]}},
        {"loc": ("m", "c"), "type": "missing", "ctx": {"relevant_for": ["VolumeMesh"]}},
        {"loc": ("d",), "type": "model_type", "ctx": {"relevant_for": ["Case"]}},
        {"loc": ("e",), "type": "model_attributes_type", "ctx": {"relevant_for": ["Case"]}},
    ]

    try:
        with validation_context.ValidationContext(
            [validation_context.SURFACE_MESH, validation_context.VOLUME_MESH]
        ):
            BaseModel(**test_data1)
    except pd.ValidationError as err:
        errors = err.errors()
        assert len(errors) == len(excpected_errors)
        for err, exp_err in zip(errors, excpected_errors):
            assert err["loc"] == exp_err["loc"]
            assert err["type"] == exp_err["type"]
            if "ctx" in exp_err.keys():
                assert err["ctx"]["relevant_for"] == exp_err["ctx"]["relevant_for"]


def test_with_sm_and_vm_and_case_context_validate():
    excpected_errors = [
        {"loc": ("m", "a"), "type": "missing"},
        {"loc": ("m", "b"), "type": "missing", "ctx": {"relevant_for": ["SurfaceMesh"]}},
        {"loc": ("m", "c"), "type": "missing", "ctx": {"relevant_for": ["VolumeMesh"]}},
        {"loc": ("m", "d"), "type": "missing", "ctx": {"relevant_for": ["Case"]}},
        {"loc": ("c",), "type": "missing", "ctx": {"relevant_for": ["Case"]}},
        {"loc": ("d",), "type": "model_type", "ctx": {"relevant_for": ["Case"]}},
        {"loc": ("e",), "type": "model_attributes_type", "ctx": {"relevant_for": ["Case"]}},
    ]

    try:
        with validation_context.ValidationContext(
            [
                validation_context.SURFACE_MESH,
                validation_context.VOLUME_MESH,
                validation_context.CASE,
            ]
        ):
            BaseModel(**test_data1)
    except pd.ValidationError as err:
        errors = err.errors()
        assert len(errors) == len(excpected_errors)
        for err, exp_err in zip(errors, excpected_errors):
            assert err["loc"] == exp_err["loc"]
            assert err["type"] == exp_err["type"]
            if "ctx" in exp_err.keys():
                assert err["ctx"]["relevant_for"] == exp_err["ctx"]["relevant_for"]


def test_correct_context_validate():

    BaseModel(**test_data2)
    with validation_context.ValidationContext(validation_context.SURFACE_MESH):
        BaseModel(**test_data2)

    with validation_context.ValidationContext(validation_context.VOLUME_MESH):
        BaseModel(**test_data2)

    with validation_context.ValidationContext(validation_context.CASE):
        BaseModel(**test_data2)

    with validation_context.ValidationContext(validation_context.ALL):
        BaseModel(**test_data2)


def test_without_context_validate_not_required():
    excpected_errors = [
        {"loc": ("e",), "type": "float_parsing", "ctx": {"relevant_for": ["Case"]}},
    ]

    try:
        Model(a="a", d=1, e="str")
    except pd.ValidationError as err:
        errors = err.errors()
        assert len(errors) == len(excpected_errors)
    for err, exp_err in zip(errors, excpected_errors):
        assert err["loc"] == exp_err["loc"]
        assert err["type"] == exp_err["type"]
        if "ctx" in exp_err.keys():
            assert err["ctx"]["relevant_for"] == exp_err["ctx"]["relevant_for"]


def test_without_context_validate_not_required_2():
    excpected_errors = [
        {"loc": ("d", "a"), "type": "string_type", "ctx": {"relevant_for": ["Case"]}},
        {"loc": ("d", "c"), "type": "string_type", "ctx": {"relevant_for": ["VolumeMesh"]}},
    ]
    _test_with_given_context_and_data(None, test_data3, excpected_errors)


def test_with_context_validate_required():
    data = dict(a="f", b=1, c=None, d=1.2)
    with validation_context.ValidationContext(validation_context.SURFACE_MESH):
        Model(**data)

    # Become invalid when validating against VOLUME_MESH
    excpected_errors = [
        {"loc": ("c",), "type": "missing", "ctx": {"relevant_for": ["VolumeMesh"]}},
    ]
    try:
        with validation_context.ValidationContext(validation_context.VOLUME_MESH):
            Model(**data)
    except pd.ValidationError as err:
        errors = err.errors()
    assert len(errors) == len(excpected_errors)
    for err, exp_err in zip(errors, excpected_errors):
        assert err["loc"] == exp_err["loc"]
        assert err["type"] == exp_err["type"]
        if "ctx" in exp_err.keys():
            assert err["ctx"]["relevant_for"] == exp_err["ctx"]["relevant_for"]
