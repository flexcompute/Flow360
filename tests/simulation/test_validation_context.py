"""
Unit tests for validation_context.py contextual_field_validator function.

This test suite validates the behavior of contextual_field_validator,
particularly the required_context parameter validation.
"""

import pytest
from pydantic import ValidationError

from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.unit_system import SI_unit_system
from flow360.component.simulation.validation.validation_context import (
    ParamsValidationInfo,
    ValidationContext,
    contextual_field_validator,
)


def test_contextual_field_validator_invalid_required_context_raises_error():
    """Test that invalid required_context attribute names raise ValueError.

    When a typo or non-existent attribute name is passed to required_context,
    the validator should raise a ValueError to catch the mistake early.
    """

    class ModelWithInvalidRequiredContext(Flow360BaseModel):
        value: str = "test"

        @contextual_field_validator("value", mode="after", required_context=["invalid_attribute"])
        @classmethod
        def validate_value(cls, v, param_info: ParamsValidationInfo):
            return v

    mock_context = ValidationContext(
        levels=None, info=ParamsValidationInfo(param_as_dict={}, referenced_expressions=[])
    )

    with SI_unit_system, mock_context:
        with pytest.raises(ValidationError) as exc_info:
            ModelWithInvalidRequiredContext(value="test")

    # Check that the error message contains the expected text
    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert "Invalid validation context attribute: invalid_attribute" in errors[0]["msg"]


def test_contextual_field_validator_valid_required_context_works():
    """Test that valid required_context attribute names work correctly."""

    class ModelWithValidRequiredContext(Flow360BaseModel):
        value: str = "test"

        @contextual_field_validator("value", mode="after", required_context=["output_dict"])
        @classmethod
        def validate_value(cls, v, param_info: ParamsValidationInfo):
            # This should only run when output_dict is not None
            return v.upper()

    mock_context = ValidationContext(
        levels=None, info=ParamsValidationInfo(param_as_dict={}, referenced_expressions=[])
    )
    # Set output_dict to a non-None value
    mock_context.info.output_dict = {}

    with SI_unit_system, mock_context:
        model = ModelWithValidRequiredContext(value="test")

    # Validator should have run and converted to uppercase
    assert model.value == "TEST"


def test_contextual_field_validator_skips_when_required_context_is_none():
    """Test that validator skips when required_context attribute is None."""

    class ModelWithRequiredContext(Flow360BaseModel):
        value: str = "test"

        @contextual_field_validator("value", mode="after", required_context=["output_dict"])
        @classmethod
        def validate_value(cls, v, param_info: ParamsValidationInfo):
            # This should NOT run when output_dict is None
            return v.upper()

    mock_context = ValidationContext(
        levels=None, info=ParamsValidationInfo(param_as_dict={}, referenced_expressions=[])
    )
    # output_dict is None by default

    with SI_unit_system, mock_context:
        model = ModelWithRequiredContext(value="test")

    # Validator should have been skipped, value unchanged
    assert model.value == "test"


def test_contextual_field_validator_multiple_required_context_all_must_exist():
    """Test that all required_context attributes must be valid."""

    class ModelWithMultipleRequiredContext(Flow360BaseModel):
        value: str = "test"

        @contextual_field_validator(
            "value", mode="after", required_context=["output_dict", "nonexistent_attr"]
        )
        @classmethod
        def validate_value(cls, v, param_info: ParamsValidationInfo):
            return v

    mock_context = ValidationContext(
        levels=None, info=ParamsValidationInfo(param_as_dict={}, referenced_expressions=[])
    )
    mock_context.info.output_dict = {}

    with SI_unit_system, mock_context:
        with pytest.raises(ValidationError) as exc_info:
            ModelWithMultipleRequiredContext(value="test")

    # Should fail on the nonexistent attribute
    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert "Invalid validation context attribute: nonexistent_attr" in errors[0]["msg"]
