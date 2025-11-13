"""
Unit tests for validation_utils.py customize_model_validator_error function

This test suite validates the behavior of customize_model_validator_error with nested
Pydantic models, focusing on three-layer model structures and multiple error scenarios.

Key Findings (documented through tests):
- Pydantic validates models bottom-up (innermost to outermost)
- Multiple errors from list items are collected in a single ValidationError
- When errors occur at multiple layers, innermost errors are captured first
- Parent model validators only run after all nested models validate successfully

Test Organization:
=================

1. Helper Functions: Validation assertion utilities
2. Shared Fixtures: Reusable model structures
3. Test Cases: Organized by complexity and validation scenarios
"""

from typing import Optional

import pytest
from pydantic import BaseModel, ValidationError, model_validator

from flow360.component.simulation.validation.validation_utils import (
    customize_model_validator_error,
)

# ============================================================================
# Helper Functions
# ============================================================================


def assert_validation_error(
    error: ValidationError,
    expected_loc: tuple,
    expected_msg_contains: str,
    expected_input=None,
    expected_type: str = "value_error",
):
    """Helper function to assert common validation error properties"""
    errors = error.errors()
    assert len(errors) == 1
    assert errors[0]["loc"] == expected_loc
    assert expected_msg_contains in errors[0]["msg"]
    assert errors[0]["type"] == expected_type
    if expected_input is not None:
        assert errors[0]["input"] == expected_input


def assert_multiple_validation_errors(
    error: ValidationError,
    expected_errors: list[dict],
):
    """
    Helper function to assert multiple validation errors.

    Args:
        error: The ValidationError exception
        expected_errors: List of dicts with keys: 'loc', 'msg_contains', 'input' (optional)
    """
    errors = error.errors()
    assert len(errors) == len(
        expected_errors
    ), f"Expected {len(expected_errors)} errors but got {len(errors)}"

    for i, expected in enumerate(expected_errors):
        assert (
            errors[i]["loc"] == expected["loc"]
        ), f"Error {i}: Expected loc {expected['loc']}, got {errors[i]['loc']}"
        assert (
            expected["msg_contains"] in errors[i]["msg"]
        ), f"Error {i}: Expected '{expected['msg_contains']}' in '{errors[i]['msg']}'"
        if "input" in expected:
            assert (
                errors[i]["input"] == expected["input"]
            ), f"Error {i}: Expected input {expected['input']}, got {errors[i]['input']}"


# ============================================================================
# Shared Fixtures
# ============================================================================


@pytest.fixture
def simple_model():
    """Single-layer model with basic validation"""

    class SimpleModel(BaseModel):
        name: str
        value: int

        @model_validator(mode="after")
        def validate_name(self):
            if self.name == "invalid":
                raise customize_model_validator_error(
                    self,
                    loc=("name",),
                    message="name cannot be 'invalid'",
                    input_value=self.name,
                )
            return self

    return SimpleModel


@pytest.fixture
def list_model():
    """Model with list validation for testing list indices in error locations"""

    class ListModel(BaseModel):
        name: str
        outputs: list[dict]

        @model_validator(mode="after")
        def validate_outputs(self):
            for i, output in enumerate(self.outputs):
                if output.get("type") == "invalid":
                    raise customize_model_validator_error(
                        self,
                        loc=("outputs", i, "type"),
                        message=f"output type 'invalid' at index {i}",
                        input_value=output.get("type"),
                    )
            return self

    return ListModel


@pytest.fixture
def two_layer_models():
    """Two-layer parent/child models with validation at both levels"""

    class ChildModel(BaseModel):
        nested_value: int

        @model_validator(mode="after")
        def validate_nested(self):
            if self.nested_value < 0:
                raise customize_model_validator_error(
                    self,
                    loc=("nested_value",),
                    message="nested_value must be non-negative",
                    input_value=self.nested_value,
                )
            return self

    class ParentModel(BaseModel):
        items: list[ChildModel]
        config_name: str

        @model_validator(mode="after")
        def validate_config_name(self):
            if self.config_name == "forbidden":
                raise customize_model_validator_error(
                    self,
                    loc=("config_name",),
                    message="config_name cannot be 'forbidden'",
                    input_value=self.config_name,
                )
            return self

    return {"child": ChildModel, "parent": ParentModel}


@pytest.fixture
def three_layer_models():
    """
    Generic three-layer model structure for comprehensive testing.
    Used for: value validation, multiple errors, cascade validation, cross-field validation
    """

    class InnerConfig(BaseModel):
        threshold: float

        @model_validator(mode="after")
        def validate_threshold(self):
            if self.threshold <= 0:
                raise customize_model_validator_error(
                    self,
                    loc=("threshold",),
                    message="Inner: threshold must be positive",
                    input_value=self.threshold,
                )
            return self

    class MiddleConfig(BaseModel):
        config_id: int
        inner_configs: list[InnerConfig]

        @model_validator(mode="after")
        def validate_config_id(self):
            if self.config_id < 0:
                raise customize_model_validator_error(
                    self,
                    loc=("config_id",),
                    message="Middle: config_id must be non-negative",
                    input_value=self.config_id,
                )
            return self

    class OuterConfig(BaseModel):
        middle_configs: list[MiddleConfig]

    return {"inner": InnerConfig, "middle": MiddleConfig, "outer": OuterConfig}


@pytest.fixture
def three_layer_item_models():
    """Three-layer Item/Section/Configuration for range validation testing"""

    class Item(BaseModel):
        name: str
        value: float

        @model_validator(mode="after")
        def validate_item(self):
            if self.value < 0:
                raise customize_model_validator_error(
                    self,
                    loc=("value",),
                    message=f"Item '{self.name}' has negative value",
                    input_value=self.value,
                )
            if self.value > 1000:
                raise customize_model_validator_error(
                    self,
                    loc=("value",),
                    message=f"Item '{self.name}' exceeds maximum",
                    input_value=self.value,
                )
            return self

    class Section(BaseModel):
        section_name: str
        items: list[Item]

        @model_validator(mode="after")
        def validate_section(self):
            if not self.items:
                raise customize_model_validator_error(
                    self,
                    loc=("items",),
                    message=f"Section '{self.section_name}' must have items",
                    input_value=self.items,
                )
            return self

        @model_validator(mode="after")
        def validate_section_name(self):
            if self.section_name == "empty":
                raise customize_model_validator_error(
                    self,
                    loc=("section_name",),
                    message="Section name cannot be empty",
                    input_value=self.section_name,
                )
            return self

    class Configuration(BaseModel):
        config_name: str
        sections: list[Section]

        @model_validator(mode="after")
        def validate_configuration(self):
            if not self.sections:
                raise customize_model_validator_error(
                    self,
                    loc=("sections",),
                    message="Configuration must have sections",
                    input_value=self.sections,
                )
            return self

    return {"item": Item, "section": Section, "configuration": Configuration}


@pytest.fixture
def three_layer_parameter_models():
    """Three-layer Parameter/Group/Set for cross-field validation testing"""

    class Parameter(BaseModel):
        param_name: str
        min_value: float
        max_value: float

        @model_validator(mode="after")
        def validate_parameter(self):
            if self.min_value >= self.max_value:
                raise customize_model_validator_error(
                    self,
                    loc=("min_value",),
                    message=f"'{self.param_name}' min must be < max",
                    input_value=self.min_value,
                )
            return self

    class ParameterGroup(BaseModel):
        group_name: str
        parameters: list[Parameter]
        enabled: bool = True

        @model_validator(mode="after")
        def validate_group(self):
            if self.enabled and not self.parameters:
                raise customize_model_validator_error(
                    self,
                    loc=("parameters",),
                    message=f"Enabled '{self.group_name}' needs parameters",
                    input_value=self.parameters,
                )
            return self

    class ParameterSet(BaseModel):
        set_name: str
        groups: list[ParameterGroup]

    return {"parameter": Parameter, "group": ParameterGroup, "set": ParameterSet}


# ============================================================================
# Test Cases - Basic Functionality
# ============================================================================


def test_simple_validation_error(simple_model):
    """Test basic validation error with simple field"""
    with pytest.raises(ValidationError) as exc_info:
        simple_model(name="invalid", value=10)

    assert_validation_error(
        exc_info.value,
        expected_loc=("name",),
        expected_msg_contains="name cannot be 'invalid'",
        expected_input="invalid",
    )


def test_simple_validation_success(simple_model):
    """Test that valid data passes validation"""
    model = simple_model(name="valid", value=10)
    assert model.name == "valid"
    assert model.value == 10


def test_error_properties(simple_model):
    """Test that error has correct title and type"""
    with pytest.raises(ValidationError) as exc_info:
        simple_model(name="invalid", value=10)

    error = exc_info.value
    assert error.title == "SimpleModel"
    assert error.errors()[0]["type"] == "value_error"
    assert "ctx" in error.errors()[0]
    assert "error" in error.errors()[0]["ctx"]


# ============================================================================
# Test Cases - Two-Layer Nested Models
# ============================================================================


def test_nested_validation_error(two_layer_models):
    """Test validation error in nested model"""
    ParentModel = two_layer_models["parent"]

    with pytest.raises(ValidationError) as exc_info:
        ParentModel(
            config_name="test",
            items=[
                {"nested_value": 1},
                {"nested_value": -5},
            ],
        )

    assert_validation_error(
        exc_info.value,
        expected_loc=("items", 1, "nested_value"),
        expected_msg_contains="nested_value must be non-negative",
        expected_input=-5,
    )


def test_parent_validation_error(two_layer_models):
    """Test validation error in parent model"""
    ParentModel = two_layer_models["parent"]

    with pytest.raises(ValidationError) as exc_info:
        ParentModel(config_name="forbidden", items=[{"nested_value": 1}])

    assert_validation_error(
        exc_info.value,
        expected_loc=("config_name",),
        expected_msg_contains="config_name cannot be 'forbidden'",
        expected_input="forbidden",
    )


def test_pydantic_auto_prepending(two_layer_models):
    """Test that Pydantic automatically prepends parent paths correctly"""
    ChildModel = two_layer_models["child"]

    class MiddleModel(BaseModel):
        inner: ChildModel

    class OuterModel(BaseModel):
        middle: MiddleModel

    # Override child validation for this specific test
    class TestChild(BaseModel):
        nested_value: int

        @model_validator(mode="after")
        def validate_value(self):
            if self.nested_value == 0:
                raise customize_model_validator_error(
                    self,
                    loc=("nested_value",),
                    message="value cannot be zero",
                    input_value=self.nested_value,
                )
            return self

    class TestMiddle(BaseModel):
        inner: TestChild

    class TestOuter(BaseModel):
        middle: TestMiddle

    with pytest.raises(ValidationError) as exc_info:
        TestOuter(middle={"inner": {"nested_value": 0}})

    assert_validation_error(
        exc_info.value,
        expected_loc=("middle", "inner", "nested_value"),
        expected_msg_contains="value cannot be zero",
    )


# ============================================================================
# Test Cases - List Handling
# ============================================================================


def test_list_index_in_location(list_model):
    """Test that list indices are properly included in error location"""
    with pytest.raises(ValidationError) as exc_info:
        list_model(
            name="test",
            outputs=[
                {"type": "valid"},
                {"type": "valid"},
                {"type": "invalid"},
            ],
        )

    assert_validation_error(
        exc_info.value,
        expected_loc=("outputs", 2, "type"),
        expected_msg_contains="output type 'invalid' at index 2",
        expected_input="invalid",
    )


def test_empty_list_validation(list_model):
    """Test validation with empty list doesn't raise error"""
    model = list_model(name="test", outputs=[])
    assert model.name == "test"
    assert model.outputs == []


# ============================================================================
# Test Cases - Multiple Nesting Levels
# ============================================================================


def test_multiple_nested_levels():
    """Test validation error location with multiple nesting levels"""

    class Level3(BaseModel):
        value: int

        @model_validator(mode="after")
        def validate_value(self):
            if self.value == 999:
                raise customize_model_validator_error(
                    self,
                    loc=("value",),
                    message="value cannot be 999",
                    input_value=self.value,
                )
            return self

    class Level2(BaseModel):
        items: list[Level3]

    class Level1(BaseModel):
        nested: list[Level2]

    with pytest.raises(ValidationError) as exc_info:
        Level1(
            nested=[
                {"items": [{"value": 1}, {"value": 2}]},
                {"items": [{"value": 3}, {"value": 999}]},
            ]
        )

    assert_validation_error(
        exc_info.value,
        expected_loc=("nested", 1, "items", 1, "value"),
        expected_msg_contains="value cannot be 999",
    )


# ============================================================================
# Test Cases - Input Value Handling
# ============================================================================


def test_without_input_value_parameter():
    """Test that omitting input_value uses model_dump() as fallback"""

    class ModelWithoutInputValue(BaseModel):
        field1: str
        field2: int

        @model_validator(mode="after")
        def validate_field1(self):
            if self.field1 == "error":
                raise customize_model_validator_error(
                    self,
                    loc=("field1",),
                    message="field1 cannot be 'error'",
                )
            return self

    with pytest.raises(ValidationError) as exc_info:
        ModelWithoutInputValue(field1="error", field2=42)

    error = exc_info.value
    errors = error.errors()

    assert len(errors) == 1
    assert errors[0]["loc"] == ("field1",)
    assert isinstance(errors[0]["input"], dict)
    assert errors[0]["input"]["field1"] == "error"
    assert errors[0]["input"]["field2"] == 42


def test_validation_error_with_none_input_value():
    """Test handling when input_value is explicitly None"""

    class NoneInputModel(BaseModel):
        field: Optional[str] = None

        @model_validator(mode="after")
        def validate_field(self):
            if self.field is None:
                raise customize_model_validator_error(
                    self,
                    loc=("field",),
                    message="field cannot be None",
                    input_value=None,
                )
            return self

    with pytest.raises(ValidationError) as exc_info:
        NoneInputModel(field=None)

    error = exc_info.value
    errors = error.errors()

    assert len(errors) == 1
    assert errors[0]["loc"] == ("field",)
    assert isinstance(errors[0]["input"], dict)


# ============================================================================
# Test Cases - Complex Scenarios
# ============================================================================


def test_custom_error_message_with_special_chars():
    """Test that custom error messages with special characters are preserved"""
    custom_message = "Special chars: @#$% and 'quotes' and \"double quotes\""

    class CustomMessageModel(BaseModel):
        field: str

        @model_validator(mode="after")
        def validate_field(self):
            if self.field == "trigger":
                raise customize_model_validator_error(
                    self,
                    loc=("field",),
                    message=custom_message,
                    input_value=self.field,
                )
            return self

    with pytest.raises(ValidationError) as exc_info:
        CustomMessageModel(field="trigger")

    assert custom_message in exc_info.value.errors()[0]["msg"]


def test_complex_location_tuple():
    """Test validation with complex location tuple including strings and integers"""

    class ComplexLocationModel(BaseModel):
        data: dict

        @model_validator(mode="after")
        def validate_data(self):
            raise customize_model_validator_error(
                self,
                loc=("data", "level1", 0, "level2", 5, "field"),
                message="complex location test",
                input_value="test_value",
            )

    with pytest.raises(ValidationError) as exc_info:
        ComplexLocationModel(data={})

    assert exc_info.value.errors()[0]["loc"] == ("data", "level1", 0, "level2", 5, "field")


# ============================================================================
# Test Cases - Three Layer Models (Using three_layer_item_models fixture)
# ============================================================================


def test_three_layer_error_at_innermost(three_layer_item_models):
    """Test error at innermost layer of three-layer model"""
    Configuration = three_layer_item_models["configuration"]

    with pytest.raises(ValidationError) as exc_info:
        Configuration(
            config_name="test",
            sections=[
                {
                    "section_name": "section1",
                    "items": [
                        {"name": "item1", "value": 10.0},
                        {"name": "item2", "value": -5.0},
                    ],
                }
            ],
        )

    assert_validation_error(
        exc_info.value,
        expected_loc=("sections", 0, "items", 1, "value"),
        expected_msg_contains="negative value",
        expected_input=-5.0,
    )


def test_three_layer_error_at_middle(three_layer_item_models):
    """Test error at middle layer of three-layer model"""
    Configuration = three_layer_item_models["configuration"]

    with pytest.raises(ValidationError) as exc_info:
        Configuration(
            config_name="test",
            sections=[{"section_name": "empty", "items": []}],
        )

    assert_validation_error(
        exc_info.value,
        expected_loc=("sections", 0, "items"),
        expected_msg_contains="must have items",
    )


def test_three_layer_error_at_outermost(three_layer_item_models):
    """Test error at outermost layer of three-layer model"""
    Configuration = three_layer_item_models["configuration"]

    with pytest.raises(ValidationError) as exc_info:
        Configuration(config_name="test", sections=[])

    assert_validation_error(
        exc_info.value,
        expected_loc=("sections",),
        expected_msg_contains="must have sections",
    )


def test_three_layer_valid_model(three_layer_item_models):
    """Test that valid three-layer model passes validation"""
    Configuration = three_layer_item_models["configuration"]

    config = Configuration(
        config_name="valid",
        sections=[
            {
                "section_name": "section1",
                "items": [
                    {"name": "item1", "value": 10.0},
                    {"name": "item2", "value": 20.0},
                ],
            },
        ],
    )

    assert config.config_name == "valid"
    assert len(config.sections) == 1
    assert len(config.sections[0].items) == 2
    assert config.sections[0].items[0].value == 10.0


def test_three_layer_cross_field_validation(three_layer_parameter_models):
    """Test three-layer model with cross-field validation"""
    ParameterSet = three_layer_parameter_models["set"]

    # Test cross-field validation error
    with pytest.raises(ValidationError) as exc_info:
        ParameterSet(
            set_name="test",
            groups=[
                {
                    "group_name": "group1",
                    "parameters": [{"param_name": "p1", "min_value": 100.0, "max_value": 50.0}],
                }
            ],
        )

    assert_validation_error(
        exc_info.value,
        expected_loc=("groups", 0, "parameters", 0, "min_value"),
        expected_msg_contains="min must be < max",
    )

    # Test valid model with disabled group
    param_set = ParameterSet(
        set_name="valid",
        groups=[
            {
                "group_name": "group1",
                "parameters": [{"param_name": "p1", "min_value": 0.0, "max_value": 100.0}],
            },
            {
                "group_name": "group2",
                "enabled": False,
                "parameters": [],
            },
        ],
    )

    assert param_set.groups[0].enabled is True
    assert param_set.groups[1].enabled is False


# ============================================================================
# Test Cases - Multiple Errors Across Layers
# ============================================================================


def test_multiple_errors_in_innermost_layer(three_layer_item_models):
    """Test that Pydantic collects ALL errors from multiple items in a list"""
    Configuration = three_layer_item_models["configuration"]

    # Test: Multiple items fail in innermost layer - ALL errors collected!
    with pytest.raises(ValidationError) as exc_info:
        Configuration(
            config_name="test",
            sections=[
                {
                    "section_name": "section1",
                    "items": [
                        {"name": "item1", "value": -5.0},  # Error 1: negative
                        {"name": "item2", "value": -10.0},  # Error 2: negative
                        {"name": "item3", "value": 1500.0},  # Error 3: exceeds maximum
                    ],
                }
            ],
        )

    # Pydantic collects all errors from list items
    assert_multiple_validation_errors(
        exc_info.value,
        [
            {
                "loc": ("sections", 0, "items", 0, "value"),
                "msg_contains": "negative value",
                "input": -5.0,
            },
            {
                "loc": ("sections", 0, "items", 1, "value"),
                "msg_contains": "negative value",
                "input": -10.0,
            },
            {
                "loc": ("sections", 0, "items", 2, "value"),
                "msg_contains": "exceeds maximum",
                "input": 1500.0,
            },
        ],
    )


def test_multiple_errors_different_items_same_layer(three_layer_item_models):
    """Test error capture when multiple different items fail validation"""
    Configuration = three_layer_item_models["configuration"]

    # Multiple items at different positions fail - both errors collected
    with pytest.raises(ValidationError) as exc_info:
        Configuration(
            config_name="test",
            sections=[
                {
                    "section_name": "section1",
                    "items": [
                        {"name": "item1", "value": 10.0},  # Valid
                        {"name": "item2", "value": -5.0},  # Invalid - error 1
                        {"name": "item3", "value": 20.0},  # Valid
                        {"name": "item4", "value": -3.0},  # Invalid - error 2
                    ],
                }
            ],
        )

    # Both invalid items' errors are captured
    assert_multiple_validation_errors(
        exc_info.value,
        [
            {
                "loc": ("sections", 0, "items", 1, "value"),
                "msg_contains": "negative value",
                "input": -5.0,
            },
            {
                "loc": ("sections", 0, "items", 3, "value"),
                "msg_contains": "negative value",
                "input": -3.0,
            },
        ],
    )


def test_errors_in_middle_and_inner_layers(three_layer_item_models):
    """Test validation order: inner models validate BEFORE parent model validators"""
    Configuration = three_layer_item_models["configuration"]

    # Case 1: Inner layer error (middle layer validation passes)
    with pytest.raises(ValidationError) as exc_info:
        Configuration(
            config_name="test",
            sections=[
                {
                    "section_name": "section1",  # Middle validation will pass
                    "items": [{"name": "item1", "value": -1.0}],  # Inner validation will fail
                }
            ],
        )

    assert_validation_error(
        exc_info.value,
        expected_loc=("sections", 0, "items", 0, "value"),
        expected_msg_contains="negative value",
        expected_input=-1.0,
    )

    # Case 2: Middle layer error (inner layer validation passes)
    with pytest.raises(ValidationError) as exc_info:
        Configuration(
            config_name="test",
            sections=[
                {
                    "section_name": "empty",  # Middle validation will fail
                    "items": [{"name": "item1", "value": 1.0}],  # Inner validation will pass
                }
            ],
        )

    assert_validation_error(
        exc_info.value,
        expected_loc=("sections", 0, "section_name"),
        expected_msg_contains="Section name cannot be empty",
        expected_input="empty",
    )

    # Case 3: IMPORTANT - Inner validates BEFORE middle!
    # Both layers would fail, but inner error is raised first (bottom-up validation)
    with pytest.raises(ValidationError) as exc_info:
        Configuration(
            config_name="test",
            sections=[
                {
                    "section_name": "empty",  # Middle would fail (empty items)
                    "items": [{"name": "item1", "value": -1.0}],  # Inner validates first and fails!
                }
            ],
        )

    # Inner error is raised before middle validator runs
    assert_validation_error(
        exc_info.value,
        expected_loc=("sections", 0, "items", 0, "value"),
        expected_msg_contains="negative value",
        expected_input=-1.0,
    )

    # Case 4: Only middle layer fails (inner passes)
    with pytest.raises(ValidationError) as exc_info:
        Configuration(
            config_name="test",
            sections=[
                {
                    "section_name": "empty",  # Middle validation will fail
                    "items": [],  # Empty items - middle layer error
                }
            ],
        )

    assert_validation_error(
        exc_info.value,
        expected_loc=("sections", 0, "items"),
        expected_msg_contains="must have items",
    )


def test_multiple_middle_layer_errors_with_inner_errors(three_layer_models):
    """
    Test validation with errors in both middle and inner layers across multiple items.
    Demonstrates that Pydantic validates bottom-up and collects ALL errors.
    """
    OuterConfig = three_layer_models["outer"]

    # IMPORTANT: Both inner and middle errors are captured!
    # Item 0: Inner fails, Item 1: Middle fails - BOTH errors collected
    with pytest.raises(ValidationError) as exc_info:
        OuterConfig(
            middle_configs=[
                {
                    "config_id": -1,  # Middle would fail
                    "inner_configs": [{"threshold": -0.5}],  # Inner validates FIRST and fails
                },
                {
                    "config_id": -2,  # Middle fails (inner passes so middle validator runs)
                    "inner_configs": [{"threshold": 1.0}],  # Inner passes
                },
            ]
        )

    # Both errors are collected: inner from item 0, middle from item 1
    assert_multiple_validation_errors(
        exc_info.value,
        [
            {
                "loc": ("middle_configs", 0, "inner_configs", 0, "threshold"),
                "msg_contains": "Inner: threshold must be positive",
                "input": -0.5,
            },
            {
                "loc": ("middle_configs", 1, "config_id"),
                "msg_contains": "Middle: config_id must be non-negative",
                "input": -2,
            },
        ],
    )

    # Case 2: Multiple inner errors across different middle items
    with pytest.raises(ValidationError) as exc_info:
        OuterConfig(
            middle_configs=[
                {
                    "config_id": 1,  # Middle passes
                    "inner_configs": [
                        {"threshold": 1.0},  # Valid
                        {"threshold": -0.5},  # Inner fails
                    ],
                },
                {
                    "config_id": 2,  # Middle passes
                    "inner_configs": [
                        {"threshold": -1.0},  # Inner fails
                        {"threshold": 1.0},  # Valid
                    ],
                },
            ]
        )

    # Both inner errors are collected
    assert_multiple_validation_errors(
        exc_info.value,
        [
            {
                "loc": ("middle_configs", 0, "inner_configs", 1, "threshold"),
                "msg_contains": "Inner: threshold must be positive",
                "input": -0.5,
            },
            {
                "loc": ("middle_configs", 1, "inner_configs", 0, "threshold"),
                "msg_contains": "Inner: threshold must be positive",
                "input": -1.0,
            },
        ],
    )


def test_three_layer_cascade_validation_order():
    """Test validation order through three layers to understand error capture behavior"""

    class Level3(BaseModel):
        l3_value: str

        @model_validator(mode="after")
        def validate_l3(self):
            if self.l3_value == "error_l3":
                raise customize_model_validator_error(
                    self,
                    loc=("l3_value",),
                    message="Level 3 validation failed",
                    input_value=self.l3_value,
                )
            return self

    class Level2(BaseModel):
        l2_value: str
        level3_items: list[Level3]

        @model_validator(mode="after")
        def validate_l2(self):
            if self.l2_value == "error_l2":
                raise customize_model_validator_error(
                    self,
                    loc=("l2_value",),
                    message="Level 2 validation failed",
                    input_value=self.l2_value,
                )
            return self

    class Level1(BaseModel):
        l1_value: str
        level2_items: list[Level2]

        @model_validator(mode="after")
        def validate_l1(self):
            if self.l1_value == "error_l1":
                raise customize_model_validator_error(
                    self,
                    loc=("l1_value",),
                    message="Level 1 validation failed",
                    input_value=self.l1_value,
                )
            return self

    # Scenario 1: Only Level 3 (innermost) fails
    with pytest.raises(ValidationError) as exc_info:
        Level1(
            l1_value="valid",
            level2_items=[
                {
                    "l2_value": "valid",
                    "level3_items": [
                        {"l3_value": "valid"},
                        {"l3_value": "error_l3"},  # Innermost error
                    ],
                }
            ],
        )

    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert errors[0]["loc"] == ("level2_items", 0, "level3_items", 1, "l3_value")
    assert "Level 3 validation failed" in errors[0]["msg"]

    # Scenario 2: CRITICAL - Both layers would fail but Level 3 validates FIRST (bottom-up)
    with pytest.raises(ValidationError) as exc_info:
        Level1(
            l1_value="valid",
            level2_items=[
                {
                    "l2_value": "error_l2",  # Would fail, but L3 validates first!
                    "level3_items": [{"l3_value": "error_l3"}],  # L3 validates FIRST (bottom-up)
                }
            ],
        )

    errors = exc_info.value.errors()
    assert len(errors) == 1
    # Level 3 error is captured because it validates before Level 2!
    assert errors[0]["loc"] == ("level2_items", 0, "level3_items", 0, "l3_value")
    assert "Level 3 validation failed" in errors[0]["msg"]

    # Scenario 2b: Only Level 2 (middle) fails (Level 3 passes)
    with pytest.raises(ValidationError) as exc_info:
        Level1(
            l1_value="valid",
            level2_items=[
                {
                    "l2_value": "error_l2",  # Middle fails
                    "level3_items": [
                        {"l3_value": "valid"}  # Level 3 passes, so Level 2 validator runs
                    ],
                }
            ],
        )

    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert errors[0]["loc"] == ("level2_items", 0, "l2_value")
    assert "Level 2 validation failed" in errors[0]["msg"]

    # Scenario 3: All layers would fail - Level 3 (innermost) error captured (bottom-up!)
    with pytest.raises(ValidationError) as exc_info:
        Level1(
            l1_value="error_l1",  # Would fail
            level2_items=[
                {
                    "l2_value": "error_l2",  # Would fail
                    "level3_items": [{"l3_value": "error_l3"}],  # L3 validates FIRST!
                }
            ],
        )

    errors = exc_info.value.errors()
    assert len(errors) == 1
    # Even when all would fail, innermost layer validates first!
    assert errors[0]["loc"] == ("level2_items", 0, "level3_items", 0, "l3_value")
    assert "Level 3 validation failed" in errors[0]["msg"]

    # Scenario 3b: Only Level 1 (outer) fails (inner layers pass)
    with pytest.raises(ValidationError) as exc_info:
        Level1(
            l1_value="error_l1",  # Outer fails
            level2_items=[
                {
                    "l2_value": "valid",  # L2 passes
                    "level3_items": [{"l3_value": "valid"}],  # L3 passes
                }
            ],
        )

    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert errors[0]["loc"] == ("l1_value",)
    assert "Level 1 validation failed" in errors[0]["msg"]

    # Scenario 4: Multiple list items with errors - BOTH are collected
    with pytest.raises(ValidationError) as exc_info:
        Level1(
            l1_value="valid",
            level2_items=[
                {
                    "l2_value": "valid",
                    "level3_items": [{"l3_value": "error_l3"}],  # Item 0: L3 fails
                },
                {
                    "l2_value": "error_l2",  # Item 1: L2 fails
                    "level3_items": [{"l3_value": "valid"}],
                },
            ],
        )

    # Both errors are collected from different list items!
    errors = exc_info.value.errors()
    assert len(errors) == 2
    assert errors[0]["loc"] == ("level2_items", 0, "level3_items", 0, "l3_value")
    assert "Level 3 validation failed" in errors[0]["msg"]
    assert errors[1]["loc"] == ("level2_items", 1, "l2_value")
    assert "Level 2 validation failed" in errors[1]["msg"]


def test_complex_three_layer_multiple_error_scenarios():
    """Test complex scenarios with multiple potential errors across all three layers"""

    class Metric(BaseModel):
        name: str
        min_val: float
        max_val: float

        @model_validator(mode="after")
        def validate_metric(self):
            # Multiple validation rules in innermost layer
            if self.min_val >= self.max_val:
                raise customize_model_validator_error(
                    self,
                    loc=("min_val",),
                    message=f"Metric '{self.name}': min must be < max",
                    input_value=self.min_val,
                )
            if self.min_val < 0:
                raise customize_model_validator_error(
                    self,
                    loc=("min_val",),
                    message=f"Metric '{self.name}': min cannot be negative",
                    input_value=self.min_val,
                )
            if self.max_val > 1000:
                raise customize_model_validator_error(
                    self,
                    loc=("max_val",),
                    message=f"Metric '{self.name}': max exceeds limit",
                    input_value=self.max_val,
                )
            return self

    class MetricGroup(BaseModel):
        group_name: str
        metrics: list[Metric]
        is_active: bool = True

        @model_validator(mode="after")
        def validate_group(self):
            if self.is_active and len(self.metrics) == 0:
                raise customize_model_validator_error(
                    self,
                    loc=("metrics",),
                    message=f"Group '{self.group_name}': active group needs metrics",
                    input_value=self.metrics,
                )
            if len(self.metrics) > 10:
                raise customize_model_validator_error(
                    self,
                    loc=("metrics",),
                    message=f"Group '{self.group_name}': too many metrics",
                    input_value=self.metrics,
                )
            return self

    class MetricConfig(BaseModel):
        config_version: int
        groups: list[MetricGroup]

        @model_validator(mode="after")
        def validate_config(self):
            if self.config_version < 1:
                raise customize_model_validator_error(
                    self,
                    loc=("config_version",),
                    message="Config version must be >= 1",
                    input_value=self.config_version,
                )
            if len(self.groups) == 0:
                raise customize_model_validator_error(
                    self,
                    loc=("groups",),
                    message="Config must have at least one group",
                    input_value=self.groups,
                )
            return self

    # Test: Inner layer error (min >= max)
    with pytest.raises(ValidationError) as exc_info:
        MetricConfig(
            config_version=1,
            groups=[
                {
                    "group_name": "group1",
                    "metrics": [
                        {"name": "m1", "min_val": 100.0, "max_val": 50.0},  # min >= max
                    ],
                }
            ],
        )

    assert_validation_error(
        exc_info.value,
        expected_loc=("groups", 0, "metrics", 0, "min_val"),
        expected_msg_contains="min must be < max",
    )

    # Test: Middle layer error (empty active group)
    with pytest.raises(ValidationError) as exc_info:
        MetricConfig(
            config_version=1,
            groups=[
                {
                    "group_name": "empty_group",
                    "is_active": True,
                    "metrics": [],  # Active but empty
                }
            ],
        )

    assert_validation_error(
        exc_info.value,
        expected_loc=("groups", 0, "metrics"),
        expected_msg_contains="active group needs metrics",
    )

    # Test: Outer layer error (invalid version)
    with pytest.raises(ValidationError) as exc_info:
        MetricConfig(
            config_version=0,  # Invalid
            groups=[
                {
                    "group_name": "group1",
                    "metrics": [{"name": "m1", "min_val": 0.0, "max_val": 100.0}],
                }
            ],
        )

    assert_validation_error(
        exc_info.value,
        expected_loc=("config_version",),
        expected_msg_contains="must be >= 1",
    )

    # Test: Valid configuration passes all layers
    config = MetricConfig(
        config_version=1,
        groups=[
            {
                "group_name": "group1",
                "metrics": [
                    {"name": "m1", "min_val": 0.0, "max_val": 100.0},
                    {"name": "m2", "min_val": 10.0, "max_val": 200.0},
                ],
            },
            {
                "group_name": "group2",
                "is_active": False,
                "metrics": [],  # OK because not active
            },
        ],
    )

    assert config.config_version == 1
    assert len(config.groups) == 2
    assert len(config.groups[0].metrics) == 2
