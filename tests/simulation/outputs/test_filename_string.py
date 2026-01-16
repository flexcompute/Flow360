"""Unit tests for FileNameString validation."""

import pydantic as pd
import pytest

from flow360.component.simulation.outputs.outputs import (
    FileNameString,
    SurfaceIntegralOutput,
)


class TestFileNameString:
    """Test suite for FileNameString validation."""

    def test_valid_filenames(self):
        """Test that valid filenames are accepted."""

        class Model(pd.BaseModel):
            name: FileNameString

        # Valid filenames
        valid_names = [
            "surface_integral",
            "test-123",
            "MyFile",
            "output_1",
            "A_B_C",
            "Surface integral output",  # Spaces are allowed
            "test file name",
        ]

        for name in valid_names:
            model = Model(name=name)
            assert model.name == name

    def test_invalid_slash(self):
        """Test that filenames with slashes are rejected."""

        class Model(pd.BaseModel):
            name: FileNameString

        with pytest.raises(
            pd.ValidationError,
            match="Filename contains invalid characters: '/'",
        ):
            Model(name="A/B")

    def test_invalid_null_byte(self):
        """Test that filenames with null bytes are rejected."""

        class Model(pd.BaseModel):
            name: FileNameString

        with pytest.raises(
            pd.ValidationError,
            match="Filename contains invalid characters",
        ):
            Model(name="test\x00file")

    def test_empty_string(self):
        """Test that empty strings are rejected."""

        class Model(pd.BaseModel):
            name: FileNameString

        with pytest.raises(
            pd.ValidationError,
            match="Filename cannot be empty",
        ):
            Model(name="")

    def test_reserved_names(self):
        """Test that reserved names are rejected."""

        class Model(pd.BaseModel):
            name: FileNameString

        with pytest.raises(
            pd.ValidationError,
            match="Filename cannot be '.' \\(reserved name\\)",
        ):
            Model(name=".")

        with pytest.raises(
            pd.ValidationError,
            match="Filename cannot be '..' \\(reserved name\\)",
        ):
            Model(name="..")


class TestSurfaceIntegralOutputName:
    """Test suite for SurfaceIntegralOutput name field with FileNameString."""

    def test_default_name(self):
        """Test that default name is valid."""

        # We can't create a full SurfaceIntegralOutput without entities,
        # but we can test that the default name would pass FileNameString validation
        class Model(pd.BaseModel):
            name: FileNameString = pd.Field("Surface integral output")

        model = Model()
        assert model.name == "Surface integral output"

    def test_invalid_name_with_slash(self):
        """Test that SurfaceIntegralOutput rejects names with slashes."""
        with pytest.raises(
            pd.ValidationError,
            match="Filename contains invalid characters: '/'",
        ):
            SurfaceIntegralOutput(
                name="A/B",
                output_fields=["PressureForce"],
                entities=[],
            )

    def test_valid_custom_name(self):
        """Test that valid custom names are accepted."""
        # This will fail on entities validation, but name should be OK
        with pytest.raises(pd.ValidationError) as exc_info:
            SurfaceIntegralOutput(
                name="my_custom_output",
                output_fields=["PressureForce"],
                entities=[],
            )

        # Check that the error is about entities, not name
        errors = exc_info.value.errors()
        name_errors = [e for e in errors if e["loc"] == ("name",)]
        assert len(name_errors) == 0, "Name validation should have passed"
