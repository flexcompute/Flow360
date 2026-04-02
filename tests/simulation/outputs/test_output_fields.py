import pytest

import flow360 as fl
from flow360 import SI_unit_system, u
from flow360.component.simulation.outputs.output_fields import (
    generate_predefined_udf,
    remove_fields_subsumed_by_primitive_vars,
)


@pytest.fixture
def simulation_params():
    """Create a simulation parameters object for testing."""
    with SI_unit_system:
        params = fl.SimulationParams(
            operating_condition=fl.AerospaceCondition(
                velocity_magnitude=100,
            ),
        )
    params._private_set_length_unit(1 * u.m)

    return params


def test_generate_field_udf_with_unit(simulation_params):
    """Test generating UDF expression for a field with units."""
    result = generate_predefined_udf("velocity_m_per_s", simulation_params)

    expected = (
        "double velocity_[3];"
        "velocity_[0] = primitiveVars[1] * velocityScale;"
        "velocity_[1] = primitiveVars[2] * velocityScale;"
        "velocity_[2] = primitiveVars[3] * velocityScale;"
        "velocity_m_per_s[0] = velocity_[0] * 340.2940058082124;"
        "velocity_m_per_s[1] = velocity_[1] * 340.2940058082124;"
        "velocity_m_per_s[2] = velocity_[2] * 340.2940058082124;"
    )

    assert result == expected


def test_generate_field_udf_velocity_components(simulation_params):
    """Test generating UDF expressions for velocity components with units."""

    result = generate_predefined_udf("velocity_x_m_per_s", simulation_params)
    expected = (
        "double velocity_x;"
        "velocity_x = primitiveVars[1] * velocityScale;"
        "velocity_x_m_per_s = velocity_x * 340.2940058082124;"
    )
    assert result == expected

    result = generate_predefined_udf("velocity_y_m_per_s", simulation_params)
    expected = (
        "double velocity_y;"
        "velocity_y = primitiveVars[2] * velocityScale;"
        "velocity_y_m_per_s = velocity_y * 340.2940058082124;"
    )
    assert result == expected

    result = generate_predefined_udf("velocity_z_m_per_s", simulation_params)
    expected = (
        "double velocity_z;"
        "velocity_z = primitiveVars[3] * velocityScale;"
        "velocity_z_m_per_s = velocity_z * 340.2940058082124;"
    )
    assert result == expected


def test_generate_field_velocity_magnitude_no_unit(simulation_params):
    """Test generating UDF expression for velocity magnitude fields."""

    result = generate_predefined_udf("velocity_magnitude", simulation_params)
    expected = (
        "double velocity[3];"
        "velocity[0] = primitiveVars[1];"
        "velocity[1] = primitiveVars[2];"
        "velocity[2] = primitiveVars[3];"
        "velocity_magnitude = magnitude(velocity) * velocityScale;"
    )
    assert result == expected


def test_generate_field_udf_velocity_magnitude(simulation_params):
    """Test generating UDF expression for velocity magnitude."""

    result = generate_predefined_udf("velocity_magnitude_m_per_s", simulation_params)
    expected = (
        "double velocity_magnitude;"
        "double velocity[3];"
        "velocity[0] = primitiveVars[1];"
        "velocity[1] = primitiveVars[2];"
        "velocity[2] = primitiveVars[3];"
        "velocity_magnitude = magnitude(velocity) * velocityScale;"
        "velocity_magnitude_m_per_s = velocity_magnitude * 340.2940058082124;"
    )
    assert result == expected


def test_generate_field_udf_pressure(simulation_params):
    """Test generating UDF expression for pressure fields."""

    result = generate_predefined_udf("pressure_pa", simulation_params)
    expected = (
        "double pressure_;double gamma = 1.4;"
        "pressure_ = (usingLiquidAsMaterial) ? (primitiveVars[4] - 1.0 / gamma) * (velocityScale * velocityScale) : primitiveVars[4];"
        "pressure_pa = pressure_ * 141855.01272652458;"
    )
    assert result == expected


def test_genereate_field_wall_shear_stress_no_unit(simulation_params):
    """Test generating UDF expression for wall shear stress fields."""

    result = generate_predefined_udf("wall_shear_stress_magnitude", simulation_params)
    expected = "wall_shear_stress_magnitude = magnitude(wallShearStress) * (velocityScale * velocityScale);"
    assert result == expected


def test_generate_field_udf_wall_shear_stress(simulation_params):
    """Test generating UDF expression for wall shear stress fields."""

    result = generate_predefined_udf("wall_shear_stress_magnitude_pa", simulation_params)
    expected = (
        "double wall_shear_stress_magnitude;"
        "wall_shear_stress_magnitude = magnitude(wallShearStress) * (velocityScale * velocityScale);"
        "wall_shear_stress_magnitude_pa = wall_shear_stress_magnitude * 141855.01272652458;"
    )
    assert result == expected


def test_generate_field_udf_precedence(simulation_params):
    """Test that longer matching keys take precedence."""
    result = generate_predefined_udf("velocity_x", simulation_params)
    expected = "velocity_x = primitiveVars[1] * velocityScale;"
    assert result == expected


def test_generate_field_udf_no_match(simulation_params):
    """Test behavior when no matching predefined expression is found."""
    result = generate_predefined_udf("non_existent_field_for_udf", simulation_params)
    assert result is None


class TestRemoveFieldsSubsumedByPrimitiveVars:
    def test_removes_pressure_and_velocity_when_primitive_vars_present(self):
        fields = ["Cp", "primitiveVars", "pressure", "velocity", "Mach"]
        result = remove_fields_subsumed_by_primitive_vars(fields)
        assert "primitiveVars" in result
        assert "pressure" not in result
        assert "velocity" not in result
        assert "Cp" in result
        assert "Mach" in result

    def test_keeps_pressure_and_velocity_when_no_primitive_vars(self):
        fields = ["pressure", "velocity", "Cp"]
        result = remove_fields_subsumed_by_primitive_vars(fields)
        assert result == ["pressure", "velocity", "Cp"]

    def test_preserves_velocity_magnitude_when_velocity_removed(self):
        fields = ["primitiveVars", "velocity", "velocity_magnitude", "pressure"]
        result = remove_fields_subsumed_by_primitive_vars(fields)
        assert "velocity" not in result
        assert "pressure" not in result
        assert "velocity_magnitude" in result
        assert "primitiveVars" in result

    def test_no_op_on_empty_list(self):
        assert remove_fields_subsumed_by_primitive_vars([]) == []

    def test_primitive_vars_only(self):
        fields = ["primitiveVars"]
        result = remove_fields_subsumed_by_primitive_vars(fields)
        assert result == ["primitiveVars"]

    def test_removes_pressure_only_when_velocity_absent(self):
        fields = ["primitiveVars", "pressure", "Cp"]
        result = remove_fields_subsumed_by_primitive_vars(fields)
        assert "pressure" not in result
        assert result == ["primitiveVars", "Cp"]

    def test_removes_velocity_only_when_pressure_absent(self):
        fields = ["primitiveVars", "velocity", "Mach"]
        result = remove_fields_subsumed_by_primitive_vars(fields)
        assert "velocity" not in result
        assert result == ["primitiveVars", "Mach"]
