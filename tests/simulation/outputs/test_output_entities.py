import re

import pydantic as pd
import pytest

from flow360 import SI_unit_system, u
from flow360.component.simulation.outputs.output_entities import Isosurface
from flow360.component.simulation.user_code.core.types import UserVariable
from flow360.component.simulation.user_code.functions import math
from flow360.component.simulation.user_code.variables import solution


def test_isosurface_field_preprocess_expression_and_solver_variable():
    """
    Test the preprocessing in the before_validator for iso field
    """

    # Test that an Isosurface field cannot be defined using an Expression.
    with pytest.raises(
        pd.ValidationError,
        match=re.escape(
            "Expression (solution.vorticity[0]) cannot be directly used as isosurface field, "
            "please define a UserVariable first."
        ),
    ):
        Isosurface(
            name="test_iso_vorticity_component", field=solution.vorticity[0], iso_value=0.8 / u.s
        )

    iso = Isosurface(
        name="test_iso_vorticity_mag",
        field=UserVariable(name="vorticity_component", value=solution.vorticity[0]),
        iso_value=0.8 / u.s,
    )
    assert iso.iso_value == 0.8 / u.s

    with pytest.raises(
        pd.ValidationError,
        match=re.escape(
            "Expression (math.magnitude(solution.velocity)) cannot be directly used as isosurface field, "
            "please define a UserVariable first."
        ),
    ):
        Isosurface(
            name="test_iso_velocity_mag",
            field=math.magnitude(solution.velocity),
            iso_value=0.8 * u.m / u.s,
        )

    # Test that an Isosurface field defined using a UserVariable.
    iso = Isosurface(
        name="test_iso_velocity_mag",
        field=UserVariable(name="velocity_mag", value=math.magnitude(solution.velocity)),
        iso_value=0.8 * u.m / u.s,
    )
    assert iso.iso_value == 0.8 * u.m / u.s

    iso = Isosurface(
        name="test_iso_rho_cgs",
        field=solution.density.in_units(new_name="rho_cgs", new_unit=u.g / u.cm**3),
        iso_value=0.8 * 10**3 * u.kg / u.m**3,
    )
    assert iso.iso_value == 0.8 * 10**3 * u.kg / u.m**3

    # Test that an Isosurface field defined using SolverVariable
    # is correctly converted to a UserVariable and passes validation.
    with SI_unit_system:
        iso = Isosurface(
            name="test_iso_density",
            field=solution.density,
            iso_value=0.8 * 10**3 * u.kg / u.m**3,
        )
    assert isinstance(iso.field, UserVariable)


def test_isosurface_field_check_expression_length():
    uv_vector = UserVariable(name="uv_vector", value=solution.velocity)
    with pytest.raises(
        ValueError,
        match=re.escape("The isosurface field (uv_vector) must be defined with a scalar variable."),
    ):
        Isosurface(name="test_iso_vector_field", field=uv_vector, iso_value=1.0)

    uv_list = UserVariable(name="uv_list", value=[solution.velocity[0], 1 * u.m / u.s])
    with pytest.raises(
        ValueError,
        match=re.escape("The isosurface field (uv_list) must be defined with a scalar variable."),
    ):
        Isosurface(name="test_iso_list", field=uv_list, iso_value=1.0)


def test_isosurface_field_check_runtime_expression():
    """
    Test that an Isosurface field defined with a UserVariable holding a constant
    value (not an Expression) raises a ValueError.
    """
    uv_float = UserVariable(name="my_const_float_var", value=10.0)
    with pytest.raises(
        ValueError,
        match=re.escape("The isosurface field (my_const_float_var) cannot be a constant value."),
    ):
        Isosurface(name="test_iso", field=uv_float, iso_value=5.0)

    uv_float_derived = UserVariable(name="uv_float_derived", value=uv_float * 2)
    with pytest.raises(
        ValueError,
        match=re.escape("The isosurface field (uv_float_derived) cannot be a constant value."),
    ):
        Isosurface(name="test_iso_expr_const", field=uv_float_derived, iso_value=20.0)

    uv_dim = UserVariable(name="my_const_dim_var", value=10.0 * u.m)
    with pytest.raises(
        ValueError,
        match=re.escape("The isosurface field (my_const_dim_var) cannot be a constant value."),
    ):
        Isosurface(name="test_iso", field=uv_dim, iso_value=1.0 * u.m)

    uv_dim_derived = UserVariable(name="uv_dim_derived", value=uv_float * uv_dim)
    with pytest.raises(
        ValueError,
        match=re.escape("The isosurface field (uv_dim_derived) cannot be a constant value."),
    ):
        Isosurface(name="test_iso_expr_const", field=uv_dim_derived, iso_value=20.0 * u.m)

    uv_dim2 = UserVariable(name="my_const_dim_var2", value=10.0 * u.s)
    uv_dim_derived2 = UserVariable(name="uv_dim_derived2", value=uv_dim * uv_dim2)
    with pytest.raises(
        ValueError,
        match=re.escape("The isosurface field (uv_dim_derived2) cannot be a constant value."),
    ):
        Isosurface(
            name="test_iso_expr_const", field=uv_dim_derived2, iso_value=10 * u.m * 10.0 * u.m
        )


def test_isosurface_single_iso_value():
    """
    Test that an Isosurface iso_value defined with single value.
    """

    uv_vel = UserVariable(name="uv_vel", value=solution.velocity[0])
    with pytest.raises(
        ValueError,
        match=re.escape("The iso_value ([1] m/s) must be defined with a single value."),
    ):
        Isosurface(name="test_iso_list", field=uv_vel, iso_value=[1 * u.m / u.s])

    with pytest.raises(
        ValueError,
        match=re.escape("The iso_value ([1 2] m/s) must be defined with a single value."),
    ):
        Isosurface(name="test_iso_unyt_array", field=uv_vel, iso_value=[1, 2] * u.m / u.s)


def test_isosurface_check_iso_value_dimensions():
    """
    Test that an Isosurface field defined with a UserVariable has the same dimensions
    as the iso_value.
    """

    uv_density = UserVariable(
        name="uv_density",
        value=solution.density.in_units(new_name="density_CGS", new_unit=u.g / u.m**3),
    )  # Density dimension
    iso = Isosurface(name="test_iso_dim_match", field=uv_density, iso_value=10 * u.kg / u.m**3)
    assert iso.iso_value == 10 * u.kg / u.m**3

    with pytest.raises(
        ValueError,
        match=re.escape(
            "The iso_value (10.0 m, dimensions:(length)) should have the "
            "same dimensions as the isosurface field (uv_density, dimensions: (mass)/(length)**3)."
        ),
    ):
        Isosurface(name="test_iso_dim_mismatch", field=uv_density, iso_value=10.0 * u.m)

    uv_Cp = UserVariable(name="uv_Cp", value=solution.Cp)
    iso = Isosurface(name="test_iso_nondim_match", field=uv_Cp, iso_value=10)
    assert iso.iso_value == 10

    with pytest.raises(
        ValueError,
        match=re.escape(
            "The iso_value (10.0 m, dimensions:(length)) should have the "
            "same dimensions as the isosurface field (uv_Cp, dimensions: 1)."
        ),
    ):
        Isosurface(name="test_iso_dim_mismatch", field=uv_Cp, iso_value=10.0 * u.m)

    uv_pressure = UserVariable(
        name="uv_pressure",
        value=0.5 * solution.Cp * solution.density * math.magnitude(solution.velocity) ** 2,
    )
    iso = Isosurface(name="test_iso_nondim_match", field=uv_pressure, iso_value=10 * u.Pa)
    assert iso.iso_value == 10

    with pytest.raises(
        ValueError,
        match=re.escape(
            "The iso_value (10.0 m, dimensions:(length)) should have the "
            "same dimensions as the isosurface field "
            "(uv_pressure, dimensions: (mass)/((length)*(time)**2))."
        ),
    ):
        Isosurface(name="test_iso_dim_mismatch", field=uv_pressure, iso_value=10.0 * u.m)
