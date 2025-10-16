import json
import os

import numpy as np

import flow360 as fl
from flow360.component.results.case_results import ActuatorDiskResultCSVModel
from flow360.component.simulation.framework.param_utils import AssetCache
from flow360.component.simulation.services import ValidationCalledBy, validate_model

from .test_helpers import compute_freestream_direction, compute_lift_direction


def test_actuator_disk_simple_coefficients():
    # Prepare a simple actuator disk CSV with one timestep
    csv_path = os.path.join(
        os.path.dirname(__file__),
        os.path.pardir,
        "data",
        "coeff_simple",
        "results",
        "actuatorDisk_output_v2.csv",
    )
    csv_path = os.path.abspath(csv_path)

    # Simple params: liquid, explicit V_ref, nonzero alpha/beta, off-axis disk and offset center
    alpha, beta = 5.0, 10.0
    axis_tuple = (2.0, 1.0, 1.0)
    center_tuple = (0.5, -0.2, 0.3)

    with fl.SI_unit_system:
        params = fl.SimulationParams(
            reference_geometry=fl.ReferenceGeometry(
                moment_center=(0, 0, 0) * fl.u.m,
                moment_length=1 * fl.u.m,
                area=2.0 * fl.u.m**2,
            ),
            operating_condition=fl.LiquidOperatingCondition(
                velocity_magnitude=10 * fl.u.m / fl.u.s,
                reference_velocity_magnitude=10 * fl.u.m / fl.u.s,
                alpha=alpha * fl.u.deg,
                beta=beta * fl.u.deg,
            ),
            models=[
                fl.ActuatorDisk(
                    entities=fl.Cylinder(
                        name="actuator_disk",
                        center=center_tuple * fl.u.m,
                        axis=axis_tuple,
                        height=1 * fl.u.m,
                        outer_radius=1.0 * fl.u.m,
                    ),
                    force_per_area=fl.ForcePerArea(
                        radius=[0, 1] * fl.u.m,
                        thrust=[0, 0] * fl.u.Pa,
                        circumferential=[0, 0] * fl.u.Pa,
                    ),
                )
            ],
            private_attribute_asset_cache=AssetCache(project_length_unit=1 * fl.u.m),
        )

    model = ActuatorDiskResultCSVModel()
    model.load_from_local(csv_path)
    coeffs = model.compute_coefficients(params=params)

    data = coeffs.as_dict()

    assert "Disk0_CFx" in data
    assert "Disk0_CFy" in data
    assert "Disk0_CFz" in data
    assert "Disk0_CMx" in data
    assert "Disk0_CMy" in data
    assert "Disk0_CMz" in data
    assert "Disk0_CL" in data
    assert "Disk0_CD" in data

    CF = np.array([data["Disk0_CFx"][0], data["Disk0_CFy"][0], data["Disk0_CFz"][0]], dtype=float)
    CM = np.array([data["Disk0_CMx"][0], data["Disk0_CMy"][0], data["Disk0_CMz"][0]], dtype=float)
    CL = float(data["Disk0_CL"][0])
    CD = float(data["Disk0_CD"][0])

    # CF direction should align with disk axis proportions: (2,1,1)
    assert np.isclose(CF[0] / CF[1], 2.0, rtol=1e-6, atol=1e-12)
    assert np.isclose(CF[0] / CF[2], 2.0, rtol=1e-6, atol=1e-12)

    # Non-zero cross-plane components and CL
    assert not np.isclose(CF[1], 0.0)
    assert not np.isclose(CF[2], 0.0)

    # Drag and lift from projections
    drag_dir = compute_freestream_direction(alpha, beta)
    lift_dir = compute_lift_direction(alpha)
    assert np.isclose(CD, float(np.dot(CF, drag_dir)), rtol=1e-6, atol=1e-12)
    assert np.isclose(CL, float(np.dot(CF, lift_dir)), rtol=1e-6, atol=1e-12)

    # Check moment decomposition: CM = r x CF + (M/F) * axis_unit * |CF|
    r = np.array(center_tuple, dtype=float) - np.array([0.0, 0.0, 0.0], dtype=float)
    r_cross_CF = np.cross(r, CF)
    axis_unit = np.array(axis_tuple, dtype=float)
    axis_unit /= np.linalg.norm(axis_unit)
    remainder = CM - r_cross_CF
    # remainder should be colinear with axis_unit
    rem_norm = float(np.linalg.norm(remainder))
    CF_mag = float(np.linalg.norm(CF))
    assert rem_norm > 0.0
    assert CF_mag > 0.0
    dir_remainder = remainder / rem_norm
    assert np.allclose(dir_remainder, axis_unit, rtol=1e-6, atol=1e-6)

    # Magnitude ratio equals M/F
    F_mag = 100.0
    M_mag = 10.0
    s = M_mag / F_mag
    assert np.isclose(rem_norm / CF_mag, s, rtol=1e-6, atol=1e-12)


def test_actuator_disk_real_case_coefficients():
    """
    Test ActuatorDisk coefficient computation with real case data.

    This test uses CSV data and parameters from an actual Flow360 simulation
    to verify that coefficient computation works correctly with real-world data.
    """
    # Load CSV file
    csv_path = os.path.join(
        os.path.dirname(__file__),
        os.path.pardir,
        "data",
        "real_case_coefficients",
        "results",
        "actuatorDisk_output_v2.csv",
    )
    csv_path = os.path.abspath(csv_path)

    # Load reference coefficients
    ref_path = os.path.join(
        os.path.dirname(__file__),
        os.path.pardir,
        "data",
        "real_case_coefficients",
        "results",
        "reference_coefficients.json",
    )
    ref_path = os.path.abspath(ref_path)

    with open(ref_path, "r") as f:
        reference_data = json.load(f)

    actuator_disk_refs = reference_data["ActuatorDisk"]

    # Load simulation params from JSON
    params_path = os.path.join(
        os.path.dirname(__file__),
        os.path.pardir,
        "data",
        "real_case_coefficients",
        "results",
        "simulation_params.json",
    )
    params_path = os.path.abspath(params_path)

    with open(params_path, "r") as f:
        params_json = f.read()

    params_as_dict = json.loads(params_json)
    params, errors, warnings = validate_model(
        params_as_dict=params_as_dict,
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type=None,
    )

    assert errors is None, f"Validation errors: {errors}"
    assert params is not None

    # Load CSV and compute coefficients
    model = ActuatorDiskResultCSVModel()
    model.load_from_local(csv_path)
    coeffs = model.compute_coefficients(params=params)

    data = coeffs.as_dict()

    # Test all entities
    for disk_name, expected_coeffs in actuator_disk_refs.items():
        # Verify all coefficient keys exist
        assert f"{disk_name}_CFx" in data
        assert f"{disk_name}_CFy" in data
        assert f"{disk_name}_CFz" in data
        assert f"{disk_name}_CMx" in data
        assert f"{disk_name}_CMy" in data
        assert f"{disk_name}_CMz" in data
        assert f"{disk_name}_CL" in data
        assert f"{disk_name}_CD" in data

        # Get computed coefficients (last timestep)
        computed_CFx = float(data[f"{disk_name}_CFx"][-1])
        computed_CFy = float(data[f"{disk_name}_CFy"][-1])
        computed_CFz = float(data[f"{disk_name}_CFz"][-1])
        computed_CMx = float(data[f"{disk_name}_CMx"][-1])
        computed_CMy = float(data[f"{disk_name}_CMy"][-1])
        computed_CMz = float(data[f"{disk_name}_CMz"][-1])
        computed_CD = float(data[f"{disk_name}_CD"][-1])
        computed_CL = float(data[f"{disk_name}_CL"][-1])

        # Compare with reference values
        assert np.isclose(
            computed_CFx, expected_coeffs["CFx"], rtol=1e-10, atol=1e-15
        ), f"{disk_name} CFx mismatch"
        assert np.isclose(
            computed_CFy, expected_coeffs["CFy"], rtol=1e-10, atol=1e-15
        ), f"{disk_name} CFy mismatch"
        assert np.isclose(
            computed_CFz, expected_coeffs["CFz"], rtol=1e-10, atol=1e-15
        ), f"{disk_name} CFz mismatch"
        assert np.isclose(
            computed_CMx, expected_coeffs["CMx"], rtol=1e-10, atol=1e-15
        ), f"{disk_name} CMx mismatch"
        assert np.isclose(
            computed_CMy, expected_coeffs["CMy"], rtol=1e-10, atol=1e-15
        ), f"{disk_name} CMy mismatch"
        assert np.isclose(
            computed_CMz, expected_coeffs["CMz"], rtol=1e-10, atol=1e-15
        ), f"{disk_name} CMz mismatch"
        assert np.isclose(
            computed_CD, expected_coeffs["CD"], rtol=1e-10, atol=1e-15
        ), f"{disk_name} CD mismatch"
        assert np.isclose(
            computed_CL, expected_coeffs["CL"], rtol=1e-10, atol=1e-15
        ), f"{disk_name} CL mismatch"
