import json
import os

import numpy as np

import flow360 as fl
from flow360.component.results.case_results import BETForcesResultCSVModel
from flow360.component.results.results_utils import _build_coeff_env
from flow360.component.simulation.framework.param_utils import AssetCache
from flow360.component.simulation.models.volume_models import BETDisk
from flow360.component.simulation.services import ValidationCalledBy, validate_model

from .test_helpers import compute_freestream_direction, compute_lift_direction


def test_bet_disk_simple_coefficients():
    # Prepare a simple BET disk CSV with one timestep
    csv_path = os.path.join(
        os.path.dirname(__file__),
        os.path.pardir,
        "data",
        "coeff_simple",
        "results",
        "bet_forces_v2.csv",
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
                fl.BETDisk(
                    entities=fl.Cylinder(
                        name="bet_disk",
                        center=center_tuple * fl.u.m,
                        axis=axis_tuple,
                        height=1 * fl.u.m,
                        outer_radius=1.0 * fl.u.m,
                    ),
                    rotation_direction_rule="leftHand",
                    number_of_blades=3,
                    omega=100 * fl.u.rpm,
                    chord_ref=14 * fl.u.inch,
                    n_loading_nodes=20,
                    mach_numbers=[0],
                    reynolds_numbers=[1000000],
                    twists=[fl.BETDiskTwist(radius=0 * fl.u.inch, twist=0 * fl.u.deg)],
                    chords=[fl.BETDiskChord(radius=0 * fl.u.inch, chord=14 * fl.u.inch)],
                    alphas=[-2, 0, 2] * fl.u.deg,
                    sectional_radiuses=[13.5, 25.5] * fl.u.inch,
                    sectional_polars=[
                        fl.BETDiskSectionalPolar(
                            lift_coeffs=[[[0.1, 0.2, 0.3]]],  # 1 Mach x 1 Re x 3 alphas
                            drag_coeffs=[[[0.01, 0.02, 0.03]]],
                        ),
                        fl.BETDiskSectionalPolar(
                            lift_coeffs=[[[0.15, 0.25, 0.35]]],  # 1 Mach x 1 Re x 3 alphas
                            drag_coeffs=[[[0.015, 0.025, 0.035]]],
                        ),
                    ],
                )
            ],
            private_attribute_asset_cache=AssetCache(project_length_unit=1 * fl.u.m),
        )

    model = BETForcesResultCSVModel()
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

    # CF direction should match input force direction proportions: (100, 50, 25)
    assert np.isclose(CF[0] / CF[1], 2.0, rtol=1e-6, atol=1e-12)
    assert np.isclose(CF[0] / CF[2], 4.0, rtol=1e-6, atol=1e-12)

    # Non-zero cross-plane components and CL
    assert not np.isclose(CF[1], 0.0)
    assert not np.isclose(CF[2], 0.0)

    # Drag and lift from projections
    drag_dir = compute_freestream_direction(alpha, beta)
    lift_dir = compute_lift_direction(alpha)
    assert np.isclose(CD, float(np.dot(CF, drag_dir)), rtol=1e-6, atol=1e-12)
    assert np.isclose(CL, float(np.dot(CF, lift_dir)), rtol=1e-6, atol=1e-12)

    # Check that CM values are non-zero and have reasonable magnitudes
    # (The exact values depend on unit conversion which may need refinement)
    assert not np.allclose(CM, 0.0, atol=1e-10)
    assert np.all(np.isfinite(CM))

    # Check that the moment decomposition includes cross-product term
    # CM should have non-zero components due to r x force term
    assert not np.isclose(CM[0], 0.0, atol=1e-10)  # Should be non-zero due to cross product
    assert not np.isclose(CM[1], 0.0, atol=1e-10)  # Should be non-zero due to cross product
    assert not np.isclose(CM[2], 0.0, atol=1e-10)  # Should be non-zero due to cross product


def test_bet_disk_real_case_coefficients():
    """
    Test BETDisk coefficient computation with real case data.

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
        "bet_forces_v2.csv",
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

    bet_disk_refs = reference_data["BETDisk"]

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
    params, errors, _ = validate_model(
        params_as_dict=params_as_dict,
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type=None,
    )

    mach_ref = params.operating_condition.mach

    coeff_env = _build_coeff_env(params)
    assert coeff_env["dynamic_pressure"] == 0.5 * mach_ref * mach_ref
    assert coeff_env["area"] == (params.reference_geometry.area / (1.0 * fl.u.cm**2)).value
    assert np.allclose(coeff_env["moment_length_vec"], [140, 140, 140])
    assert np.allclose(coeff_env["moment_center_global"], [0, 0, 0])
    assert np.allclose(coeff_env["lift_dir"], [-0.25881905, 0.0, 0.96592583])
    assert np.allclose(coeff_env["drag_dir"], [0.96592583, -0.0, 0.25881905])

    assert errors is None, f"Validation errors: {errors}"
    assert params is not None

    # Load CSV and compute coefficients
    model = BETForcesResultCSVModel()
    model.load_from_local(csv_path)
    coeffs = model.compute_coefficients(params=params)

    data = coeffs.as_dict()

    # Test all entities
    for disk_name, expected_coeffs in bet_disk_refs.items():
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
