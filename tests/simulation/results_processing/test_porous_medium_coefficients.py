import json
import os

import numpy as np

import flow360 as fl
from flow360.component.results.case_results import PorousMediumResultCSVModel
from flow360.component.simulation.framework.param_utils import AssetCache
from flow360.component.simulation.services import ValidationCalledBy, validate_model

from .test_helpers import compute_freestream_direction, compute_lift_direction


def test_porous_medium_simple_coefficients():
    # Prepare a simple porous medium CSV with one timestep
    csv_path = os.path.join(
        os.path.dirname(__file__),
        os.path.pardir,
        "data",
        "coeff_simple",
        "results",
        "porous_media_output_v2.csv",
    )
    csv_path = os.path.abspath(csv_path)

    # Simple params: liquid, explicit V_ref, nonzero alpha/beta
    alpha, beta = 5.0, 10.0

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
                fl.PorousMedium(
                    entities=[
                        fl.Box.from_principal_axes(
                            name="porous_zone",
                            axes=[(1, 0, 0), (0, 1, 0)],
                            center=(0, 0, 0) * fl.u.m,
                            size=(0.2, 0.3, 2) * fl.u.m,
                        )
                    ],
                    darcy_coefficient=(1e6, 0, 0) / fl.u.m**2,
                    forchheimer_coefficient=(1, 0, 0) / fl.u.m,
                )
            ],
            private_attribute_asset_cache=AssetCache(project_length_unit=1 * fl.u.m),
        )

    model = PorousMediumResultCSVModel()
    model.load_from_local(csv_path)
    coeffs = model.compute_coefficients(params=params)

    data = coeffs.as_dict()

    assert "zone_0_CFx" in data
    assert "zone_0_CFy" in data
    assert "zone_0_CFz" in data
    assert "zone_0_CMx" in data
    assert "zone_0_CMy" in data
    assert "zone_0_CMz" in data
    assert "zone_0_CL" in data
    assert "zone_0_CD" in data

    CF = np.array(
        [data["zone_0_CFx"][0], data["zone_0_CFy"][0], data["zone_0_CFz"][0]], dtype=float
    )
    CM = np.array(
        [data["zone_0_CMx"][0], data["zone_0_CMy"][0], data["zone_0_CMz"][0]], dtype=float
    )
    CL = float(data["zone_0_CL"][0])
    CD = float(data["zone_0_CD"][0])

    # CF direction should align with force proportions: (50, 25, 25) = (2, 1, 1)
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

    # CM direction should align with moment proportions: (5, 2.5, 2.5) = (2, 1, 1)
    # Note: moments are already relative to global moment center from solver
    assert np.isclose(CM[0] / CM[1], 2.0, rtol=1e-6, atol=1e-12)
    assert np.isclose(CM[0] / CM[2], 2.0, rtol=1e-6, atol=1e-12)


def test_porous_medium_simple_coefficients_with_generic_reference_condition():
    # Prepare a simple porous medium CSV with one timestep
    csv_path = os.path.join(
        os.path.dirname(__file__),
        os.path.pardir,
        "data",
        "coeff_simple",
        "results",
        "porous_media_output_v2.csv",
    )
    csv_path = os.path.abspath(csv_path)

    # Generic reference condition has no alpha/beta. In coefficient computations we assume:
    # lift direction = (0, 0, 1), drag direction = (1, 0, 0)
    with fl.SI_unit_system:
        params = fl.SimulationParams(
            reference_geometry=fl.ReferenceGeometry(
                moment_center=(0, 0, 0) * fl.u.m,
                moment_length=1 * fl.u.m,
                area=2.0 * fl.u.m**2,
            ),
            operating_condition=fl.GenericReferenceCondition(
                velocity_magnitude=10 * fl.u.m / fl.u.s
            ),
            models=[
                fl.PorousMedium(
                    entities=[
                        fl.Box.from_principal_axes(
                            name="porous_zone",
                            axes=[(1, 0, 0), (0, 1, 0)],
                            center=(0, 0, 0) * fl.u.m,
                            size=(0.2, 0.3, 2) * fl.u.m,
                        )
                    ],
                    darcy_coefficient=(1e6, 0, 0) / fl.u.m**2,
                    forchheimer_coefficient=(1, 0, 0) / fl.u.m,
                )
            ],
            private_attribute_asset_cache=AssetCache(project_length_unit=1 * fl.u.m),
        )

    model = PorousMediumResultCSVModel()
    model.load_from_local(csv_path)
    coeffs = model.compute_coefficients(params=params)

    data = coeffs.as_dict()
    CF = np.array(
        [data["zone_0_CFx"][0], data["zone_0_CFy"][0], data["zone_0_CFz"][0]], dtype=float
    )

    # Drag/lift projections use default axes for GenericReferenceCondition
    drag_dir = np.array([1.0, 0.0, 0.0], dtype=float)
    lift_dir = np.array([0.0, 0.0, 1.0], dtype=float)
    CD = float(data["zone_0_CD"][0])
    CL = float(data["zone_0_CL"][0])
    assert np.isclose(CD, float(np.dot(CF, drag_dir)), rtol=1e-6, atol=1e-12)
    assert np.isclose(CL, float(np.dot(CF, lift_dir)), rtol=1e-6, atol=1e-12)


def test_porous_medium_real_case_coefficients():
    """
    Test PorousMedium coefficient computation with real case data.

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
        "porous_media_output_v2.csv",
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

    porous_medium_refs = reference_data["PorousMedium"]

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
    model = PorousMediumResultCSVModel()
    model.load_from_local(csv_path)
    coeffs = model.compute_coefficients(params=params)

    data = coeffs.as_dict()

    # Test all entities
    for zone_name, expected_coeffs in porous_medium_refs.items():
        # Verify all coefficient keys exist
        assert f"{zone_name}_CFx" in data
        assert f"{zone_name}_CFy" in data
        assert f"{zone_name}_CFz" in data
        assert f"{zone_name}_CMx" in data
        assert f"{zone_name}_CMy" in data
        assert f"{zone_name}_CMz" in data
        assert f"{zone_name}_CL" in data
        assert f"{zone_name}_CD" in data

        # Get computed coefficients (last timestep)
        computed_CFx = float(data[f"{zone_name}_CFx"][-1])
        computed_CFy = float(data[f"{zone_name}_CFy"][-1])
        computed_CFz = float(data[f"{zone_name}_CFz"][-1])
        computed_CMx = float(data[f"{zone_name}_CMx"][-1])
        computed_CMy = float(data[f"{zone_name}_CMy"][-1])
        computed_CMz = float(data[f"{zone_name}_CMz"][-1])
        computed_CD = float(data[f"{zone_name}_CD"][-1])
        computed_CL = float(data[f"{zone_name}_CL"][-1])

        # Compare with reference values
        assert np.isclose(
            computed_CFx, expected_coeffs["CFx"], rtol=1e-10, atol=1e-15
        ), f"{zone_name} CFx mismatch"
        assert np.isclose(
            computed_CFy, expected_coeffs["CFy"], rtol=1e-10, atol=1e-15
        ), f"{zone_name} CFy mismatch"
        assert np.isclose(
            computed_CFz, expected_coeffs["CFz"], rtol=1e-10, atol=1e-15
        ), f"{zone_name} CFz mismatch"
        assert np.isclose(
            computed_CMx, expected_coeffs["CMx"], rtol=1e-10, atol=1e-15
        ), f"{zone_name} CMx mismatch"
        assert np.isclose(
            computed_CMy, expected_coeffs["CMy"], rtol=1e-10, atol=1e-15
        ), f"{zone_name} CMy mismatch"
        assert np.isclose(
            computed_CMz, expected_coeffs["CMz"], rtol=1e-10, atol=1e-15
        ), f"{zone_name} CMz mismatch"
        assert np.isclose(
            computed_CD, expected_coeffs["CD"], rtol=1e-10, atol=1e-15
        ), f"{zone_name} CD mismatch"
        assert np.isclose(
            computed_CL, expected_coeffs["CL"], rtol=1e-10, atol=1e-15
        ), f"{zone_name} CL mismatch"


def test_porous_medium_generic_volume_header_matching():
    """
    Ensure porous medium coefficient computation works with non-`zone_` CSV headers,
    e.g. names like `blk-2_Force_x` and `blk-2_Moment_y`.
    """
    import tempfile

    # Create a temporary CSV with a GenericVolume-style name containing a hyphen
    csv_content = (
        "physical_step,pseudo_step,blk-2_Force_x,blk-2_Force_y,blk-2_Force_z,blk-2_Moment_x,blk-2_Moment_y,blk-2_Moment_z\n"
        "0,0,0,0,0,0,0,0\n"
        "0,10,0,2.0,0,-0.5,0,1.0\n"
        "0,20,0,4.0,0,-1.0,0,2.0\n"
    )

    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tmp:
        tmp.write(csv_content)
        temp_csv_path = tmp.name

    try:
        # Minimal params to enable coefficient computation
        with fl.SI_unit_system:
            params = fl.SimulationParams(
                reference_geometry=fl.ReferenceGeometry(
                    moment_center=(0, 0, 0) * fl.u.m,
                    moment_length=1 * fl.u.m,
                    area=1.0 * fl.u.m**2,
                ),
                operating_condition=fl.LiquidOperatingCondition(
                    velocity_magnitude=10 * fl.u.m / fl.u.s,
                    reference_velocity_magnitude=10 * fl.u.m / fl.u.s,
                    alpha=0.0 * fl.u.deg,
                    beta=0.0 * fl.u.deg,
                ),
                models=[
                    # The computation does not require exact entity-name matching here,
                    # but we include a PorousMedium model for completeness.
                    fl.PorousMedium(
                        entities=[
                            fl.Box.from_principal_axes(
                                name="dummy",
                                axes=[(1, 0, 0), (0, 1, 0)],
                                center=(0, 0, 0) * fl.u.m,
                                size=(0.1, 0.1, 0.1) * fl.u.m,
                            )
                        ],
                        darcy_coefficient=(1e6, 0, 0) / fl.u.m**2,
                        forchheimer_coefficient=(1, 0, 0) / fl.u.m,
                    )
                ],
                private_attribute_asset_cache=AssetCache(project_length_unit=1 * fl.u.m),
            )

        model = PorousMediumResultCSVModel()
        model.load_from_local(temp_csv_path)
        coeffs = model.compute_coefficients(params=params)
        data = coeffs.as_dict()

        # Keys should be derived from the CSV header prefix "blk-2"
        assert "blk-2_CFx" in data
        assert "blk-2_CFy" in data
        assert "blk-2_CFz" in data
        assert "blk-2_CMx" in data
        assert "blk-2_CMy" in data
        assert "blk-2_CMz" in data
        assert "blk-2_CL" in data
        assert "blk-2_CD" in data

        # Time series length should match the number of data rows (3)
        assert len(data["blk-2_CFx"]) == 3
        assert len(data["blk-2_CFy"]) == 3
        assert len(data["blk-2_CFz"]) == 3
        assert len(data["blk-2_CMx"]) == 3
        assert len(data["blk-2_CMy"]) == 3
        assert len(data["blk-2_CMz"]) == 3
        assert len(data["blk-2_CL"]) == 3
        assert len(data["blk-2_CD"]) == 3
    finally:
        if os.path.exists(temp_csv_path):
            os.remove(temp_csv_path)
