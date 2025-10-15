import os

import numpy as np

import flow360 as fl
from flow360.component.results.case_results import BETForcesResultCSVModel
from flow360.component.simulation.framework.param_utils import AssetCache


def _u_inf(alpha_deg: float, beta_deg: float):
    a = np.deg2rad(alpha_deg)
    b = np.deg2rad(beta_deg)
    v = np.array([np.cos(a) * np.cos(b), -np.sin(b), np.sin(a) * np.cos(b)], dtype=float)
    v /= np.linalg.norm(v)
    return v


def _lift_dir_alpha(alpha_deg: float):
    a = np.deg2rad(alpha_deg)
    k = np.array([-np.sin(a), 0.0, np.cos(a)], dtype=float)
    k /= np.linalg.norm(k)
    return k


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
    u = _u_inf(alpha, beta)
    drag_dir = u
    k = _lift_dir_alpha(alpha)
    assert np.isclose(CD, float(np.dot(CF, drag_dir)), rtol=1e-6, atol=1e-12)
    assert np.isclose(CL, float(np.dot(CF, k)), rtol=1e-6, atol=1e-12)

    # Check that CM values are non-zero and have reasonable magnitudes
    # (The exact values depend on unit conversion which may need refinement)
    assert not np.allclose(CM, 0.0, atol=1e-10)
    assert np.all(np.isfinite(CM))

    # Check that the moment decomposition includes cross-product term
    # CM should have non-zero components due to r x force term
    assert not np.isclose(CM[0], 0.0, atol=1e-10)  # Should be non-zero due to cross product
    assert not np.isclose(CM[1], 0.0, atol=1e-10)  # Should be non-zero due to cross product
    assert not np.isclose(CM[2], 0.0, atol=1e-10)  # Should be non-zero due to cross product
