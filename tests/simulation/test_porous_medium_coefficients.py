import os

import numpy as np

import flow360 as fl
from flow360.component.results.case_results import PorousMediumResultCSVModel
from flow360.component.simulation.framework.param_utils import AssetCache


def _u_inf(alpha_deg: float, beta_deg: float):
    a = np.deg2rad(alpha_deg)
    b = np.deg2rad(beta_deg)
    v = np.array([np.cos(a) * np.cos(b), np.sin(b), np.sin(a) * np.cos(b)], dtype=float)
    v /= np.linalg.norm(v)
    return v


def _lift_dir_alpha(alpha_deg: float):
    a = np.deg2rad(alpha_deg)
    k = np.array([-np.sin(a), 0.0, np.cos(a)], dtype=float)
    k /= np.linalg.norm(k)
    return k


def test_porous_medium_simple_coefficients():
    # Prepare a simple porous medium CSV with one timestep
    csv_path = os.path.join(
        os.path.dirname(__file__), "data", "coeff_simple", "results", "porous_media_output_v2.csv"
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
    u = _u_inf(alpha, beta)
    drag_dir = -u
    k = _lift_dir_alpha(alpha)
    assert np.isclose(CD, float(np.dot(CF, drag_dir)), rtol=1e-6, atol=1e-12)
    assert np.isclose(CL, float(np.dot(CF, k)), rtol=1e-6, atol=1e-12)

    # CM direction should align with moment proportions: (5, 2.5, 2.5) = (2, 1, 1)
    # Note: moments are already relative to global moment center from solver
    assert np.isclose(CM[0] / CM[1], 2.0, rtol=1e-6, atol=1e-12)
    assert np.isclose(CM[0] / CM[2], 2.0, rtol=1e-6, atol=1e-12)
