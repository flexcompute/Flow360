import json
import os

import numpy as np

import flow360 as fl
from flow360.component.results.case_results import BETForcesResultCSVModel
from flow360.component.simulation.framework.param_utils import AssetCache
from flow360.component.simulation.models.volume_models import BETDisk
from flow360.component.simulation.services import ValidationCalledBy, validate_model

from .test_helpers import compute_freestream_direction, compute_lift_direction


def test_bet_disk_simple_header_rename():
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
    old_data = model.as_dict()

    new_csv = model.rename_header(params=params, pattern="$BETName_$CylinderName")
    new_data = new_csv.as_dict()

    assert "BET disk_bet_disk_Force_x" in new_data
    assert "BET disk_bet_disk_Force_y" in new_data
    assert "BET disk_bet_disk_Force_z" in new_data
    assert "BET disk_bet_disk_Moment_x" in new_data
    assert "BET disk_bet_disk_Moment_y" in new_data
    assert "BET disk_bet_disk_Moment_z" in new_data

    for header_name, value in new_data.items():
        old_key = header_name.replace("BET disk_bet_disk", "Disk0")
        new_value = value[0]
        old_value = old_data[old_key][0]
        assert np.isclose(new_value, old_value, rtol=1e-6, atol=1e-12)


def test_bet_disk_real_case_header_rename():
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

    params, errors, warnings = validate_model(
        params_as_dict=params_as_dict,
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type=None,
    )

    assert errors is None, f"Validation errors: {errors}"
    assert params is not None

    # Load CSV and compute coefficients
    model = BETForcesResultCSVModel()
    model.load_from_local(csv_path)
    old_data = model.as_dict()

    new_csv = model.rename_header(params=params)

    new_data = new_csv.as_dict()

    bet_disks = []
    for model in params.models:
        if isinstance(model, BETDisk):
            bet_disks.append(model)
    assert bet_disks != []

    diskCount = 0
    disk_rename_map = {}
    for i, disk in enumerate(bet_disks):
        for j, cylinder in enumerate(disk.entities.stored_entities):
            disk_name = f"{disk.name}_{cylinder.name}"
            disk_rename_map[f"Disk{diskCount}"] = f"{disk_name}"
            diskCount = diskCount + 1

    assert "physical_step" in new_data
    assert "pseudo_step" in new_data

    for old_key, old_value in old_data.items():
        found = False
        new_disk_key = None
        for old_name, new_name in disk_rename_map.items():
            if old_name in old_key:
                found = True
                new_disk_key = old_key.replace(old_name, new_name)
                break
        if not found:
            new_disk_key = old_key

        assert new_disk_key in new_data

        new_value = new_data[new_disk_key]

        assert len(old_value) == len(new_value)

        for i in range(len(old_value)):
            np.isclose(old_value[i], new_value[i], rtol=1e-6, atol=1e-12)
