import json
import os

import pytest

import flow360 as fl


def test_bet_disk_updater_and_override(tmp_path):
    # Create a temporary JSON file with old version and extra fields
    filename = tmp_path / "BET_Disk_example.json"

    # Minimal content recreating the structure of the example file
    data = {
        "version": "25.7.5",
        "_id": "1076e3c2-d31e-4d2b-97bb-d0d21801fa7d",
        "name": "BET 57",
        "type": "BETDisk",
        "entities": {
            "stored_entities": [
                {
                    "private_attribute_entity_type_name": "Cylinder",
                    "private_attribute_id": "77a28c82-1dfd-40fd-bf3b-1a8385aa10e0",
                    "private_attribute_registry_bucket_name": "DraftEntities",
                    "name": "BET 5",
                    "axis": [0, 0, 1],
                    "center": {"value": [2.7, -6, 1.06], "units": "m"},
                    "height": {"value": 0.2, "units": "m"},
                    "outer_radius": {"value": 1.5, "units": "m"},
                },
                {
                    "private_attribute_entity_type_name": "Cylinder",
                    "private_attribute_id": "4a28e9b9-4124-489c-a86a-2eabb02fa6c9",
                    "private_attribute_registry_bucket_name": "DraftEntities",
                    "name": "BET 7",
                    "axis": [0, 0, 1],
                    "center": {"value": [2.7, 2.65, 1.06], "units": "m"},
                    "height": {"value": 0.2, "units": "m"},
                    "outer_radius": {"value": 1.5, "units": "m"},
                },
            ]
        },
        "rotation_direction_rule": "leftHand",
        "number_of_blades": 2,
        "omega": {"value": 800, "units": "rpm"},
        "chord_ref": {"value": 0.14, "units": "m"},
        "n_loading_nodes": 20,
        "blade_line_chord": {"value": 0.25, "units": "m"},
        "initial_blade_direction": [1, 0, 0],
        "tip_gap": "inf",
        "mach_numbers": [0],
        "reynolds_numbers": [1000000],
        "alphas": {"value": [-180, 180], "units": "degree"},
        "sectional_radiuses": {"value": [0.1, 1.5], "units": "m"},
        "sectional_polars": [
            {"lift_coeffs": [[[0.0, 0.0]]], "drag_coeffs": [[[0.0, 0.0]]]},
            {"lift_coeffs": [[[0.0, 0.0]]], "drag_coeffs": [[[0.0, 0.0]]]},
        ],
        "twists": [
            {"radius": {"value": 0.1, "units": "m"}, "twist": {"value": 0, "units": "degree"}},
            {"radius": {"value": 1.5, "units": "m"}, "twist": {"value": 0, "units": "degree"}},
        ],
        "chords": [
            {"radius": {"value": 0.1, "units": "m"}, "chord": {"value": 0.1, "units": "m"}},
            {"radius": {"value": 1.5, "units": "m"}, "chord": {"value": 0.1, "units": "m"}},
        ],
    }

    with open(filename, "w") as f:
        json.dump(data, f)

    # Define override value
    new_omega = 12345 * fl.u.rpm

    # Load from file with updater support and override
    disk = fl.BETDisk.from_file(str(filename), omega=new_omega)

    # 1. Verify override worked
    assert disk.omega == new_omega

    # 2. Verify basic properties loaded from JSON
    assert disk.name == "BET 57"
    assert disk.number_of_blades == 2
    assert len(disk.entities.stored_entities) == 2
