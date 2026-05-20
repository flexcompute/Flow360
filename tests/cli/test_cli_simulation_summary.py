import json

from click.testing import CliRunner

from flow360.cli import flow360


def _surface_entity(name):
    return {
        "name": name,
        "private_attribute_entity_type_name": "Surface",
        "private_attribute_sub_components": [],
    }


def _minimal_simulation(models):
    return {
        "version": "25.10.3b1",
        "unit_system": {"name": "SI"},
        "operating_condition": {
            "type_name": "AerospaceCondition",
            "alpha": {"value": 5.0 * 3.141592653589793 / 180, "display_unit": "degree"},
            "beta": {"value": 0.0, "display_unit": "degree"},
            "velocity_magnitude": {"value": 50.0},
            "thermal_state": {
                "type_name": "ThermalState",
                "temperature": {"value": 288.15},
                "density": {"value": 1.225},
            },
        },
        "models": models,
        "time_stepping": {"type_name": "Steady", "max_steps": 1000},
    }


def test_simulation_summary_extracts_solver_and_operating_condition():
    from flow360.cli.simulation_summary import summarize_simulation

    summary = summarize_simulation(
        _minimal_simulation(
            [
                {
                    "type": "Fluid",
                    "navier_stokes_solver": {"type_name": "Compressible"},
                    "turbulence_model_solver": {"type_name": "SpalartAllmaras"},
                }
            ]
        )
    )

    # SI value (radians) is the wire-format payload; display_unit ("degree") gets
    # pruned by the summary because the default-input also defaults to degree.
    assert summary["operating_condition"]["alpha"] == {"value": 5.0 * 3.141592653589793 / 180}
    assert "beta" not in summary["operating_condition"]
    assert summary["time_stepping"]["type_name"] == "Steady"
    assert summary["models"] == [{"type": "Fluid"}]


def test_simulation_summary_groups_identical_surface_models_by_settings():
    from flow360.cli.simulation_summary import summarize_simulation

    summary = summarize_simulation(
        _minimal_simulation(
            [
                {
                    "type": "Wall",
                    "name": "Wall",
                    "use_wall_function": False,
                    "entities": {
                        "stored_entities": [
                            _surface_entity("wing"),
                            _surface_entity("fuselage"),
                        ]
                    },
                },
                {
                    "type": "Wall",
                    "name": "Wall",
                    "use_wall_function": False,
                    "entities": {"stored_entities": [_surface_entity("tail")]},
                },
            ]
        )
    )

    assert summary["models"] == [
        {
            "_count": 2,
            "_names": ["Wall"],
            "entities": {"_count": 3, "_sample": ["wing", "fuselage", "tail"]},
            "type": "Wall",
        }
    ]


def test_simulation_summary_ignores_invalid_private_cache():
    from flow360.cli.simulation_summary import summarize_simulation

    simulation = _minimal_simulation([])
    simulation["private_attribute_asset_cache"] = {
        "variable_context": [{"name": "bad", "value": {"expression": "rho + missing_symbol"}}]
    }

    summary = summarize_simulation(simulation)

    assert "private_attribute_asset_cache" not in summary
    assert "models" not in summary


def test_simulation_summary_prunes_absent_zero_defaults():
    from flow360.cli.simulation_summary import summarize_simulation

    simulation = _minimal_simulation([])
    simulation["meshing"] = {
        "type_name": "MeshingParams",
        "gap_treatment_strength": 0,
    }

    summary = summarize_simulation(simulation)

    assert summary["meshing"] == {"type_name": "MeshingParams"}


def test_clean_empty_filters_items_after_recursive_cleanup():
    from flow360.cli.simulation_summary import _clean_empty

    assert _clean_empty([{"nested": {}}, {"keep": {"value": 1}}, [], None]) == [
        {"keep": {"value": 1}}
    ]


def test_case_summary_outputs_compact_json(monkeypatch):
    from flow360.cli import assets as assets_cli

    runner = CliRunner()
    monkeypatch.setattr(
        assets_cli,
        "_get_asset_simulation_params",
        lambda webapi_cls, asset_id: _minimal_simulation(
            [
                {
                    "type": "Fluid",
                    "navier_stokes_solver": {"type_name": "Compressible"},
                    "turbulence_model_solver": {"type_name": "SpalartAllmaras"},
                }
            ]
        ),
    )

    result = runner.invoke(flow360, ["case", "summary", "case-123"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["id"] == "case-123"
    assert payload["summary"]["models"] == [{"type": "Fluid"}]


def test_mesh_summary_commands_are_available(monkeypatch):
    from flow360.cli import assets as assets_cli

    runner = CliRunner()
    monkeypatch.setattr(
        assets_cli,
        "_get_asset_simulation_params",
        lambda webapi_cls, asset_id: _minimal_simulation([]),
    )

    for command, resource_id in [
        ("geometry", "geo-123"),
        ("surface-mesh", "sm-123"),
        ("volume-mesh", "vm-123"),
    ]:
        result = runner.invoke(flow360, [command, "summary", resource_id])

        assert result.exit_code == 0
        assert json.loads(result.output)["id"] == resource_id
