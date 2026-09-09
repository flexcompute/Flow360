"""Behavior regression tests for the pre-submission length-scale-mismatch guardrail."""

import json
from io import StringIO

import pytest
from flow360_schema.models.asset_cache import AssetCache

import flow360 as fl
import flow360.component.project_utils as project_utils
from flow360.component.project_utils import (
    GEOMETRY_ACCURACY_TOO_COARSE_RATIO,
    GEOMETRY_ACCURACY_TOO_FINE_RATIO,
    MAX_EDGE_LENGTH_TOO_FINE_RATIO,
    _collect_length_settings,
    enforce_length_scale_sanity,
)
from flow360.component.simulation.warning_bypass import (
    LENGTH_SCALE_MISMATCH,
    is_warning_bypassed,
    warning_bypass,
)

# Fixture geometry has bounding box [[-1, 0, -1], [1, 1, 1]] -> largest dimension 2, project_length_unit 1 m.
# Threshold is ratio > 500. So surface_max_edge_length 1e-2 m -> ratio 200 (safe); 1e-5 m -> ratio 2e5 (triggers).
_SAFE_EDGE_LENGTH = 1e-2
_MISMATCH_EDGE_LENGTH = 1e-5


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


@pytest.fixture
def asset_cache():
    with open("./params/data/geometry_metadata_asset_cache.json") as fp:
        return AssetCache(**json.load(fp))


def _build_params(asset_cache, *, defaults_edge_length, refinements=None, geometry_accuracy=None):
    with fl.SI_unit_system:
        defaults = fl.MeshingDefaults(
            boundary_layer_first_layer_thickness=1e-4,
            surface_max_edge_length=defaults_edge_length,
        )
        if geometry_accuracy is not None:
            defaults.geometry_accuracy = geometry_accuracy * fl.u.m
        return fl.SimulationParams(
            meshing=fl.MeshingParams(
                defaults=defaults,
                refinements=refinements or [],
                volume_zones=[fl.AutomatedFarfield(method="auto")],
            ),
            operating_condition=fl.AerospaceCondition(velocity_magnitude=0.2),
            private_attribute_asset_cache=asset_cache,
        )


# --- warning_bypass primitive ---


def test_warning_bypass_single_string():
    assert is_warning_bypassed(LENGTH_SCALE_MISMATCH) is False
    with warning_bypass(LENGTH_SCALE_MISMATCH):
        assert is_warning_bypassed(LENGTH_SCALE_MISMATCH) is True
    assert is_warning_bypassed(LENGTH_SCALE_MISMATCH) is False


def test_warning_bypass_list_form():
    with warning_bypass([LENGTH_SCALE_MISMATCH]):
        assert is_warning_bypassed(LENGTH_SCALE_MISMATCH) is True


def test_warning_bypass_nesting_additive_and_reset():
    with warning_bypass([]):
        assert is_warning_bypassed(LENGTH_SCALE_MISMATCH) is False
        with warning_bypass(LENGTH_SCALE_MISMATCH):
            assert is_warning_bypassed(LENGTH_SCALE_MISMATCH) is True
        # inner scope reset, outer (empty) restored
        assert is_warning_bypassed(LENGTH_SCALE_MISMATCH) is False


# --- guardrail behavior ---


def test_under_threshold_proceeds_without_prompt(asset_cache, monkeypatch):
    # No stdin: if it prompted, input() would EOFError -> block. It must not prompt.
    monkeypatch.setattr("sys.stdin", StringIO(""))
    params = _build_params(asset_cache, defaults_edge_length=_SAFE_EDGE_LENGTH)
    assert enforce_length_scale_sanity(params) is True


def test_mismatch_confirm_yes_proceeds(asset_cache, monkeypatch):
    monkeypatch.setattr("sys.stdin", StringIO("y\n"))
    params = _build_params(asset_cache, defaults_edge_length=_MISMATCH_EDGE_LENGTH)
    assert enforce_length_scale_sanity(params) is True


def test_mismatch_confirm_no_blocks(asset_cache, monkeypatch):
    monkeypatch.setattr("sys.stdin", StringIO("n\n"))
    params = _build_params(asset_cache, defaults_edge_length=_MISMATCH_EDGE_LENGTH)
    assert enforce_length_scale_sanity(params) is False


def test_mismatch_non_interactive_blocks(asset_cache, monkeypatch):
    monkeypatch.setattr("sys.stdin", StringIO(""))  # EOF -> cannot confirm
    params = _build_params(asset_cache, defaults_edge_length=_MISMATCH_EDGE_LENGTH)
    assert enforce_length_scale_sanity(params) is False


def test_mismatch_bypassed_proceeds_without_prompt(asset_cache, monkeypatch):
    monkeypatch.setattr("sys.stdin", StringIO(""))  # would EOF-block if it prompted
    params = _build_params(asset_cache, defaults_edge_length=_MISMATCH_EDGE_LENGTH)
    with warning_bypass(LENGTH_SCALE_MISMATCH):
        assert enforce_length_scale_sanity(params) is True


def test_smallest_refinement_is_binding(asset_cache, monkeypatch):
    # Defaults safe, but a per-face refinement is implausibly fine -> must block.
    monkeypatch.setattr("sys.stdin", StringIO("n\n"))
    boundary = asset_cache.project_entity_info.get_boundaries()[0]
    with fl.SI_unit_system:
        refinement = fl.SurfaceRefinement(faces=[boundary], max_edge_length=_MISMATCH_EDGE_LENGTH)
    params = _build_params(
        asset_cache, defaults_edge_length=_SAFE_EDGE_LENGTH, refinements=[refinement]
    )
    assert enforce_length_scale_sanity(params) is False


def test_refinement_geometry_accuracy_is_checked(asset_cache, monkeypatch):
    # Per-face GeometryRefinement.geometry_accuracy too fine -> must block (not only defaults).
    monkeypatch.setattr("sys.stdin", StringIO("n\n"))
    boundary = asset_cache.project_entity_info.get_boundaries()[0]
    with fl.SI_unit_system:
        refinement = fl.GeometryRefinement(faces=[boundary], geometry_accuracy=1e-5 * fl.u.m)
    params = _build_params(
        asset_cache, defaults_edge_length=_SAFE_EDGE_LENGTH, refinements=[refinement]
    )
    assert enforce_length_scale_sanity(params) is False


def test_no_meshing_is_noop(asset_cache, monkeypatch):
    monkeypatch.setattr("sys.stdin", StringIO(""))
    with fl.SI_unit_system:
        params = fl.SimulationParams(
            operating_condition=fl.AerospaceCondition(velocity_magnitude=0.2),
            private_attribute_asset_cache=asset_cache,
        )
    assert enforce_length_scale_sanity(params) is True


def test_geometry_accuracy_too_fine_gates(asset_cache, monkeypatch):
    # too fine: limit 1/100000. value 1e-5 m -> ratio 2e5 > 1e5 -> gate (confirm), declined -> block.
    monkeypatch.setattr("sys.stdin", StringIO("n\n"))
    params = _build_params(
        asset_cache, defaults_edge_length=_SAFE_EDGE_LENGTH, geometry_accuracy=1e-5
    )
    assert enforce_length_scale_sanity(params) is False


def test_geometry_accuracy_in_band_no_warning(asset_cache, monkeypatch):
    # ratio 2000: between coarse (<100) and fine (>1e5) -> no warning, no gate.
    monkeypatch.setattr("sys.stdin", StringIO(""))  # EOF would block if it wrongly prompted
    params = _build_params(
        asset_cache, defaults_edge_length=_SAFE_EDGE_LENGTH, geometry_accuracy=1e-3
    )
    assert enforce_length_scale_sanity(params) is True


def test_geometry_accuracy_too_coarse_soft_warns_without_gate(asset_cache, monkeypatch):
    # too coarse: ratio 2/0.1 = 20 < 100 -> soft log warning, but NO prompt and proceeds.
    monkeypatch.setattr("sys.stdin", StringIO(""))  # EOF would block if it wrongly prompted
    messages = []
    monkeypatch.setattr(
        project_utils.log, "warning", lambda msg, *a, **k: messages.append(str(msg))
    )
    params = _build_params(
        asset_cache, defaults_edge_length=_SAFE_EDGE_LENGTH, geometry_accuracy=0.1
    )
    assert enforce_length_scale_sanity(params) is True
    assert any("coarse" in m for m in messages)


def test_collect_length_settings_gathers_defaults_and_refinements(asset_cache):
    boundary = asset_cache.project_entity_info.get_boundaries()[0]
    with fl.SI_unit_system:
        refinement = fl.SurfaceRefinement(faces=[boundary], max_edge_length=_MISMATCH_EDGE_LENGTH)
    params = _build_params(
        asset_cache, defaults_edge_length=_SAFE_EDGE_LENGTH, refinements=[refinement]
    )
    labels = [label for label, _ in _collect_length_settings(params.meshing)]
    assert labels.count("max_edge_length") == 2  # defaults + one refinement


def test_ratio_limit_constants():
    assert MAX_EDGE_LENGTH_TOO_FINE_RATIO == 500
    assert GEOMETRY_ACCURACY_TOO_FINE_RATIO == 100000
    assert GEOMETRY_ACCURACY_TOO_COARSE_RATIO == 100
