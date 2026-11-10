"""Round-trip every simulation-params JSON in the client test corpus through the compact wire format.

Catches entity-type coverage gaps in the serializer/materializer and updater
match/backfill edge cases that single-purpose tests miss.
"""

import json
from pathlib import Path

import pytest
from flow360_schema.framework.expression.registry import clear_context
from flow360_schema.testing import (
    assert_unique_entity_instances,
    round_trip_simulation_dict,
)

CORPUS_ROOT = Path(__file__).resolve().parents[1]

# Fixtures that are intentionally not valid params (error-message tests etc.)
# or that fail full validation identically on master (pre-existing fixture rot).
SKIP = {
    "simulation/converter/ref/ref_monitor.json": (
        "v1-to-v2 converter output is a pre-upload intermediate with no entity_info by "
        "design; the upload path attaches entity_info later, so this ref is not wire data"
    ),
    "data/simulation/simulation_pre_24_11_1_symmetry.json": (
        "never full-loads on master either (invalid entity_info silently degraded there); "
        "draft Box carries a legacy 'rotation' shape no updater converts"
    ),
    "data/simulation/simulation_pre_24_11_7.json": (
        "fails identically on master: duplicate ProbeOutput name (ungated Case validator)"
    ),
    "data/simulation/simulation_pre_25_7_2.json": (
        "fails identically on master: geometry_accuracy requires GAI (ungated validator)"
    ),
    "simulation/data/simulation_with_wrong_expr_syntax.json": (
        "intentional error fixture (bad expression syntax)"
    ),
    "simulation/service/data/updater_should_pass.json": (
        "fails identically on master: pre-25.6 initial_condition/transition-model shapes "
        "no updater converts; updater-step unit-test input only"
    ),
    "simulation/service/params.json": (
        "never loads on master either: extra_forbidden private_attribute_registry_bucket_name"
    ),
}


@pytest.fixture(autouse=True)
def _isolated_variable_space():
    # The expression variable context is process-global; earlier tests leak
    # user variables that collide with corpus files declaring the same names.
    clear_context()
    yield
    clear_context()


def _corpus():
    files = []
    for path in sorted(CORPUS_ROOT.rglob("*.json")):
        try:
            content = json.loads(path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(content, dict) and "unit_system" in content and "version" in content:
            files.append(path)
    return files


_CORPUS = _corpus()


def test_corpus_is_not_empty():
    assert len(_CORPUS) >= 10, f"corpus enumeration collapsed: {_CORPUS}"


@pytest.mark.parametrize("path", _CORPUS, ids=lambda p: p.relative_to(CORPUS_ROOT).as_posix())
def test_ref_corpus_round_trip(path):
    relative = path.relative_to(CORPUS_ROOT).as_posix()
    if relative in SKIP:
        pytest.skip(SKIP[relative])
    params_dict = json.loads(path.read_text())

    _, params_reloaded = round_trip_simulation_dict(params_dict)

    assert_unique_entity_instances(params_reloaded)
