import json

import pytest

from flow360.component.simulation.web.draft import Draft
from flow360.exceptions import Flow360WebError


def _validate_with_envelope(monkeypatch, envelope, *, root_item_type="Geometry", up_to="Case"):
    monkeypatch.setattr(Draft, "post_envelope", lambda self, json, path=None, method=None: envelope)
    draft = Draft(draft_id="00000000-0000-0000-0000-000000000000")
    return draft.validate_simulation_json(root_item_type=root_item_type, up_to=up_to)


def _failure_envelope(errors, warnings=None):
    detail = json.dumps(
        {"status": "fail", "errors": errors, "warnings": warnings or [], "schemaVersion": "25.11.7"}
    )
    return {"code": "5000000107", "detail": detail, "data": None, "error": "Fail to validate"}


def test_validation_success_returns_warnings(monkeypatch):
    envelope = {"data": {"fullSolverVersion": "release-25.11", "warnings": ["w1"]}, "code": "0"}

    errors, warnings = _validate_with_envelope(monkeypatch, envelope)

    assert errors is None
    assert warnings == ["w1"]


def test_validation_errors_filtered_by_requested_levels(monkeypatch):
    envelope = _failure_envelope(
        [
            {
                "loc": ["meshing"],
                "msg": "Field required",
                "type": "missing",
                "ctx": {"relevant_for": ["SurfaceMesh", "VolumeMesh"]},
            },
            {
                "loc": ["models", 0, "surfaces"],
                "msg": "Case only",
                "type": "missing",
                "ctx": {"relevant_for": ["Case"]},
            },
            {"loc": ["version"], "msg": "untagged applies everywhere", "type": "value_error"},
        ]
    )

    # Mesh-only submit: the Case-tagged error is not relevant; untagged errors always are.
    errors, _ = _validate_with_envelope(monkeypatch, envelope, up_to="SurfaceMesh")
    assert [error["msg"] for error in errors] == ["Field required", "untagged applies everywhere"]

    # Full submit keeps all three.
    errors, _ = _validate_with_envelope(monkeypatch, envelope, up_to="Case")
    assert len(errors) == 3


def test_validation_valid_when_every_error_is_out_of_scope(monkeypatch):
    envelope = _failure_envelope(
        [
            {
                "loc": ["models", 0, "surfaces"],
                "msg": "Case only",
                "type": "missing",
                "ctx": {"relevant_for": ["Case"]},
            }
        ],
        warnings=[{"loc": [], "msg": "heads up", "type": "warning"}],
    )

    errors, warnings = _validate_with_envelope(monkeypatch, envelope, up_to="SurfaceMesh")

    assert errors is None
    assert warnings == [{"loc": [], "msg": "heads up", "type": "warning"}]


def test_validation_without_a_verdict_hard_fails(monkeypatch):
    # No data, no recognizable validation-failure code: the submit path requires a
    # verdict and must stop rather than proceed unvalidated.
    envelope = {"code": "5000000001", "detail": "internal error", "data": None}

    with pytest.raises(Flow360WebError, match="did not return a verdict"):
        _validate_with_envelope(monkeypatch, envelope)
