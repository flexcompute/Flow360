"""Tests for DraftRunRequest, specifically the priority and job_type fields."""

import pytest
from pydantic import ValidationError

from flow360.cloud.flow360_requests import DraftRunRequest


def _make_request(**overrides):
    """Helper to build a DraftRunRequest with sensible defaults."""
    defaults = dict(
        source_item_type="Geometry",
        up_to="Case",
        use_in_house=False,
        use_gai=False,
        force_creation_config=None,
        job_type=None,
        priority=None,
    )
    defaults.update(overrides)
    return DraftRunRequest(**defaults)


class TestDraftRunRequestPriority:
    def test_priority_default_is_none(self):
        req = _make_request()
        assert req.priority is None

    def test_priority_valid_range(self):
        for val in (1, 5, 10):
            req = _make_request(priority=val)
            assert req.priority == val

    def test_priority_below_min_raises(self):
        with pytest.raises(ValidationError):
            _make_request(priority=0)

    def test_priority_above_max_raises(self):
        with pytest.raises(ValidationError):
            _make_request(priority=11)

    def test_priority_none_not_in_serialized_body(self):
        req = _make_request()
        body = req.model_dump(by_alias=True)
        # priority is None; caller (draft.py) pops it before sending
        assert body.get("priority") is None

    def test_priority_included_in_serialized_body(self):
        req = _make_request(priority=7)
        body = req.model_dump(by_alias=True)
        assert body["priority"] == 7

    def test_job_type_and_priority_together(self):
        req = _make_request(job_type="TIME_SHARED_VGPU", priority=3)
        body = req.model_dump(by_alias=True)
        assert body["jobType"] == "TIME_SHARED_VGPU"
        assert body["priority"] == 3
