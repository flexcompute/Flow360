import pytest

from flow360.component.simulation.web import draft as draft_module
from flow360.component.simulation.web.draft import Draft
from flow360.exceptions import Flow360WebError

DRAFT_ID = "00000000-0000-0000-0000-000000000000"


def test_draft_lookup_reports_a_missing_draft_as_gone(monkeypatch):
    calls = []

    def fake_get(self, path=None, method=None, json=None, params=None):
        calls.append((path, params))
        return None

    monkeypatch.setattr(Draft, "get", fake_get)

    assert Draft(draft_id=DRAFT_ID).exists_in_cloud() is False
    assert calls == [("v2/drafts/find", {"id": DRAFT_ID})]


def test_draft_lookup_reports_an_existing_draft(monkeypatch):
    monkeypatch.setattr(
        Draft, "get", lambda self, path=None, method=None, json=None, params=None: {"id": DRAFT_ID}
    )

    assert Draft(draft_id=DRAFT_ID).exists_in_cloud() is True


def _wait_with_lookups(monkeypatch, lookups, **kwargs):
    """Run the poll loop against a scripted sequence of lookup outcomes."""
    remaining = list(lookups)
    slept = []

    def fake_exists(self):
        outcome = remaining.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(draft_module.time, "sleep", slept.append)
    monkeypatch.setattr(Draft, "exists_in_cloud", fake_exists)
    confirmed = Draft(draft_id=DRAFT_ID).wait_until_deleted(**kwargs)
    return confirmed, slept, remaining


def test_wait_stops_as_soon_as_the_draft_is_gone(monkeypatch):
    confirmed, slept, remaining = _wait_with_lookups(
        monkeypatch, [True, False, True], timeout_seconds=60, poll_interval_seconds=5
    )

    assert confirmed is True
    assert slept == [5]
    assert remaining == [True]


def test_wait_keeps_polling_through_a_failed_lookup(monkeypatch):
    # A lookup that errors out is inconclusive, not a verdict.
    confirmed, _, remaining = _wait_with_lookups(
        monkeypatch,
        [Flow360WebError("transient"), False],
        timeout_seconds=60,
        poll_interval_seconds=5,
    )

    assert confirmed is True
    assert remaining == []


def test_wait_gives_up_at_the_deadline(monkeypatch):
    # Still there at the deadline means unresolved; the loop must be bounded and must
    # not sleep past it.
    confirmed, slept, _ = _wait_with_lookups(
        monkeypatch, [True] * 4, timeout_seconds=0, poll_interval_seconds=5
    )

    assert confirmed is False
    assert slept == []
