from tests.simulation.conftest import _approx_equal


def test_approx_equal_distinguishes_bool_from_numeric_types():
    assert _approx_equal(True, True)
    assert _approx_equal(False, False)
    assert not _approx_equal(True, 1)
    assert not _approx_equal(False, 0.0)
