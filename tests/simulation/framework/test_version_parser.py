import pytest

from flow360.component.simulation.framework.updater_utils import Flow360Version


def test_init_valid_version():
    v = Flow360Version("24.11.2")
    assert v.major == 24
    assert v.minor == 11
    assert v.patch == 2


@pytest.mark.parametrize(
    "invalid_version",
    ["", "1.2", "1.2.3.4", "abc.def.gh"],  # Only two parts  # Four parts  # Non-integer
)
def test_init_invalid_version(invalid_version):
    with pytest.raises(ValueError):
        Flow360Version(invalid_version)


def test_equality():
    v1 = Flow360Version("1.2.3")
    v2 = Flow360Version("1.2.3")
    v3 = Flow360Version("1.2.4")

    assert v1 == v2
    assert v1 != v3


def test_comparison():
    v1 = Flow360Version("1.2.3")
    v2 = Flow360Version("1.2.4")
    v3 = Flow360Version("2.0.0")

    # Test <, <=, >, >=
    assert v1 < v2
    assert v1 <= v2
    assert v2 < v3
    assert v2 <= v3
    assert v3 > v1
    assert v3 >= v1
    # Additional comparisons
    assert v1 < v3
    assert v1 != v2
    assert v2 != v3
