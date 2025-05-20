import re

import pydantic as pd
import pytest

from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.framework.updater_utils import (
    Flow360Version,
    deprecation_reminder,
)


def test_init_valid_version():
    v = Flow360Version("24.11.2")
    assert v.major == 24
    assert v.minor == 11
    assert v.patch == 2

    v = Flow360Version("25.2.0b1")
    assert v.major == 25
    assert v.minor == 2
    assert v.patch == 0


@pytest.mark.parametrize(
    "invalid_version",
    [
        "",
        "1.2",
        "1.2.3.4",
        "abc.def.gh",
        "24.11.2b",
    ],  # Only two parts  # Four parts  # Non-integer #No beta version
)
def test_init_invalid_version(invalid_version):
    with pytest.raises(ValueError):
        Flow360Version(invalid_version)


def test_equality():
    v1 = Flow360Version("1.2.3")
    v2 = Flow360Version("1.2.3")
    v3 = Flow360Version("1.2.4")
    v4 = Flow360Version("1.2.3b4")

    assert v1 == v2
    assert v1 == v4
    assert v1 != v3


def test_comparison():
    v1 = Flow360Version("1.2.3")
    v2 = Flow360Version("1.2.4")
    v3 = Flow360Version("2.0.0")
    v4 = Flow360Version("1.2.3b4")

    # Test <, <=, >, >=
    assert v1 < v2
    assert v1 <= v2
    assert v4 < v2
    assert v4 <= v2
    assert v2 < v3
    assert v2 <= v3
    assert v3 > v1
    assert v3 >= v1
    assert v3 > v4
    assert v3 >= v4

    # Additional comparisons
    assert v1 < v3
    assert v1 != v2
    assert v2 != v3


def test_deprecation_reminder():
    class SomeModel(Flow360BaseModel):
        field_a: int = 1

        @pd.model_validator(mode="after")
        @deprecation_reminder("20.1.2")
        def _deprecation_validator(self):
            return self

    with pytest.raises(
        ValueError,
        match=re.escape(
            "[INTERNAL] This validator or function is detecting/handling deprecated schema that was scheduled "
            "to be removed since 20.1.2. Please deprecate the schema now, write updater and remove related checks."
        ),
    ):
        SomeModel(field_a=123)
