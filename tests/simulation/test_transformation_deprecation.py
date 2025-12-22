import pytest

import flow360.component.simulation.units as u
from flow360.component.simulation.primitives import GeometryBodyGroup
from flow360.exceptions import Flow360DeprecationError


def test_transformation_deprecation_warning(capsys):
    # Setup
    bg = GeometryBodyGroup(
        name="test_bg", private_attribute_tag_key="tag", private_attribute_sub_components=[]
    )

    # Test getter raises deprecation error
    with pytest.raises(Flow360DeprecationError) as exc_info:
        val = bg.transformation

    assert "GeometryBodyGroup.transformation is deprecated" in str(exc_info.value)
    assert "Please use CoordinateSystem" in str(exc_info.value)

    # Test setter raises deprecation error
    with pytest.raises(Flow360DeprecationError) as exc_info:
        bg.transformation = "something"

    assert "GeometryBodyGroup.transformation is deprecated" in str(exc_info.value)
    assert "Please use CoordinateSystem" in str(exc_info.value)
