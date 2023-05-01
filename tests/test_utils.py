import pytest

from flow360.component.utils import validate_type
from flow360.component.volume_mesh import VolumeMeshMeta
from flow360.exceptions import TypeError


def test_validate_type():
    validate_type("str", "meta", str)
    with pytest.raises(TypeError):
        validate_type("str", "meta", VolumeMeshMeta)
