import pytest

from flow360.component.utils import is_valid_uuid, validate_type
from flow360.component.volume_mesh import VolumeMeshMeta
from flow360.exceptions import TypeError, ValueError


def test_validate_type():
    validate_type("str", "meta", str)
    with pytest.raises(TypeError):
        validate_type("str", "meta", VolumeMeshMeta)


def test_valid_uuid():
    is_valid_uuid("123e4567-e89b-12d3-a456-426614174000")
    is_valid_uuid("folder-123e4567-e89b-12d3-a456-426614174000")
    with pytest.raises(ValueError):
        is_valid_uuid("not-a-valid-uuid")

    with pytest.raises(ValueError):
        is_valid_uuid(None)

    is_valid_uuid(None, allow_none=True)
