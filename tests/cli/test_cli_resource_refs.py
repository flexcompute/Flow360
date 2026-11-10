import pytest

from flow360.cli.resource_refs import (
    ResourceRefError,
    parse_resource_ref,
    require_resource_type,
)


@pytest.mark.parametrize(
    ("resource_id", "resource_type"),
    [
        ("prj-123", "Project"),
        ("geo-123", "Geometry"),
        ("sm-123", "SurfaceMesh"),
        ("vm-123", "VolumeMesh"),
        ("case-123", "Case"),
        ("dft-123", "Draft"),
        ("folder-123", "Folder"),
        ("ROOT.FLOW360", "Folder"),
        ("ROOT.FLOW360.123", "Folder"),
    ],
)
def test_parse_resource_ref_detects_type_from_prefix(resource_id, resource_type):
    resource_ref = parse_resource_ref(resource_id)

    assert resource_ref.id == resource_id
    assert resource_ref.resource_type == resource_type


def test_parse_resource_ref_trims_outer_whitespace():
    resource_ref = parse_resource_ref("  dft-123  ")

    assert resource_ref.id == "dft-123"
    assert resource_ref.resource_type == "Draft"


def test_parse_resource_ref_rejects_unknown_prefix():
    with pytest.raises(ResourceRefError, match="Unsupported resource ID prefix"):
        parse_resource_ref("foo-123")


def test_parse_resource_ref_rejects_malformed_id():
    with pytest.raises(ResourceRefError, match="expected '<prefix>-...' shape"):
        parse_resource_ref("foo")


def test_require_resource_type_rejects_wrong_kind():
    with pytest.raises(ResourceRefError, match="Expected a Draft ID"):
        require_resource_type("prj-123", "Draft")
