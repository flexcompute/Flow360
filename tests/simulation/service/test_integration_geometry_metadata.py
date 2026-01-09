import os

import pytest

from flow360.component.geometry import Geometry
from flow360.component.simulation.framework.entity_registry import EntityRegistryView
from flow360.component.simulation.primitives import Edge, Surface
from flow360.log import set_logging_level

set_logging_level("DEBUG")

# === Box.csm: ===
# box 0 0 0 1 1 1
# attribute ByBody $Box1
# attribute ByTheType $IamFaces
# attribute IsFirstBox $Yes

# select edge
# attribute ByBody $Box1
# attribute ByTheType $IamEdges

# box 0 0 0 1 1 1
# attribute ByBody $Box2
# attribute ByTheType $IamFaces
# translate 2 0 0

# select edge
# attribute ByBody $Box2
# attribute ByTheType $IamEdges

# select face 3
# attribute RandomName $IamIsolated


def _get_property_values(view: EntityRegistryView, property_name: str) -> list[str]:
    return [entity.__getattribute__(property_name) for entity in view._entities]


@pytest.mark.usefixtures("s3_download_override")
def test_multi_body_geometry(mock_id, mock_response):
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    geometry = Geometry.from_cloud(mock_id)

    # Test grouping both edge and face at the same time.
    geometry.group_faces_by_tag(tag_name="ByTheType")
    geometry.group_edges_by_tag(tag_name="ByTheType")
    surface_bucket = geometry.internal_registry.view(Surface)
    edge_bucket = geometry.internal_registry.view(Edge)
    assert _get_property_values(surface_bucket, "name") == ["IamFaces"]
    assert set(_get_property_values(surface_bucket, "private_attribute_sub_components")[0]) == {
        "body01_face0002",
        "body02_face0004",
        "body02_face0006",
        "body02_face0005",
        "body01_face0004",
        "body01_face0005",
        "body01_face0006",
        "body02_face0001",
        "body02_face0002",
        "body02_face0003",
        "body01_face0003",
        "body01_face0001",
    }
    assert _get_property_values(edge_bucket, "name") == ["IamEdges"]
    assert set(_get_property_values(edge_bucket, "private_attribute_sub_components")[0]) == {
        "body01_edge0001",
        "body01_edge0002",
        "body01_edge0003",
        "body01_edge0004",
        "body01_edge0005",
        "body01_edge0006",
        "body01_edge0007",
        "body01_edge0008",
        "body01_edge0009",
        "body01_edge0010",
        "body01_edge0011",
        "body01_edge0012",
        "body02_edge0001",
        "body02_edge0002",
        "body02_edge0003",
        "body02_edge0004",
        "body02_edge0005",
        "body02_edge0006",
        "body02_edge0007",
        "body02_edge0008",
        "body02_edge0009",
        "body02_edge0010",
        "body02_edge0011",
        "body02_edge0012",
    }

    # Test by body grouping
    geometry.reset_face_grouping()
    geometry.reset_edge_grouping()
    geometry.group_faces_by_tag(tag_name="ByBody")
    geometry.group_edges_by_tag(tag_name="ByBody")
    surface_bucket = geometry.internal_registry.view(Surface)
    edge_bucket = geometry.internal_registry.view(Edge)

    assert set(_get_property_values(surface_bucket, "name")) == {"Box1", "Box2"}
    assert set(_get_property_values(surface_bucket, "private_attribute_sub_components")[0]) == {
        "body01_face0005",
        "body01_face0002",
        "body01_face0003",
        "body01_face0006",
        "body01_face0001",
        "body01_face0004",
    }
    assert set(_get_property_values(surface_bucket, "private_attribute_sub_components")[1]) == {
        "body02_face0004",
        "body02_face0005",
        "body02_face0002",
        "body02_face0003",
        "body02_face0006",
        "body02_face0001",
    }
    assert set(_get_property_values(edge_bucket, "name")) == {"Box1", "Box2"}
    assert set(_get_property_values(edge_bucket, "private_attribute_sub_components")[0]) == {
        "body01_edge0001",
        "body01_edge0002",
        "body01_edge0003",
        "body01_edge0004",
        "body01_edge0005",
        "body01_edge0006",
        "body01_edge0007",
        "body01_edge0008",
        "body01_edge0009",
        "body01_edge0010",
        "body01_edge0011",
        "body01_edge0012",
    }
    assert set(_get_property_values(edge_bucket, "private_attribute_sub_components")[1]) == {
        "body02_edge0001",
        "body02_edge0002",
        "body02_edge0003",
        "body02_edge0004",
        "body02_edge0005",
        "body02_edge0006",
        "body02_edge0007",
        "body02_edge0008",
        "body02_edge0009",
        "body02_edge0010",
        "body02_edge0011",
        "body02_edge0012",
    }

    geometry.reset_face_grouping()
    geometry.reset_edge_grouping()
    geometry.group_faces_by_tag(tag_name="RandomName")
    surface_bucket = geometry.internal_registry.view(Surface)
    assert set(_get_property_values(surface_bucket, "name")) == {
        "IamIsolated",
        "body01_face0001",
        "body01_face0002",
        "body01_face0003",
        "body01_face0004",
        "body01_face0005",
        "body01_face0006",
        "body02_face0001",
        "body02_face0002",
        "body02_face0004",
        "body02_face0005",
        "body02_face0006",
    }

    ##-- Test the __getitem__ method
    geometry.reset_face_grouping()
    geometry.reset_edge_grouping()
    geometry.group_faces_by_tag(tag_name="ByBody")
    geometry.group_edges_by_tag(tag_name="ByBody")

    assert geometry["Box1"][0].private_attribute_entity_type_name == "Surface"
    assert geometry["Box1"][0].name == "Box1"
    assert geometry["Box1"][0].private_attribute_tag_key == "ByBody"
    assert geometry["Box1"][0].private_attribute_sub_components == [
        "body01_face0001",
        "body01_face0002",
        "body01_face0003",
        "body01_face0004",
        "body01_face0005",
        "body01_face0006",
    ]

    assert geometry["Box1"][1].private_attribute_entity_type_name == "Edge"
    assert geometry["Box1"][1].name == "Box1"
    assert geometry["Box1"][1].private_attribute_tag_key == "ByBody"
    assert geometry["Box1"][1].private_attribute_sub_components == [
        "body01_edge0001",
        "body01_edge0002",
        "body01_edge0003",
        "body01_edge0004",
        "body01_edge0005",
        "body01_edge0006",
        "body01_edge0007",
        "body01_edge0008",
        "body01_edge0009",
        "body01_edge0010",
        "body01_edge0011",
        "body01_edge0012",
    ]
