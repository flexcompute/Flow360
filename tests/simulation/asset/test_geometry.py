import os
import re
import unittest

import pytest

from flow360 import exceptions as ex
from flow360.component.geometry import Geometry, GeometryMeta
from flow360.component.resource_base import local_metadata_builder
from flow360.component.simulation.primitives import SnappyBody, Surface
from flow360.examples import Cylinder3D

assertions = unittest.TestCase("__init__")

geo_meta = {
    "id": "geo-fcbe1113-a70b-43b9-a4f3-bbeb122d64fb",
    "name": "airplane_simple_obtained_from_csm_by_esp",
    "s3_path": "s3://mesh-bucket/users/user-29083u29irfjsdkfns/geo-fcbe1113-a70b-43b9-a4f3-bbeb122d64fb",
}


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


@pytest.fixture
def stl_geo_meta():
    geo_meta = {
        "id": "geo-b2ca24af-f60d-4fb3-8120-c653f3e65be6",
        "name": "stl_mixed_convention",
        "s3_path": "s3://flow360meshes-v1/users/user-id",
    }

    geometry = Geometry.from_local_storage(
        geometry_id=geo_meta["id"],
        local_storage_path=os.path.join("../../data", geo_meta["id"]),
        meta_data=GeometryMeta(
            **local_metadata_builder(
                id=geo_meta["id"],
                name=geo_meta["name"],
                cloud_path_prefix=geo_meta["s3_path"].rsplit("/", 1)[0],
                status="processed",
            )
        ),
    )
    return geometry


def test_draft_geometry_from_file():
    with pytest.raises(
        ex.Flow360FileError,
        match=re.escape(
            "The given file: file.unsupported is not a supported geometry or surface mesh file. Allowed file suffixes are: ['.csm', '.egads', '.sat', '.sab', '.asat', '.asab', '.iam', '.catpart', '.catproduct', '.gt', '.prt', '.prt.*', '.asm.*', '.par', '.asm', '.psm', '.sldprt', '.sldasm', '.stp', '.step', '.x_t', '.xmt_txt', '.x_b', '.xmt_bin', '.3dm', '.ipt', '.ugrid', '.cgns', '.stl']"
        ),
    ):
        sm = Geometry.from_file("file.unsupported")

    with pytest.raises(ex.Flow360FileError, match="not found"):
        sm = Geometry.from_file("data/geometry/no_exist.stp")

    Cylinder3D.get_files()
    sm = Geometry.from_file(
        Cylinder3D.geometry,
        project_name="my_geo",
        solver_version="Mock_version-99.9.9",
        length_unit="cm",
    )
    assert sm.project_name == "my_geo"
    assert sm.length_unit == "cm"


def test_geometry_rename_edges():

    geometry = Geometry.from_local_storage(
        geometry_id=geo_meta["id"],
        local_storage_path=os.path.join("../../data", geo_meta["id"]),
        meta_data=GeometryMeta(
            **local_metadata_builder(
                id=geo_meta["id"],
                name=geo_meta["name"],
                cloud_path_prefix=geo_meta["s3_path"].rsplit("/", 1)[0],
                status="processed",
            )
        ),
    )

    # Test rename edge
    with pytest.raises(
        ex.Flow360ValueError,
        match=(
            f"Renaming failed: Could not find edge grouping info in the draft's simulation settings."
            "Please group them first before renaming the entities."
        ),
    ):
        geometry.rename_edges(
            current_name_pattern="body00001_edge00001",
            new_name_prefix="body00001_edge00001_rename",
        )

    geometry.group_edges_by_tag("edgeId")
    assert geometry.edge_group_tag == "edgeId"
    with pytest.raises(
        ex.Flow360ValueError,
        match=(
            "Renaming failed: No entity is found to match the input name pattern: body00001_edge00001_typo."
        ),
    ):
        geometry.rename_edges(
            current_name_pattern="body00001_edge00001_typo",
            new_name_prefix="body00001_edge00001_rename",
        )

    with pytest.raises(
        ex.Flow360ValueError,
        match=("Renaming failed: An entity with the new name: body00001_edge00002 already exists."),
    ):
        geometry.rename_edges(
            current_name_pattern="body00001_edge00001",
            new_name_prefix="body00001_edge00002",
        )

    geometry.rename_edges(current_name_pattern="body00001_edge0003*", new_name_prefix="newEdge")
    for i in range(4):
        assert geometry[f"newEdge_000{i+1}"].private_attribute_id == f"body00001_edge0003{i}"

    geometry.rename_edges(current_name_pattern="newEdge_*", new_name_prefix="newEdgeV2")
    for i in range(4):
        assert geometry[f"newEdgeV2_000{i+1}"].private_attribute_id == f"body00001_edge0003{i}"

    geometry.reset_edge_grouping()
    assert geometry.edge_group_tag == None
    with pytest.raises(
        ex.Flow360ValueError,
        match=(
            f"Renaming failed: Could not find edge grouping info in the draft's simulation settings."
            "Please group them first before renaming the entities."
        ),
    ):
        geometry.rename_edges(
            current_name_pattern="newEdge_0002",
            new_name_prefix="newEdge_0012",
        )
    with pytest.raises(
        ValueError,
        match=(f"No entity found in registry with given name/naming pattern: 'newEdge_0001'."),
    ):
        assert geometry["newEdge_0001"]


def test_geometry_rename_surfaces():

    geometry = Geometry.from_local_storage(
        geometry_id=geo_meta["id"],
        local_storage_path=os.path.join("../../data", geo_meta["id"]),
        meta_data=GeometryMeta(
            **local_metadata_builder(
                id=geo_meta["id"],
                name=geo_meta["name"],
                cloud_path_prefix=geo_meta["s3_path"].rsplit("/", 1)[0],
                status="processed",
            )
        ),
    )

    # Test rename face
    with pytest.raises(
        ex.Flow360ValueError,
        match=(
            f"Renaming failed: Could not find face grouping info in the draft's simulation settings."
            "Please group them first before renaming the entities."
        ),
    ):
        geometry.rename_surfaces(
            current_name_pattern="body00001_face00001",
            new_name_prefix="body00001_face00001_rename",
        )

    geometry.group_faces_by_tag("faceId")
    assert geometry.face_group_tag == "faceId"
    with pytest.raises(
        ex.Flow360ValueError,
        match=(
            "Renaming failed: No entity is found to match the input name pattern: body00001_face00001_typo."
        ),
    ):
        geometry.rename_surfaces(
            current_name_pattern="body00001_face00001_typo",
            new_name_prefix="body00001_face00001_rename",
        )

    with pytest.raises(
        ex.Flow360ValueError,
        match=("Renaming failed: An entity with the new name: body00001_face00002 already exists."),
    ):
        geometry.rename_surfaces(
            current_name_pattern="body00001_face00001",
            new_name_prefix="body00001_face00002",
        )

    geometry.rename_surfaces(current_name_pattern="farfield_*", new_name_prefix="newFarfield")

    assert (
        geometry["newFarfield_0001"].private_attribute_id
        == "farfield_only_sphere_volume_mesh.lb8.ugrid_1"
    )
    assert (
        geometry["newFarfield_0002"].private_attribute_id
        == "farfield_only_sphere_volume_mesh.lb8.ugrid_2"
    )

    geometry.reset_face_grouping()
    assert geometry.face_group_tag == None
    with pytest.raises(
        ex.Flow360ValueError,
        match=(
            f"Renaming failed: Could not find face grouping info in the draft's simulation settings."
            "Please group them first before renaming the entities."
        ),
    ):
        geometry.rename_surfaces(
            current_name_pattern="newFarfield_0002",
            new_name_prefix="newFarfield_0003",
        )

    with pytest.raises(
        ValueError,
        match=(f"No entity found in registry with given name/naming pattern: 'newFarfield_0001'."),
    ):
        assert geometry["newFarfield_0001"]


def test_geometry_rename_body_groups():

    geometry = Geometry.from_local_storage(
        geometry_id=geo_meta["id"],
        local_storage_path=os.path.join("../../data", geo_meta["id"]),
        meta_data=GeometryMeta(
            **local_metadata_builder(
                id=geo_meta["id"],
                name=geo_meta["name"],
                cloud_path_prefix=geo_meta["s3_path"].rsplit("/", 1)[0],
                status="processed",
            )
        ),
    )
    with pytest.raises(
        ex.Flow360ValueError,
        match=(
            f"Renaming failed: Could not find body grouping info in the draft's simulation settings."
            "Please group them first before renaming the entities."
        ),
    ):
        geometry.rename_body_groups(
            current_name_pattern="airplane_simple_obtained_from_csm_by_esp.step",
            new_name_prefix="step_body",
        )

    geometry.group_bodies_by_tag("groupByFile")
    assert geometry.body_group_tag == "groupByFile"
    with pytest.raises(
        ex.Flow360ValueError,
        match=(
            "Renaming failed: No entity is found to match the input name pattern: "
            "airplane_simple_obtained_from_csm_by_esp.step_typo."
        ),
    ):
        geometry.rename_body_groups(
            current_name_pattern="airplane_simple_obtained_from_csm_by_esp.step_typo",
            new_name_prefix="airplane_mesh",
        )

    with pytest.raises(
        ex.Flow360ValueError,
        match=(
            "Renaming failed: An entity with the new name: "
            "farfield_only_sphere_volume_mesh.lb8.ugrid already exists."
        ),
    ):
        geometry.rename_body_groups(
            current_name_pattern="airplane_simple_obtained_from_csm_by_esp.step",
            new_name_prefix="farfield_only_sphere_volume_mesh.lb8.ugrid",
        )

    geometry.rename_body_groups(
        current_name_pattern="airplane_simple_obtained_from_csm_by_esp.step",
        new_name_prefix="airplane_mesh",
    )
    assert (
        geometry["airplane_mesh"].private_attribute_id
        == "airplane_simple_obtained_from_csm_by_esp.step"
    )

    geometry.group_bodies_by_tag("FCsource")
    assert geometry.body_group_tag == "FCsource"
    geometry.rename_body_groups("airplane*", "newAirplane")

    assert (
        geometry["newAirplane_0001"].private_attribute_id
        == "airplane_simple_obtained_from_csm_by_esp.step"
    )
    assert geometry["newAirplane_0002"].private_attribute_id == "airplane_translate_in_z_-5.stl"

    geometry.reset_body_grouping()
    assert geometry.body_group_tag == None
    with pytest.raises(
        ex.Flow360ValueError,
        match=(
            f"Renaming failed: Could not find body grouping info in the draft's simulation settings."
            "Please group them first before renaming the entities."
        ),
    ):
        geometry.rename_body_groups(
            current_name_pattern="newAirplane_0002",
            new_name_prefix="newAirplane_0003",
        )

    with pytest.raises(
        ValueError,
        match=(f"No entity found in registry with given name/naming pattern: 'newAirplane_0002'."),
    ):
        assert geometry["newAirplane_0002"]

def test_geometry_group_for_snappy(stl_geo_meta):
    geo: Geometry = stl_geo_meta

    geo.group_faces_for_snappy()

    # body with one region
    assert isinstance(geo["rr-wh-rim-lhs"], SnappyBody)
    assert len(geo["rr-wh-rim-lhs"]["*"]) == 1
    assert isinstance(geo["rr-wh-rim-lhs"]["*"][0], Surface)
    assert geo["rr-wh-rim-lhs"]["*"][0].name == "rr-wh-rim-lhs"

    # body with more regions
    assert all([isinstance(region, Surface) for region in geo["tunnel"]["*"]])
    assert len(geo["tunnel"]["*"]) == 5

    # registry wildcard
    assert len(geo["uf*"]) == 2
    assert len(geo["velocity*"]) == 10

    # double indexing with wildcard
    assert len(geo["*nn*"]["*"]) == 6

def test_snappy_grouping_not_found_messages(stl_geo_meta):
    geo: Geometry = stl_geo_meta

    geo.group_faces_for_snappy()

    with pytest.raises(
        ValueError,
        match=(f"No entity found in registry with given name/naming pattern: 'dummy'."),
    ):
        assert geo["dummy"]

    with pytest.raises(
        ValueError,
        match=(f"No entity found in registry for parent entities: body-inner-nlyr, tunnel with given name/naming pattern: 'dummy'."),
    ):
        assert geo["*nn*"]["dummy"]

    with pytest.raises(
        KeyError
    ):
        assert geo["body-nose"]["dummy*"]