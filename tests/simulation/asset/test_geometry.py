import copy
import re
import unittest

import pytest

from flow360 import exceptions as ex
from flow360.component.geometry import Geometry
from flow360.examples import Cylinder3D

assertions = unittest.TestCase("__init__")


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


def test_draft_geometry_from_file():
    with pytest.raises(
        ex.Flow360FileError,
        match=re.escape(
            "The given file: file.unsupported is not a supported geometry file. Allowed file suffixes are: ['.csm', '.egads', '.sat', '.sab', '.asat', '.asab', '.iam', '.catpart', '.catproduct', '.gt', '.prt', '.prt.*', '.asm.*', '.par', '.asm', '.psm', '.sldprt', '.sldasm', '.stp', '.step', '.x_t', '.xmt_txt', '.x_b', '.xmt_bin', '.3dm', '.ipt']"
        ),
    ):
        sm = Geometry.from_file("file.unsupported")

    with pytest.raises(ex.Flow360FileError, match="not found"):
        sm = Geometry.from_file("data/geometry/no_exist.step")

    Cylinder3D.get_files()
    sm = Geometry.from_file(
        Cylinder3D.geometry,
        project_name="my_geo",
        solver_version="Mock_version-99.9.9",
        length_unit="cm",
    )
    assert sm.project_name == "my_geo"
    assert sm.length_unit == "cm"


def test_geometry_rename_entity(mock_id, mock_response):

    geometry_initial = Geometry.from_cloud(id="geo-2877e124-96ff-473d-864b-11eec8648d42")

    geometry = copy.deepcopy(geometry_initial)
    entity_type_name = "edge"
    with pytest.raises(
        ex.Flow360ValueError,
        match=(
            f"Renaming failed: Could not find {entity_type_name} grouping info in the draft's simulation settings."
            "Please group them first before remaning the entities."
        ),
    ):
        geometry._rename_entity(
            entity_type_name=entity_type_name,
            current_name_pattern="wing",
            new_name_prefix="NewWing",
        )

    geometry.group_edges_by_tag("edgeName")
    with pytest.raises(
        ex.Flow360ValueError,
        match=(
            "Renaming failed: No entity is found to match the input name pattern: trailingEdgeTypo."
        ),
    ):
        geometry._rename_entity(
            entity_type_name=entity_type_name,
            current_name_pattern="trailingEdgeTypo",
            new_name_prefix="NewTrailingEdge",
        )

    with pytest.raises(
        ex.Flow360ValueError,
        match=("Renaming failed: An entity with the new name: leadingEdge already exists."),
    ):
        geometry._rename_entity(
            entity_type_name=entity_type_name,
            current_name_pattern="trailingEdge",
            new_name_prefix="leadingEdge",
        )

    geometry._rename_entity(
        entity_type_name=entity_type_name,
        current_name_pattern="trailingEdge",
        new_name_prefix="newTrailingEdge",
    )
    assert geometry["newTrailingEdge"].private_attribute_id == "trailingEdge"

    geometry = copy.deepcopy(geometry_initial)
    geometry.group_edges_by_tag("edgeName")
    geometry._rename_entity(
        entity_type_name=entity_type_name,
        current_name_pattern="*Edge",
        new_name_prefix="newEdges",
    )

    assert geometry["newEdges_0001"].private_attribute_id == "leadingEdge"
    assert geometry["newEdges_0002"].private_attribute_id == "trailingEdge"
