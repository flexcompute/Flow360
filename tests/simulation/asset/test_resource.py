import pytest

from flow360.component.case import Case, CaseMeta
from flow360.component.resource_base import (
    AssetMetaBaseModel,
    AssetMetaBaseModelV2,
    local_metadata_builder,
)


def test_category_tag():
    asset_v1 = AssetMetaBaseModel(
        name="test",
        user_id="test",
        id="id",
        parent_id=None,
        solver_version=None,
        status="unknown",
    )

    asset_v2 = AssetMetaBaseModelV2(**local_metadata_builder("test", "test"))

    assert asset_v1.category_tag is None
    assert asset_v2.category_tag is None

    asset_v1.tags.append("a")
    asset_v1.tags.append("b")

    asset_v2.tags.append("a")
    asset_v2.tags.append("b")

    assert asset_v1.category_tag == "a"
    assert asset_v2.category_tag == "a"


def test_add_tag():
    resource = Case._from_meta(
        meta=CaseMeta(
            id="case-11111111-1111-1111-1111-111111111111",
            name="name",
            status="completed",
            userId="user-id",
            caseMeshId="mesh-id",
            cloud_path_prefix="s3://flow360cases-v1/users/user-id",
        )
    )

    assert resource.info.tags == []
    assert resource.info.category_tag == None

    resource.add_tag("a")

    assert resource.info.tags == ["a"]
    assert resource.info.category_tag == "a"

    resource.add_tag("b")

    assert resource.info.tags == ["a", "b"]
    assert resource.info.category_tag == "a"

    resource.add_tag("c", category_tag=True)

    assert resource.info.tags == ["c", "a", "b"]
    assert resource.info.category_tag == "c"

    resource.add_tag("a")

    assert resource.info.tags == ["c", "b", "a"]

    resource.add_tag("a", category_tag=True)

    assert resource.info.tags == ["a", "c", "b"]
