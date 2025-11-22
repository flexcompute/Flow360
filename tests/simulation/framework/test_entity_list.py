import re
from typing import ClassVar, Literal

import pydantic as pd
import pytest

import flow360 as fl
from flow360.component.simulation.entity_info import SurfaceMeshEntityInfo
from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.framework.entity_base import EntityBase, EntityList
from flow360.component.simulation.framework.param_utils import AssetCache
from flow360.component.simulation.primitives import GenericVolume, Surface


class _SurfaceEntityBase(EntityBase):
    """Base class for surface-like entities (CAD or mesh)."""

    entity_bucket: ClassVar[str] = "surfaces"
    private_attribute_entity_type_name: Literal["_SurfaceEntityBase"] = pd.Field(
        "_SurfaceEntityBase", frozen=True
    )


class TempSurface(_SurfaceEntityBase):
    private_attribute_entity_type_name: Literal["TempSurface"] = pd.Field(
        "TempSurface", frozen=True
    )


class _ParamsStub:
    def __init__(self, asset_cache: AssetCache):
        self.private_attribute_asset_cache = asset_cache


def _build_preview_context(boundary_names: list[str]):
    with fl.SI_unit_system:
        boundaries = [
            Surface(name=name, private_attribute_id=f"{name}_id") for name in boundary_names
        ]
    entity_info = SurfaceMeshEntityInfo(boundaries=boundaries)
    asset_cache = AssetCache(project_entity_info=entity_info)
    return boundaries, _ParamsStub(asset_cache)


def test_entity_list_deserializer_handles_mixed_types_and_selectors():
    """
    Test: EntityList deserializer correctly processes a mixed list of entities and selectors.
    - Verifies that EntityList can accept a list containing both entity instances and selectors.
    - Verifies that entity objects are placed in `stored_entities`.
    - Verifies that EntitySelector objects are placed in `selectors`.
    - Verifies that the types are validated against the EntityList's generic parameters.
    """
    with fl.SI_unit_system:
        selector = Surface.match("*", name="all_surfaces")
        surface_entity = Surface(name="my_surface")
        temp_surface_entity = TempSurface(name="my_temp_surface")
        # This entity should be filtered out as it's not a valid type for this list
        volume_entity = GenericVolume(name="my_volume")

        # Use model_validate to correctly trigger the "before" mode validator
        entity_list = EntityList[Surface, TempSurface].model_validate(
            [selector, surface_entity, temp_surface_entity, volume_entity]
        )

    assert len(entity_list.stored_entities) == 2
    assert entity_list.stored_entities[0] == surface_entity
    assert entity_list.stored_entities[1] == temp_surface_entity

    assert len(entity_list.selectors) == 1
    assert entity_list.selectors[0] == selector


def test_entity_list_discrimination():
    """
    Test: EntityList correctly uses the discriminator field for Pydantic model validation.
    """

    class ConfusingEntity1(EntityBase):
        entity_bucket: ClassVar[str] = "confusing"
        some_value: int = pd.Field(1, gt=1)
        private_attribute_entity_type_name: Literal["ConfusingEntity1"] = pd.Field(
            "ConfusingEntity1", frozen=True
        )

    class ConfusingEntity2(EntityBase):
        entity_bucket: ClassVar[str] = "confusing"
        some_value: int = pd.Field(1, gt=2)
        private_attribute_entity_type_name: Literal["ConfusingEntity2"] = pd.Field(
            "ConfusingEntity2", frozen=True
        )

    class MyModel(Flow360BaseModel):
        entities: EntityList[ConfusingEntity1, ConfusingEntity2]

    # Ensure EntityList requires the discriminator
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Unable to extract tag using discriminator 'private_attribute_entity_type_name'"
        ),
    ):
        MyModel(
            entities={
                "stored_entities": [
                    {
                        "name": "discriminator_is_missing",
                        "some_value": 3,
                    }
                ],
            }
        )

    # Ensure EntityList validates against the correct model based on the discriminator
    with pytest.raises(pd.ValidationError) as err:
        MyModel(
            entities={
                "stored_entities": [
                    {
                        "name": "should_be_confusing_entity_1",
                        "private_attribute_entity_type_name": "ConfusingEntity1",
                        "some_value": 1,  # This violates the gt=1 constraint of ConfusingEntity1
                    }
                ],
            }
        )

    validation_errors = err.value.errors()
    # Pydantic should only try to validate against ConfusingEntity1, resulting in one error.
    # Without discrimination, it would have failed checks for both models.
    assert len(validation_errors) == 1
    assert validation_errors[0]["msg"] == "Input should be greater than 1"
    assert validation_errors[0]["loc"] == (
        "entities",
        "stored_entities",
        0,
        "ConfusingEntity1",
        "some_value",
    )


def test_entity_list_invalid_inputs():
    """
    Test: EntityList deserializer handles various invalid inputs gracefully.
    """
    # 1. Test invalid entity type in list (e.g., int)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Type(<class 'int'>) of input to `entities` (1) is not valid. Expected entity instance."
        ),
    ):
        EntityList[Surface].model_validate([1])

    # 2. Test empty list
    with pytest.raises(
        ValueError,
        match=re.escape("Invalid input type to `entities`, list is empty."),
    ):
        EntityList[Surface].model_validate([])

    # 3. Test None input
    with pytest.raises(
        pd.ValidationError,
        match="Input should be a valid list",
    ):
        EntityList[Surface].model_validate(None)

    # 4. Test list containing only invalid types
    with pytest.raises(
        ValueError,
        match=re.escape("Can not find any valid entity of type ['Surface'] from the input."),
    ):
        with fl.SI_unit_system:
            EntityList[Surface].model_validate([GenericVolume(name="a_volume")])


def test_preview_selection_returns_names_by_default():
    boundaries, params_stub = _build_preview_context(["tail", "wing_leading", "wing_trailing"])
    selector = Surface.match("wing*", name="wing_surfaces")

    entity_list = EntityList[Surface].model_validate([boundaries[0], selector])

    previewed_names = entity_list.preview_selection(params_stub)

    assert previewed_names == ["tail", "wing_leading", "wing_trailing"]


def test_preview_selection_returns_instances_when_requested():
    boundaries, params_stub = _build_preview_context(["body00001", "body00002"])
    selector = Surface.match("body00002", name="second_body")

    entity_list = EntityList[Surface].model_validate([selector])
    entity_list.stored_entities = [boundaries[0].model_dump(mode="json", exclude_none=True)]

    expanded_entities = entity_list.preview_selection(params_stub, return_names=False)

    assert [entity.name for entity in expanded_entities] == ["body00001", "body00002"]
    assert all(isinstance(entity, Surface) for entity in expanded_entities)
