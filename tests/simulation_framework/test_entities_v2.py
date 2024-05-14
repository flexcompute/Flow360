from typing import Optional

import pydantic as pd
import pytest

from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.framework.entity_base import EntityBase, EntityList
from flow360.component.simulation.framework.entity_registry import EntityRegistry
from flow360.component.simulation.primitives import Box, Cylinder
from flow360.log import set_logging_level

set_logging_level("DEBUG")


class TempSurface(EntityBase):
    _entity_type = "Surfaces"


class ZoneList(Flow360BaseModel):
    entities: EntityList[Box, Cylinder, str] = pd.Field(["*"])


@pytest.fixture
def cleanup():
    EntityRegistry.clear()


@pytest.fixture
def my_cylinder1():
    return Cylinder(
        name="zone/Cylinder1",
        height=11,
        axis=(1, 0, 0),
        inner_radius=1,
        outer_radius=2,
        center=(1, 2, 3),
    )


@pytest.fixture
def my_cylinder2():
    return Cylinder(
        name="zone/Cylinder2",
        height=12,
        axis=(1, 0, 0),
        inner_radius=1,
        outer_radius=2,
        center=(1, 2, 3),
    )


@pytest.fixture
def my_cylinder3():
    return Cylinder(
        name="zone/Cylinder3",
        height=13,
        axis=(1, 0, 0),
        inner_radius=1,
        outer_radius=2,
        center=(1, 2, 3),
    )


@pytest.fixture
def my_box_zone1():
    return Box(name="zone/Box1", center=(1, 2), size=(123, 1, 0.1), axes=((1, 0, 0), (0, 1, 0)))


@pytest.fixture
def my_surface1():
    return TempSurface(name="Box_in_name_but_is_surface")


@pytest.fixture
def my_box_zone2():
    return Box(name="zone/Box2", center=(1, 22), size=(123, 1, 0.1), axes=((1, 0, 0), (1, 1, 0)))


@pytest.fixture
def my_box_zone3():
    return Box(name="zone/Box3", center=(1, 23), size=(123, 1, 0.1), axes=((1, 0, 0), (0, 1, 0)))


def test_entity_expansion(
    cleanup, my_cylinder1, my_cylinder2, my_cylinder3, my_box_zone1, my_surface1
):
    expanded_entities = ZoneList(
        entities=["*Box*", my_cylinder3, my_box_zone1]
    ).entities.get_expanded_entities()
    # Matches "Box*" but not Union[Cylinder, Box, str]
    assert my_surface1 not in expanded_entities
    # Does not match anything
    assert my_cylinder2 not in expanded_entities
    # Matches my_cylinder3
    assert my_cylinder3 in expanded_entities
    # Matches "Box*"
    assert my_box_zone1 in expanded_entities
    assert (
        len(expanded_entities) == 2
    )  # expansion of "*Box*" and my_box_zone1 does not result in duplicates


def test_by_reference_registry(cleanup, my_cylinder3):
    my_cylinder3.height = 131
    for entity in EntityRegistry.get_entities(Cylinder):
        if isinstance(entity, Cylinder) and entity.name == "zone/Cylinder3":
            assert entity.height == 131


def test_hard_copy_expansion(cleanup, my_cylinder3):
    expanded_entities = ZoneList(entities=["*Box*", my_cylinder3]).entities.get_expanded_entities()
    my_cylinder3.height = 1012
    for entity in expanded_entities:
        if isinstance(entity, Cylinder) and entity.name == "zone/Cylinder3":
            assert entity.height == 13  # unchanged


def test_multiple_expansion(cleanup, my_cylinder3, my_box_zone1, request):
    my_fd_zone = ZoneList(entities=["*Box*", my_cylinder3])
    # Suppose afterwards some thing triggered the expansion
    expanded_entities = my_fd_zone.entities.get_expanded_entities()
    assert my_box_zone1 in expanded_entities
    assert my_cylinder3 in expanded_entities
    # Suppose now user loads the mesh and it populated the EntityRegistry with more zones.
    my_box_zone2 = request.getfixturevalue("my_box_zone2")
    my_box_zone3 = request.getfixturevalue("my_box_zone3")
    # And now we expand again
    expanded_entities = my_fd_zone.entities.get_expanded_entities()
    assert my_box_zone1 in expanded_entities
    assert my_cylinder3 in expanded_entities
    assert my_box_zone2 in expanded_entities
    assert my_box_zone3 in expanded_entities
    assert len(expanded_entities) == 4  # 3 Box, 1 Cylinder


def test_duplicate_entities(cleanup, my_cylinder3, my_box_zone1):
    expanded_entities = ZoneList(entities=["*Box*", my_cylinder3]).entities.get_expanded_entities()
    assert my_box_zone1 in expanded_entities
    assert my_cylinder3 in expanded_entities
    try:
        my_cylinder3.copy()
    except ValueError as e:
        assert (
            "Change is necessary when copying an entity as there cannot be two identical entities at the same time. Please use update parameter to change the entity attributes."
            in str(e)
        )
    try:
        my_cylinder3.copy(update={"height": 1.0234})
    except ValueError as e:
        assert (
            "Copying an entity requires a new name to be specified. Please provide a new name in the update dictionary."
            in str(e)
        )
    try:
        my_cylinder3.copy(update={"height": 1.0234, "name": my_cylinder3.name})
    except ValueError as e:
        assert (
            "Copying an entity requires a new name to be specified. Please provide a new name in the update dictionary."
            in str(e)
        )
    try:
        ZoneList(entities=["*Box*", my_cylinder3, my_cylinder3])
    except ValueError as e:
        assert "Duplicate entity found, name: zone/Cylinder3" in str(e)

    try:
        ZoneList(entities=["zone/Box1", "zone/Box1", my_cylinder3])
    except ValueError as e:
        assert "Duplicate entity found: zone/Box1" in str(e)
    EntityRegistry.clear()


def test_copying_entity(cleanup, my_cylinder3, my_box_zone1, request):
    expanded_entities = ZoneList(entities=["*Box*", my_cylinder3]).entities.get_expanded_entities()

    assert len(EntityRegistry.get_entities(Cylinder)) == 2
    my_cylinder3_2 = my_cylinder3.copy(update={"height": 8119, "name": "zone/Cylinder3-2"})
    assert len(EntityRegistry.get_entities(Cylinder)) == 3

    for entity in EntityRegistry.get_entities(Cylinder):
        if entity.name == "zone/Cylinder3-2":
            assert entity.height == 8119

    for entity in expanded_entities:
        if isinstance(entity, Cylinder) and entity.name == "zone/Cylinder3-2":
            assert entity.height == 8119


def test_get_entities(
    cleanup,
    my_cylinder1,
    my_cylinder2,
    my_cylinder3,
    my_box_zone1,
    my_surface1,
    my_box_zone2,
    my_box_zone3,
):
    all_box_entities = EntityRegistry.get_entities(Box)
    assert len(all_box_entities) == 6
    assert my_box_zone1 in all_box_entities
    assert my_box_zone2 in all_box_entities
    assert my_box_zone3 in all_box_entities
    assert my_cylinder1 in all_box_entities  # my_cylinder1 is not a Box but is a _volumeZoneBase
    assert my_cylinder2 in all_box_entities
    assert my_cylinder3 in all_box_entities


def test_default_entities(
    cleanup,
    my_cylinder1,
    my_cylinder2,
    my_cylinder3,
    my_box_zone1,
    my_surface1,
    my_box_zone2,
    my_box_zone3,
):
    expanded_entities = ZoneList(entities=["*"]).entities.get_expanded_entities()
    assert my_cylinder1 in expanded_entities
    assert my_cylinder2 in expanded_entities
    assert my_cylinder3 in expanded_entities
    assert my_box_zone1 in expanded_entities
    assert my_box_zone2 in expanded_entities
    assert my_box_zone3 in expanded_entities
    assert len(expanded_entities) == 6

    my_fd_zone_2 = ZoneList()
    expanded_entities_2 = my_fd_zone_2.entities.get_expanded_entities()
    assert expanded_entities_2 == expanded_entities


def test_entities_input_interface(cleanup, my_cylinder1, my_cylinder2):
    expanded_entities = ZoneList(entities=my_cylinder1).entities.get_expanded_entities()
    assert expanded_entities == [my_cylinder1]
    expanded_entities = ZoneList(entities="zone/Cylinder2").entities.get_expanded_entities()
    assert expanded_entities == [my_cylinder2]
    try:
        expanded_entities = ZoneList(entities=1).entities.get_expanded_entities()
    except ValueError as e:
        assert "Invalid input type to `entities`: <class 'int'>" in str(e)

    try:
        expanded_entities = ZoneList(entities=[1, "*"]).entities.get_expanded_entities()
    except pd.ValidationError as e:
        assert "Input should be a valid dictionary or instance of Box" in str(e)
        assert "Input should be a valid dictionary or instance of Cylinder" in str(e)
        assert "Input should be a valid string" in str(e)

    try:
        expanded_entities = ZoneList(entities=[]).entities.get_expanded_entities()
    except ValueError as e:
        assert "Invalid input type to `entities`, list is empty." in str(e)

    try:
        expanded_entities = ZoneList(entities=None).entities.get_expanded_entities()
    except pd.ValidationError as e:
        assert "Value error, Invalid input type to `entities`: <class 'NoneType'>" in str(e)

    try:
        expanded_entities = ZoneList(
            entities=["Non_existing_volume"]
        ).entities.get_expanded_entities()
    except ValueError as e:
        assert (
            "Failed to find any matching entity with ['Non_existing_volume']. Please check the input to entities."
            in str(e)
        )
