import re
from abc import ABCMeta
from typing import List, Literal, Union

import pydantic as pd
import pytest

import flow360.component.simulation.units as u
from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.framework.entity_base import EntityBase, EntityList
from flow360.component.simulation.framework.entity_registry import EntityRegistry
from flow360.component.simulation.primitives import (
    Box,
    Cylinder,
    GenericSurface,
    GenericVolume,
    _SurfaceEntityBase,
)
from flow360.log import set_logging_level

set_logging_level("DEBUG")


class AssetBase(metaclass=ABCMeta):
    _registry: EntityRegistry

    def __init__(self):
        self._registry = EntityRegistry()

    def __getitem__(self, key: str) -> list[EntityBase]:
        """Use [] to access the registry"""
        if isinstance(key, str) == False:
            raise ValueError(f"Entity naming pattern: {key} is not a string.")
        found_entities = self._registry.find_by_name_pattern(key)
        if found_entities == []:
            raise ValueError(
                f"Failed to find any matching entity with {key}. Please check your input."
            )
        return found_entities


class TempVolumeMesh(AssetBase):
    """Mimicing the final VolumeMesh class"""

    fname: str

    def _get_meta_data(self):
        if self.fname == "volMesh-1.cgns":
            return {
                "zones": {"zone_1": {}, "zone_2": {}, "zone_3": {}},
                "surfaces": {"surface_1": {}, "surface_2": {}},
            }
        elif self.fname == "volMesh-2.cgns":
            return {
                "zones": {"zone_4": {}, "zone_5": {}, "zone_6": {}},
                "surfaces": {"surface_3": {}, "surface_4": {}},
            }
        else:
            raise ValueError("Invalid file name")

    def _populate_registry(self):
        for zone_name in self._get_meta_data()["zones"]:
            self._registry.register(GenericVolume(name=zone_name))
        for surface_name in self._get_meta_data()["surfaces"]:
            self._registry.register(GenericSurface(name=surface_name))

    def __init__(self, file_name: str):
        super().__init__()
        self.fname = file_name
        self._populate_registry()


class TempSurface(_SurfaceEntityBase):
    pass


class TempFluidDynamics(Flow360BaseModel):
    entities: EntityList[GenericVolume, Box, Cylinder, str] = pd.Field(
        alias="volumes", default=None
    )


class TempWallBC(Flow360BaseModel):
    entities: EntityList[GenericSurface, TempSurface, str] = pd.Field(alias="surfaces", default=[])


def _get_supplementary_registry(far_field_type: str):
    """
    Given the supplied partly validated dict (values), populate the supplementary registry
    """
    _supplementary_registry = EntityRegistry()
    if far_field_type == "auto":
        _supplementary_registry.register(TempSurface(name="farfield"))
    return _supplementary_registry


class TempSimulationParam(Flow360BaseModel):

    far_field_type: Literal["auto", "user-defined"] = pd.Field()

    models: List[Union[TempFluidDynamics, TempWallBC]] = pd.Field()

    def preprocess(self):
        """
        Supply self._supplementary_registry to the construction of
        TempFluidDynamics etc so that the class can perform proper validation
        """
        _supplementary_registry = _get_supplementary_registry(self.far_field_type)
        for model in self.models:
            model.entities.preprocess(_supplementary_registry, mesh_unit=1 * u.m)

        return self


@pytest.fixture
def my_cylinder1():
    return Cylinder(
        name="zone/Cylinder1",
        height=11 * u.cm,
        axis=(1, 0, 0),
        inner_radius=1 * u.ft,
        outer_radius=2 * u.ft,
        center=(1, 2, 3) * u.ft,
    )


@pytest.fixture
def my_cylinder2():
    return Cylinder(
        name="zone/Cylinder2",
        height=12 * u.nm,
        axis=(1, 0, 0),
        inner_radius=1 * u.nm,
        outer_radius=2 * u.nm,
        center=(1, 2, 3) * u.nm,
    )


@pytest.fixture
def my_box_zone1():
    return Box(
        name="zone/Box1",
        axes=((-1, 0, 0), (1, 0, 0)),
        center=(1, 2, 3) * u.mm,
        size=(0.1, 0.01, 0.001) * u.mm,
    )


@pytest.fixture
def my_box_zone2():
    return Box(
        name="zone/Box2",
        axes=((-1, 0, 0), (1, 1, 0)),
        center=(3, 2, 3) * u.um,
        size=(0.1, 0.01, 0.001) * u.um,
    )


@pytest.fixture
def my_surface1():
    return TempSurface(name="MySurface1")


@pytest.fixture
def my_volume_mesh1():
    return TempVolumeMesh(file_name="volMesh-1.cgns")


@pytest.fixture
def my_volume_mesh2():
    return TempVolumeMesh(file_name="volMesh-2.cgns")


##:: ---------------- Entity tests ----------------


def unset_entity_type():
    def IncompleteEntity(EntityBase):
        pass

    with pytest.raises(
        NotImplementedError,
        match=re.escape("_entity_type is not defined in the entity class."),
    ):
        IncompleteEntity(name="IncompleteEntity")


def test_wrong_ways_of_copying_entity(my_cylinder1):
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Change is necessary when calling .copy() as there cannot be two identical entities at the same time. Please use update parameter to change the entity attributes."
        ),
    ):
        my_cylinder1.copy()

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Copying an entity requires a new name to be specified. Please provide a new name in the update dictionary."
        ),
    ):
        my_cylinder1.copy(update={"height": 1.0234})

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Copying an entity requires a new name to be specified. Please provide a new name in the update dictionary."
        ),
    ):
        my_cylinder1.copy(update={"height": 1.0234, "name": my_cylinder1.name})

    assert (
        len(TempFluidDynamics(entities=[my_cylinder1, my_cylinder1]).entities.stored_entities) == 1
    )

    assert (
        len(
            TempFluidDynamics(
                entities=["zone/Box1", "zone/Box1", my_cylinder1]
            ).entities.stored_entities
        )
        == 2
    )


def test_copying_entity(my_cylinder1):
    my_cylinder3_2 = my_cylinder1.copy(update={"height": 8119 * u.m, "name": "zone/Cylinder3-2"})
    print(my_cylinder3_2)
    assert my_cylinder3_2.height == 8119 * u.m


##:: ---------------- EntityList/Registry tests ----------------


@pytest.mark.usefixtures("array_equality_override")
def test_entities_expansion(my_cylinder1, my_cylinder2, my_box_zone1, my_surface1):
    # 0. No supplied registry but trying to use str
    with pytest.raises(
        ValueError,
        match=re.escape("Internal error, registry is not supplied for entity (Box*) expansion."),
    ):
        expanded_entities = TempFluidDynamics(
            entities=["Box*", my_cylinder1, my_box_zone1]
        ).entities._get_expanded_entities()

    # 1. No supplied registry
    expanded_entities = TempFluidDynamics(
        entities=[my_cylinder1, my_box_zone1]
    ).entities._get_expanded_entities()
    assert my_cylinder1 in expanded_entities
    assert my_box_zone1 in expanded_entities
    assert len(expanded_entities) == 2

    # 2. With supplied registry and has implicit duplicates
    _supplementary_registry = EntityRegistry()
    _supplementary_registry.register(
        Box(
            name="Implicitly_generated_Box_zone1",
            axes=((-1, 0, 0), (1, 1, 0)),
            center=(32, 2, 3) * u.cm,
            size=(0.1, 0.01, 0.001) * u.cm,
        )
    )
    _supplementary_registry.register(
        Box(
            name="Implicitly_generated_Box_zone2",
            axes=((-1, 0, 0), (1, 1, 0)),
            center=(31, 2, 3) * u.cm,
            size=(0.1, 0.01, 0.001) * u.cm,
        )
    )
    expanded_entities = TempFluidDynamics(
        entities=[my_cylinder1, my_box_zone1, "*Box*"]
    ).entities._get_expanded_entities(_supplementary_registry)
    selected_entity_names = [entity.name for entity in expanded_entities]
    assert "Implicitly_generated_Box_zone1" in selected_entity_names
    assert "Implicitly_generated_Box_zone2" in selected_entity_names
    assert "zone/Box1" in selected_entity_names
    assert "zone/Cylinder1" in selected_entity_names
    assert len(selected_entity_names) == 4  # 2 new boxes, 1 cylinder, 1 box


def test_by_reference_registry(my_cylinder2):
    registry = EntityRegistry()
    registry.register(my_cylinder2)
    my_cylinder2.height = 131 * u.m
    for entity in registry.get_all_entities_of_given_type(Cylinder):
        if isinstance(entity, Cylinder) and entity.name == "zone/Cylinder2":
            assert entity.height == 131 * u.m


def test_by_value_expansion(my_cylinder2):
    expanded_entities = TempFluidDynamics(entities=[my_cylinder2]).entities._get_expanded_entities()
    my_cylinder2.height = 1012 * u.cm
    for entity in expanded_entities:
        if isinstance(entity, Cylinder) and entity.name == "zone/Cylinder2":
            assert entity.height == 12 * u.nm  # unchanged


def test_get_entities(
    my_cylinder1,
    my_cylinder2,
    my_box_zone1,
    my_box_zone2,
):
    registry = EntityRegistry()
    registry.register(my_cylinder1)
    registry.register(my_cylinder2)
    registry.register(my_box_zone1)
    registry.register(my_box_zone2)
    all_box_entities = registry.get_all_entities_of_given_type(Box)
    assert len(all_box_entities) == 4
    assert my_box_zone1 in all_box_entities
    assert my_box_zone2 in all_box_entities
    # my_cylinder1 is not a Box but is a _volumeZoneBase and EntityRegistry registers by base type
    assert my_cylinder1 in all_box_entities
    assert my_cylinder2 in all_box_entities


def test_changing_final_attributes(my_box_zone1):
    with pytest.raises(AttributeError, match=re.escape("Cannot modify _entity_type")):
        my_box_zone1.entity_type = "WrongSubClass"

    with pytest.raises(AttributeError, match=re.escape("Cannot modify _auto_constructed")):
        my_box_zone1.auto_constructed = True


def test_entities_input_interface(my_cylinder1, my_cylinder2, my_volume_mesh1):
    # 1. Using reference of single asset entity
    expanded_entities = TempFluidDynamics(
        entities=my_volume_mesh1["zone*"]
    ).entities._get_expanded_entities()
    assert len(expanded_entities) == 3
    assert expanded_entities == my_volume_mesh1["zone*"]

    # 2. test using invalid entity input (UGRID convention example)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Type(<class 'int'>) of input to `entities` (1) is not valid. Expected str or entity instance."
        ),
    ):
        expanded_entities = TempFluidDynamics(entities=1).entities._get_expanded_entities()
    # 3. test empty list
    with pytest.raises(
        ValueError,
        match=re.escape("Invalid input type to `entities`, list is empty."),
    ):
        expanded_entities = TempFluidDynamics(entities=[]).entities._get_expanded_entities()

    # 4. test None
    expanded_entities = TempFluidDynamics(entities=None).entities._get_expanded_entities()
    assert expanded_entities is None

    # 5. test non-existing entity
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Failed to find any matching entity with ['Non_existing_volume']. Please check the input to entities."
        ),
    ):
        expanded_entities = TempFluidDynamics(
            entities=["Non_existing_volume"]
        ).entities._get_expanded_entities(EntityRegistry())

    with pytest.raises(
        ValueError,
        match=re.escape("Failed to find any matching entity with asdf. Please check your input."),
    ):
        my_volume_mesh1["asdf"]


def test_skipped_entities(my_cylinder1, my_cylinder2):
    TempFluidDynamics()
    assert TempFluidDynamics().entities.stored_entities is None


def test_duplicate_entities(my_volume_mesh1):
    user_override_cylinder = Cylinder(
        name="zone_1",
        height=12 * u.m,
        axis=(1, 0, 0),
        inner_radius=1 * u.m,
        outer_radius=2 * u.m,
        center=(1, 2, 3) * u.m,
    )

    expanded_entities = TempFluidDynamics(
        entities=[
            my_volume_mesh1["zone*"],
            user_override_cylinder,
            user_override_cylinder,
            user_override_cylinder,
            my_volume_mesh1["*"],
        ]
    ).entities._get_expanded_entities()

    assert len(expanded_entities) == 3
    assert user_override_cylinder in expanded_entities


def test_entire_worklfow(my_cylinder1, my_volume_mesh1):

    my_param = TempSimulationParam(
        far_field_type="auto",
        models=[
            TempFluidDynamics(
                entities=[
                    my_cylinder1,
                    my_cylinder1,
                    my_cylinder1,
                    my_volume_mesh1["*"],
                    my_volume_mesh1["*zone*"],
                ]
            ),
            TempWallBC(surfaces=[my_volume_mesh1["*"], "*", "farfield"]),
        ],
    )

    my_param.preprocess()

    fluid_dynamics_entity_names = [
        entity.name for entity in my_param.models[0].entities.stored_entities
    ]

    wall_entity_names = [entity.name for entity in my_param.models[1].entities.stored_entities]
    assert "zone/Cylinder1" in fluid_dynamics_entity_names
    assert "zone_1" in fluid_dynamics_entity_names
    assert "zone_2" in fluid_dynamics_entity_names
    assert "zone_3" in fluid_dynamics_entity_names
    assert len(fluid_dynamics_entity_names) == 4

    assert "surface_1" in wall_entity_names
    assert "surface_2" in wall_entity_names
    assert "farfield" in wall_entity_names
    assert len(wall_entity_names) == 3
