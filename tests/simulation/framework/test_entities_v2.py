import re
from copy import deepcopy
from typing import Annotated, List, Literal, Optional, Union

import numpy as np
import pydantic as pd
import pytest

import flow360.component.simulation.units as u
from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.framework.entity_base import (
    EntityBase,
    EntityList,
    MergeConflictError,
    _merge_objects,
)
from flow360.component.simulation.framework.entity_registry import EntityRegistry
from flow360.component.simulation.framework.param_utils import (
    AssetCache,
    register_entity_list,
)
from flow360.component.simulation.primitives import (
    Box,
    Cylinder,
    GenericVolume,
    Surface,
    _SurfaceEntityBase,
)
from flow360.component.simulation.simulation_params import _ParamModelBase
from flow360.component.simulation.unit_system import LengthType, SI_unit_system
from flow360.component.simulation.utils import model_attribute_unlock
from flow360.log import set_logging_level
from tests.simulation.conftest import AssetBase

set_logging_level("DEBUG")


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


@pytest.fixture
def array_equality_override():
    pass  # No-op fixture to override the original one


class TempVolumeMesh(AssetBase):
    """Mimicing the final VolumeMesh class"""

    fname: str

    def _get_meta_data(self):
        if self.fname == "volMesh-1.cgns":
            return {
                "zones": {
                    "zone_1": {
                        "boundaryNames": [
                            "surface_1",
                        ],
                    },
                    "zone_2": {
                        "boundaryNames": [
                            "surface_2",
                        ],
                    },
                    "zone_3": {
                        "boundaryNames": [
                            "surface_3",
                        ],
                    },
                },
                "surfaces": {"surface_1": {}, "surface_2": {}, "surface_3": {}},
            }
        elif self.fname == "volMesh-2.cgns":
            return {
                "zones": {
                    "zone_4": {
                        "boundaryNames": [
                            "surface_4",
                        ],
                    },
                    "zone_5": {
                        "boundaryNames": [
                            "surface_5",
                        ],
                    },
                    "zone_6": {
                        "boundaryNames": [
                            "surface_6",
                        ],
                    },
                    "zone_1": {
                        "boundaryNames": [
                            "surface_1",
                        ],
                    },
                },
                "surfaces": {"surface_4": {}, "surface_5": {}, "surface_6": {}, "surface_1": {}},
            }
        elif self.fname == "volMesh-with_interface.cgns":
            return {
                "zones": {
                    "farfield": {
                        "boundaryNames": [
                            "farfield/farfield",
                            "farfield/rotIntf",
                        ],
                        "donorInterfaceNames": ["innerZone/rotIntf-1"],
                        "donorZoneNames": ["innerZone"],
                        "receiverInterfaceNames": ["farfield/rotIntf"],
                    },
                    "innerZone": {
                        "boundaryNames": [
                            "innerZone/rotIntf-1",
                            "innerZone/rotIntf-2",
                        ],
                        "donorInterfaceNames": ["farFieldBlock/rotIntf", "mostinnerZone/rotIntf"],
                        "donorZoneNames": ["farFieldBlock", "mostinnerZone"],
                        "receiverInterfaceNames": ["innerZone/rotIntf-1", "innerZone/rotIntf-2"],
                    },
                    "mostinnerZone": {
                        "boundaryNames": [
                            "mostinnerZone/rotIntf",
                            "my_wall_1",
                            "my_wall_2",
                            "my_wall_3",
                        ],
                        "donorInterfaceNames": ["innerZone/rotIntf-2"],
                        "donorZoneNames": ["innerZone"],
                        "receiverInterfaceNames": ["mostinnerZone/rotIntf"],
                    },
                },
                "surfaces": {
                    "farfield/farfield": {},
                    "farfield/rotIntf": {},
                    "innerZone/rotIntf-1": {},
                    "innerZone/rotIntf-2": {},
                    "mostinnerZone/rotIntf": {},
                    "my_wall_1": {},
                    "my_wall_2": {},
                    "my_wall_3": {},
                },
            }
        else:
            raise ValueError("Invalid file name")

    def _populate_registry(self):
        for zone_name, zone_meta in self._get_meta_data()["zones"].items():
            all_my_boundaries = [item for item in zone_meta["boundaryNames"]]
            self.internal_registry.register(
                GenericVolume(
                    name=zone_name, private_attribute_zone_boundary_names=all_my_boundaries
                )
            )
        # get interfaces
        interfaces = set()
        for zone_name, zone_meta in self._get_meta_data()["zones"].items():
            for surface_name in (
                zone_meta["donorInterfaceNames"] if "donorInterfaceNames" in zone_meta else []
            ):
                interfaces.add(surface_name)
            for surface_name in (
                zone_meta["receiverInterfaceNames"] if "receiverInterfaceNames" in zone_meta else []
            ):
                interfaces.add(surface_name)

        for surface_name in self._get_meta_data()["surfaces"]:
            self.internal_registry.register(
                Surface(
                    name=surface_name, private_attribute_is_interface=surface_name in interfaces
                )
            )

    def __init__(self, file_name: str):
        super().__init__()
        self.fname = file_name
        self._populate_registry()


class TempSurface(_SurfaceEntityBase):
    private_attribute_entity_type_name: Literal["TempSurface"] = pd.Field(
        "TempSurface", frozen=True
    )


class TempFluidDynamics(Flow360BaseModel):
    entities: EntityList[GenericVolume, Box, Cylinder] = pd.Field(alias="volumes", default=None)


class TempWallBC(Flow360BaseModel):
    entities: EntityList[Surface, TempSurface] = pd.Field(alias="surfaces", default=[])


class TempRotation(Flow360BaseModel):
    entities: EntityList[GenericVolume, Cylinder] = pd.Field(alias="volumes")
    parent_volume: Optional[Union[GenericVolume]] = pd.Field(None)


class TempUserDefinedDynamic(Flow360BaseModel):
    name: str = pd.Field()
    input_boundary_patches: Optional[EntityList[Surface]] = pd.Field(None)
    output_target: Optional[Cylinder] = pd.Field(
        None
    )  # Limited to `Cylinder` for now as we have only tested using UDD to control rotation.


class TempSimulationParam(_ParamModelBase):

    far_field_type: Literal["auto", "user-defined"] = pd.Field()

    models: List[Union[TempFluidDynamics, TempWallBC, TempRotation]] = pd.Field()
    udd: Optional[TempUserDefinedDynamic] = pd.Field(None)
    private_attribute_asset_cache: AssetCache = pd.Field(AssetCache(), frozen=True)

    def preprocess(self):
        """
        Supply self._supplementary_registry to the construction of
        TempFluidDynamics etc so that the class can perform proper validation
        """
        with model_attribute_unlock(self.private_attribute_asset_cache, "project_length_unit"):
            self.private_attribute_asset_cache.project_length_unit = LengthType.validate(1 * u.m)

        for model in self.models:
            model.entities.preprocess(params=self)

        return self

    def get_used_entity_registry(self) -> EntityRegistry:
        """Recursively register all entities listed in EntityList to the asset cache."""
        # pylint: disable=no-member
        registry = EntityRegistry()
        register_entity_list(self, registry)
        return registry


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
    return Box.from_principal_axes(
        name="zone/Box1",
        axes=((-1, 0, 0), (0, 1, 0)),
        center=(1, 2, 3) * u.mm,
        size=(0.1, 0.01, 0.001) * u.mm,
    )


@pytest.fixture
def my_box_zone2():
    return Box.from_principal_axes(
        name="zone/Box2",
        axes=((0, 0, 1), (1, 1, 0)),
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


@pytest.fixture
def my_volume_mesh_with_interface():
    return TempVolumeMesh(file_name="volMesh-with_interface.cgns")


##:: ---------------- Entity tests ----------------


def unset_entity_type():
    def IncompleteEntity(EntityBase):
        pass

    with pytest.raises(
        NotImplementedError,
        match=re.escape(
            "private_attribute_registry_bucket_name is not defined in the entity class."
        ),
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
        len(
            TempFluidDynamics(
                entities=[my_cylinder1, my_cylinder1]
            ).entities._get_expanded_entities(create_hard_copy=False)
        )
        == 1
    )


def test_copying_entity(my_cylinder1):
    my_cylinder3_2 = my_cylinder1.copy(update={"height": 8119 * u.m, "name": "zone/Cylinder3-2"})
    print(my_cylinder3_2)
    assert my_cylinder3_2.height == 8119 * u.m


##:: ---------------- EntityList/Registry tests ----------------


def test_EntityList_discrimination():
    class ConfusingEntity1(EntityBase):
        some_value: int = pd.Field(1, gt=1)
        private_attribute_entity_type_name: Literal["ConfusingEntity1"] = pd.Field(
            "ConfusingEntity1", frozen=True
        )
        private_attribute_registry_bucket_name: Literal["UnitTestEntityType"] = pd.Field(
            "UnitTestEntityType", frozen=True
        )

    class ConfusingEntity2(EntityBase):
        some_value: int = pd.Field(1, gt=2)
        private_attribute_entity_type_name: Literal["ConfusingEntity2"] = pd.Field(
            "ConfusingEntity2", frozen=True
        )
        private_attribute_registry_bucket_name: Literal["UnitTestEntityType"] = pd.Field(
            "UnitTestEntityType", frozen=True
        )

    class MyModel(Flow360BaseModel):
        entities: EntityList[ConfusingEntity1, ConfusingEntity2] = pd.Field()

    # Ensure EntityList is looking for the discriminator
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Unable to extract tag using discriminator 'private_attribute_entity_type_name'"
        ),
    ):
        MyModel(
            **{
                "entities": {
                    "stored_entities": [
                        {
                            "name": "I should be deserialize as ConfusingEntity1",
                            "some_value": 1,
                        }
                    ],
                }
            }
        )

    # Ensure EntityList is only trying to validate against ConfusingEntity1
    try:
        MyModel(
            **{
                "entities": {
                    "stored_entities": [
                        {
                            "name": "I should be deserialize as ConfusingEntity1",
                            "private_attribute_entity_type_name": "ConfusingEntity1",
                            "some_value": 1,
                        }
                    ],
                }
            }
        )
    except pd.ValidationError as err:
        validation_errors = err.errors()
    # Without discrimination, above deserialization would have failed both
    # ConfusingEntitys' checks and result in 3 errors:
    # 1. some_value is less than 1 (from ConfusingEntity1)
    # 2. some_value is less than 2 (from ConfusingEntity2)
    # 3. private_attribute_entity_type_name is incorrect (from ConfusingEntity2)
    # But now we enforce Pydantic to only check against ConfusingEntity1
    assert validation_errors[0]["msg"] == "Input should be greater than 1"
    assert validation_errors[0]["loc"] == (
        "entities",
        "stored_entities",
        0,
        "ConfusingEntity1",
        "some_value",
    )
    assert len(validation_errors) == 1


def test_entities_expansion(my_cylinder1, my_box_zone1):
    expanded_entities = TempFluidDynamics(
        entities=[my_cylinder1, my_cylinder1, my_box_zone1]
    ).entities._get_expanded_entities(create_hard_copy=False)
    assert my_cylinder1 in expanded_entities
    assert my_box_zone1 in expanded_entities
    assert len(expanded_entities) == 2


def test_by_reference_registry(my_cylinder2):
    my_fd = TempFluidDynamics(entities=[my_cylinder2])

    registry = EntityRegistry()
    registry.register(my_cylinder2)
    entities = registry.get_bucket(by_type=Cylinder).entities  # get the entities now before change
    # [Registry] External changes --> Internal
    my_cylinder2.height = 131 * u.m
    for entity in entities:
        if isinstance(entity, Cylinder) and entity.name == "zone/Cylinder2":
            assert entity.height == 131 * u.m

    # [Registry] Internal changes --> External
    my_cylinder2_ref = registry.find_single_entity_by_name("zone/Cylinder2")
    my_cylinder2_ref.height = 132 * u.m
    assert my_cylinder2.height == 132 * u.m

    assert my_fd.entities.stored_entities[0].height == 132 * u.m


def test_by_value_expansion(my_cylinder2):
    expanded_entities = TempFluidDynamics(entities=[my_cylinder2]).entities._get_expanded_entities(
        create_hard_copy=True
    )
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
    all_box_entities = registry.get_bucket(by_type=Box).entities
    assert len(all_box_entities) == 4
    assert my_box_zone1 in all_box_entities
    assert my_box_zone2 in all_box_entities
    # my_cylinder1 is not a Box but is a _volumeZoneBase and EntityRegistry registers by base type
    assert my_cylinder1 in all_box_entities
    assert my_cylinder2 in all_box_entities

    registry = EntityRegistry()
    registry.register(Surface(name="AA_ground_close"))
    registry.register(Surface(name="BB"))
    registry.register(Surface(name="CC_ground"))
    items = registry.find_by_naming_pattern("*ground", enforce_output_as_list=True)
    assert len(items) == 1
    assert items[0].name == "CC_ground"


def test_entities_input_interface(my_volume_mesh1):
    # 1. Using reference of single asset entity
    expanded_entities = TempFluidDynamics(
        entities=my_volume_mesh1["zone*"]
    ).entities._get_expanded_entities(create_hard_copy=True)
    assert len(expanded_entities) == 3
    assert expanded_entities == my_volume_mesh1["zone*"]

    # 2. test using invalid entity input (UGRID convention example)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Type(<class 'int'>) of input to `entities` (1) is not valid. Expected str or entity instance."
        ),
    ):
        expanded_entities = TempFluidDynamics(entities=1).entities._get_expanded_entities(
            create_hard_copy=True
        )
    # 3. test empty list
    with pytest.raises(
        ValueError,
        match=re.escape("Invalid input type to `entities`, list is empty."),
    ):
        expanded_entities = TempFluidDynamics(entities=[]).entities._get_expanded_entities(
            create_hard_copy=True
        )

    # 4. test None
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Input should be a valid list [type=list_type, input_value=None, input_type=NoneType]"
        ),
    ):
        expanded_entities = TempFluidDynamics(entities=None).entities._get_expanded_entities(
            create_hard_copy=True
        )

    with pytest.raises(
        ValueError,
        match=re.escape("Failed to find any matching entity with asdf. Please check your input."),
    ):
        my_volume_mesh1["asdf"]


def test_entire_worklfow(my_cylinder1, my_volume_mesh1):
    with SI_unit_system:
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
                TempWallBC(surfaces=[my_volume_mesh1["*"]]),
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
    assert "surface_3" in wall_entity_names
    assert len(wall_entity_names) == 3


def test_multiple_param_creation_and_asset_registry(
    my_cylinder1, my_box_zone2, my_box_zone1, my_volume_mesh1, my_volume_mesh2
):  # Make sure that no entities from the first param are present in the second param
    with SI_unit_system:
        my_param1 = TempSimulationParam(
            far_field_type="auto",
            models=[
                TempFluidDynamics(
                    entities=[
                        my_cylinder1,
                        my_cylinder1,
                        my_cylinder1,
                        my_volume_mesh1["*"],
                    ]
                ),
                TempWallBC(surfaces=[my_volume_mesh1["*"]]),
            ],
        )

    ref_registry = EntityRegistry()
    ref_registry.register(my_cylinder1)
    ref_registry.register(my_volume_mesh1["zone_1"])
    ref_registry.register(my_volume_mesh1["zone_2"])
    ref_registry.register(my_volume_mesh1["zone_3"])
    ref_registry.register(my_volume_mesh1["surface_1"])
    ref_registry.register(my_volume_mesh1["surface_2"])
    ref_registry.register(my_volume_mesh1["surface_3"])

    assert my_param1.get_used_entity_registry() == ref_registry

    TempFluidDynamics(entities=[my_box_zone2])  # This should not be added to the registry

    with SI_unit_system:
        my_param2 = TempSimulationParam(
            far_field_type="auto",
            models=[
                TempFluidDynamics(
                    entities=[
                        my_box_zone1,
                        my_box_zone1,
                        my_volume_mesh2["*"],
                    ]
                ),
                TempWallBC(surfaces=[my_volume_mesh2["*"]]),
            ],
        )

    ref_registry = EntityRegistry()
    ref_registry.register(my_box_zone1)
    ref_registry.register(my_volume_mesh2["zone_4"])
    ref_registry.register(my_volume_mesh2["zone_5"])
    ref_registry.register(my_volume_mesh2["zone_6"])
    ref_registry.register(my_volume_mesh2["zone_1"])
    ref_registry.register(my_volume_mesh2["surface_4"])
    ref_registry.register(my_volume_mesh2["surface_5"])
    ref_registry.register(my_volume_mesh2["surface_6"])
    ref_registry.register(my_volume_mesh2["surface_1"])
    print(ref_registry)
    assert my_param2.get_used_entity_registry() == ref_registry


def test_entities_change_reflection_in_param_registry(my_cylinder1, my_volume_mesh1):
    # Make sure the changes in the entity are always reflected in the registry
    with SI_unit_system:
        my_param1 = TempSimulationParam(
            far_field_type="auto",
            models=[
                TempFluidDynamics(
                    entities=[
                        my_cylinder1,
                        my_cylinder1,
                        my_cylinder1,
                        my_volume_mesh1["*"],
                    ]
                ),
                TempWallBC(surfaces=[my_volume_mesh1["*"]]),
            ],
        )
    my_cylinder1.center = (3, 2, 1) * u.m
    used_entity_registry = EntityRegistry()
    register_entity_list(my_param1, used_entity_registry)
    my_cylinder1_ref = used_entity_registry.find_single_entity_by_name("zone/Cylinder1")
    assert all(my_cylinder1_ref.center == [3, 2, 1] * u.m)


def test_entities_merging_logic(my_volume_mesh_with_interface):
    ##:: Scenario 1: Merge Generic with Generic
    my_generic_base = GenericVolume(name="my_generic_volume", axis=(0, 1, 0))
    my_generic_merged = deepcopy(my_generic_base)
    my_generic_merged = _merge_objects(
        my_generic_merged,
        GenericVolume(
            name="my_generic_volume", private_attribute_zone_boundary_names=["my_wall_1"]
        ),
    )
    assert my_generic_merged.private_attribute_zone_boundary_names.items == ["my_wall_1"]
    assert my_generic_merged.axis == (0, 1, 0)

    ##:: Scenario 2: Merge Generic with Generic with conflict
    with pytest.raises(
        MergeConflictError,
        match=re.escape(r"Conflict on attribute 'axis':"),
    ):
        my_generic_merged = deepcopy(my_generic_base)
        my_generic_merged = _merge_objects(
            my_generic_merged,
            GenericVolume(name="my_generic_volume", axis=(0, 2, 1)),
        )

    ##:: Scenario 3: Merge Generic with NonGeneric
    my_generic_merged = deepcopy(my_generic_base)
    my_generic_merged = _merge_objects(
        my_generic_merged,
        Cylinder(
            name="my_generic_volume",
            height=11 * u.cm,
            axis=(0, 1, 0),
            inner_radius=1 * u.ft,
            outer_radius=2 * u.ft,
            center=(1, 2, 3) * u.ft,
        ),
    )
    assert isinstance(my_generic_merged, Cylinder)
    assert my_generic_merged.height == 11 * u.cm
    assert my_generic_merged.axis == (0, 1, 0)
    assert my_generic_merged.inner_radius == 1 * u.ft
    assert my_generic_merged.outer_radius == 2 * u.ft
    assert all(my_generic_merged.center == (1, 2, 3) * u.ft)

    ##:: Scenario 4: Merge NonGeneric with Generic
    # reverse the order does not change the result
    my_generic_merged = deepcopy(my_generic_base)
    my_generic_merged = _merge_objects(
        Cylinder(
            name="my_generic_volume",
            height=11 * u.cm,
            axis=(0, 1, 0),
            inner_radius=1 * u.ft,
            outer_radius=2 * u.ft,
            center=(1, 2, 3) * u.ft,
        ),
        my_generic_merged,
    )
    assert isinstance(my_generic_merged, Cylinder)
    assert my_generic_merged.height == 11 * u.cm
    assert my_generic_merged.axis == (0, 1, 0)
    assert my_generic_merged.inner_radius == 1 * u.ft
    assert my_generic_merged.outer_radius == 2 * u.ft
    assert all(my_generic_merged.center == (1, 2, 3) * u.ft)

    # ##:: Scenario 4: Merge NonGeneric with Generic with conflict
    with pytest.raises(
        MergeConflictError,
        match=re.escape(r"Conflict on attribute 'axis':"),
    ):
        my_generic_merged = deepcopy(my_generic_base)
        _merge_objects(
            my_generic_merged,
            GenericVolume(name="my_generic_volume", axis=(0, 2, 1)),
        )

    ##:: Scenario 5: Merge NonGeneric with NonGeneric
    my_cylinder1 = Cylinder(
        name="innerZone",
        height=11 * u.cm,
        axis=(1, 0, 0),
        inner_radius=1 * u.ft,
        outer_radius=2 * u.ft,
        center=(1, 2, 3) * u.ft,
    )

    # Only valid if they are exactly the same
    merged = _merge_objects(
        my_cylinder1,
        my_cylinder1,
    )
    assert merged == my_cylinder1

    ##:: Scenario 6: Merge NonGeneric with NonGeneric with conflict
    with pytest.raises(
        MergeConflictError,
        match=re.escape(r"Conflict on attribute 'height':"),
    ):
        my_generic_merged = deepcopy(my_generic_base)
        merged = _merge_objects(
            my_cylinder1,
            Cylinder(
                name="innerZone",
                height=12 * u.cm,
                axis=(1, 0, 0),
                inner_radius=1 * u.ft,
                outer_radius=2 * u.ft,
                center=(1, 2, 3) * u.ft,
            ),
        )

    ##:: Scenario 7: Merge NonGeneric with NonGeneric with different class
    with pytest.raises(
        MergeConflictError,
        match=re.escape(r"Cannot merge objects of different classes: Cylinder and Box."),
    ):
        my_generic_merged = deepcopy(my_generic_base)
        merged = _merge_objects(
            my_cylinder1,
            Box.from_principal_axes(
                name="innerZone",
                axes=((-1, 0, 0), (0, 1, 0)),
                center=(1, 2, 3) * u.mm,
                size=(0.1, 0.01, 0.001) * u.mm,
            ),
        )

    ##:: Scenario 8: No user specified attributes in the Generic type entities and
    ##::             now we merge the user overload with it.
    user_override_cylinder = Cylinder(
        name="innerZone",
        height=12 * u.m,
        axis=(1, 0, 0),
        inner_radius=1 * u.m,
        outer_radius=2 * u.m,
        center=(1, 2, 3) * u.m,
    )

    with SI_unit_system:
        my_param = TempSimulationParam(
            far_field_type="user-defined",
            models=[
                TempFluidDynamics(
                    entities=[
                        user_override_cylinder,
                        my_volume_mesh_with_interface["*"],
                    ]
                ),
                TempWallBC(surfaces=[my_volume_mesh_with_interface["*"]]),
            ],
        )

    target_entity_param_reg = my_param.get_used_entity_registry().find_single_entity_by_name(
        "innerZone"
    )

    target_entity_mesh_reg = (
        my_volume_mesh_with_interface.internal_registry.find_single_entity_by_name("innerZone")
    )

    assert (
        len(my_param.models[0].entities._get_expanded_entities(create_hard_copy=True)) == 3
    )  # 1 cylinder, 2 generic zones
    assert isinstance(target_entity_param_reg, Cylinder)

    # Note: mesh still register the original one because it was not used at all.
    assert isinstance(target_entity_mesh_reg, GenericVolume)


def test_entity_registry_serialization_and_deserialization():
    # This is already tested in to_file_from_file tests in param unit test.
    pass


def test_update_asset_registry(my_volume_mesh_with_interface):
    user_override_cylinder = Cylinder(
        name="innerZone",
        height=12 * u.m,
        axis=(1, 0, 0),
        inner_radius=1 * u.m,
        outer_radius=2 * u.m,
        center=(1, 2, 3) * u.m,
    )
    backup = deepcopy(
        my_volume_mesh_with_interface.internal_registry.find_single_entity_by_name("innerZone")
    )
    assert my_volume_mesh_with_interface.internal_registry.contains(backup)

    my_volume_mesh_with_interface.internal_registry.replace_existing_with(user_override_cylinder)

    assert my_volume_mesh_with_interface.internal_registry.contains(user_override_cylinder)


def test_corner_cases_for_entity_registry_thoroughness(my_cylinder1, my_volume_mesh_with_interface):
    with SI_unit_system:
        my_param = TempSimulationParam(
            far_field_type="auto",
            models=[
                TempRotation(
                    entities=[my_volume_mesh_with_interface["innerZone"]],
                    parent_volume=my_volume_mesh_with_interface["mostinnerZone"],
                ),
            ],
            udd=TempUserDefinedDynamic(
                name="pseudo",
                input_boundary_patches=[my_volume_mesh_with_interface["*"]],
                output_target=my_cylinder1,
            ),
        )
    my_reg = my_param.get_used_entity_registry()
    # output_target
    assert my_reg.contains(my_cylinder1)
    # input_boundary_patches
    for surface_name in [
        "farfield/farfield",
        "farfield/rotIntf",
        "innerZone/rotIntf-1",
        "innerZone/rotIntf-2",
        "mostinnerZone/rotIntf",
        "my_wall_1",
        "my_wall_2",
        "my_wall_3",
    ]:
        assert my_reg.contains(my_volume_mesh_with_interface[surface_name])

    # parent_volume
    assert my_reg.contains(my_volume_mesh_with_interface["mostinnerZone"])
    # entities
    assert my_reg.contains(my_volume_mesh_with_interface["innerZone"])
    assert my_reg.entity_count() == 11


def compare_boxes(box1, box2):
    return (
        np.isclose(np.linalg.norm(np.cross(box1.axis_of_rotation, box2.axis_of_rotation)), 0)
        and np.isclose(
            np.mod(box1.angle_of_rotation.value, 2 * np.pi),
            np.mod(box2.angle_of_rotation.value, 2 * np.pi),
        )
        and np.all(np.isclose(box1.center.value, box2.center.value))
        and np.all(np.isclose(box1.size.value, box2.size.value))
        and np.all(
            np.isclose(
                np.asarray(box1.private_attribute_input_cache.axes, dtype=float),
                np.asarray(box2.private_attribute_input_cache.axes, dtype=float),
            )
        )
    )


def test_box_multi_constructor():
    box1 = Box(
        name="box1",
        center=(0, 0, 0) * u.m,
        size=(1, 1, 1) * u.m,
        axis_of_rotation=(1, 1, 0),
        angle_of_rotation=np.pi * u.rad,
    )
    box2 = Box.from_principal_axes(
        name="box2", center=(0, 0, 0) * u.m, size=(1, 1, 1) * u.m, axes=((0, 1, 0), (1, 0, 0))
    )
    assert compare_boxes(box1, box2)

    box3 = Box(
        name="box3",
        center=(0, 0, 0) * u.m,
        size=(1, 1, 1) * u.m,
        axis_of_rotation=(0.1, 0.5, 0.2),
        angle_of_rotation=np.pi / 6 * u.rad,
    )
    box4 = Box.from_principal_axes(
        name="box4",
        center=(0, 0, 0) * u.m,
        size=(1, 1, 1) * u.m,
        axes=(
            (0.8704912236582907, 0.20490328520431558, -0.4475038248399343),
            (-0.16024508646579513, 0.9776709006307398, 0.13594529165604813),
        ),
    )
    assert compare_boxes(box3, box4)

    box5 = Box.from_principal_axes(
        name="box5", center=(0, 0, 0) * u.m, size=(1, 1, 1) * u.m, axes=((1, 0, 0), (0, 1, 0))
    )
    assert np.isclose(box5.angle_of_rotation.value, 0)


##:: ---------------- Entity specific validaitons ----------------


def test_box_validation():
    with pytest.raises(
        ValueError, match=re.escape("The two axes are not orthogonal, dot product is 1.")
    ):
        Box.from_principal_axes(
            name="box6", center=(0, 0, 0) * u.m, size=(1, 1, 1) * u.m, axes=((1, 0, 0), (1, 0, 0))
        )

    with pytest.raises(ValueError, match=re.escape("'[  1   1 -10] m' cannot have negative value")):
        Box(
            name="box6",
            center=(0, 0, 0) * u.m,
            size=(1, 1, -10) * u.m,
            axis_of_rotation=(1, 0, 0),
            angle_of_rotation=10 * u.deg,
        )

    with pytest.raises(
        ValueError, match=re.escape("'(1, 1, -10) flow360_length_unit' cannot have negative value")
    ):
        Box(
            name="box6",
            center=(0, 0, 0) * u.m,
            size=(1, 1, -10) * u.flow360_length_unit,
            axis_of_rotation=(1, 0, 0),
            angle_of_rotation=10 * u.deg,
        )


def test_cylinder_validation():
    with pytest.raises(
        ValueError,
        match=re.escape("Cylinder inner radius (1000.0 m) must be less than outer radius (2.0 m)"),
    ):
        Cylinder(
            name="cyl",
            center=(0, 0, 0) * u.m,
            height=2 * u.m,
            axis=(1, 0, 0),
            inner_radius=1000 * u.m,
            outer_radius=2 * u.m,
        )
