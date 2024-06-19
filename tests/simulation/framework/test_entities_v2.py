import re
from copy import deepcopy
from typing import List, Literal, Union

import pydantic as pd
import pytest

import flow360.component.simulation.units as u
from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.framework.entity_base import (
    EntityList,
    MergeConflictError,
    _merge_objects,
)
from flow360.component.simulation.framework.entity_registry import EntityRegistry
from flow360.component.simulation.primitives import (
    Box,
    Cylinder,
    GenericSurface,
    GenericVolume,
    _SurfaceEntityBase,
)
from flow360.component.simulation.simulation_params import _ParamModelBase
from flow360.component.simulation.unit_system import SI_unit_system
from flow360.log import set_logging_level
from tests.simulation.conftest import AssetBase
from tests.utils import compare_to_ref

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
                            "innerZone/interface",
                        ],
                    },
                    "innerZone": {
                        "boundaryNames": [
                            "innerZone/interface",
                            "mostinnerZone/interface",
                        ],
                    },
                    "mostinnerZone": {
                        "boundaryNames": [
                            "mostinnerZone/interface",
                            "my_wall_1",
                            "my_wall_2",
                            "my_wall_3",
                        ],
                    },
                },
                "surfaces": {
                    "farfield/farfield": {},
                    "innerZone/interface": {},
                    "mostinnerZone/interface": {},
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
        for surface_name in self._get_meta_data()["surfaces"]:
            apperance = 0
            for zone_name, zone_meta in self._get_meta_data()["zones"].items():
                if surface_name in zone_meta["boundaryNames"]:
                    apperance += 1
            i_am_interface = apperance > 1
            assert apperance > 0 and apperance < 3
            self.internal_registry.register(
                GenericSurface(name=surface_name, private_attribute_is_interface=i_am_interface)
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


class TempSimulationParam(_ParamModelBase):

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


def test_entities_expansion(my_cylinder1, my_box_zone1):
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
    my_fd = TempFluidDynamics(entities=[my_cylinder2])

    registry = EntityRegistry()
    registry.register(my_cylinder2)

    # [Registry] External changes --> Internal
    my_cylinder2.height = 131 * u.m
    for entity in registry.get_all_entities_of_given_bucket(Cylinder):
        if isinstance(entity, Cylinder) and entity.name == "zone/Cylinder2":
            assert entity.height == 131 * u.m

    # [Registry] Internal changes --> External
    my_cylinder2_ref = registry.find_by_name("zone/Cylinder2")
    my_cylinder2_ref.height = 132 * u.m
    assert my_cylinder2.height == 132 * u.m

    assert my_fd.entities.stored_entities[0].height == 132 * u.m


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
    all_box_entities = registry.get_all_entities_of_given_bucket(Box)
    assert len(all_box_entities) == 4
    assert my_box_zone1 in all_box_entities
    assert my_box_zone2 in all_box_entities
    # my_cylinder1 is not a Box but is a _volumeZoneBase and EntityRegistry registers by base type
    assert my_cylinder1 in all_box_entities
    assert my_cylinder2 in all_box_entities


def test_entities_input_interface(my_volume_mesh1):
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


def test_skipped_entities():
    TempFluidDynamics()
    assert TempFluidDynamics().entities.stored_entities is None


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
    assert "surface_3" in wall_entity_names
    assert "farfield" in wall_entity_names
    assert len(wall_entity_names) == 4


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
                TempWallBC(surfaces=[my_volume_mesh1["*"], "*"]),
            ],
        )

    ref_registry = EntityRegistry()
    ref_registry.register(my_cylinder1)
    ref_registry.register(my_volume_mesh1["zone_1"][0])
    ref_registry.register(my_volume_mesh1["zone_2"][0])
    ref_registry.register(my_volume_mesh1["zone_3"][0])
    ref_registry.register(my_volume_mesh1["surface_1"][0])
    ref_registry.register(my_volume_mesh1["surface_2"][0])
    ref_registry.register(my_volume_mesh1["surface_3"][0])

    assert my_param1.private_attribute_asset_cache.asset_entity_registry == ref_registry

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
                TempWallBC(surfaces=[my_volume_mesh2["*"], "*"]),
            ],
        )

    ref_registry = EntityRegistry()
    ref_registry.register(my_box_zone1)
    ref_registry.register(my_volume_mesh2["zone_4"][0])
    ref_registry.register(my_volume_mesh2["zone_5"][0])
    ref_registry.register(my_volume_mesh2["zone_6"][0])
    ref_registry.register(my_volume_mesh2["zone_1"][0])
    ref_registry.register(my_volume_mesh2["surface_4"][0])
    ref_registry.register(my_volume_mesh2["surface_5"][0])
    ref_registry.register(my_volume_mesh2["surface_6"][0])
    ref_registry.register(my_volume_mesh2["surface_1"][0])

    assert my_param2.private_attribute_asset_cache.asset_entity_registry == ref_registry


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
                TempWallBC(surfaces=[my_volume_mesh1["*"], "*"]),
            ],
        )
    my_cylinder1.center = (3, 2, 1) * u.m
    my_cylinder1_ref = my_param1.private_attribute_asset_cache.asset_entity_registry.find_by_name(
        "zone/Cylinder1"
    )
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

    ##:: Scenario 4: Merge NonGeneric with Generic with conflict
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
        match=re.escape(r"Cannot merge objects of different class:"),
    ):
        my_generic_merged = deepcopy(my_generic_base)
        merged = _merge_objects(
            my_cylinder1,
            Box(
                name="innerZone",
                axes=((-1, 0, 0), (1, 0, 0)),
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

    target_entity_param_reg = (
        my_param.private_attribute_asset_cache.asset_entity_registry.find_by_name("innerZone")
    )

    target_entity_mesh_reg = my_volume_mesh_with_interface.internal_registry.find_by_name(
        "innerZone"
    )

    assert (
        len(my_param.models[0].entities._get_expanded_entities()) == 3
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
    backup = deepcopy(my_volume_mesh_with_interface.internal_registry.find_by_name("innerZone"))
    assert my_volume_mesh_with_interface.internal_registry.contains(backup)

    my_volume_mesh_with_interface.internal_registry.replace_existing_with(user_override_cylinder)

    assert my_volume_mesh_with_interface.internal_registry.contains(user_override_cylinder)
