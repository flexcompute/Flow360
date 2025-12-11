import re
from copy import deepcopy
from typing import List, Literal, Optional, Union

import numpy as np
import pydantic as pd
import pytest

import flow360.component.simulation.units as u
from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.framework.entity_base import EntityBase, EntityList
from flow360.component.simulation.framework.entity_registry import EntityRegistry
from flow360.component.simulation.framework.param_utils import (
    AssetCache,
    register_entity_list,
)
from flow360.component.simulation.outputs.output_entities import PointArray2D
from flow360.component.simulation.outputs.outputs import StreamlineOutput
from flow360.component.simulation.primitives import (
    Box,
    Cylinder,
    Edge,
    GenericVolume,
    Surface,
    _SurfaceEntityBase,
)
from flow360.component.simulation.simulation_params import (
    SimulationParams,
    _ParamModelBase,
)
from flow360.component.simulation.unit_system import LengthType, SI_unit_system
from flow360.component.simulation.utils import model_attribute_unlock
from tests.simulation.conftest import AssetBase


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


@pytest.fixture
def array_equality_override():
    pass  # No-op fixture to override the original one


class TempVolumeMesh(AssetBase):
    """Mimicking the final VolumeMesh class"""

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
        known_frozen_hashes = set()
        for zone_name, zone_meta in self._get_meta_data()["zones"].items():
            all_my_boundaries = [item for item in zone_meta["boundaryNames"]]
            known_frozen_hashes = self.internal_registry.fast_register(
                GenericVolume(
                    name=zone_name, private_attribute_zone_boundary_names=all_my_boundaries
                ),
                known_frozen_hashes,
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

        known_frozen_hashes = set()
        for surface_name in self._get_meta_data()["surfaces"]:
            known_frozen_hashes = self.internal_registry.fast_register(
                Surface(
                    name=surface_name, private_attribute_is_interface=surface_name in interfaces
                ),
                known_frozen_hashes,
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

    @property
    def base_length(self) -> LengthType:
        return self.private_attribute_asset_cache.project_length_unit.to("m")

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


##:: ---------------- EntityList/Registry tests ----------------


def test_by_reference_registry(my_cylinder2):
    """Test that the entity registry contains reference not deepcopy of the entities."""
    my_fd = TempFluidDynamics(entities=[my_cylinder2])

    registry = EntityRegistry()
    registry.fast_register(my_cylinder2, set())
    entities = list(registry.view(Cylinder))  # get the entities now before change
    # [Registry] External changes --> Internal
    my_cylinder2.height = 131 * u.m
    for entity in entities:
        if isinstance(entity, Cylinder) and entity.name == "zone/Cylinder2":
            assert entity.height == 131 * u.m

    # [Registry] Internal changes --> External
    my_cylinder2_ref = registry.find_by_naming_pattern(
        pattern="zone/Cylinder2", enforce_output_as_list=False
    )
    my_cylinder2_ref.height = 132 * u.m
    assert my_cylinder2.height == 132 * u.m

    assert my_fd.entities.stored_entities[0].height == 132 * u.m


def test_entity_registry_item_retrieval(
    my_cylinder1,
    my_cylinder2,
    my_box_zone1,
    my_box_zone2,
):
    known_frozen_hashes = set()
    registry = EntityRegistry()
    known_frozen_hashes = registry.fast_register(my_cylinder1, known_frozen_hashes)
    known_frozen_hashes = registry.fast_register(my_cylinder2, known_frozen_hashes)
    known_frozen_hashes = registry.fast_register(my_box_zone1, known_frozen_hashes)
    known_frozen_hashes = registry.fast_register(my_box_zone2, known_frozen_hashes)
    all_box_entities = list(registry.view(Box))
    # Note: After switching to type-based storage, Box and Cylinder are separate types
    assert len(all_box_entities) == 2
    assert my_box_zone1 in all_box_entities
    assert my_box_zone2 in all_box_entities
    assert my_cylinder1 not in all_box_entities
    assert my_cylinder2 not in all_box_entities

    known_frozen_hashes = set()
    registry = EntityRegistry()
    known_frozen_hashes = registry.fast_register(
        Surface(name="AA_ground_close"), known_frozen_hashes
    )
    known_frozen_hashes = registry.fast_register(Surface(name="BB"), known_frozen_hashes)
    known_frozen_hashes = registry.fast_register(Surface(name="CC_ground"), known_frozen_hashes)
    items = registry.find_by_naming_pattern("*ground", enforce_output_as_list=True)
    assert len(items) == 1
    assert items[0].name == "CC_ground"


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

    known_frozen_hashes = set()
    ref_registry = EntityRegistry()
    known_frozen_hashes = ref_registry.fast_register(my_cylinder1, known_frozen_hashes)
    known_frozen_hashes = ref_registry.fast_register(my_volume_mesh1["zone_1"], known_frozen_hashes)
    known_frozen_hashes = ref_registry.fast_register(my_volume_mesh1["zone_2"], known_frozen_hashes)
    known_frozen_hashes = ref_registry.fast_register(my_volume_mesh1["zone_3"], known_frozen_hashes)
    known_frozen_hashes = ref_registry.fast_register(
        my_volume_mesh1["surface_1"], known_frozen_hashes
    )
    known_frozen_hashes = ref_registry.fast_register(
        my_volume_mesh1["surface_2"], known_frozen_hashes
    )
    known_frozen_hashes = ref_registry.fast_register(
        my_volume_mesh1["surface_3"], known_frozen_hashes
    )

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

    known_frozen_hashes = set()
    ref_registry = EntityRegistry()
    known_frozen_hashes = ref_registry.fast_register(my_box_zone1, known_frozen_hashes)
    known_frozen_hashes = ref_registry.fast_register(my_volume_mesh2["zone_4"], known_frozen_hashes)
    known_frozen_hashes = ref_registry.fast_register(my_volume_mesh2["zone_5"], known_frozen_hashes)
    known_frozen_hashes = ref_registry.fast_register(my_volume_mesh2["zone_6"], known_frozen_hashes)
    known_frozen_hashes = ref_registry.fast_register(my_volume_mesh2["zone_1"], known_frozen_hashes)
    known_frozen_hashes = ref_registry.fast_register(
        my_volume_mesh2["surface_4"], known_frozen_hashes
    )
    known_frozen_hashes = ref_registry.fast_register(
        my_volume_mesh2["surface_5"], known_frozen_hashes
    )
    known_frozen_hashes = ref_registry.fast_register(
        my_volume_mesh2["surface_6"], known_frozen_hashes
    )
    known_frozen_hashes = ref_registry.fast_register(
        my_volume_mesh2["surface_1"], known_frozen_hashes
    )
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
    my_cylinder1_ref = used_entity_registry.find_by_naming_pattern(
        pattern="zone/Cylinder1", enforce_output_as_list=False
    )
    assert all(my_cylinder1_ref.center == [3, 2, 1] * u.m)


def test_entity_registry_find_by_id():
    registry = EntityRegistry()

    genericVolume_entity = GenericVolume(name="123", private_attribute_id="original_zone_name")
    surface_entity1 = Surface(name="123", private_attribute_id="original_surface_name")
    surface_entity2 = Surface(name="1234", private_attribute_id="original_surface_name2")
    edge_entity = Edge(name="123", private_attribute_id="original_edge_name")
    with SI_unit_system:
        box_entity = Box(
            name="123bOx",
            center=(0, 0, 0) * u.m,
            size=(1, 1, 1) * u.m,
            axis_of_rotation=(1, 1, 0),
            angle_of_rotation=np.pi * u.rad,
            private_attribute_id="original_box_name",
        )

    known_frozen_hashes = set()
    known_frozen_hashes = registry.fast_register(genericVolume_entity, known_frozen_hashes)
    known_frozen_hashes = registry.fast_register(surface_entity1, known_frozen_hashes)
    known_frozen_hashes = registry.fast_register(surface_entity2, known_frozen_hashes)
    known_frozen_hashes = registry.fast_register(edge_entity, known_frozen_hashes)
    known_frozen_hashes = registry.fast_register(box_entity, known_frozen_hashes)

    modified_genericVolume_entity = GenericVolume(
        name="999", private_attribute_id="original_zone_name"
    )
    modified_surface_entity1 = Surface(name="999", private_attribute_id="original_surface_name")
    modified_surface_entity2 = Surface(name="9992", private_attribute_id="original_surface_name2")
    modified_edge_entity = Edge(name="999", private_attribute_id="original_edge_name")
    with SI_unit_system:
        modified_box_entity = Box(
            name="999",
            center=(0, 0, 0) * u.m,
            size=(1, 1, 1) * u.m,
            axis_of_rotation=(1, 1, 0),
            angle_of_rotation=np.pi * u.rad,
            private_attribute_id="original_box_name",
        )

    for modified_item, original_item in zip(
        [
            modified_genericVolume_entity,
            modified_surface_entity1,
            modified_surface_entity2,
            modified_edge_entity,
            modified_box_entity,
        ],
        [genericVolume_entity, surface_entity1, surface_entity2, edge_entity, box_entity],
    ):
        assert (
            registry.find_by_asset_id(
                entity_id=modified_item.id, entity_class=modified_item.__class__
            )
            == original_item
        )
