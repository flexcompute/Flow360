from typing import List, Literal, Optional

import pydantic as pd
import pytest
from flow360_schema.framework.physical_dimensions import Length

import flow360.component.simulation.units as u
from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.framework.entity_base import EntityList
from flow360.component.simulation.framework.entity_registry import EntityRegistry
from flow360.component.simulation.framework.param_utils import (
    AssetCache,
    register_entity_list,
)
from flow360.component.simulation.primitives import Cylinder, GenericVolume, Surface
from flow360.component.simulation.simulation_params import _ParamModelBase
from flow360.component.simulation.unit_system import SI_unit_system
from tests.simulation.conftest import AssetBase


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


class TempRotation(Flow360BaseModel):
    entities: EntityList[GenericVolume, Cylinder] = pd.Field(alias="volumes")
    parent_volume: Optional[GenericVolume] = pd.Field(None)


class TempUserDefinedDynamic(Flow360BaseModel):
    name: str = pd.Field()
    input_boundary_patches: Optional[EntityList[Surface]] = pd.Field(None)
    output_target: Optional[Cylinder] = pd.Field(
        None
    )  # Limited to `Cylinder` for now as we have only tested using UDD to control rotation.


class TempSimulationParam(_ParamModelBase):

    far_field_type: Literal["auto", "user-defined"] = pd.Field()

    models: List[TempRotation] = pd.Field()
    udd: Optional[TempUserDefinedDynamic] = pd.Field(None)
    private_attribute_asset_cache: AssetCache = pd.Field(AssetCache(), frozen=True)

    @property
    def base_length(self) -> Length.Float64:
        return self.private_attribute_asset_cache.project_length_unit.to("m")

    def preprocess(self):
        """
        Supply self._supplementary_registry to the construction of
        TempFluidDynamics etc so that the class can perform proper validation
        """
        self.private_attribute_asset_cache._force_set_attr("project_length_unit", 1 * u.m)

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
def my_volume_mesh1():
    return TempVolumeMesh(file_name="volMesh-1.cgns")


@pytest.fixture
def my_volume_mesh_with_interface():
    return TempVolumeMesh(file_name="volMesh-with_interface.cgns")


def test_asset_getitem(my_volume_mesh1):
    """Test the __getitem__ interface of asset objects."""
    # 1. Using reference of single asset entity
    expanded_entities = my_volume_mesh1["zone*"]
    assert len(expanded_entities) == 3

    try:
        my_volume_mesh1["asdf"]
    except ValueError as error:
        assert (
            str(error) == "Failed to find any matching entity with asdf. Please check your input."
        )
    else:
        raise AssertionError("Expected __getitem__ lookup failure for missing entity pattern.")


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
