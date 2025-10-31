import re

import pytest

from flow360.component.simulation.entity_info import SurfaceMeshEntityInfo
from flow360.component.simulation.framework.param_utils import AssetCache
from flow360.component.simulation.meshing_param.params import (
    MeshingDefaults,
    MeshingParams,
)
from flow360.component.simulation.meshing_param.volume_params import (
    CustomZones,
    UserDefinedFarfield,
)
from flow360.component.simulation.models.material import aluminum
from flow360.component.simulation.models.volume_models import Solid
from flow360.component.simulation.primitives import CustomVolume, Surface
from flow360.component.simulation.services import ValidationCalledBy, validate_model
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.unit_system import SI_unit_system


def _build_params_with_custom_volume(element_type: str):
    zone = CustomVolume(name="solid_zone", boundaries=[Surface(name="face1")])
    return SimulationParams(
        meshing=MeshingParams(
            defaults=MeshingDefaults(
                boundary_layer_first_layer_thickness=1e-4,
                planar_face_tolerance=1e-4,
            ),
            volume_zones=[
                CustomZones(
                    name="custom_zones",
                    entities=[zone],
                    element_type=element_type,
                ),
                UserDefinedFarfield(),
            ],
        ),
        models=[
            Solid(
                entities=[zone],
                material=aluminum,
            )
        ],
        private_attribute_asset_cache=AssetCache(
            use_inhouse_mesher=True,
            project_entity_info=SurfaceMeshEntityInfo(boundaries=[Surface(name="face1")]),
        ),
    )


def test_solid_custom_volume_requires_tetrahedra_raises_on_mixed():
    with SI_unit_system:
        params = _build_params_with_custom_volume(element_type="mixed")

    _, errors, _ = validate_model(
        params_as_dict=params.model_dump(mode="json"),
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type="SurfaceMesh",
        validation_level="All",
    )

    assert errors is not None and len(errors) == 1
    assert errors[0]["msg"] == (
        "Value error, CustomVolume 'solid_zone' must be meshed with tetrahedra-only elements."
    )
    # Location should point to Solid.entities
    assert errors[0]["loc"][0] == "models"
    assert errors[0]["loc"][2] == "entities"


def test_solid_custom_volume_allows_tetrahedra():
    with SI_unit_system:
        params = _build_params_with_custom_volume(element_type="tetrahedra")

    _, errors, _ = validate_model(
        params_as_dict=params.model_dump(mode="json"),
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type="SurfaceMesh",
        validation_level="All",
    )

    assert errors is None
