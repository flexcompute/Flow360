import pytest

import flow360.component.simulation.units as u
from flow360.component.project_utils import _set_up_params_non_persistent_entity_info
from flow360.component.simulation.entity_info import SurfaceMeshEntityInfo
from flow360.component.simulation.meshing_param.params import MeshingParams
from flow360.component.simulation.meshing_param.volume_params import (
    CustomZones,
    RotationVolume,
    UserDefinedFarfield,
)
from flow360.component.simulation.primitives import (
    AxisymmetricBody,
    CustomVolume,
    Surface,
)
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.unit_system import SI_unit_system


def _get_basic_entity_info():
    return SurfaceMeshEntityInfo(boundaries=[Surface(name="face1"), Surface(name="face2")])


def test_custom_volume_added_to_draft_entities():
    with SI_unit_system:
        custom_volume = CustomVolume(name="cv1", boundaries=[Surface(name="face1")])
        params = SimulationParams(
            meshing=MeshingParams(
                volume_zones=[
                    CustomZones(name="custom_zones", entities=[custom_volume]),
                    UserDefinedFarfield(name="ff"),
                ]
            )
        )
    entity_info = _get_basic_entity_info()
    updated = _set_up_params_non_persistent_entity_info(entity_info, params)
    assert any(isinstance(e, CustomVolume) and e.name == "cv1" for e in updated.draft_entities)


def test_axisymmetric_body_added_to_draft_entities():
    with SI_unit_system:
        axis_body = AxisymmetricBody(
            name="axis_body",
            center=(0, 0, 0) * u.m,
            axis=(0, 0, 1),
            profile_curve=[
                (-1.0, 0.0) * u.m,
                (-0.5, 0.2) * u.m,
                (0.5, 0.2) * u.m,
                (1.0, 0.0) * u.m,
            ],
        )
        params = SimulationParams(
            meshing=MeshingParams(
                volume_zones=[
                    RotationVolume(
                        entities=axis_body,
                        spacing_axial=0.1 * u.m,
                        spacing_radial=0.1 * u.m,
                        spacing_circumferential=0.1 * u.m,
                    ),
                    UserDefinedFarfield(name="ff"),
                ]
            )
        )
    entity_info = _get_basic_entity_info()
    updated = _set_up_params_non_persistent_entity_info(entity_info, params)
    assert any(
        isinstance(e, AxisymmetricBody) and e.name == "axis_body" for e in updated.draft_entities
    )
