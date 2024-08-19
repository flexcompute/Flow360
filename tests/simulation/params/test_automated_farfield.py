import re

import pytest

from flow360.component.simulation.framework.unique_list import UniqueStringList
from flow360.component.simulation.meshing_param.face_params import SurfaceRefinement
from flow360.component.simulation.meshing_param.params import MeshingParams
from flow360.component.simulation.meshing_param.volume_params import (
    AutomatedFarfield,
    RotationCylinder,
)
from flow360.component.simulation.models.surface_models import (
    Freestream,
    SlipWall,
    SymmetryPlane,
    Wall,
)
from flow360.component.simulation.outputs.outputs import (
    SurfaceIntegralOutput,
    SurfaceOutput,
)
from flow360.component.simulation.primitives import Cylinder, Surface
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.unit_system import SI_unit_system


def test_automated_farfield_names():
    with SI_unit_system:
        my_farfield = AutomatedFarfield(name="my_farfield")
        _ = SimulationParams(
            meshing=MeshingParams(
                volume_zones=[
                    my_farfield,
                ]
            ),
        )

    assert my_farfield.private_attribute_entity.name == "fluid"
    assert isinstance(
        my_farfield.private_attribute_entity.private_attribute_zone_boundary_names, UniqueStringList
    )
    assert sorted(
        my_farfield.private_attribute_entity.private_attribute_zone_boundary_names.items
    ) == sorted(["farfield", "symmetric"])

    # Warning: volume_zones.append(RotationCylinder(...)) will not change the zone name
    # because the append() will not trigger the validators. It is probably fine since we construct `SimulationParams`
    # again in transltors anyway.
    with SI_unit_system:
        my_cylinder = Cylinder(
            name="zone/Cylinder1",
            height=11,
            axis=(1, 0, 0),
            inner_radius=1,
            outer_radius=2,
            center=(1, 2, 3),
        )
        my_farfield = AutomatedFarfield(name="my_farfield")
        _ = SimulationParams(
            meshing=MeshingParams(
                volume_zones=[
                    my_farfield,
                    RotationCylinder(
                        entities=my_cylinder,
                        spacing_axial=0.1,
                        spacing_radial=0.1,
                        spacing_circumferential=0.1,
                    ),
                ]
            ),
        )

    assert my_farfield.private_attribute_entity.name == "stationaryBlock"
    assert set(
        my_farfield.private_attribute_entity.private_attribute_zone_boundary_names.items
    ) == set(["farfield", "symmetric"])

    with SI_unit_system:
        my_farfield = AutomatedFarfield(method="quasi-3d")
        _ = SimulationParams(
            meshing=MeshingParams(
                volume_zones=[
                    my_farfield,
                ]
            ),
        )
    assert set(
        my_farfield.private_attribute_entity.private_attribute_zone_boundary_names.items
    ) == set(["farfield", "symmetric-1", "symmetric-2"])


def test_automated_farfield_surface_usage():
    # Test use of GhostSurface in meshing
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Can not find any valid entity of type ['Surface', 'ForAll'] from the input."
        ),
    ):
        with SI_unit_system:
            my_farfield = AutomatedFarfield(name="my_farfield")
            _ = SimulationParams(
                meshing=MeshingParams(
                    volume_zones=[
                        my_farfield,
                    ],
                    refinements=[
                        SurfaceRefinement(
                            name="does not work",
                            entities=[my_farfield.farfield],
                            max_edge_length=1e-4,
                        )
                    ],
                ),
            )

    # Test use of GhostSurface in boundary conditions
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Can not find any valid entity of type ['Surface', 'ForAll'] from the input."
        ),
    ):
        with SI_unit_system:
            my_farfield = AutomatedFarfield(name="my_farfield")
            _ = SimulationParams(
                meshing=MeshingParams(
                    volume_zones=[
                        my_farfield,
                    ],
                    refinements=[
                        SurfaceRefinement(
                            name="does not work",
                            entities=[my_farfield.farfield],
                            max_edge_length=1e-4,
                        )
                    ],
                ),
                models=[Wall(name="wall", surface=my_farfield.farfield)],
            )

    with SI_unit_system:
        my_farfield = AutomatedFarfield(name="my_farfield")
        _ = SimulationParams(
            models=[
                SlipWall(name="slipwall", entities=my_farfield.farfield),
                SymmetryPlane(name="symm_plane", entities=my_farfield.symmetry_planes),
            ],
        )

    with SI_unit_system:
        my_farfield = AutomatedFarfield(name="my_farfield")
        _ = SimulationParams(
            models=[
                Freestream(name="fs", entities=my_farfield.farfield),
            ],
        )

    # Test use of GhostSurface in SurfaceOutput
    with SI_unit_system:
        my_farfield = AutomatedFarfield(name="my_farfield")
        _ = SimulationParams(
            outputs=[
                SurfaceOutput(entities=my_farfield.farfield, output_fields=["Cp"]),
                SurfaceIntegralOutput(
                    name="prb 110",
                    entities=[
                        my_farfield.symmetry_planes,
                        Surface(name="surface2"),
                    ],
                    output_fields=["Cp"],
                ),
            ],
        )

    # Test that the GhostSurface will have updated full name through model_validators after entity register has been constructed
    with SI_unit_system:
        my_farfield = AutomatedFarfield(name="my_farfield", method="quasi-3d")
        param = SimulationParams(
            models=[
                Freestream(name="fs", entities=my_farfield.farfield),
                SymmetryPlane(name="symm_plane", entities=my_farfield.symmetry_planes[1]),
            ],
            meshing=MeshingParams(
                volume_zones=[
                    my_farfield,
                ],
            ),
        )
    param._get_used_entity_registry()
    assert (
        param.models[0].entities.stored_entities[0].private_attribute_full_name == "fluid/farfield"
    )
    assert (
        param.models[1].entities.stored_entities[0].private_attribute_full_name
        == "fluid/symmetric-2"
    )


def test_automated_farfield_import_export():

    my_farfield = AutomatedFarfield(name="my_farfield")
    model_as_dict = my_farfield.model_dump()
    assert "private_attribute_entity" not in model_as_dict.keys()

    model_as_dict = {"name": "my_farfield", "method": "auto"}
    my_farfield = AutomatedFarfield(**model_as_dict)

    model_as_dict = {"name": "my_farfield"}
    my_farfield = AutomatedFarfield(**model_as_dict)

    with pytest.raises(
        ValueError,
        match=re.escape("Unable to extract tag using discriminator 'type'"),
    ):
        MeshingParams(**{"volume_zones": [model_as_dict]})

    model_as_dict = {"name": "my_farfield", "type": "AutomatedFarfield"}
    meshing = MeshingParams(**{"volume_zones": [model_as_dict]})
    assert isinstance(meshing.volume_zones[0], AutomatedFarfield)
