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


def test_automated_farfield_surface_usage():
    # Test use of GhostSurface in meshing
    with pytest.raises(
        ValueError,
        match=re.escape("Can not find any valid entity of type ['Surface'] from the input."),
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
        match=re.escape("Can not find any valid entity of type ['Surface'] from the input."),
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
