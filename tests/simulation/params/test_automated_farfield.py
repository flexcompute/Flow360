import re

import pytest
import unyt as u

from flow360.component.project_utils import set_up_params_for_uploading
from flow360.component.resource_base import local_metadata_builder
from flow360.component.simulation import services
from flow360.component.simulation.meshing_param.face_params import SurfaceRefinement
from flow360.component.simulation.meshing_param.params import (
    MeshingDefaults,
    MeshingParams,
)
from flow360.component.simulation.meshing_param.volume_params import AutomatedFarfield
from flow360.component.simulation.models.surface_models import (
    Freestream,
    SlipWall,
    SymmetryPlane,
    Wall,
)
from flow360.component.simulation.operating_condition.operating_condition import (
    AerospaceCondition,
)
from flow360.component.simulation.outputs.outputs import (
    SurfaceIntegralOutput,
    SurfaceOutput,
    UserDefinedField,
)
from flow360.component.simulation.primitives import Surface
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.unit_system import SI_unit_system
from flow360.component.surface_mesh_v2 import SurfaceMeshMetaV2, SurfaceMeshV2


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


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
                    output_fields=["Cpt_user_defined"],
                ),
            ],
            user_defined_fields=[UserDefinedField(name="Cpt_user_defined", expression="Cp-123")],
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


def test_symmetric_existence():
    def _run_validation(params):
        params = set_up_params_for_uploading(
            params=params,
            root_asset=sm,
            length_unit=1 * u.m,
            use_beta_mesher=True,
            use_geometry_AI=False,
        )

        _, errors, _ = services.validate_model(
            params_as_dict=params.model_dump(exclude_none=True),
            validated_by=services.ValidationCalledBy.LOCAL,
            root_item_type="SurfaceMesh",
            validation_level="All",
        )

        return errors

    sm = SurfaceMeshV2.from_local_storage(
        local_storage_path="data/surface_mesh",
        meta_data=SurfaceMeshMetaV2(
            **local_metadata_builder(
                id="aaa",
                name="aaa",
                cloud_path_prefix="aaa",
            )
        ),
    )
    farfield = AutomatedFarfield()
    with SI_unit_system:
        params = SimulationParams(
            operating_condition=AerospaceCondition(velocity_magnitude=1000),
            meshing=MeshingParams(
                defaults=MeshingDefaults(
                    boundary_layer_first_layer_thickness=0.001,
                    boundary_layer_growth_rate=1.1,
                ),
                volume_zones=[farfield],
            ),
            models=[
                Wall(surfaces=sm["*"]),
                Freestream(surfaces=[farfield.farfield]),
            ],
        )

    # Valid Symmetric but did not use it
    errors = _run_validation(params)
    assert len(errors) == 1
    assert (
        "The following boundaries do not have a boundary condition: symmetric." in errors[0]["msg"]
    )

    params.models.append(SymmetryPlane(surfaces=[farfield.symmetry_planes]))
    errors = _run_validation(params)
    assert errors is None

    # Invalid Symmetric
    params.meshing.defaults.planar_face_tolerance = 1e-100
    errors = _run_validation(params)
    assert len(errors) == 1
    assert (
        "`symmetric` boundary not usable: model spans y=[-4.08e-05, 1.16e+03], "
        "but tolerance from y=0 is 1.00e-100 x 2.53e+03 = 2.53e-97." in errors[0]["msg"]
    )
