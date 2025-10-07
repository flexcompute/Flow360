import pydantic as pd
import pytest

from flow360 import u
from flow360.component.simulation.meshing_param.meshing_specs import (
    SnappySurfaceMeshingDefaults,
)
from flow360.component.simulation.meshing_param.params import (
    BetaVolumeMeshingParams,
    MeshingParams,
    ModularMeshingWorkflow,
    SnappySurfaceMeshingParams,
)
from flow360.component.simulation.meshing_param.surface_mesh_refinements import (
    SnappyBodyRefinement,
)
from flow360.component.simulation.meshing_param.volume_params import (
    AutomatedFarfield,
    AxisymmetricRefinement,
    RotationCylinder,
    UniformRefinement,
    UserDefinedFarfield,
)
from flow360.component.simulation.primitives import (
    Cylinder,
    SeedpointZone,
    SnappyBody,
    Surface,
)
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.unit_system import CGS_unit_system, SI_unit_system


def test_disable_multiple_cylinder_in_one_ratataion_cylinder():
    with pytest.raises(
        pd.ValidationError,
        match="Only single instance is allowed in entities for each RotationCylinder.",
    ):
        with CGS_unit_system:
            cylinder_1 = Cylinder(
                name="1",
                outer_radius=12,
                height=2,
                axis=(0, 1, 0),
                center=(0, 5, 0),
            )
            cylinder_2 = Cylinder(
                name="2",
                outer_radius=2,
                height=2,
                axis=(0, 1, 0),
                center=(0, 5, 0),
            )
            SimulationParams(
                meshing=MeshingParams(
                    volume_zones=[
                        RotationCylinder(
                            entities=[cylinder_1, cylinder_2],
                            spacing_axial=20,
                            spacing_radial=0.2,
                            spacing_circumferential=20,
                            enclosed_entities=[
                                Surface(name="hub"),
                            ],
                        ),
                        AutomatedFarfield(),
                    ],
                )
            )

        with CGS_unit_system:
            cylinder_1 = Cylinder(
                name="1",
                outer_radius=12,
                height=2,
                axis=(0, 1, 0),
                center=(0, 5, 0),
            )
            cylinder_2 = Cylinder(
                name="2",
                outer_radius=2,
                height=2,
                axis=(0, 1, 0),
                center=(0, 5, 0),
            )
            SimulationParams(
                meshing=ModularMeshingWorkflow(
                    volume_meshing=BetaVolumeMeshingParams(
                        volume_zones=[
                            RotationCylinder(
                                entities=[cylinder_1, cylinder_2],
                                spacing_axial=20,
                                spacing_radial=0.2,
                                spacing_circumferential=20,
                                enclosed_entities=[
                                    Surface(name="hub"),
                                ],
                            ),
                            AutomatedFarfield(),
                        ],
                    )
                )
            )


def test_limit_cylinder_entity_name_length_in_rotation_cylinder():
    with pytest.raises(
        pd.ValidationError,
        match=r"The name \(very_long_cylinder_name\) of `Cylinder` entity in `RotationCylinder`"
        + " exceeds 18 characters limit.",
    ):
        with CGS_unit_system:
            cylinder = Cylinder(
                name="very_long_cylinder_name",
                outer_radius=12,
                height=2,
                axis=(0, 1, 0),
                center=(0, 5, 0),
            )
            SimulationParams(
                meshing=MeshingParams(
                    volume_zones=[
                        RotationCylinder(
                            entities=[cylinder],
                            spacing_axial=20,
                            spacing_radial=0.2,
                            spacing_circumferential=20,
                            enclosed_entities=[
                                Surface(name="hub"),
                            ],
                        ),
                        AutomatedFarfield(),
                    ],
                )
            )

        with CGS_unit_system:
            cylinder = Cylinder(
                name="very_long_cylinder_name",
                outer_radius=12,
                height=2,
                axis=(0, 1, 0),
                center=(0, 5, 0),
            )
            SimulationParams(
                meshing=ModularMeshingWorkflow(
                    volume_meshing=BetaVolumeMeshingParams(
                        volume_zones=[
                            RotationCylinder(
                                entities=[cylinder],
                                spacing_axial=20,
                                spacing_radial=0.2,
                                spacing_circumferential=20,
                                enclosed_entities=[
                                    Surface(name="hub"),
                                ],
                            ),
                            AutomatedFarfield(),
                        ],
                    )
                )
            )


def test_reuse_of_same_cylinder():
    with pytest.raises(
        pd.ValidationError,
        match=r"Using Volume entity `I am reused` in `AxisymmetricRefinement`, `RotationCylinder` at the same time is not allowed.",
    ):
        with CGS_unit_system:
            cylinder = Cylinder(
                name="I am reused",
                outer_radius=1,
                height=12,
                axis=(0, 1, 0),
                center=(0, 5, 0),
            )
            SimulationParams(
                meshing=MeshingParams(
                    volume_zones=[
                        RotationCylinder(
                            entities=[cylinder],
                            spacing_axial=20,
                            spacing_radial=0.2,
                            spacing_circumferential=20,
                            enclosed_entities=[
                                Surface(name="hub"),
                            ],
                        ),
                        AutomatedFarfield(),
                    ],
                    refinements=[
                        AxisymmetricRefinement(
                            entities=[cylinder],
                            spacing_axial=0.1,
                            spacing_radial=0.2,
                            spacing_circumferential=0.3,
                        )
                    ],
                )
            )

        with CGS_unit_system:
            cylinder = Cylinder(
                name="I am reused",
                outer_radius=1,
                height=12,
                axis=(0, 1, 0),
                center=(0, 5, 0),
            )
            SimulationParams(
                meshing=ModularMeshingWorkflow(
                    volume_meshing=BetaVolumeMeshingParams(
                        refinements=[
                            AxisymmetricRefinement(
                                entities=[cylinder],
                                spacing_axial=0.1,
                                spacing_radial=0.2,
                                spacing_circumferential=0.3,
                            )
                        ],
                    ),
                    zones=[
                        RotationCylinder(
                            entities=[cylinder],
                            spacing_axial=20,
                            spacing_radial=0.2,
                            spacing_circumferential=20,
                            enclosed_entities=[
                                Surface(name="hub"),
                            ],
                        ),
                        AutomatedFarfield(),
                    ],
                )
            )

    with CGS_unit_system:
        cylinder = Cylinder(
            name="Okay to reuse",
            outer_radius=1,
            height=12,
            axis=(0, 1, 0),
            center=(0, 5, 0),
        )
        SimulationParams(
            meshing=MeshingParams(
                volume_zones=[
                    RotationCylinder(
                        entities=[cylinder],
                        spacing_axial=20,
                        spacing_radial=0.2,
                        spacing_circumferential=20,
                        enclosed_entities=[
                            Surface(name="hub"),
                        ],
                    ),
                    AutomatedFarfield(),
                ],
                refinements=[
                    UniformRefinement(
                        entities=[cylinder],
                        spacing=0.1,
                    )
                ],
            )
        )

    with CGS_unit_system:
        cylinder = Cylinder(
            name="Okay to reuse",
            outer_radius=1,
            height=12,
            axis=(0, 1, 0),
            center=(0, 5, 0),
        )
        SimulationParams(
            meshing=ModularMeshingWorkflow(
                volume_meshing=BetaVolumeMeshingParams(
                    refinements=[
                        UniformRefinement(
                            entities=[cylinder],
                            spacing=0.1,
                        )
                    ],
                ),
                zones=[
                    RotationCylinder(
                        entities=[cylinder],
                        spacing_axial=20,
                        spacing_radial=0.2,
                        spacing_circumferential=20,
                        enclosed_entities=[
                            Surface(name="hub"),
                        ],
                    ),
                    AutomatedFarfield(),
                ],
            )
        )

    with pytest.raises(
        pd.ValidationError,
        match=r"Using Volume entity `I am reused` in `AxisymmetricRefinement`, `UniformRefinement` at the same time is not allowed.",
    ):
        with CGS_unit_system:
            cylinder = Cylinder(
                name="I am reused",
                outer_radius=1,
                height=12,
                axis=(0, 1, 0),
                center=(0, 5, 0),
            )
            SimulationParams(
                meshing=MeshingParams(
                    refinements=[
                        UniformRefinement(entities=[cylinder], spacing=0.1),
                        AxisymmetricRefinement(
                            entities=[cylinder],
                            spacing_axial=0.1,
                            spacing_radial=0.1,
                            spacing_circumferential=0.1,
                        ),
                    ],
                )
            )

    with pytest.raises(
        pd.ValidationError,
        match=r"Using Volume entity `I am reused` in `AxisymmetricRefinement`, `UniformRefinement` at the same time is not allowed.",
    ):
        with CGS_unit_system:
            cylinder = Cylinder(
                name="I am reused",
                outer_radius=1,
                height=12,
                axis=(0, 1, 0),
                center=(0, 5, 0),
            )
            SimulationParams(
                meshing=ModularMeshingWorkflow(
                    volume_meshing=BetaVolumeMeshingParams(
                        refinements=[
                            UniformRefinement(entities=[cylinder], spacing=0.1),
                            AxisymmetricRefinement(
                                entities=[cylinder],
                                spacing_axial=0.1,
                                spacing_radial=0.1,
                                spacing_circumferential=0.1,
                            ),
                        ],
                    ),
                    zones=[AutomatedFarfield()],
                )
            )

    with pytest.raises(
        pd.ValidationError,
        match=r" Volume entity `I am reused` is used multiple times in `UniformRefinement`.",
    ):
        with CGS_unit_system:
            cylinder = Cylinder(
                name="I am reused",
                outer_radius=1,
                height=12,
                axis=(0, 1, 0),
                center=(0, 5, 0),
            )
            SimulationParams(
                meshing=MeshingParams(
                    refinements=[
                        UniformRefinement(entities=[cylinder], spacing=0.1),
                        UniformRefinement(entities=[cylinder], spacing=0.2),
                    ],
                )
            )
    with pytest.raises(
        pd.ValidationError,
        match=r" Volume entity `I am reused` is used multiple times in `UniformRefinement`.",
    ):
        with CGS_unit_system:
            cylinder = Cylinder(
                name="I am reused",
                outer_radius=1,
                height=12,
                axis=(0, 1, 0),
                center=(0, 5, 0),
            )
            SimulationParams(
                meshing=ModularMeshingWorkflow(
                    volume_meshing=BetaVolumeMeshingParams(
                        refinements=[
                            UniformRefinement(entities=[cylinder], spacing=0.1),
                            UniformRefinement(entities=[cylinder], spacing=0.2),
                        ],
                    ),
                    zones=[AutomatedFarfield()],
                )
            )


def test_require_mesh_zones():
    with SI_unit_system:
        ModularMeshingWorkflow(
            surface_meshing=SnappySurfaceMeshingParams(
                defaults=SnappySurfaceMeshingDefaults(
                    min_spacing=1 * u.mm, max_spacing=5 * u.mm, gap_resolution=0.001 * u.mm
                ),
            ),
            zones=[AutomatedFarfield()],
        )

    with SI_unit_system:
        ModularMeshingWorkflow(
            surface_meshing=SnappySurfaceMeshingParams(
                defaults=SnappySurfaceMeshingDefaults(
                    min_spacing=1 * u.mm, max_spacing=5 * u.mm, gap_resolution=0.01 * u.mm
                ),
            ),
            zones=[SeedpointZone(name="fluid", point_in_mesh=(0, 0, 0) * u.mm)],
        )

    with pytest.raises(ValueError):
        with SI_unit_system:
            ModularMeshingWorkflow(
                surface_meshing=SnappySurfaceMeshingParams(
                    defaults=SnappySurfaceMeshingDefaults(
                        min_spacing=1 * u.mm, max_spacing=5 * u.mm, gap_resolution=0.01 * u.mm
                    )
                ),
                zones=[UserDefinedFarfield()],
            )


def test_bad_refinements():
    with pytest.raises(ValueError):
        surface_meshing = SnappySurfaceMeshingParams(
            defaults=SnappySurfaceMeshingDefaults(
                min_spacing=1 * u.mm, max_spacing=5 * u.mm, gap_resolution=0.01 * u.mm
            ),
            refinements=[
                SnappyBodyRefinement(
                    min_spacing=6 * u.mm, bodies=[SnappyBody(name="bbb", stored_entities=[])]
                )
            ],
        )

    with pytest.raises(ValueError):
        surface_meshing = SnappySurfaceMeshingParams(
            defaults=SnappySurfaceMeshingDefaults(
                min_spacing=1 * u.mm, max_spacing=5 * u.mm, gap_resolution=0.01 * u.mm
            ),
            refinements=[
                SnappyBodyRefinement(
                    max_spacing=0.5 * u.mm, bodies=[SnappyBody(name="bbb", stored_entities=[])]
                )
            ],
        )
