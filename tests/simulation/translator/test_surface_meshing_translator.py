import json
import os

import pytest

import flow360.component.simulation.units as u
from flow360.component.geometry import Geometry, GeometryMeta
from flow360.component.project import create_draft
from flow360.component.project_utils import (
    set_up_params_for_uploading,
    validate_params_with_context,
)
from flow360.component.resource_base import local_metadata_builder
from flow360.component.simulation.draft_context.coordinate_system_manager import (
    CoordinateSystemAssignmentGroup,
    CoordinateSystemEntityRef,
    CoordinateSystemStatus,
)
from flow360.component.simulation.entity_info import GeometryEntityInfo
from flow360.component.simulation.entity_operation import CoordinateSystem
from flow360.component.simulation.framework.param_utils import AssetCache
from flow360.component.simulation.framework.updater_utils import compare_values
from flow360.component.simulation.meshing_param import snappy
from flow360.component.simulation.meshing_param.edge_params import (
    AngleBasedRefinement,
    AspectRatioBasedRefinement,
    HeightBasedRefinement,
    ProjectAnisoSpacing,
    SurfaceEdgeRefinement,
)
from flow360.component.simulation.meshing_param.face_params import (
    GeometryRefinement,
    SurfaceRefinement,
)
from flow360.component.simulation.meshing_param.meshing_specs import (
    VolumeMeshingDefaults,
)
from flow360.component.simulation.meshing_param.params import (
    MeshingDefaults,
    MeshingParams,
    ModularMeshingWorkflow,
    VolumeMeshingParams,
)
from flow360.component.simulation.meshing_param.volume_params import (
    AutomatedFarfield,
    CustomZones,
    RotationVolume,
    UniformRefinement,
    WheelBelts,
    WindTunnelFarfield,
)
from flow360.component.simulation.operating_condition.operating_condition import (
    AerospaceCondition,
)
from flow360.component.simulation.primitives import (
    Box,
    Cylinder,
    Edge,
    MirroredGeometryBodyGroup,
    MirroredSurface,
    SeedpointVolume,
    Surface,
)
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.translator.surface_meshing_translator import (
    get_surface_meshing_json,
)
from flow360.component.simulation.unit_system import (
    LengthType,
    SI_unit_system,
    imperial_unit_system,
)
from tests.simulation.conftest import AssetBase


class TempGeometry(AssetBase):
    """Mimicing the final VolumeMesh class"""

    fname: str
    mesh_unit: LengthType.Positive

    def _get_meta_data(self):
        if self.fname == "om6wing.csm":
            return {
                "edges": {
                    "body01_edge001": {},
                    "body01_edge002": {},
                    "body01_edge003": {},
                    "body01_edge004": {},
                },
                "surfaces": {"body01_face001": {}},
                "mesh_unit": {"units": "m", "value": 1.0},
            }
        elif self.fname == "geometry.egads":
            return {
                "surfaces": {
                    "body01_face001": {},
                    "body01_face002": {},
                    "body01_face003": {},
                    "body01_face004": {},
                    "body01_face005": {},
                    "body01_face006": {},
                    "body01_face007": {},
                    "body01_face008": {},
                    "body01_face009": {},
                    "body01_face010": {},
                },
                "mesh_unit": {"units": "m", "value": 1.0},
            }
        elif self.fname == "rotor.csm":
            return {
                "surfaces": {
                    "body01_face001": {},
                    "body01_face002": {},
                    "body01_face003": {},
                },
                "edges": {
                    "body01_edge001": {},
                    "body01_edge002": {},
                    "body01_edge003": {},
                    "body01_edge004": {},
                    "body01_edge005": {},
                    "body01_edge006": {},
                    "body01_edge007": {},
                },
                "mesh_unit": {"units": "inch", "value": 1.0},
            }
        elif self.fname == "tester.stl":
            return {
                "surfaces": {
                    "body0::patch0": {},
                    "body0::patch1": {},
                    "body1::patch0": {},
                    "body1::patch1": {},
                    "body1::patch2": {},
                    "body2::patch0": {},
                    "body3::patch0": {},
                },
                "mesh_unit": {"units": "mm", "value": 1.0},
            }
        elif self.fname == "tester_no_naming.stl":
            return {
                "surfaces": {
                    "body01_face001": {},
                    "body01_face002": {},
                    "body01_face003": {},
                },
                "mesh_unit": {"units": "mm", "value": 1.0},
            }
        else:
            raise ValueError("Invalid file name")

    def _get_entity_info(self):
        if self.fname == "om6wing.csm":
            return GeometryEntityInfo(
                face_ids=["body01_face001"],
                edge_ids=[
                    "body01_edge001",
                    "body01_edge002",
                    "body01_edge003",
                    "body01_edge004",
                ],
                face_attribute_names=["dummy"],
                face_group_tag="dummy",
                grouped_faces=[
                    [Surface(name="wing", private_attribute_sub_components=["body01_face001"])]
                ],
            )

        elif self.fname == "geometry.egads":
            return GeometryEntityInfo(
                face_ids=[
                    "body01_face001",
                    "body01_face002",
                    "body01_face003",
                    "body01_face004",
                    "body01_face005",
                    "body01_face006",
                    "body01_face007",
                    "body01_face008",
                    "body01_face009",
                    "body01_face010",
                ],
                edge_ids=[],
                face_attribute_names=["dummy"],
                face_group_tag="dummy",
                grouped_faces=[
                    [
                        Surface(
                            name="Wing",
                            private_attribute_sub_components=[
                                "body01_face001",
                                "body01_face002",
                                "body01_face003",
                                "body01_face004",
                            ],
                        ),
                        Surface(
                            name="Fuselage",
                            private_attribute_sub_components=[
                                "body01_face005",
                                "body01_face006",
                                "body01_face007",
                            ],
                        ),
                        Surface(
                            name="Stab",
                            private_attribute_sub_components=[
                                "body01_face008",
                                "body01_face009",
                            ],
                        ),
                        Surface(
                            name="Fin",
                            private_attribute_sub_components=[
                                "body01_face010",
                            ],
                        ),
                    ]
                ],
            )
        elif self.fname == "rotor.csm":
            return GeometryEntityInfo(
                face_ids=["body01_face001", "body01_face002", "body01_face003"],
                edge_ids=[
                    "body01_edge001",
                    "body01_edge002",
                    "body01_edge003",
                    "body01_edge004",
                    "body01_edge005",
                    "body01_edge006",
                    "body01_edge007",
                ],
                face_attribute_names=["dummy"],
                face_group_tag="dummy",
                grouped_faces=[
                    [
                        Surface(
                            name="rotor",
                            private_attribute_sub_components=[
                                "body01_face001",
                                "body01_face002",
                                "body01_face003",
                            ],
                        )
                    ]
                ],
            )
            r
        elif self.fname == "tester.stl":
            return GeometryEntityInfo(
                face_ids=[
                    "body0::patch0",
                    "body0::patch1",
                    "body1::patch0",
                    "body1::patch1",
                    "body1::patch2",
                    "body2::patch0",
                    "body3::patch0",
                ],
                edge_ids=[],
                face_attribute_names=["faceId"],
                face_group_tag="dummy",
                grouped_faces=[
                    [
                        Surface(
                            name="body0::patch0",
                            private_attribute_sub_components=["body0::patch0"],
                        ),
                        Surface(
                            name="body0::patch1",
                            private_attribute_sub_components=["body0::patch1"],
                        ),
                        Surface(
                            name="body1::patch0",
                            private_attribute_sub_components=["body1::patch0"],
                        ),
                        Surface(
                            name="body1::patch1",
                            private_attribute_sub_components=["body1::patch1"],
                        ),
                        Surface(
                            name="body1::patch2",
                            private_attribute_sub_components=["body1::patch2"],
                        ),
                        Surface(
                            name="body2::patch0",
                            private_attribute_sub_components=["body2::patch0"],
                        ),
                        Surface(
                            name="body3::patch0",
                            private_attribute_sub_components=["body3::patch0"],
                        ),
                    ]
                ],
            )
        elif self.fname == "tester_no_naming.stl":
            return GeometryEntityInfo(
                face_ids=["body01_face001", "body01_face002", "body01_face003"],
                edge_ids=[],
                face_attribute_names=["faceId"],
                face_group_tag="faceId",
                grouped_faces=[
                    [
                        Surface(
                            name="body01_face001",
                            private_attribute_sub_components=["body01_face001"],
                        ),
                        Surface(
                            name="body01_face002",
                            private_attribute_sub_components=["body01_face002"],
                        ),
                        Surface(
                            name="body01_face003",
                            private_attribute_sub_components=["body01_face003"],
                        ),
                    ]
                ],
            )
        else:
            raise ValueError("Invalid file name")

    def _populate_registry(self):
        self.mesh_unit = LengthType.validate(self._get_meta_data()["mesh_unit"])
        if self.snappy:
            self.internal_registry = self._get_entity_info()._group_entity_by_tag(
                "face", "faceId", self.internal_registry
            )
            self.snappy_bodies = self._get_entity_info()._group_faces_by_snappy_format()
        else:
            for zone_name in (
                self._get_meta_data()["edges"] if "edges" in self._get_meta_data() else []
            ):
                # pylint: disable=fixme
                # TODO: private_attribute_sub_components is hacked to be just the grouped name,
                # TODO: this should actually be the list of edgeIDs/faceIDs
                self.internal_registry.register(
                    Edge(name=zone_name, private_attribute_sub_components=[zone_name])
                )
            for surface_name in (
                self._get_meta_data()["surfaces"] if "surfaces" in self._get_meta_data() else []
            ):
                self.internal_registry.register(
                    Surface(name=surface_name, private_attribute_sub_components=[surface_name])
                )

    def __init__(self, file_name: str, for_snappy=False):
        super().__init__()
        self.fname = file_name
        self.snappy = for_snappy
        self._populate_registry()


@pytest.fixture()
def om6wing_tutorial_global_plus_local_override():
    my_geometry = TempGeometry("om6wing.csm")
    with SI_unit_system:
        param = SimulationParams(
            private_attribute_asset_cache=AssetCache(
                project_entity_info=my_geometry._get_entity_info()
            ),
            meshing=MeshingParams(
                defaults=MeshingDefaults(
                    surface_edge_growth_rate=1.07,
                    curvature_resolution_angle=10 * u.deg,
                    surface_max_edge_length=15 * u.cm,
                    edge_split_layers=0,
                ),
                refinements=[
                    SurfaceRefinement(
                        entities=[my_geometry["body01_face001"]],
                        max_edge_length=14 * u.cm,
                    ),
                    SurfaceEdgeRefinement(
                        entities=[my_geometry["body01_edge001"], my_geometry["body01_edge002"]],
                        method=HeightBasedRefinement(value=3e-2 * u.cm),
                    ),
                    SurfaceEdgeRefinement(
                        entities=[my_geometry["body01_edge003"], my_geometry["body01_edge004"]],
                        method=ProjectAnisoSpacing(),
                    ),
                ],
            ),
        )
    return param


@pytest.fixture()
def om6wing_tutorial_aspect_ratio():
    my_geometry = TempGeometry("om6wing.csm")
    with SI_unit_system:
        param = SimulationParams(
            private_attribute_asset_cache=AssetCache(
                project_entity_info=my_geometry._get_entity_info()
            ),
            meshing=MeshingParams(
                defaults=MeshingDefaults(
                    surface_edge_growth_rate=1.07,
                    curvature_resolution_angle=10 * u.deg,
                    surface_max_edge_length=15 * u.cm,
                    edge_split_layers=0,
                ),
                refinements=[
                    SurfaceRefinement(
                        entities=[my_geometry["body01_face001"]],
                        max_edge_length=14 * u.cm,
                    ),
                    SurfaceEdgeRefinement(
                        entities=[my_geometry["body01_edge001"], my_geometry["body01_edge002"]],
                        method=HeightBasedRefinement(value=3e-2 * u.cm),
                    ),
                    SurfaceEdgeRefinement(
                        entities=[my_geometry["body01_edge003"], my_geometry["body01_edge004"]],
                        method=AspectRatioBasedRefinement(value=10),
                    ),
                ],
            ),
        )
    return param


@pytest.fixture()
def get_om6wing_geometry():
    return TempGeometry("om6wing.csm")


@pytest.fixture()
def om6wing_tutorial_global_only():
    my_geometry = TempGeometry("om6wing.csm")
    with SI_unit_system:
        param = SimulationParams(
            private_attribute_asset_cache=AssetCache(
                project_entity_info=my_geometry._get_entity_info()
            ),
            meshing=MeshingParams(
                defaults=MeshingDefaults(
                    surface_edge_growth_rate=1.07,
                    curvature_resolution_angle=10 * u.deg,
                    surface_max_edge_length=15 * u.cm,
                    edge_split_layers=0,
                ),
                refinements=[
                    SurfaceEdgeRefinement(
                        entities=[my_geometry["body01_edge001"], my_geometry["body01_edge002"]],
                        method=HeightBasedRefinement(value=3e-2 * u.cm),
                    ),
                    SurfaceEdgeRefinement(
                        entities=[my_geometry["body01_edge003"], my_geometry["body01_edge004"]],
                        method=ProjectAnisoSpacing(),
                    ),
                ],
            ),
        )
    return param


@pytest.fixture()
def get_airplane_geometry():
    return TempGeometry("geometry.egads")


@pytest.fixture()
def get_rotor_geometry():
    return TempGeometry("rotor.csm")


@pytest.fixture()
def get_snappy_geometry():
    return TempGeometry("tester.stl")


@pytest.fixture()
def airplane_surface_mesh():
    my_geometry = TempGeometry("geometry.egads")
    from numpy import pi

    with SI_unit_system:
        param = SimulationParams(
            private_attribute_asset_cache=AssetCache(
                project_entity_info=my_geometry._get_entity_info()
            ),
            meshing=MeshingParams(
                defaults=MeshingDefaults(
                    surface_edge_growth_rate=1.2,
                    surface_max_edge_length=100 * u.cm,
                    curvature_resolution_angle=pi / 12 * u.rad,
                    edge_split_layers=0,
                ),
                refinements=[
                    SurfaceRefinement(
                        entities=[
                            my_geometry["body01_face001"],
                            my_geometry["body01_face002"],
                            my_geometry["body01_face003"],
                            my_geometry["body01_face004"],
                        ],
                        max_edge_length=1.5 * u.m,
                    ),
                    SurfaceRefinement(
                        entities=[
                            my_geometry["body01_face005"],
                            my_geometry["body01_face006"],
                            my_geometry["body01_face007"],
                        ],
                        max_edge_length=700 * u.mm,
                    ),
                    SurfaceRefinement(
                        entities=[my_geometry["body01_face008"], my_geometry["body01_face009"]],
                        max_edge_length=0.5 * u.m,
                    ),
                    SurfaceRefinement(
                        entities=[my_geometry["body01_face010"]],
                        max_edge_length=50 * u.cm,
                    ),
                ],
            ),
        )
    return param


@pytest.fixture()
def rotor_surface_mesh():
    rotor_geometry = TempGeometry("rotor.csm")
    with imperial_unit_system:
        param = SimulationParams(
            private_attribute_asset_cache=AssetCache(
                project_entity_info=rotor_geometry._get_entity_info()
            ),
            meshing=MeshingParams(
                defaults=MeshingDefaults(
                    surface_edge_growth_rate=1.2,
                    surface_max_edge_length=10,
                    curvature_resolution_angle=15 * u.deg,
                    edge_split_layers=0,
                ),
                refinements=[
                    SurfaceRefinement(
                        entities=[rotor_geometry["body01_face003"]],
                        max_edge_length=0.1 * u.inch,
                    ),
                    SurfaceRefinement(
                        entities=[
                            rotor_geometry["body01_face001"],
                            rotor_geometry["body01_face002"],
                        ],
                        max_edge_length=10 * u.inch,
                    ),
                    SurfaceEdgeRefinement(
                        entities=[rotor_geometry["body01_edge001"]],
                        method=AngleBasedRefinement(value=1 * u.degree),
                    ),
                    SurfaceEdgeRefinement(
                        entities=[
                            rotor_geometry["body01_edge002"],
                            rotor_geometry["body01_edge003"],
                        ],
                        method=HeightBasedRefinement(value=0.05 * u.inch),
                    ),
                    SurfaceEdgeRefinement(
                        entities=[
                            rotor_geometry["body01_edge004"],
                            rotor_geometry["body01_edge006"],
                        ],
                        method=ProjectAnisoSpacing(),
                    ),
                    SurfaceEdgeRefinement(
                        entities=[rotor_geometry["body01_edge005"]],
                        method=HeightBasedRefinement(value=0.01 * u.inch),
                    ),
                    SurfaceEdgeRefinement(
                        entities=[rotor_geometry["body01_edge007"]],
                        method=HeightBasedRefinement(value=0.01 * u.inch),
                    ),
                ],
            ),
        )
    return param


@pytest.fixture()
def snappy_all_defaults():
    test_geometry = TempGeometry("tester.stl", True)
    with SI_unit_system:
        surf_meshing_params = snappy.SurfaceMeshingParams(
            defaults=snappy.SurfaceMeshingDefaults(
                min_spacing=3 * u.mm, max_spacing=4 * u.mm, gap_resolution=1 * u.mm
            )
        )

        param = SimulationParams(
            private_attribute_asset_cache=AssetCache(
                project_entity_info=test_geometry._get_entity_info(), project_length_unit=1 * u.mm
            ),
            meshing=ModularMeshingWorkflow(
                surface_meshing=surf_meshing_params, zones=[AutomatedFarfield()]
            ),
        )
    return param


@pytest.fixture()
def snappy_basic_refinements():
    test_geometry = TempGeometry("tester.stl", True)
    with SI_unit_system:
        surf_meshing_params = snappy.SurfaceMeshingParams(
            defaults=snappy.SurfaceMeshingDefaults(
                min_spacing=3 * u.mm, max_spacing=4 * u.mm, gap_resolution=1 * u.mm
            ),
            base_spacing=3.5 * u.mm,
            refinements=[
                snappy.BodyRefinement(
                    gap_resolution=2 * u.mm,
                    min_spacing=5 * u.mm,
                    max_spacing=10 * u.mm,
                    bodies=[
                        test_geometry.snappy_bodies["body1"],
                        test_geometry.snappy_bodies["body3"],
                    ],
                ),
                snappy.BodyRefinement(
                    gap_resolution=0.5 * u.mm,
                    min_spacing=1 * u.mm,
                    max_spacing=2 * u.mm,
                    bodies=[test_geometry.snappy_bodies["body2"]],
                    proximity_spacing=0.2 * u.mm,
                ),
                snappy.RegionRefinement(
                    min_spacing=20 * u.mm,
                    max_spacing=40 * u.mm,
                    proximity_spacing=3 * u.mm,
                    regions=[
                        test_geometry.snappy_bodies["body0"]["patch0"],
                        test_geometry["body1::patch1"],
                    ],
                ),
                snappy.SurfaceEdgeRefinement(
                    spacing=4 * u.mm,
                    min_elem=3,
                    included_angle=120 * u.deg,
                    entities=test_geometry.snappy_bodies["body1"],
                    geometric_test_only=True,
                ),
                snappy.SurfaceEdgeRefinement(
                    spacing=[4 * u.mm],
                    distances=[5 * u.mm],
                    min_elem=3,
                    included_angle=120 * u.deg,
                    entities=[test_geometry.snappy_bodies["body0"]["patch0"]],
                ),
                snappy.SurfaceEdgeRefinement(
                    spacing=[3 * u.mm, 5e-3 * u.m],
                    distances=[1 * u.mm, 3e-3 * u.m],
                    min_len=6 * u.mm,
                    entities=[
                        test_geometry.snappy_bodies["*"]["patch1"],
                        test_geometry.snappy_bodies["body3"],
                    ],
                    retain_on_smoothing=False,
                ),
                UniformRefinement(
                    spacing=2 * u.mm,
                    entities=[
                        Box(name="box0", center=[0, 30, 60] * u.mm, size=[20, 30, 40] * u.mm),
                        Cylinder(
                            name="cyl0",
                            axis=[0, 0, 1],
                            center=[10, 20, 30] * u.mm,
                            height=60 * u.mm,
                            outer_radius=20 * u.mm,
                        ),
                    ],
                ),
                UniformRefinement(
                    spacing=8 * u.mm,
                    entities=[
                        Cylinder(
                            name="cyl1",
                            axis=[-0.26, 0.45, -0.43],
                            center=[10, 20, 30] * u.mm,
                            height=60 * u.mm,
                            outer_radius=34 * u.mm,
                        )
                    ],
                ),
            ],
            smooth_controls=snappy.SmoothControls(),
        )

        param = SimulationParams(
            private_attribute_asset_cache=AssetCache(
                project_entity_info=test_geometry._get_entity_info(), project_length_unit=1 * u.mm
            ),
            meshing=ModularMeshingWorkflow(
                surface_meshing=surf_meshing_params, zones=[AutomatedFarfield()]
            ),
        )
    return param


@pytest.fixture()
def snappy_coupled_refinements():
    test_geometry = TempGeometry("tester.stl", True)
    with SI_unit_system:
        surf_meshing_params = snappy.SurfaceMeshingParams(
            defaults=snappy.SurfaceMeshingDefaults(
                min_spacing=3 * u.mm, max_spacing=4 * u.mm, gap_resolution=1 * u.mm
            ),
            base_spacing=5 * u.mm,
            refinements=[],
            smooth_controls=snappy.SmoothControls(),
        )
        vol_meshing_params = VolumeMeshingParams(
            defaults=VolumeMeshingDefaults(
                boundary_layer_first_layer_thickness=1 * u.mm, boundary_layer_growth_rate=1.2
            ),
            refinements=[
                UniformRefinement(
                    spacing=2 * u.mm,
                    entities=[
                        Box(name="box0", center=[0, 30, 60] * u.mm, size=[20, 30, 40] * u.mm),
                        Cylinder(
                            name="cyl0",
                            axis=[0, 0, 1],
                            center=[10, 20, 30] * u.mm,
                            height=60 * u.mm,
                            outer_radius=20 * u.mm,
                        ),
                    ],
                    project_to_surface=True,
                ),
                UniformRefinement(
                    spacing=8 * u.mm,
                    entities=[
                        Cylinder(
                            name="cyl1",
                            axis=[-0.26, 0.45, -0.43],
                            center=[10, 20, 30] * u.mm,
                            height=60 * u.mm,
                            outer_radius=34 * u.mm,
                        )
                    ],
                    project_to_surface=False,
                ),
            ],
        )
        param = SimulationParams(
            private_attribute_asset_cache=AssetCache(
                project_entity_info=test_geometry._get_entity_info(), project_length_unit=1 * u.mm
            ),
            meshing=ModularMeshingWorkflow(
                surface_meshing=surf_meshing_params,
                volume_meshing=vol_meshing_params,
                zones=[
                    CustomZones(
                        entities=[SeedpointVolume(name="farfield", point_in_mesh=[0, 0, 0] * u.mm)]
                    )
                ],
            ),
        )
    return param


@pytest.fixture()
def snappy_refinements_multiple_regions():
    test_geometry = TempGeometry("tester.stl", True)
    with SI_unit_system:
        surf_meshing_params = snappy.SurfaceMeshingParams(
            defaults=snappy.SurfaceMeshingDefaults(
                min_spacing=2.999999992 * u.mm, max_spacing=4 * u.mm, gap_resolution=1 * u.mm
            ),
            base_spacing=3 * u.mm,
            refinements=[
                snappy.RegionRefinement(
                    min_spacing=20 * u.mm,
                    max_spacing=40 * u.mm,
                    proximity_spacing=3 * u.mm,
                    regions=[
                        test_geometry["body1::patch0"],
                        test_geometry.snappy_bodies["body1"]["patch1"],
                        test_geometry["body1::patch2"],
                    ],
                ),
                snappy.RegionRefinement(
                    min_spacing=10 * u.mm, max_spacing=40 * u.mm, regions=test_geometry["body0::*"]
                ),
                snappy.RegionRefinement(
                    min_spacing=5 * u.mm,
                    max_spacing=40 * u.mm,
                    regions=[
                        test_geometry.snappy_bodies["body2"]["*"],
                        test_geometry["body3::patch0"],
                    ],
                ),
                snappy.SurfaceEdgeRefinement(
                    spacing=4 * u.mm,
                    min_elem=3,
                    included_angle=120 * u.deg,
                    entities=[test_geometry["body0::patch0"], test_geometry["body0::patch1"]],
                    retain_on_smoothing=False,
                    geometric_test_only=True,
                ),
            ],
            smooth_controls=snappy.SmoothControls(),
        )

        param = SimulationParams(
            private_attribute_asset_cache=AssetCache(
                project_entity_info=test_geometry._get_entity_info(), project_length_unit=1 * u.mm
            ),
            meshing=ModularMeshingWorkflow(
                surface_meshing=surf_meshing_params, zones=[AutomatedFarfield()]
            ),
        )
    return param


@pytest.fixture()
def snappy_refinements_no_regions():
    test_geometry = TempGeometry("tester_no_naming.stl", True)
    with SI_unit_system:
        surf_meshing_params = snappy.SurfaceMeshingParams(
            defaults=snappy.SurfaceMeshingDefaults(
                min_spacing=3 * u.mm, max_spacing=4 * u.mm, gap_resolution=1 * u.mm
            ),
            refinements=[
                snappy.BodyRefinement(
                    gap_resolution=2 * u.mm,
                    min_spacing=5 * u.mm,
                    max_spacing=10 * u.mm,
                    bodies=[test_geometry.snappy_bodies["body01_face001"]],
                ),
                snappy.BodyRefinement(
                    gap_resolution=0.5 * u.mm,
                    min_spacing=1 * u.mm,
                    max_spacing=2 * u.mm,
                    bodies=[test_geometry.snappy_bodies["body01_face002"]],
                    proximity_spacing=0.2 * u.mm,
                ),
                snappy.SurfaceEdgeRefinement(
                    spacing=4 * u.mm,
                    min_elem=3,
                    included_angle=120 * u.deg,
                    entities=[test_geometry.snappy_bodies["body01_face003"]],
                ),
            ],
            smooth_controls=snappy.SmoothControls(),
        )

        param = SimulationParams(
            private_attribute_asset_cache=AssetCache(
                project_entity_info=test_geometry._get_entity_info(), project_length_unit=1 * u.mm
            ),
            meshing=ModularMeshingWorkflow(
                surface_meshing=surf_meshing_params,
                zones=[
                    CustomZones(
                        entities=[SeedpointVolume(name="fluid", point_in_mesh=[0, 0, 0] * u.m)]
                    )
                ],
            ),
        )
    return param


@pytest.fixture()
def snappy_settings():
    test_geometry = TempGeometry("tester.stl", True)
    with SI_unit_system:
        surf_meshing_params = snappy.SurfaceMeshingParams(
            defaults=snappy.SurfaceMeshingDefaults(
                min_spacing=3 * u.mm, max_spacing=4 * u.mm, gap_resolution=1 * u.mm
            ),
            quality_metrics=snappy.QualityMetrics(
                max_non_orthogonality=55 * u.deg,
                max_boundary_skewness=30 * u.deg,
                max_internal_skewness=70 * u.deg,
                max_concavity=20 * u.deg,
                min_pyramid_cell_volume=1e-2,
                min_tet_quality=0.15,
                min_face_area=2 * u.mm * u.mm,
                min_twist=0.3,
                min_determinant=0.5,
                min_volume_ratio=0.1,
                min_face_weight=0.3,
                min_triangle_twist=0.1,
                n_smooth_scale=6,
                error_reduction=0.4,
                zmetric_threshold=0.9,
                feature_edge_deduplication_tolerance=0.25,
                min_volume_collapse_ratio=0.5,
            ),
            snap_controls=snappy.SnapControls(
                n_smooth_patch=5,
                tolerance=4,
                n_solve_iter=20,
                n_relax_iter=2,
                n_feature_snap_iter=10,
                multi_region_feature_snap=False,
                strict_region_snap=True,
            ),
            castellated_mesh_controls=snappy.CastellatedMeshControls(
                resolve_feature_angle=10 * u.deg, n_cells_between_levels=3, min_refinement_cells=50
            ),
            smooth_controls=snappy.SmoothControls(lambda_factor=0.3, mu_factor=0.31, iterations=5),
        )

        param = SimulationParams(
            private_attribute_asset_cache=AssetCache(
                project_entity_info=test_geometry._get_entity_info(), project_length_unit=1 * u.mm
            ),
            meshing=ModularMeshingWorkflow(
                surface_meshing=surf_meshing_params,
                zones=[
                    CustomZones(
                        entities=[
                            SeedpointVolume(name="fluid", point_in_mesh=[0, 0, 0] * u.m),
                            SeedpointVolume(
                                name="solid", point_in_mesh=[0.001, 0.002, 0.003] * u.m
                            ),
                        ]
                    )
                ],
            ),
        )
    return param


@pytest.fixture()
def snappy_settings_off_position():
    test_geometry = TempGeometry("tester.stl", True)
    with SI_unit_system:
        surf_meshing_params = snappy.SurfaceMeshingParams(
            defaults=snappy.SurfaceMeshingDefaults(
                min_spacing=3 * u.mm, max_spacing=4 * u.mm, gap_resolution=1 * u.mm
            ),
            quality_metrics=snappy.QualityMetrics(
                max_non_ortho=False,
                max_boundary_skewness=False,
                max_internal_skewness=False,
                max_concave=False,
                min_vol=False,
                min_tet_quality=False,
                min_area=False,
                min_twist=False,
                min_determinant=False,
                min_vol_ratio=False,
                min_face_weight=False,
                min_triangle_twist=False,
                zmetric_threshold=False,
                feature_edge_deduplication_tolerance=False,
                min_vol_collapse_ratio=False,
            ),
            snap_controls=snappy.SnapControls(
                n_smooth_patch=5,
                tolerance=4,
                n_solve_iter=20,
                n_relax_iter=2,
                n_feature_snap_iter=10,
                multi_region_feature_snap=False,
                strict_region_snap=True,
            ),
            castellated_mesh_controls=snappy.CastellatedMeshControls(
                resolve_feature_angle=10 * u.deg, n_cells_between_levels=3, min_refinement_cells=50
            ),
            smooth_controls=False,
        )

        param = SimulationParams(
            private_attribute_asset_cache=AssetCache(
                project_entity_info=test_geometry._get_entity_info(), project_length_unit=1 * u.mm
            ),
            meshing=ModularMeshingWorkflow(
                surface_meshing=surf_meshing_params,
                zones=[
                    CustomZones(
                        entities=[
                            SeedpointVolume(name="fluid", point_in_mesh=[0, 0, 0] * u.m),
                            SeedpointVolume(
                                name="solid", point_in_mesh=[0.001, 0.002, 0.003] * u.m
                            ),
                        ]
                    )
                ],
            ),
        )
    return param


def _translate_and_compare(param, mesh_unit, ref_json_file: str, atol=1e-15):
    param, err, warnings = validate_params_with_context(param, "Geometry", "SurfaceMesh")
    assert err is None, f"Validation error: {err}"
    assert warnings == [], f"Unexpected warnings during validation: {warnings}"
    translated = get_surface_meshing_json(param, mesh_unit=mesh_unit)
    with open(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "ref/surface_meshing", ref_json_file
        )
    ) as fh:
        ref_dict = json.load(fh)

    # It is important that the list in the configs are sorted beforehand
    # as the hash values for reuse resource rely on that

    # check if everything is serializable
    json.dumps(translated)
    assert compare_values(ref_dict, translated, atol=atol)


def test_om6wing_tutorial(
    get_om6wing_geometry, om6wing_tutorial_global_plus_local_override, om6wing_tutorial_global_only
):
    _translate_and_compare(
        om6wing_tutorial_global_plus_local_override,
        get_om6wing_geometry.mesh_unit,
        "om6wing_tutorial_ignore_global.json",
    )
    _translate_and_compare(
        om6wing_tutorial_global_only,
        get_om6wing_geometry.mesh_unit,
        "om6wing_tutorial_use_global.json",
    )


def test_om6wing_tutorial_aspect_ratio(get_om6wing_geometry, om6wing_tutorial_aspect_ratio):
    params = om6wing_tutorial_aspect_ratio
    _translate_and_compare(
        om6wing_tutorial_aspect_ratio,
        get_om6wing_geometry.mesh_unit,
        "om6wing_tutorial_aspect_ratio.json",
    )


def test_airplane_surface_mesh(get_airplane_geometry, airplane_surface_mesh):
    _translate_and_compare(
        airplane_surface_mesh,
        get_airplane_geometry.mesh_unit,
        "airplane.json",
    )


def test_rotor_surface_mesh(get_rotor_geometry, rotor_surface_mesh):
    _translate_and_compare(
        rotor_surface_mesh,
        get_rotor_geometry.mesh_unit,
        "rotor.json",
    )


def test_snappy_default(get_snappy_geometry, snappy_all_defaults):
    _translate_and_compare(
        snappy_all_defaults, get_snappy_geometry.mesh_unit, "default_snappy.json", atol=1e-6
    )


def test_snappy_basic(get_snappy_geometry, snappy_basic_refinements):
    _translate_and_compare(
        snappy_basic_refinements,
        get_snappy_geometry.mesh_unit,
        "snappy_basic_refinements.json",
        atol=1e-6,
    )


def test_snappy_coupled(get_snappy_geometry, snappy_coupled_refinements):
    _translate_and_compare(
        snappy_coupled_refinements,
        get_snappy_geometry.mesh_unit,
        "snappy_coupled_refinements.json",
        atol=1e-6,
    )


def test_snappy_multiple_regions(get_snappy_geometry, snappy_refinements_multiple_regions):
    _translate_and_compare(
        snappy_refinements_multiple_regions,
        get_snappy_geometry.mesh_unit,
        "snappy_refinements_multiple_regions.json",
        atol=1e-6,
    )


def test_snappy_settings(get_snappy_geometry, snappy_settings):
    _translate_and_compare(
        snappy_settings, get_snappy_geometry.mesh_unit, "snappy_settings.json", atol=1e-6
    )


def test_snappy_settings_off_position(get_snappy_geometry, snappy_settings_off_position):
    _translate_and_compare(
        snappy_settings_off_position, get_snappy_geometry.mesh_unit, "snappy_settings_off_pos.json"
    )


def test_snappy_no_refinements(get_snappy_geometry, snappy_refinements_no_regions):
    _translate_and_compare(
        snappy_refinements_no_regions,
        get_snappy_geometry.mesh_unit,
        "snappy_no_regions.json",
        atol=1e-6,
    )


def test_gai_surface_mesher_refinements():
    geometry = Geometry.from_local_storage(
        geometry_id="geo-e5c01a98-2180-449e-b255-d60162854a83",
        local_storage_path=os.path.join(
            os.path.dirname(__file__), "data", "gai_geometry_entity_info"
        ),
        meta_data=GeometryMeta(
            **local_metadata_builder(
                id="geo-e5c01a98-2180-449e-b255-d60162854a83",
                name="aaa",
                cloud_path_prefix="aaa",
                status="processed",
            )
        ),
    )
    geometry.group_faces_by_tag("faceId")
    geometry.group_edges_by_tag("edgeId")
    geometry.group_bodies_by_tag("groupByFile")

    with create_draft(new_run_from=geometry) as draft:
        with SI_unit_system:
            # Coordinate systems replace body-group transformations.
            coordinate_system = CoordinateSystem(
                name="body_group_cs",
                reference_point=[0, 0, 0] * u.m,
                axis_of_rotation=(0, 0, 1),
                angle_of_rotation=0 * u.deg,
                scale=(1.0, 1.0, 1.0),
                translation=[10, 20, 30] * u.m,
            )

            draft.coordinate_systems.assign(
                entities=[
                    draft.body_groups["cube-holes.egads"],
                    draft.body_groups["cylinder.stl"],
                ],
                coordinate_system=coordinate_system,
            )

            farfield = AutomatedFarfield(domain_type="half_body_positive_y")
            params = SimulationParams(
                meshing=MeshingParams(
                    defaults=MeshingDefaults(
                        geometry_accuracy=0.05 * u.m,  # GAI only setting
                        surface_max_edge_length=0.2,
                        boundary_layer_first_layer_thickness=0.01,
                        surface_max_aspect_ratio=0.01,
                        surface_max_adaptation_iterations=19,
                    ),
                    volume_zones=[farfield],
                    refinements=[
                        SurfaceRefinement(
                            name="renamed_surface",
                            max_edge_length=0.1,
                            faces=[geometry["*"]],
                            curvature_resolution_angle=5.0 * u.deg,  # GAI only setting
                            resolve_face_boundaries=True,  # GAI only setting
                        ),
                        GeometryRefinement(
                            name="Local_override",
                            geometry_accuracy=0.05 * u.m,
                            min_passage_size=0.1 * u.m,
                            faces=[geometry["body00001_face00001"]],
                        ),
                    ],
                ),
                operating_condition=AerospaceCondition(
                    velocity_magnitude=10 * u.m / u.s,
                ),
            )

            params = set_up_params_for_uploading(
                root_asset=geometry,
                length_unit=1 * u.m,
                params=params,
                use_beta_mesher=True,
                use_geometry_AI=True,
            )

    _translate_and_compare(
        params,
        1 * u.m,
        "gai_surface_mesher.json",
    )


def test_gai_translator_hashing_ignores_id():
    """Test that hash calculation ignores private_attribute_id fields."""

    hashes = []
    json_dicts = []

    # Create the same configuration twice in a loop
    # Each iteration generates different UUIDs for entities with private_attribute_id
    for i in range(2):
        with SI_unit_system:
            # Cylinder has private_attribute_id with generate_uuid factory
            cylinder = Cylinder(
                name="test_cylinder",
                center=[0, 0, 0] * u.m,
                axis=[0, 0, 1],
                height=10 * u.m,
                outer_radius=5 * u.m,
            )

            params = SimulationParams(
                meshing=MeshingParams(
                    defaults=MeshingDefaults(
                        surface_max_edge_length=0.2,
                    ),
                    refinements=[
                        UniformRefinement(
                            name="cylinder_refinement", entities=[cylinder], spacing=0.1 * u.m
                        )
                    ],
                )
            )

        # Export to dict
        params_dict = params.model_dump(mode="json")
        json_dicts.append(params_dict)

        # Calculate hash
        hash_value = SimulationParams._calculate_hash(params_dict)
        hashes.append(hash_value)

    # Verify JSONs are different (due to different UUIDs)
    json_str_0 = json.dumps(json_dicts[0], sort_keys=True)
    json_str_1 = json.dumps(json_dicts[1], sort_keys=True)
    assert (
        json_str_0 != json_str_1
    ), "JSON strings should differ due to different private_attribute_id (UUID) values"

    # Verify hashes are identical (UUID ignored in hash calculation)
    assert (
        hashes[0] == hashes[1]
    ), f"Hashes should be identical despite different UUIDs:\n  Hash 1: {hashes[0]}\n  Hash 2: {hashes[1]}"


def test_gai_analytic_wind_tunnel_farfield():
    with SI_unit_system:
        wind_tunnel = WindTunnelFarfield(
            width=10,
            height=10,
            inlet_x_position=-5,
            outlet_x_position=15,
            floor_z_position=0,
            floor_type=WheelBelts(
                central_belt_x_range=(-1, 6),
                central_belt_width=1.2,
                front_wheel_belt_x_range=(-0.3, 0.5),
                front_wheel_belt_y_range=(0.7, 1.2),
                rear_wheel_belt_x_range=(2.6, 3.8),
                rear_wheel_belt_y_range=(0.7, 1.2),
            ),
        )
        meshing_params = MeshingParams(
            defaults=MeshingDefaults(
                surface_max_aspect_ratio=10,
                curvature_resolution_angle=15 * u.deg,
                geometry_accuracy=1e-2,
                boundary_layer_first_layer_thickness=1e-4,
                boundary_layer_growth_rate=1.2,
                planar_face_tolerance=1e-3,
                surface_max_edge_length=0.2,
            ),
            volume_zones=[wind_tunnel],
        )
        with open(
            os.path.join(
                os.path.dirname(__file__), "data", "gai_windtunnel_farfield_info", "simulation.json"
            ),
            "r",
        ) as fh:
            asset_cache = AssetCache.model_validate(
                json.load(fh).pop("private_attribute_asset_cache")
            )

        params = SimulationParams(
            meshing=meshing_params,
            operating_condition=AerospaceCondition(velocity_magnitude=30 * u.m / u.s),
            private_attribute_asset_cache=asset_cache,
        )

    _translate_and_compare(
        params,
        1 * u.m,
        "gai_windtunnel.json",
    )


def test_sliding_interface_tolerance_gai():
    """Test that sliding_interface_tolerance is included in GAI filtered JSON."""
    geometry = Geometry.from_local_storage(
        geometry_id="geo-e5c01a98-2180-449e-b255-d60162854a83",
        local_storage_path=os.path.join(
            os.path.dirname(__file__), "data", "gai_geometry_entity_info"
        ),
        meta_data=GeometryMeta(
            **local_metadata_builder(
                id="geo-e5c01a98-2180-449e-b255-d60162854a83",
                name="aaa",
                cloud_path_prefix="aaa",
                status="processed",
            )
        ),
    )
    geometry.group_faces_by_tag("faceId")
    geometry.group_edges_by_tag("edgeId")
    geometry.group_bodies_by_tag("groupByFile")

    with open(
        os.path.join(
            os.path.dirname(__file__), "data", "gai_geometry_entity_info", "simulation.json"
        ),
        "r",
    ) as fh:
        asset_cache = AssetCache.model_validate(json.load(fh).pop("private_attribute_asset_cache"))

    with SI_unit_system:
        farfield = AutomatedFarfield(domain_type="half_body_positive_y")
        rotating_volume = RotationVolume(
            name="rotating_volume",
            spacing_axial=0.1 * u.m,
            spacing_circumferential=0.1 * u.m,
            spacing_radial=0.1 * u.m,
            entities=[
                Cylinder(
                    name="cylinder",
                    center=[0, 0, 0] * u.m,
                    axis=[0, 0, 1],
                    height=10 * u.m,
                    outer_radius=5 * u.m,
                    private_attribute_id="b8d08e11-e837-4cc7-95b3-f92e05e71a65",
                )
            ],
        )
        params = SimulationParams(
            meshing=MeshingParams(
                defaults=MeshingDefaults(
                    geometry_accuracy=0.05 * u.m,
                    surface_max_edge_length=0.2,
                    sliding_interface_tolerance=3e-3,
                ),
                volume_zones=[farfield, rotating_volume],
            ),
            private_attribute_asset_cache=asset_cache,
        )

    _, err, warnings = validate_params_with_context(params, "Geometry", "SurfaceMesh")
    assert err is None, f"Validation error: {err}"
    assert warnings == [], f"Unexpected warnings during validation: {warnings}"
    translated = get_surface_meshing_json(params, mesh_unit=1 * u.m)

    # Verify sliding_interface_tolerance is in the translated JSON
    assert "meshing" in translated
    assert "defaults" in translated["meshing"]
    assert "sliding_interface_tolerance" in translated["meshing"]["defaults"]
    assert translated["meshing"]["defaults"]["sliding_interface_tolerance"] == 3e-3
    assert "volume_zones" in translated["meshing"]
    assert translated["meshing"]["volume_zones"][1] == {
        "entities": {
            "stored_entities": [
                {
                    "axis": [0.0, 0.0, 1.0],
                    "center": {"units": "1.0*m", "value": [0.0, 0.0, 0.0]},
                    "height": {"units": "1.0*m", "value": 10.0},
                    "inner_radius": {"units": "1.0*m", "value": 0.0},
                    "name": "cylinder",
                    "outer_radius": {"units": "1.0*m", "value": 5.0},
                    "private_attribute_entity_type_name": "Cylinder",
                    "private_attribute_id": "b8d08e11-e837-4cc7-95b3-f92e05e71a65",
                    "private_attribute_zone_boundary_names": {"items": []},
                }
            ]
        },
        "name": "rotating_volume",
        "spacing_axial": {"units": "1.0*m", "value": 0.1},
        "spacing_circumferential": {"units": "1.0*m", "value": 0.1},
        "spacing_radial": {"units": "1.0*m", "value": 0.1},
        "type": "RotationVolume",
    }


def test_gai_mirror_status_translation():
    """Test that mirror_status is properly translated into GAI surface meshing JSON.

    This test verifies:
    1. mirror_status is included in the translated JSON
    2. All dimensional values (e.g., MirrorPlane.center) are serialized with proper units
    """
    from flow360.component.simulation.draft_context.mirror import (
        MirrorPlane,
        MirrorStatus,
    )

    geometry = Geometry.from_local_storage(
        geometry_id="geo-e5c01a98-2180-449e-b255-d60162854a83",
        local_storage_path=os.path.join(
            os.path.dirname(__file__), "data", "gai_geometry_entity_info"
        ),
        meta_data=GeometryMeta(
            **local_metadata_builder(
                id="geo-e5c01a98-2180-449e-b255-d60162854a83",
                name="mirror_test",
                cloud_path_prefix="mirror_test",
                status="processed",
            )
        ),
    )
    geometry.group_faces_by_tag("faceId")
    geometry.group_edges_by_tag("edgeId")
    geometry.group_bodies_by_tag("groupByFile")

    with open(
        os.path.join(
            os.path.dirname(__file__), "data", "gai_geometry_entity_info", "simulation.json"
        ),
        "r",
    ) as fh:
        asset_cache_dict = json.load(fh).pop("private_attribute_asset_cache")

    # Create mirror status with dimensional values
    # Use fixed IDs to ensure reproducibility
    mirror_plane = MirrorPlane(
        name="y_symmetry_plane",
        normal=(0, 1, 0),
        center=[0.5, 0, 0.25] * u.m,
        private_attribute_id="mirror-plane-test-id-001",
    )

    mirrored_body_group = MirroredGeometryBodyGroup(
        name="cube-holes.egads_<mirror>",
        geometry_body_group_id="cube-holes.egads",
        mirror_plane_id="mirror-plane-test-id-001",
        private_attribute_id="mirrored-body-group-test-id-001",
    )

    mirrored_surface = MirroredSurface(
        name="body00001_face00001_<mirror>",
        surface_id="body00001_face00001",
        mirror_plane_id="mirror-plane-test-id-001",
        private_attribute_id="mirrored-surface-test-id-001",
    )

    mirror_status = MirrorStatus(
        mirror_planes=[mirror_plane],
        mirrored_geometry_body_groups=[mirrored_body_group],
        mirrored_surfaces=[mirrored_surface],
    )

    # Add mirror_status to asset_cache
    asset_cache_dict["mirror_status"] = mirror_status.model_dump(mode="json")
    asset_cache = AssetCache.model_validate(asset_cache_dict)

    with SI_unit_system:
        farfield = AutomatedFarfield(domain_type="half_body_positive_y")
        params = SimulationParams(
            meshing=MeshingParams(
                defaults=MeshingDefaults(
                    geometry_accuracy=0.05 * u.m,
                    surface_max_edge_length=0.2,
                ),
                volume_zones=[farfield],
            ),
            private_attribute_asset_cache=asset_cache,
        )

    _, err, warnings = validate_params_with_context(params, "Geometry", "SurfaceMesh")
    assert err is None, f"Validation error: {err}"
    assert warnings == [], f"Unexpected warnings during validation: {warnings}"

    translated = get_surface_meshing_json(params, mesh_unit=1 * u.m)

    # Verify the JSON is serializable
    json.dumps(translated)

    # Assert mirror_status is in the translated JSON
    assert "private_attribute_asset_cache" in translated
    assert "mirror_status" in translated["private_attribute_asset_cache"]

    mirror_status_json = translated["private_attribute_asset_cache"]["mirror_status"]

    # Assert mirror_planes is present and has expected structure
    assert "mirror_planes" in mirror_status_json
    assert len(mirror_status_json["mirror_planes"]) == 1

    plane_json = mirror_status_json["mirror_planes"][0]

    # Verify plane name and normal
    assert plane_json["name"] == "y_symmetry_plane"
    assert plane_json["normal"] == [0.0, 1.0, 0.0]

    # KEY ASSERTION: Verify dimensional value (center) has proper units format
    assert "center" in plane_json
    center = plane_json["center"]
    assert isinstance(center, dict), "center should be a dict with value and units"
    assert "value" in center, "center must have 'value' key"
    assert "units" in center, "center must have 'units' key"

    # Verify the values are correct (converted to Flow360 units - meters)
    assert center["value"] == [0.5, 0.0, 0.25], f"Expected [0.5, 0.0, 0.25], got {center['value']}"
    # Units should be in meter format (could be "m" or "1.0*m" depending on serialization)
    assert "m" in center["units"], f"Expected meter units, got {center['units']}"

    # Assert mirrored entities are present
    assert "mirrored_geometry_body_groups" in mirror_status_json
    assert len(mirror_status_json["mirrored_geometry_body_groups"]) == 1
    assert (
        mirror_status_json["mirrored_geometry_body_groups"][0]["name"]
        == "cube-holes.egads_<mirror>"
    )

    assert "mirrored_surfaces" in mirror_status_json
    assert len(mirror_status_json["mirrored_surfaces"]) == 1
    assert mirror_status_json["mirrored_surfaces"][0]["name"] == "body00001_face00001_<mirror>"


def test_gai_mirror_status_translation_idempotency():
    """Test that mirror_status translation is idempotent.

    This test verifies that translating the same configuration twice produces
    identical JSON output, even though the entities have different UUIDs each time.
    The private_attribute_id fields should be stripped from the output.
    """
    from flow360.component.simulation.draft_context.mirror import (
        MirrorPlane,
        MirrorStatus,
    )

    translated_jsons = []

    for _ in range(2):
        geometry = Geometry.from_local_storage(
            geometry_id="geo-e5c01a98-2180-449e-b255-d60162854a83",
            local_storage_path=os.path.join(
                os.path.dirname(__file__), "data", "gai_geometry_entity_info"
            ),
            meta_data=GeometryMeta(
                **local_metadata_builder(
                    id="geo-e5c01a98-2180-449e-b255-d60162854a83",
                    name="idempotency_test",
                    cloud_path_prefix="idempotency_test",
                    status="processed",
                )
            ),
        )
        geometry.group_faces_by_tag("faceId")
        geometry.group_edges_by_tag("edgeId")
        geometry.group_bodies_by_tag("groupByFile")

        with open(
            os.path.join(
                os.path.dirname(__file__), "data", "gai_geometry_entity_info", "simulation.json"
            ),
            "r",
        ) as fh:
            asset_cache_dict = json.load(fh).pop("private_attribute_asset_cache")

        # Create mirror status - each iteration generates new UUIDs
        # because we don't specify private_attribute_id
        mirror_plane = MirrorPlane(
            name="test_mirror_plane",
            normal=(1, 0, 0),
            center=[0, 0, 0] * u.m,
        )

        mirrored_body_group = MirroredGeometryBodyGroup(
            name="cube-holes.egads_<mirror>",
            geometry_body_group_id="cube-holes.egads",
            mirror_plane_id=mirror_plane.private_attribute_id,
        )

        mirrored_surface = MirroredSurface(
            name="body00001_face00001_<mirror>",
            surface_id="body00001_face00001",
            mirror_plane_id=mirror_plane.private_attribute_id,
        )

        mirror_status = MirrorStatus(
            mirror_planes=[mirror_plane],
            mirrored_geometry_body_groups=[mirrored_body_group],
            mirrored_surfaces=[mirrored_surface],
        )

        asset_cache_dict["mirror_status"] = mirror_status.model_dump(mode="json")
        asset_cache = AssetCache.model_validate(asset_cache_dict)

        with SI_unit_system:
            farfield = AutomatedFarfield(domain_type="half_body_positive_y")
            params = SimulationParams(
                meshing=MeshingParams(
                    defaults=MeshingDefaults(
                        geometry_accuracy=0.05 * u.m,
                        surface_max_edge_length=0.2,
                    ),
                    volume_zones=[farfield],
                ),
                private_attribute_asset_cache=asset_cache,
            )

        _, err, warnings = validate_params_with_context(params, "Geometry", "SurfaceMesh")
        assert err is None, f"Validation error: {err}"
        assert warnings == [], f"Unexpected warnings during validation: {warnings}"

        translated = get_surface_meshing_json(params, mesh_unit=1 * u.m)
        translated_jsons.append(translated)

    # Convert both to JSON strings for comparison
    json_str_0 = json.dumps(translated_jsons[0], sort_keys=True)
    json_str_1 = json.dumps(translated_jsons[1], sort_keys=True)

    # The two translations should be identical (idempotent)
    assert json_str_0 == json_str_1, (
        "Translated JSONs should be identical (idempotent) despite different UUIDs.\n"
        f"First translation:\n{json.dumps(translated_jsons[0], indent=2)}\n"
        f"Second translation:\n{json.dumps(translated_jsons[1], indent=2)}"
    )

    # Verify that no UUID-style IDs remain in mirror_status
    # (deterministic IDs like "mirror-{name}" are expected and should be present)
    mirror_status_json = translated_jsons[0]["private_attribute_asset_cache"]["mirror_status"]

    def check_no_uuids(obj, path=""):
        """Check that no UUID-style strings (with hyphens and hex chars) are present."""
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, str) and len(v) == 36 and v.count("-") == 4:
                    # Looks like a UUID (e.g., "e9f942f7-3152-4f4e-a68c-32280465cb77")
                    assert False, f"UUID-style string '{v}' found at {path}.{k}"
                check_no_uuids(v, f"{path}.{k}")
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                check_no_uuids(item, f"{path}[{i}]")

    check_no_uuids(mirror_status_json, "mirror_status")

    # Verify that mirroring relationships are preserved with deterministic IDs
    mirror_planes = mirror_status_json.get("mirror_planes", [])
    mirrored_body_groups = mirror_status_json.get("mirrored_geometry_body_groups", [])
    mirrored_surfaces = mirror_status_json.get("mirrored_surfaces", [])

    # Check that mirror planes have deterministic IDs
    assert len(mirror_planes) == 1, "Expected one mirror plane"
    plane = mirror_planes[0]
    expected_plane_id = f"mirror-{plane['name']}"
    assert (
        plane["private_attribute_id"] == expected_plane_id
    ), f"Mirror plane should have deterministic ID 'mirror-{{name}}', got {plane['private_attribute_id']}"

    # Check that mirrored body groups reference the correct mirror plane
    assert len(mirrored_body_groups) == 1, "Expected one mirrored body group"
    body_group = mirrored_body_groups[0]
    assert (
        "private_attribute_id" not in body_group
    ), "Mirrored body group should not have private_attribute_id"
    assert (
        body_group["mirror_plane_id"] == expected_plane_id
    ), f"Mirrored body group should reference mirror plane '{expected_plane_id}'"

    # Check that mirrored surfaces reference the correct mirror plane
    assert len(mirrored_surfaces) == 1, "Expected one mirrored surface"
    surface = mirrored_surfaces[0]
    assert (
        "private_attribute_id" not in surface
    ), "Mirrored surface should not have private_attribute_id"
    assert (
        surface["mirror_plane_id"] == expected_plane_id
    ), f"Mirrored surface should reference mirror plane '{expected_plane_id}'"


def test_gai_no_stationary_enclosed_entities():
    param_dict = {
        "private_attribute_asset_cache": {
            "use_inhouse_mesher": True,
            "use_geometry_AI": True,
            "project_entity_info": {"type_name": "GeometryEntityInfo"},
        },
    }

    with SI_unit_system:
        # Create entities
        cylinder = Cylinder(
            name="intersectingCylinder",
            center=(0, 0, 0) * u.m,
            outer_radius=1 * u.m,
            height=2 * u.m,
            axis=(0, 0, 1),
        )
        sphere_surface = Surface(name="sphere.lb8.ugrid")

        # Create RotationVolume with enclosed_entities
        rotation_volume = RotationVolume(
            name="RotationVolume",
            spacing_axial=0.5 * u.m,
            spacing_circumferential=0.3 * u.m,
            spacing_radial=1.5 * u.m,
            entities=[cylinder],
            enclosed_entities=[sphere_surface],
            stationary_enclosed_entities=[sphere_surface],
        )

        # Create meshing params
        meshing = MeshingParams(
            defaults=MeshingDefaults(
                surface_max_edge_length=0.1,
                geometry_accuracy=0.01,
            ),
            volume_zones=[
                AutomatedFarfield(name="Farfield"),
                rotation_volume,
            ],
        )

        params = SimulationParams(
            meshing=meshing,
            private_attribute_asset_cache=AssetCache.model_validate(
                param_dict["private_attribute_asset_cache"]
            ),
        )

    translated_json = get_surface_meshing_json(params, 1 * u.m)
    assert "volume_zones" in translated_json["meshing"]
    volume_zones = translated_json["meshing"]["volume_zones"]
    for zone in volume_zones:
        if zone["type"] in ("RotationVolume", "RotationCylinder"):
            assert "stationary_enclosed_entities" not in zone
