import json
import os

import pytest

import flow360.component.simulation.units as u
from flow360.component.geometry import Geometry, GeometryMeta
from flow360.component.project_utils import validate_params_with_context
from flow360.component.resource_base import local_metadata_builder
from flow360.component.simulation.entity_info import GeometryEntityInfo
from flow360.component.simulation.entity_operation import Transformation
from flow360.component.simulation.framework.param_utils import AssetCache
from flow360.component.simulation.framework.updater_utils import compare_values
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
    BetaVolumeMeshingDefaults,
    SnappyCastellatedMeshControls,
    SnappyQualityMetrics,
    SnappySmoothControls,
    SnappySnapControls,
    SnappySurfaceMeshingDefaults,
)
from flow360.component.simulation.meshing_param.params import (
    BetaVolumeMeshingParams,
    MeshingDefaults,
    MeshingParams,
    ModularMeshingWorkflow,
    SnappySurfaceMeshingParams,
)
from flow360.component.simulation.meshing_param.surface_mesh_refinements import (
    SnappyBodyRefinement,
    SnappyRegionRefinement,
    SnappySurfaceEdgeRefinement,
)
from flow360.component.simulation.meshing_param.volume_params import (
    AutomatedFarfield,
    UniformRefinement,
)
from flow360.component.simulation.operating_condition.operating_condition import (
    AerospaceCondition,
)
from flow360.component.simulation.primitives import (
    Box,
    Cylinder,
    Edge,
    SeedpointZone,
    SnappyBody,
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
            self.internal_registry = self._get_entity_info()._group_faces_by_snappy_format()
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
    rotor_geopmetry = TempGeometry("rotor.csm")
    with imperial_unit_system:
        param = SimulationParams(
            private_attribute_asset_cache=AssetCache(
                project_entity_info=rotor_geopmetry._get_entity_info()
            ),
            meshing=MeshingParams(
                defaults=MeshingDefaults(
                    surface_edge_growth_rate=1.2,
                    surface_max_edge_length=10,
                    curvature_resolution_angle=15 * u.deg,
                ),
                refinements=[
                    SurfaceRefinement(
                        entities=[rotor_geopmetry["body01_face003"]],
                        max_edge_length=0.1 * u.inch,
                    ),
                    SurfaceRefinement(
                        entities=[
                            rotor_geopmetry["body01_face001"],
                            rotor_geopmetry["body01_face002"],
                        ],
                        max_edge_length=10 * u.inch,
                    ),
                    SurfaceEdgeRefinement(
                        entities=[rotor_geopmetry["body01_edge001"]],
                        method=AngleBasedRefinement(value=1 * u.degree),
                    ),
                    SurfaceEdgeRefinement(
                        entities=[
                            rotor_geopmetry["body01_edge002"],
                            rotor_geopmetry["body01_edge003"],
                        ],
                        method=HeightBasedRefinement(value=0.05 * u.inch),
                    ),
                    SurfaceEdgeRefinement(
                        entities=[
                            rotor_geopmetry["body01_edge004"],
                            rotor_geopmetry["body01_edge006"],
                        ],
                        method=ProjectAnisoSpacing(),
                    ),
                    SurfaceEdgeRefinement(
                        entities=[rotor_geopmetry["body01_edge005"]],
                        method=HeightBasedRefinement(value=0.01 * u.inch),
                    ),
                    SurfaceEdgeRefinement(
                        entities=[rotor_geopmetry["body01_edge007"]],
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
        surf_meshing_params = SnappySurfaceMeshingParams(
            defaults=SnappySurfaceMeshingDefaults(
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
        surf_meshing_params = SnappySurfaceMeshingParams(
            defaults=SnappySurfaceMeshingDefaults(
                min_spacing=3 * u.mm, max_spacing=4 * u.mm, gap_resolution=1 * u.mm
            ),
            base_spacing=3.5 * u.mm,
            refinements=[
                SnappyBodyRefinement(
                    gap_resolution=2 * u.mm,
                    min_spacing=5 * u.mm,
                    max_spacing=10 * u.mm,
                    bodies=[test_geometry["body1"], test_geometry["body3"]],
                ),
                SnappyBodyRefinement(
                    gap_resolution=0.5 * u.mm,
                    min_spacing=1 * u.mm,
                    max_spacing=2 * u.mm,
                    bodies=[test_geometry["body2"]],
                    proximity_spacing=0.2 * u.mm,
                ),
                SnappyRegionRefinement(
                    min_spacing=20 * u.mm,
                    max_spacing=40 * u.mm,
                    proximity_spacing=3 * u.mm,
                    regions=[
                        test_geometry["body0"]["patch0"],
                        test_geometry["body1"]["patch1"],
                    ],
                ),
                SnappySurfaceEdgeRefinement(
                    spacing=4 * u.mm,
                    min_elem=3,
                    included_angle=120 * u.deg,
                    bodies=test_geometry["body1"],
                ),
                SnappySurfaceEdgeRefinement(
                    spacing=4 * u.mm,
                    min_elem=3,
                    included_angle=120 * u.deg,
                    regions=[test_geometry["body0"]["patch0"]],
                ),
                SnappySurfaceEdgeRefinement(
                    spacing=[3 * u.mm, 5 * u.mm],
                    distances=[1 * u.mm, 3 * u.mm],
                    min_len=6 * u.mm,
                    regions=[test_geometry["*"]["patch1"]],
                    bodies=[test_geometry["body3"]],
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
            smooth_controls=SnappySmoothControls(),
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
        surf_meshing_params = SnappySurfaceMeshingParams(
            defaults=SnappySurfaceMeshingDefaults(
                min_spacing=3 * u.mm, max_spacing=4 * u.mm, gap_resolution=1 * u.mm
            ),
            base_spacing=5 * u.mm,
            refinements=[],
            smooth_controls=SnappySmoothControls(),
        )
        vol_meshing_params = BetaVolumeMeshingParams(
            defaults=BetaVolumeMeshingDefaults(
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
                zones=[SeedpointZone(name="farfield", point_in_mesh=[0, 0, 0] * u.mm)],
            ),
        )
    return param


@pytest.fixture()
def snappy_refinements_multiple_regions():
    test_geometry = TempGeometry("tester.stl", True)
    with SI_unit_system:
        surf_meshing_params = SnappySurfaceMeshingParams(
            defaults=SnappySurfaceMeshingDefaults(
                min_spacing=2.999999992 * u.mm, max_spacing=4 * u.mm, gap_resolution=1 * u.mm
            ),
            base_spacing=3 * u.mm,
            refinements=[
                SnappyRegionRefinement(
                    min_spacing=20 * u.mm,
                    max_spacing=40 * u.mm,
                    proximity_spacing=3 * u.mm,
                    regions=[
                        test_geometry["body1"]["patch0"],
                        test_geometry["body1"]["patch1"],
                        test_geometry["body1"]["patch2"],
                    ],
                ),
                SnappyRegionRefinement(
                    min_spacing=10 * u.mm, max_spacing=40 * u.mm, regions=test_geometry["body0"]
                ),
                SnappyRegionRefinement(
                    min_spacing=5 * u.mm,
                    max_spacing=40 * u.mm,
                    regions=[test_geometry["body2"], test_geometry["body3"]["patch0"]],
                ),
                SnappySurfaceEdgeRefinement(
                    spacing=4 * u.mm,
                    min_elem=3,
                    included_angle=120 * u.deg,
                    regions=[test_geometry["body0"]["patch0"], test_geometry["body0"]["patch1"]],
                    retain_on_smoothing=False,
                ),
            ],
            smooth_controls=SnappySmoothControls(),
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
        surf_meshing_params = SnappySurfaceMeshingParams(
            defaults=SnappySurfaceMeshingDefaults(
                min_spacing=3 * u.mm, max_spacing=4 * u.mm, gap_resolution=1 * u.mm
            ),
            refinements=[
                SnappyBodyRefinement(
                    gap_resolution=2 * u.mm,
                    min_spacing=5 * u.mm,
                    max_spacing=10 * u.mm,
                    bodies=[test_geometry["body01_face001"]],
                ),
                SnappyBodyRefinement(
                    gap_resolution=0.5 * u.mm,
                    min_spacing=1 * u.mm,
                    max_spacing=2 * u.mm,
                    bodies=[test_geometry["body01_face002"]],
                    proximity_spacing=0.2 * u.mm,
                ),
                SnappySurfaceEdgeRefinement(
                    spacing=4 * u.mm,
                    min_elem=3,
                    included_angle=120 * u.deg,
                    bodies=[test_geometry["body01_face003"]],
                ),
            ],
            smooth_controls=SnappySmoothControls(),
        )

        param = SimulationParams(
            private_attribute_asset_cache=AssetCache(
                project_entity_info=test_geometry._get_entity_info(), project_length_unit=1 * u.mm
            ),
            meshing=ModularMeshingWorkflow(
                surface_meshing=surf_meshing_params,
                zones=[SeedpointZone(name="fluid", point_in_mesh=[0, 0, 0] * u.m)],
            ),
        )
    return param


@pytest.fixture()
def snappy_settings():
    test_geometry = TempGeometry("tester.stl", True)
    with SI_unit_system:
        surf_meshing_params = SnappySurfaceMeshingParams(
            defaults=SnappySurfaceMeshingDefaults(
                min_spacing=3 * u.mm, max_spacing=4 * u.mm, gap_resolution=1 * u.mm
            ),
            quality_metrics=SnappyQualityMetrics(
                max_non_ortho=55 * u.deg,
                max_boundary_skewness=30 * u.deg,
                max_internal_skewness=70 * u.deg,
                max_concave=20 * u.deg,
                min_vol=1e-2,
                min_tet_quality=0.15,
                min_area=2 * u.mm * u.mm,
                min_twist=0.3,
                min_determinant=0.5,
                min_vol_ratio=0.1,
                min_face_weight=0.3,
                min_triangle_twist=0.1,
                n_smooth_scale=6,
                error_reduction=0.4,
                min_vol_collapse_ratio=0.5,
            ),
            snap_controls=SnappySnapControls(
                n_smooth_patch=5,
                tolerance=4,
                n_solve_iter=20,
                n_relax_iter=2,
                n_feature_snap_iter=10,
                multi_region_feature_snap=False,
                strict_region_snap=True,
            ),
            castellated_mesh_controls=SnappyCastellatedMeshControls(
                resolve_feature_angle=10 * u.deg, n_cells_between_levels=3, min_refinement_cells=50
            ),
            smooth_controls=SnappySmoothControls(lambda_factor=0.3, mu_factor=0.31, iterations=5),
        )

        param = SimulationParams(
            private_attribute_asset_cache=AssetCache(
                project_entity_info=test_geometry._get_entity_info(), project_length_unit=1 * u.mm
            ),
            meshing=ModularMeshingWorkflow(
                surface_meshing=surf_meshing_params,
                zones=[
                    SeedpointZone(name="fluid", point_in_mesh=[0, 0, 0] * u.m),
                    SeedpointZone(name="solid", point_in_mesh=[0.001, 0.002, 0.003] * u.m),
                ],
            ),
        )
    return param


@pytest.fixture()
def snappy_settings_off_position():
    test_geometry = TempGeometry("tester.stl", True)
    with SI_unit_system:
        surf_meshing_params = SnappySurfaceMeshingParams(
            defaults=SnappySurfaceMeshingDefaults(
                min_spacing=3 * u.mm, max_spacing=4 * u.mm, gap_resolution=1 * u.mm
            ),
            quality_metrics=SnappyQualityMetrics(
                max_non_ortho=None,
                max_boundary_skewness=None,
                max_internal_skewness=None,
                max_concave=None,
                min_vol=None,
                min_tet_quality=None,
                min_area=None,
                min_twist=None,
                min_determinant=None,
                min_vol_ratio=None,
                min_face_weight=None,
                min_triangle_twist=None,
                n_smooth_scale=None,
                error_reduction=None,
                min_vol_collapse_ratio=None,
            ),
            snap_controls=SnappySnapControls(
                n_smooth_patch=5,
                tolerance=4,
                n_solve_iter=20,
                n_relax_iter=2,
                n_feature_snap_iter=10,
                multi_region_feature_snap=False,
                strict_region_snap=True,
            ),
            castellated_mesh_controls=SnappyCastellatedMeshControls(
                resolve_feature_angle=10 * u.deg, n_cells_between_levels=3, min_refinement_cells=50
            ),
            smooth_controls=SnappySmoothControls(
                lambda_factor=None, mu_factor=None, iterations=None
            ),
        )

        param = SimulationParams(
            private_attribute_asset_cache=AssetCache(
                project_entity_info=test_geometry._get_entity_info(), project_length_unit=1 * u.mm
            ),
            meshing=ModularMeshingWorkflow(
                surface_meshing=surf_meshing_params,
                zones=[
                    SeedpointZone(name="fluid", point_in_mesh=[0, 0, 0] * u.m),
                    SeedpointZone(name="solid", point_in_mesh=[0.001, 0.002, 0.003] * u.m),
                ],
            ),
        )
    return param


def deep_sort_lists(obj):
    """
    Recursively sort all lists in a JSON-like object to ensure consistent ordering.

    Args:
        obj: Any JSON-like object (dict, list, str, int, float, bool, None)

    Returns:
        A new object with all lists sorted
    """
    if isinstance(obj, dict):
        return {k: deep_sort_lists(v) for k, v in sorted(obj.items())}
    elif isinstance(obj, list):
        # Sort the list and recursively sort its elements
        sorted_items = [deep_sort_lists(item) for item in obj]

        # Create a stable sorting key that works for complex nested structures
        def sort_key(item):
            if isinstance(item, dict):
                # For dictionaries, create a canonical string representation
                return json.dumps(item, sort_keys=True, separators=(",", ":"))
            elif isinstance(item, list):
                # For lists, create a canonical string representation
                return json.dumps(item, sort_keys=True, separators=(",", ":"))
            else:
                # For primitives, use string representation
                return str(item)

        return sorted(sorted_items, key=sort_key)
    else:
        return obj


def _translate_and_compare(param, mesh_unit, ref_json_file: str, atol=1e-15):
    param, _ = validate_params_with_context(param, "Geometry", "SurfaceMesh")
    translated = get_surface_meshing_json(param, mesh_unit=mesh_unit)
    with open(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "ref/surface_meshing", ref_json_file
        )
    ) as fh:
        ref_dict = json.load(fh)

    ref_dict, translated = deep_sort_lists(ref_dict), deep_sort_lists(translated)
    # check if everything is seriazable
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
        snappy_all_defaults, get_snappy_geometry.mesh_unit, "default_snappy.json"
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
    )


def test_snappy_settings(get_snappy_geometry, snappy_settings):
    _translate_and_compare(snappy_settings, get_snappy_geometry.mesh_unit, "snappy_settings.json")


def test_snappy_settings_off_position(get_snappy_geometry, snappy_settings_off_position):
    _translate_and_compare(
        snappy_settings_off_position, get_snappy_geometry.mesh_unit, "snappy_settings_off_pos.json"
    )


def test_snappy_no_refinements(get_snappy_geometry, snappy_refinements_no_regions):
    _translate_and_compare(
        snappy_refinements_no_regions, get_snappy_geometry.mesh_unit, "snappy_no_regions.json"
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

    with open(
        os.path.join(
            os.path.dirname(__file__), "data", "gai_geometry_entity_info", "simulation.json"
        ),
        "r",
    ) as fh:
        asset_cache = AssetCache.model_validate(json.load(fh).pop("private_attribute_asset_cache"))

    with SI_unit_system:
        # Rotate around z-axis for 90 deg, and scale in 3 axes
        transformation = Transformation(
            origin=[0, 0, 0] * u.m,
            axis_of_rotation=(0, 0, 1),
            angle_of_rotation=90 * u.deg,
            scale=(4.0, 3.0, 2.0),
            translation=[0, 0, 0] * u.m,
        )
        geometry["cube-holes.egads"].transformation = transformation
        geometry["cylinder.stl"].transformation = transformation
        farfield = AutomatedFarfield()
        params = SimulationParams(
            meshing=MeshingParams(
                defaults=MeshingDefaults(
                    geometry_accuracy=0.05 * u.m,  # GAI setting
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
                    ),
                    GeometryRefinement(
                        name="Local_override",
                        geometry_accuracy=0.05 * u.m,
                        faces=[geometry["body00001_face00001"]],
                    ),
                ],
            ),
            operating_condition=AerospaceCondition(
                velocity_magnitude=10 * u.m / u.s,
            ),
            private_attribute_asset_cache=asset_cache,
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
