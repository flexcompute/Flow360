import json
import os

import pytest
import flow360.component.simulation.units as u
from flow360.component.simulation.entity_info import GeometryEntityInfo
from flow360.component.simulation.framework.param_utils import AssetCache
from flow360.component.simulation.framework.updater_utils import compare_values
from flow360.component.simulation.meshing_param.edge_params import (
    AngleBasedRefinement,
    AspectRatioBasedRefinement,
    HeightBasedRefinement,
    ProjectAnisoSpacing,
    SurfaceEdgeRefinement,
)
from flow360.component.simulation.meshing_param.face_params import SurfaceRefinement
from flow360.component.simulation.meshing_param.surface_mesh_refinements import (
    SnappyBodyRefinement,
    SnappySurfaceEdgeRefinement,
    SnappyRegionRefinement
)
from flow360.component.simulation.meshing_param.params import (
    MeshingDefaults,
    MeshingParams,
    SnappySurfaceMeshingParams,
    BetaVolumeMeshingParams,
    ModularMeshingWorkflow
)
from flow360.component.simulation.meshing_param.meshing_specs import (
    SnappySurfaceDefaults,
    SnappyCastellatedMeshControls,
    SnappyQualityMetrics,
    SnappySnapControls,
    SnappySmoothControls
)
from flow360.component.simulation.primitives import Edge, Surface, SnappyBody, Box, MeshZone
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.translator.surface_meshing_translator import (
    get_surface_meshing_json,
)
from flow360.component.simulation.unit_system import (
    LengthType,
    SI_unit_system,
    imperial_unit_system,
)
from flow360.component.simulation.meshing_param.volume_params import UserDefinedFarfield, AutomatedFarfield

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
                    "body3::patch0": {}
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
                    "body3::patch0"
                ],
                edge_ids=[],
                face_attribute_names=["dummy"],
                face_group_tag="dummy",
                grouped_faces=[
                    [
                        Surface(
                            name="body0::patch0",
                            private_attribute_sub_components=[
                                "body0::patch0"
                            ],
                        ),
                        Surface(
                            name="body0::patch1",
                            private_attribute_sub_components=[
                                "body0::patch1"
                            ],
                        ),
                        Surface(
                            name="body1::patch0",
                            private_attribute_sub_components=[
                                "body1::patch0"
                            ],
                        ),
                        Surface(
                            name="body1::patch1",
                            private_attribute_sub_components=[
                                "body1::patch1"
                            ],
                        ),
                        Surface(
                            name="body1::patch2",
                            private_attribute_sub_components=[
                                "body1::patch2"
                            ],
                        ),
                        Surface(
                            name="body2::patch0",
                            private_attribute_sub_components=[
                                "body2::patch0"
                            ],
                        ),
                        Surface(
                            name="body3::patch0",
                            private_attribute_sub_components=[
                                "body3::patch0"
                            ],
                        )
                    ]
                ],
            )
        else:
            raise ValueError("Invalid file name")

    def _populate_registry(self):
        self.mesh_unit = LengthType.validate(self._get_meta_data()["mesh_unit"])
        for zone_name in self._get_meta_data()["edges"] if "edges" in self._get_meta_data() else []:
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

    def __init__(self, file_name: str):
        super().__init__()
        self.fname = file_name
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
    test_geometry = TempGeometry("tester.stl")
    with SI_unit_system:
        surf_meshing_params = SnappySurfaceMeshingParams(
            defaults=SnappySurfaceDefaults(
                min_spacing=3 * u.mm,
                max_spacing=4 * u.mm,
                gap_resolution= 1 * u.mm
            )
        )

        param = SimulationParams(
            private_attribute_asset_cache=AssetCache(
                project_entity_info=test_geometry._get_entity_info()
            ),
            meshing=ModularMeshingWorkflow(
                surface_meshing=surf_meshing_params
            )
        )
    return param

@pytest.fixture()
def snappy_basic_refinements():
    test_geometry = TempGeometry("tester.stl")
    with SI_unit_system:
        surf_meshing_params = SnappySurfaceMeshingParams(
            defaults=SnappySurfaceDefaults(
                min_spacing=3*u.mm,
                max_spacing=4*u.mm,
                gap_resolution=1*u.mm
            ),
            refinements=[
                SnappyBodyRefinement(
                    gap_resolution=2*u.mm,
                    min_spacing=5*u.mm,
                    max_spacing=10*u.mm,
                    bodies=[SnappyBody(body_name="body1"), SnappyBody(body_name="body3")]
                ),
                SnappyBodyRefinement(
                    gap_resolution=0.5*u.mm,
                    min_spacing=1*u.mm,
                    max_spacing=2*u.mm,
                    bodies=[SnappyBody(body_name="body2")],
                    proximity_spacing=0.2*u.mm
                ),
                SnappyRegionRefinement(
                    min_spacing=20*u.mm,
                    max_spacing=40*u.mm,
                    proximity_spacing=3*u.mm,
                    regions=[
                        test_geometry["body0::patch0"],
                        test_geometry["body1::patch1"],
                    ]
                ),
                SnappySurfaceEdgeRefinement(
                    spacing=4*u.mm,
                    min_elem=3,
                    included_angle=120*u.deg,
                    regions=[test_geometry["body0::patch0"]],
                    bodies=[SnappyBody(body_name="body1")]
                ),
                SnappySurfaceEdgeRefinement(
                    spacing=[3*u.mm, 5*u.mm],
                    distances=[1*u.mm, 3*u.mm],
                    min_len=6*u.mm,
                    regions=[test_geometry["*patch1"]],
                    bodies=[SnappyBody(body_name="body3")]
                )
            ],
            smooth_controls=SnappySmoothControls()
        )

        param = SimulationParams(
            private_attribute_asset_cache=AssetCache(
                project_entity_info=test_geometry._get_entity_info()
            ),
            meshing=ModularMeshingWorkflow(
                surface_meshing=surf_meshing_params
            )
        )
    return param

@pytest.fixture()
def snappy_settings():
    test_geometry = TempGeometry("tester.stl")
    with SI_unit_system:
        surf_meshing_params = SnappySurfaceMeshingParams(
            defaults=SnappySurfaceDefaults(
                min_spacing=3 * u.mm,
                max_spacing=4 * u.mm,
                gap_resolution= 1 * u.mm
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
                min_vol_collapse_ratio=0.5
            ),
            snap_controls=SnappySnapControls(
                n_smooth_patch=5,
                tolerance=4,
                n_solve_iter=20,
                n_relax_iter=2,
                n_feature_snap_iter=10,
                multi_region_feature_snap=False,
                strict_region_snap=True
            ),
            castellated_mesh_controls=SnappyCastellatedMeshControls(
                resolve_feature_angle=10 *u.deg,
                n_cells_between_levels=3,
                min_refinement_cells=50
            ),
            bounding_box=Box(name="enclosure", center=(0, 0, 0) * u.m, size=(0.4, 0.8, 0.6) * u.m),
            smooth_controls=SnappySmoothControls(
                lambda_factor=0.3,
                mu_factor=0.31,
                iterations=5,
                min_elem=3,
                min_len=30*u.mm,
                included_angle=120*u.deg
            ),
            zones=[
                MeshZone(name="fluid", point_in_mesh=[0, 0, 0]*u.m), 
                MeshZone(name="solid", point_in_mesh=[0.001, 0.002, 0.003]*u.m)
            ]
        )

        param = SimulationParams(
            private_attribute_asset_cache=AssetCache(
                project_entity_info=test_geometry._get_entity_info()
            ),
            meshing=ModularMeshingWorkflow(
                surface_meshing=surf_meshing_params,
            )
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
                return json.dumps(item, sort_keys=True, separators=(',', ':'))
            elif isinstance(item, list):
                # For lists, create a canonical string representation
                return json.dumps(item, sort_keys=True, separators=(',', ':'))
            else:
                # For primitives, use string representation
                return str(item)
        
        return sorted(sorted_items, key=sort_key)
    else:
        return obj

def _translate_and_compare(param, mesh_unit, ref_json_file: str):
    translated = get_surface_meshing_json(param, mesh_unit=mesh_unit)
    with open(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "ref/surface_meshing", ref_json_file
        )
    ) as fh:
        ref_dict = json.load(fh)

    ref_dict, translated = deep_sort_lists(ref_dict), deep_sort_lists(translated)
    assert compare_values(ref_dict, translated)


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
    print(params.model_dump_json(indent=4))
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
        snappy_all_defaults,
        get_snappy_geometry.mesh_unit,
        "default_snappy.json"
    )

def test_snappy_basic(get_snappy_geometry, snappy_basic_refinements):
    _translate_and_compare(
        snappy_basic_refinements,
        get_snappy_geometry.mesh_unit,
        "snappy_basic_refinements.json"
    )

def test_snappy_settings(get_snappy_geometry, snappy_settings):
    _translate_and_compare(
        snappy_settings,
        get_snappy_geometry.mesh_unit,
        "snappy_settings.json"
    )
