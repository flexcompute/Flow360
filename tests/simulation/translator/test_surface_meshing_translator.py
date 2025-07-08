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
from flow360.component.simulation.meshing_param.params import (
    MeshingDefaults,
    MeshingParams,
    WrappingSettings,
    GeometrySettings,
    MeshQuality,
    SnapControls,
)
from flow360.component.simulation.primitives import Edge, Surface
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
def airplane_surface_mesh_with_wrapping():
    my_geometry = TempGeometry("geometry.egads")
    from numpy import pi

    with SI_unit_system:
        param = SimulationParams(
            private_attribute_asset_cache=AssetCache(
                project_entity_info=my_geometry._get_entity_info()
            ),
            meshing=MeshingParams(
                defaults=MeshingDefaults(
                    surface_max_edge_length=100 * u.cm,
                ),
                wrapping_settings=WrappingSettings(
                    geometry=[
                        GeometrySettings(
                            entities=[
                                my_geometry["*"],
                            ],
                            spec={
                                "spacing": {"min": 5, "max": 50},
                                "edges": {
                                    "edgeSpacing": 2,
                                    "includedAngle": 140,
                                    "minElem": 3,
                                    "minLen": 10
                                },
                                "gap": 0.001,
                                "regions": [],
                                "gapSpacingReduction": None,
                                "includedAngle": 140
                            }
                        )
                    ],
                    mesh_quality=MeshQuality(
                        max_non_ortho=85,
                        max_boundary_skewness=20,
                        max_internal_skewness=50,
                        max_concave=50,
                        min_vol=-1e+30,
                        min_tet_quality=-1e+30,
                        min_area=-1,
                        min_twist=-2,
                        min_determinant=0,
                        min_vol_ratio=0,
                        min_face_weight=0,
                        min_triangle_twist=-1,
                        n_smooth_scale=4,
                        error_reduction=0.75,
                        min_vol_collapse_ratio=0
                    ),
                    snap_controls=SnapControls(
                        tolerance=2,
                        n_feature_snap_iter=15,
                        multi_region_feature_snap=True
                    ),
                    location_in_mesh=(3000, 0, 800) * u.m,
                    cad_is_fluid=True
                )
            ),
        )
    return param


def _translate_and_compare(param, mesh_unit, ref_json_file: str):
    translated = get_surface_meshing_json(param, mesh_unit=mesh_unit)
    with open(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "ref/surface_meshing", ref_json_file
        )
    ) as fh:
        ref_dict = json.load(fh)
    
    print("\n" + "="*80)
    print("REFERENCE JSON:")
    print("="*80)
    print(json.dumps(ref_dict, indent=2))
    
    print("\n" + "="*80)
    print("GENERATED JSON:")
    print("="*80)
    print(json.dumps(translated, indent=2))
    
    print("\n" + "="*80)
    print("COMPARISON RESULT:")
    print("="*80)
    result = compare_values(ref_dict, translated)
    print(f"Comparison result: {result}")
    
    assert result


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


def test_wrapping_settings(get_airplane_geometry, airplane_surface_mesh_with_wrapping):
    _translate_and_compare(
        airplane_surface_mesh_with_wrapping,
        get_airplane_geometry.mesh_unit,
        "surface_mesh_snappy_config.json",
    )
