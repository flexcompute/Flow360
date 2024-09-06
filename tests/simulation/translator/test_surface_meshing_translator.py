import json
import os

import pytest

import flow360.component.simulation.units as u
from flow360.component.simulation.entity_info import GeometryEntityInfo
from flow360.component.simulation.framework.param_utils import AssetCache
from flow360.component.simulation.meshing_param.edge_params import (
    AngleBasedRefinement,
    HeightBasedRefinement,
    ProjectAnisoSpacing,
    SurfaceEdgeRefinement,
)
from flow360.component.simulation.meshing_param.face_params import SurfaceRefinement
from flow360.component.simulation.meshing_param.params import (
    MeshingDefaults,
    MeshingParams,
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
from tests.utils import compare_values


class TempGeometry(AssetBase):
    """Mimicing the final VolumeMesh class"""

    fname: str
    mesh_unit: LengthType.Positive

    def _get_meta_data(self):
        if self.fname == "om6wing.csm":
            return {
                "edges": {
                    "wingLeadingEdge": {},
                    "wingTrailingEdge": {},
                    "rootAirfoilEdge": {},
                    "tipAirfoilEdge": {},
                },
                "surfaces": {"wing": {}},
                "mesh_unit": {"units": "m", "value": 1.0},
            }
        elif self.fname == "geometry.egads":
            return {
                "surfaces": {
                    "Outer_Wing_mirrored": {},
                    "Stab_mirrored": {},
                    "Outer_Wing": {},
                    "Fuselage_H": {},
                    "Inner_Wing": {},
                    "Fin": {},
                    "Inner_Wing_mirrored": {},
                    "Stab": {},
                    "Fuselage_H_mirrored": {},
                    "Fuselage_V": {},
                },
                "mesh_unit": {"units": "m", "value": 1.0},
            }
        elif self.fname == "rotor.csm":
            return {
                "surfaces": {
                    "hub": {},
                    "blade": {},
                    "tip": {},
                },
                "edges": {
                    "leadingEdge": {},
                    "trailingEdge": {},
                    "tipEdge": {},
                    "bladeSplitEdge": {},
                    "hubCircle": {},
                    "hubSplitEdge": {},
                    "junctionEdge": {},
                },
                "mesh_unit": {"units": "inch", "value": 1.0},
            }
        else:
            raise ValueError("Invalid file name")

    def _get_entity_info(self):
        if self.fname == "om6wing.csm":
            return GeometryEntityInfo(
                face_ids=["wing"],
                edge_ids=[
                    "wingLeadingEdge",
                    "wingTrailingEdge",
                    "rootAirfoilEdge",
                    "tipAirfoilEdge",
                ],
                face_attribute_names=["dummy"],
                face_group_tag="dummy",
                grouped_faces=[[Surface(name="wing", private_attribute_sub_components=["wing"])]],
            )

        elif self.fname == "geometry.egads":
            return GeometryEntityInfo(
                face_ids=[
                    "Outer_Wing_mirrored",
                    "Stab_mirrored",
                    "Outer_Wing",
                    "Fuselage_H",
                    "Inner_Wing",
                    "Fin",
                    "Inner_Wing_mirrored",
                    "Stab",
                    "Fuselage_H_mirrored",
                    "Fuselage_V",
                ],
                edge_ids=[],
                face_attribute_names=["dummy"],
                face_group_tag="dummy",
                grouped_faces=[
                    [
                        Surface(
                            name="Wing",
                            private_attribute_sub_components=[
                                "Outer_Wing_mirrored",
                                "Outer_Wing",
                                "Inner_Wing",
                                "Inner_Wing_mirrored",
                            ],
                        ),
                        Surface(
                            name="Fuselage",
                            private_attribute_sub_components=[
                                "Fuselage_H_mirrored",
                                "Fuselage_H",
                                "Fuselage_V",
                            ],
                        ),
                        Surface(
                            name="Stab", private_attribute_sub_components=["Stab", "Stab_mirrored"]
                        ),
                        Surface(name="Fin", private_attribute_sub_components=["Fin"]),
                    ]
                ],
            )
        elif self.fname == "rotor.csm":
            return GeometryEntityInfo(
                face_ids=["hub", "blade", "tip"],
                edge_ids=[
                    "leadingEdge",
                    "trailingEdge",
                    "tipEdge",
                    "bladeSplitEdge",
                    "hubCircle",
                    "hubSplitEdge",
                    "junctionEdge",
                ],
                face_attribute_names=["dummy"],
                face_group_tag="dummy",
                grouped_faces=[
                    [
                        Surface(
                            name="rotor",
                            private_attribute_sub_components=["hub", "blade", "tip"],
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
                        entities=[my_geometry["wing"]],
                        max_edge_length=14 * u.cm,
                    ),
                    SurfaceEdgeRefinement(
                        entities=[my_geometry["wing*Edge"]],
                        method=HeightBasedRefinement(value=3e-2 * u.cm),
                    ),
                    SurfaceEdgeRefinement(
                        entities=[my_geometry["*AirfoilEdge"]],
                        method=ProjectAnisoSpacing(),
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
                        entities=[my_geometry["wing*Edge"]],
                        method=HeightBasedRefinement(value=3e-2 * u.cm),
                    ),
                    SurfaceEdgeRefinement(
                        entities=[my_geometry["*AirfoilEdge"]],
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
                        entities=[my_geometry["*Wing*"]],
                        max_edge_length=1.5 * u.m,
                    ),
                    SurfaceRefinement(
                        entities=[my_geometry["Fuselage*"]],
                        max_edge_length=700 * u.mm,
                    ),
                    SurfaceRefinement(
                        entities=[my_geometry["Stab*"]],
                        max_edge_length=0.5 * u.m,
                    ),
                    SurfaceRefinement(
                        entities=[my_geometry["Fin"]],
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
                        entities=[rotor_geopmetry["tip"]],
                        max_edge_length=0.1 * u.inch,
                    ),
                    SurfaceRefinement(
                        entities=[rotor_geopmetry["blade"], rotor_geopmetry["hub"]],
                        max_edge_length=10 * u.inch,
                    ),
                    SurfaceEdgeRefinement(
                        entities=[rotor_geopmetry["leadingEdge"]],
                        method=AngleBasedRefinement(value=1 * u.degree),
                    ),
                    SurfaceEdgeRefinement(
                        entities=[rotor_geopmetry["t*Edge"]],
                        method=HeightBasedRefinement(value=0.05 * u.inch),
                    ),
                    SurfaceEdgeRefinement(
                        entities=[
                            rotor_geopmetry["bladeSplitEdge"],
                            rotor_geopmetry["hubSplitEdge"],
                        ],
                        method=ProjectAnisoSpacing(),
                    ),
                    SurfaceEdgeRefinement(
                        entities=[rotor_geopmetry["hubCircle"]],
                        method=HeightBasedRefinement(value=0.01 * u.inch),
                    ),
                    SurfaceEdgeRefinement(
                        entities=[rotor_geopmetry["junctionEdge"]],
                        method=HeightBasedRefinement(value=0.01 * u.inch),
                    ),
                ],
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
