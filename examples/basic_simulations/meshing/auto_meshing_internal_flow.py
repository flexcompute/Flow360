import flow360 as fl
from flow360.examples import TutorialAutoMeshingInternalFlow

TutorialAutoMeshingInternalFlow.get_files()
project = fl.Project.from_geometry(
    TutorialAutoMeshingInternalFlow.geometry,
    name="Tutorial Auto Meshing Internal Flow from Python",
)
geometry = project.geometry

# show face groupings
geometry.show_available_groupings(verbose_mode=True)
geometry.group_faces_by_tag("faceName")

with fl.SI_unit_system:
    farfield = fl.UserDefinedFarfield()
    params = fl.SimulationParams(
        meshing=fl.MeshingParams(
            defaults=fl.MeshingDefaults(
                surface_max_edge_length=1.2 * fl.u.m,
                curvature_resolution_angle=15 * fl.u.deg,
                surface_edge_growth_rate=1.2,
                boundary_layer_first_layer_thickness=1e-6,
                boundary_layer_growth_rate=1.2,
            ),
            refinement_factor=1.0,
            volume_zones=[farfield],
            refinements=[
                fl.SurfaceRefinement(
                    name="sphere", max_edge_length=0.1, faces=[geometry["sphere"]]
                ),
                fl.SurfaceRefinement(name="strut", max_edge_length=0.01, faces=[geometry["strut"]]),
                fl.BoundaryLayer(
                    name="floor", first_layer_thickness=1e-5, faces=[geometry["floor"]]
                ),
                fl.PassiveSpacing(
                    name="adjacent2floor", type="projected", faces=[geometry["adjacent2floor"]]
                ),
                fl.PassiveSpacing(name="ceiling", type="unchanged", faces=[geometry["ceiling"]]),
            ],
        ),
    )

project.generate_surface_mesh(params)
project.generate_volume_mesh(params)
