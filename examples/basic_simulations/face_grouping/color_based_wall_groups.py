"""
A single STEP file with two colored spheres, grouped by color into Wall
boundaries via the rule-based face-grouping color condition, authored as
DRAFT STATE: ``create_draft(face_grouping=FaceGroupingRules(...))``.

The V1 CAD importer splits this model into 4 bodies, each carrying 2 green and
2 red faces. One ``green`` and one ``red`` group across all four bodies is fine:
a face group may span bodies; what it may not span is model files (one group per
model file -- see ``two_models_wall_groups.py``).
"""

import flow360 as fl
from flow360.examples import TwoNamedSpheresColorGrouping

TwoNamedSpheresColorGrouping.get_files()
project = fl.Project.from_geometry(
    TwoNamedSpheresColorGrouping.geometry,
    name="Two Named Spheres Color Grouping from Python",
    cad_importer_version="v1",
    length_unit="m",
)
geometry = project.geometry

root = fl.face_grouping.root
rules = fl.face_grouping.FaceGroupingRules(
    groups=[
        fl.face_grouping.FaceGroupRule(name="green", rule=root.faces(color="#00FF00")),
        fl.face_grouping.FaceGroupRule(name="red", rule=root.faces(color="#FF0000")),
    ]
)

with fl.create_draft(new_run_from=geometry, face_grouping=rules) as draft:
    with fl.SI_unit_system:
        farfield = fl.AutomatedFarfield()
        params = fl.SimulationParams(
            meshing=fl.MeshingParams(
                defaults=fl.MeshingDefaults(
                    geometry_accuracy=0.01 * fl.u.m,
                    target_surface_node_count=10000,
                    surface_max_edge_length=0.1 * fl.u.m,
                    boundary_layer_first_layer_thickness=1e-3 * fl.u.m,
                ),
                volume_zones=[farfield],
            ),
            reference_geometry=fl.ReferenceGeometry(
                area=1, moment_center=[0, 0, 0], moment_length=[1, 1, 1]
            ),
            operating_condition=fl.AerospaceCondition.from_mach_reynolds(
                mach=0.1, reynolds_mesh_unit=1000, project_length_unit=fl.u.m
            ),
            time_stepping=fl.Steady(max_steps=500),
            models=[
                fl.Fluid(
                    navier_stokes_solver=fl.NavierStokesSolver(absolute_tolerance=1e-9),
                    turbulence_model_solver=fl.NoneSolver(),
                ),
                fl.Wall(surfaces=[draft.surfaces["green"]]),
                fl.Wall(surfaces=[draft.surfaces["red"]]),
                fl.Freestream(surfaces=[farfield.farfield]),
            ],
            outputs=[
                fl.SurfaceOutput(
                    surfaces=[draft.surfaces["*"]], output_fields=["Cp", "Cf", "yPlus"]
                ),
                fl.VolumeOutput(output_fields=["Mach", "Cp"]),
            ],
        )

    project.run_case(
        params, "Two Named Spheres Color Grouping case from Python", use_geometry_AI=True
    )
