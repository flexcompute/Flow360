"""
Two separate sphere STEP files, each grouped into its own Wall boundary via
rule-based face grouping: a file-scoped ``model_files()`` rule for the first file, and
the ``is_remaining`` catch-all for whatever is left — here the whole second file (a
group may not span model files, so the leftovers must all live in one). Authored
as DRAFT STATE: ``create_draft(face_grouping=FaceGroupingRules(...))`` resolves
the rules against the geometry's topology tree and carries the groups on the
draft — the geometry cloud asset itself stays immutable.
"""

import flow360 as fl
from flow360.examples import TwoSpheresFaceGrouping

TwoSpheresFaceGrouping.get_files()
project = fl.Project.from_geometry(
    TwoSpheresFaceGrouping.geometry,
    name="Two Spheres Face Grouping from Python",
    cad_importer_version="v1",
    length_unit="m",
)
geometry = project.geometry

# The first file by rule; everything left (the second file) as the catch-all.
root = fl.face_grouping.root
rules = fl.face_grouping.FaceGroupingRules(
    groups=[
        fl.face_grouping.FaceGroupRule(
            name="sphere_1", rule=root.model_files(name="sphere_r2mm.step").faces()
        ),
        fl.face_grouping.FaceGroupRule(name="sphere_2", is_remaining=True),
    ]
)

with fl.create_draft(new_run_from=geometry, face_grouping=rules) as draft:
    with fl.SI_unit_system:
        farfield = fl.AutomatedFarfield()
        params = fl.SimulationParams(
            meshing=fl.MeshingParams(
                defaults=fl.MeshingDefaults(
                    geometry_accuracy=0.02 * fl.u.m,
                    target_surface_node_count=10000,
                    surface_max_edge_length=1 * fl.u.m,
                    curvature_resolution_angle=12 * fl.u.deg,
                    surface_edge_growth_rate=1.2,
                    boundary_layer_growth_rate=1.2,
                    boundary_layer_first_layer_thickness=0.001 * fl.u.m,
                ),
                volume_zones=[farfield],
            ),
            reference_geometry=fl.ReferenceGeometry(
                area=1, moment_center=[0, 0, 0], moment_length=[1, 1, 1]
            ),
            operating_condition=fl.AerospaceCondition(velocity_magnitude=2),
            time_stepping=fl.Steady(max_steps=500),
            models=[
                fl.Fluid(
                    navier_stokes_solver=fl.NavierStokesSolver(absolute_tolerance=1e-10),
                    turbulence_model_solver=fl.SpalartAllmaras(absolute_tolerance=1e-8),
                ),
                fl.Wall(surfaces=[draft.surfaces["sphere_1"], draft.surfaces["sphere_2"]]),
                fl.Freestream(surfaces=[farfield.farfield]),
            ],
            outputs=[
                fl.SurfaceOutput(
                    surfaces=[draft.surfaces["sphere_1"], draft.surfaces["sphere_2"]],
                    output_fields=["Cp", "yPlus", "Cf", "CfVec"],
                ),
            ],
        )

    # Two geometry files require Geometry AI (the legacy mesher only accepts one).
    project.run_case(params, "Two Spheres Face Grouping case from Python", use_geometry_AI=True)
