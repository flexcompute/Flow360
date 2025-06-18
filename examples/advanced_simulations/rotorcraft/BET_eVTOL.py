import flow360 as fl
from flow360.examples import BETEVTOL

BETEVTOL.get_files()

project = fl.Project.from_geometry("evtol.egads", name="eVTOL")


geometry = project.geometry
geometry.group_edges_by_tag("edgeId")
geometry.group_faces_by_tag("faceId")


with fl.SI_unit_system:
    box1 = fl.Box(name="Box 1", center=[2, 0, 0.5], size=[12, 16, 4])
    box2 = fl.Box(name="Box 2", center=[8, 0, 0.5], size=[24, 32, 8])
    cylinder1 = fl.Cylinder(
        name="BET 1", center=[-1.95, -6, 0.57], axis=[-1, 0, 0], outer_radius=1.5, height=0.2
    )
    cylinder2 = fl.Cylinder(
        name="BET 2", center=[-1.95, -2.65, 0.51], axis=[-1, 0, 0], outer_radius=1.5, height=0.2
    )
    cylinder3 = fl.Cylinder(
        name="BET 3", center=[-1.95, 2.65, 0.51], axis=[-1, 0, 0], outer_radius=1.5, height=0.2
    )
    cylinder4 = fl.Cylinder(
        name="BET 4", center=[-1.95, 6, 0.57], axis=[-1, 0, 0], outer_radius=1.5, height=0.2
    )
    cylinder5 = fl.Cylinder(
        name="BET 5", center=[2.7, -6, 1.06], axis=[0, 0, 1], outer_radius=1.5, height=0.2
    )
    cylinder6 = fl.Cylinder(
        name="BET 6", center=[2.7, -2.65, 1.06], axis=[0, 0, 1], outer_radius=1.5, height=0.2
    )
    cylinder7 = fl.Cylinder(
        name="BET 7", center=[2.7, 2.65, 1.06], axis=[0, 0, 1], outer_radius=1.5, height=0.2
    )
    cylinder8 = fl.Cylinder(
        name="BET 8", center=[2.7, 6, 1.06], axis=[0, 0, 1], outer_radius=1.5, height=0.2
    )
    slices = [
        fl.Slice(name=f"Slice BET {i+1}", normal=[0, 1, 0], origin=[0, originY, 0])
        for i, originY in enumerate([-6, -2.65, 2.65, 6])
    ]
    farfield = fl.AutomatedFarfield(name="Farfield")
    params = fl.SimulationParams(
        meshing=fl.MeshingParams(
            defaults=fl.MeshingDefaults(
                surface_max_edge_length=0.05, boundary_layer_first_layer_thickness=0.01 * fl.u.mm
            ),
            volume_zones=[farfield],
            refinements=[
                fl.AxisymmetricRefinement(
                    name="BET refinement",
                    spacing_axial=0.01,
                    spacing_radial=0.03,
                    spacing_circumferential=0.03,
                    entities=[
                        cylinder1,
                        cylinder2,
                        cylinder3,
                        cylinder4,
                        cylinder5,
                        cylinder6,
                        cylinder7,
                        cylinder8,
                    ],
                ),
                fl.UniformRefinement(name="Uniform refinement 0.05", spacing=0.05, entities=box1),
                fl.UniformRefinement(name="Uniform refinement 0.1", spacing=0.1, entities=box2),
            ],
        ),
        reference_geometry=fl.ReferenceGeometry(
            area=16.8, moment_center=[0, 0, 0], moment_length=[1.4, 1.4, 1.4]
        ),
        operating_condition=fl.AerospaceCondition(velocity_magnitude=70, alpha=15 * fl.u.deg),
        models=[
            fl.Wall(name="Wall", surfaces=[geometry["*"]]),
            fl.Freestream(name="Freestream", surfaces=[farfield.farfield]),
            fl.Fluid(
                navier_stokes_solver=fl.NavierStokesSolver(relative_tolerance=0.01),
                turbulence_model_solver=fl.SpalartAllmaras(
                    relative_tolerance=0.01,
                    rotation_correction=True,
                    hybrid_model=fl.DetachedEddySimulation(),
                ),
            ),
            fl.BETDisk.from_file(BETEVTOL.extra["disk13"]),
            fl.BETDisk.from_file(BETEVTOL.extra["disk24"]),
            fl.BETDisk.from_file(BETEVTOL.extra["disk57"]),
            fl.BETDisk.from_file(BETEVTOL.extra["disk68"]),
        ],
        time_stepping=fl.Unsteady(
            steps=1000,
            step_size=0.0004,
            max_pseudo_steps=30,
            CFL=fl.AdaptiveCFL(min=1, max=3000, max_relative_change=50, convergence_limiting_factor=0.1),
        ),
        outputs=[
            fl.SurfaceOutput(
                name="Surface output",
                output_fields=["Cp", "yPlus", "Cf", "CfVec"],
                surfaces=[geometry["*"]],
            ),
            fl.SliceOutput(
                name="Slice output", output_fields=["Cp", "Mach", "vorticity"], slices=[*slices]
            ),
        ],
    )


project.run_case(params, name="BET eVTOL case")
