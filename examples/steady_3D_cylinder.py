import flow360 as fl
from flow360.examples import Cylinder3D

Cylinder3D.get_files()

project = fl.Project.from_file(
    files=fl.GeometryFiles(Cylinder3D.geometry), name="Steady 3D Cylinder from Python"
)

geo = project.geometry

geo.show_available_groupings()
geo.group_faces_by_tag("faceId")

with fl.SI_unit_system:
    farfield = fl.AutomatedFarfield(name="Farfield")
    params = fl.SimulationParams(
        meshing=fl.MeshingParams(
            defaults=fl.MeshingDefaults(
                surface_max_edge_length=1,
                curvature_resolution_angle=15 * fl.u.deg,
                surface_edge_growth_rate=1.2,
                boundary_layer_first_layer_thickness=0.01,
            ),
            volume_zones=[farfield],
        ),
        reference_geometry=fl.ReferenceGeometry(
            area=340, moment_center=[0, 0, 0], moment_length=[1, 1, 1]
        ),
        operating_condition=fl.operating_condition_from_mach_reynolds(
            reynolds=5, mach=0.1, project_length_unit=fl.u.m
        ),
        time_stepping=fl.Steady(),
        models=[
            fl.Fluid(
                navier_stokes_solver=fl.NavierStokesSolver(
                    absolute_tolerance=1e-11, linear_solver=fl.LinearSolver(max_iterations=35)
                ),
                turbulence_model_solver=fl.NoneSolver(),
            ),
            fl.Wall(
                name="NoSlipWall",
                surfaces=[
                    geo["*"],
                ],
            ),
            fl.Freestream(name="Freestream", surfaces=farfield.farfield),
        ],
        outputs=[
            fl.SurfaceOutput(
                name="SurfaceOutput",
                output_fields=["Cp", "Cf", "CfVec", "yPlus", "nodeForcesPerUnitArea"],
                surfaces=[geo["*"]],
            ),
            fl.VolumeOutput(name="VolumeOutput", output_fields=["primitiveVars", "qcriterion"]),
        ],
    )

project.run_case(params, "Steady 3D Cylinder case from Python")
