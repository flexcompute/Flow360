import flow360 as fl
from flow360.examples import TutorialUDDForcesMoments

TutorialUDDForcesMoments.get_files()

project = fl.Project.from_geometry(
    TutorialUDDForcesMoments.geometry,
    name="Tutorial UDD forces and moments from Python",
)
geometry = project.geometry

geometry.show_available_groupings()
geometry.group_edges_by_tag("edgeName")
geometry.group_faces_by_tag("groupName")

with fl.SI_unit_system:
    box1 = fl.Box(name="box1", size=[2, 6, 3], center=[6.5, 9, 0], axis_of_rotation=[0, 1, 0])
    box2 = fl.Box(name="box2", size=[2, 6, 3], center=[6.5, -9, 0], axis_of_rotation=[0, 1, 0])
    box3 = fl.Box(name="box3", size=[4, 8, 3], center=[12, 0, 2], axis_of_rotation=[0, 1, 0])
    farfield = fl.AutomatedFarfield()
    params = fl.SimulationParams(
        meshing=fl.MeshingParams(
            defaults=fl.MeshingDefaults(
                surface_max_edge_length=0.5,
                curvature_resolution_angle=10 * fl.u.deg,
                boundary_layer_first_layer_thickness=2e-6,
                boundary_layer_growth_rate=1.2,
            ),
            refinement_factor=1,
            refinements=[
                fl.SurfaceEdgeRefinement(
                    method=fl.AngleBasedRefinement(value=1 * fl.u.deg),
                    edges=[geometry["leadingEdge"]],
                ),
                fl.SurfaceEdgeRefinement(
                    method=fl.HeightBasedRefinement(value=5e-3), edges=[geometry["trailingEdge"]]
                ),
                fl.SurfaceRefinement(max_edge_length=0.5, faces=[geometry["wing*"]]),
                fl.UniformRefinement(
                    name="box_refinement1", entities=[box1, box2, box3], spacing=0.2
                ),
            ],
            volume_zones=[farfield],
        ),
        reference_geometry=fl.ReferenceGeometry(
            area=60, moment_center=[5.7542, 0, 0], moment_length=[1, 1, 1]
        ),
        operating_condition=fl.AerospaceCondition(
            velocity_magnitude=50,
            alpha=10 * fl.u.deg,
            atmosphere=fl.ThermalState(temperature=288.15),
        ),
        models=[
            fl.Fluid(
                navier_stokes_solver=fl.NavierStokesSolver(
                    absolute_tolerance=1e-9, linear_solver=fl.LinearSolver(max_iterations=35)
                ),
                turbulence_model_solver=fl.SpalartAllmaras(
                    linear_solver=fl.LinearSolver(max_iterations=25)
                ),
            ),
            fl.Wall(
                surfaces=[geometry["*Left"], geometry["*Right"], geometry["fuselage"]],
            ),
            fl.Freestream(surfaces=[farfield.farfield]),
        ],
        time_stepping=fl.Steady(max_steps=5000),
        outputs=[
            fl.VolumeOutput(
                output_fields=["primitiveVars", "vorticity", "qcriterion", "Cp", "Mach"]
            ),
            fl.SurfaceOutput(
                surfaces=[geometry["*"]],
                output_fields=["primitiveVars", "Cf", "wallDistance", "Cp", "CfVec", "yPlus"],
            ),
        ],
        user_defined_dynamics=[
            fl.UserDefinedDynamic(
                name="rightAileronHingeTorque",
                input_vars=["forceX", "forceY", "forceZ", "momentX", "momentY", "momentZ"],
                constants={
                    "density_kgpm3": 1.225,
                    "c_inf_mps": 340.29400580821283,
                    "l_grid_unit": 1,
                    "newCenterX": 5.7542,
                    "newCenterY": 7,
                    "newCenterZ": 0,
                    "newAxisX": 0,
                    "newAxisY": 1,
                    "newAxisZ": 0,
                },
                state_vars_initial_value=["0.0", "0.0", "0.0", "0.0", "0.0"],
                update_law=[
                    "density_kgpm3 * c_inf_mps * c_inf_mps * l_grid_unit * l_grid_unit * l_grid_unit;",
                    "(momentX - ((newCenterY - momentCenterY) * forceZ - (newCenterZ - momentCenterZ) * forceY)) * state[0];",
                    "(momentY + ((newCenterX - momentCenterX) * forceZ - (newCenterZ - momentCenterZ) * forceX)) * state[0];",
                    "(momentZ - ((newCenterX - momentCenterX) * forceY - (newCenterY - momentCenterY) * forceX)) * state[0];",
                    "state[1] * newAxisX + state[2] * newAxisY + state[3] * newAxisZ;",
                ],
                input_boundary_patches=[geometry["aileronRight"]],
            ),
            fl.UserDefinedDynamic(
                name="leftAileronHingeTorque",
                input_vars=["forceX", "forceY", "forceZ", "momentX", "momentY", "momentZ"],
                constants={
                    "density_kgpm3": 1.225,
                    "c_inf_mps": 340.29400580821283,
                    "l_grid_unit": 1,
                    "newCenterX": 5.7542,
                    "newCenterY": -7,
                    "newCenterZ": 0,
                    "newAxisX": 0,
                    "newAxisY": -1,
                    "newAxisZ": 0,
                },
                state_vars_initial_value=["0.0", "0.0", "0.0", "0.0", "0.0"],
                update_law=[
                    "density_kgpm3 * c_inf_mps * c_inf_mps * l_grid_unit * l_grid_unit * l_grid_unit",
                    "(momentX - ((newCenterY - momentCenterY) * forceZ - (newCenterZ - momentCenterZ) * forceY)) * state[0];",
                    "(momentY + ((newCenterX - momentCenterX) * forceZ - (newCenterZ - momentCenterZ) * forceX)) * state[0];",
                    "(momentZ - ((newCenterX - momentCenterX) * forceY - (newCenterY - momentCenterY) * forceX)) * state[0];",
                    "state[1] * newAxisX + state[2] * newAxisY + state[3] * newAxisZ ",
                ],
                input_boundary_patches=[geometry["aileronLeft"]],
            ),
            fl.UserDefinedDynamic(
                name="rightRudderHingeTorque",
                input_vars=["forceX", "forceY", "forceZ", "momentX", "momentY", "momentZ"],
                constants={
                    "density_kgpm3": 1.225,
                    "c_inf_mps": 340.29400580821283,
                    "l_grid_unit": 1,
                    "newCenterX": 12.01,
                    "newCenterY": 0.861,
                    "newCenterZ": 0.861,
                    "newAxisX": 0,
                    "newAxisY": 0.7071,
                    "newAxisZ": 0.7071,
                },
                state_vars_initial_value=["0.0", "0.0", "0.0", "0.0", "0.0"],
                update_law=[
                    "density_kgpm3 * c_inf_mps * c_inf_mps * l_grid_unit * l_grid_unit * l_grid_unit",
                    "(momentX - ((newCenterY - momentCenterY) * forceZ - (newCenterZ - momentCenterZ) * forceY)) * state[0];",
                    "(momentY + ((newCenterX - momentCenterX) * forceZ - (newCenterZ - momentCenterZ) * forceX)) * state[0];",
                    "(momentZ - ((newCenterX - momentCenterX) * forceY - (newCenterY - momentCenterY) * forceX)) * state[0];",
                    "state[1] * newAxisX + state[2] * newAxisY + state[3] * newAxisZ ",
                ],
                input_boundary_patches=[geometry["rudderRight"]],
            ),
            fl.UserDefinedDynamic(
                name="leftRudderHingeTorque",
                input_vars=["forceX", "forceY", "forceZ", "momentX", "momentY", "momentZ"],
                constants={
                    "density_kgpm3": 1.225,
                    "c_inf_mps": 340.29400580821283,
                    "l_grid_unit": 1,
                    "newCenterX": 12.01,
                    "newCenterY": -0.861,
                    "newCenterZ": 0.861,
                    "newAxisX": 0,
                    "newAxisY": -0.7071,
                    "newAxisZ": 0.7071,
                },
                state_vars_initial_value=["0.0", "0.0", "0.0", "0.0", "0.0"],
                update_law=[
                    "density_kgpm3 * c_inf_mps * c_inf_mps * l_grid_unit * l_grid_unit * l_grid_unit",
                    "(momentX - ((newCenterY - momentCenterY) * forceZ - (newCenterZ - momentCenterZ) * forceY)) * state[0];",
                    "(momentY + ((newCenterX - momentCenterX) * forceZ - (newCenterZ - momentCenterZ) * forceX)) * state[0];",
                    "(momentZ - ((newCenterX - momentCenterX) * forceY - (newCenterY - momentCenterY) * forceX)) * state[0];",
                    "state[1] * newAxisX + state[2] * newAxisY + state[3] * newAxisZ ",
                ],
                input_boundary_patches=[geometry["rudderLeft"]],
            ),
        ],
    )

project.run_case(params, name="Case of tutorial UDD forces and moments from Python")
