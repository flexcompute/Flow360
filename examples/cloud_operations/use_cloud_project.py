import flow360 as fl

my_project = fl.Project.from_cloud("PROJECT_ID_HERE")
# Applicable for projects with Geometry being the starting point.
geo = my_project.geometry

geo.group_faces_by_tag("groupName")

# Submit a case with changed freestream velocity and angle of attack
with fl.SI_unit_system:
    far_field_zone = fl.AutomatedFarfield()

    params = fl.SimulationParams(
        meshing=fl.MeshingParams(
            defaults=fl.MeshingDefaults(
                boundary_layer_first_layer_thickness=0.001,
                surface_max_edge_length=1,
            ),
            volume_zones=[far_field_zone],
        ),
        reference_geometry=fl.ReferenceGeometry(),
        operating_condition=fl.AerospaceCondition(
            velocity_magnitude=105,  # Changed
            alpha=10 * fl.u.deg,  # Changed
        ),
        time_stepping=fl.Steady(max_steps=1000),
        models=[
            fl.Wall(
                surfaces=[geo["*"]],
            ),
            fl.Freestream(
                surfaces=[far_field_zone.farfield],
            ),
        ],
        outputs=[
            fl.SurfaceOutput(
                surfaces=geo["*"],
                output_fields=["Cp", "Cf", "yPlus", "CfVec"],
            )
        ],
    )

my_project.run_case(
    params=params, name="Case of Simple Airplane from Python with modifidied freestream"
)
