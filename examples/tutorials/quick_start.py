# Import necessary modules from the Flow360 library
import flow360 as fl
from flow360 import SI_unit_system, u

project = fl.Project.from_file("volumeMesh.cgns", name="VM")
vm = project.volume_mesh

# Step 4: Define simulation parameters within a specific unit system
with SI_unit_system:
    # Define an automated far-field boundary condition for the simulation
    far_field_zone = fl.AutomatedFarfield()

    # Set up the main simulation parameters
    params = fl.SimulationParams(
        # Reference geometry parameters for the simulation (e.g., center of pressure)
        reference_geometry=fl.ReferenceGeometry(),
        # Operating conditions: setting speed and angle of attack for the simulation
        operating_condition=fl.AerospaceCondition(
            velocity_magnitude=100,  # Velocity of 100 m/s
            alpha=5 * u.deg,  # Angle of attack of 5 degrees
        ),
        # Time-stepping configuration: specifying steady-state with a maximum step limit
        time_stepping=fl.Steady(max_steps=1000),
        # Define models for the simulation, such as walls and freestream conditions
        models=[
            fl.Wall(
                surfaces=[vm["*"]],  # Apply wall boundary conditions to all surfaces in geometry
                name="Wall",
            ),
            fl.Freestream(
                surfaces=[
                    far_field_zone.farfield
                ],  # Apply freestream boundary to the far-field zone
                name="Freestream",
            ),
        ],
        # Define output parameters for the simulation
        outputs=[
            fl.SurfaceOutput(
                surfaces=vm["*"],  # Select all surfaces for output
                output_fields=["Cp", "Cf", "yPlus", "CfVec"],  # Output fields for post-processing
            )
        ],
    )

# Step 5: Run the simulation case with the specified parameters
project.run_case(params=params, name="1")


# case_params = fl.Case.from_cloud("case-b889f655-155c-4df8-bc9c-54de4dff2a63").params
