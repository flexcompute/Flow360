# Import necessary modules from the Flow360 library
import flow360 as fl
from flow360 import SI_unit_system, u
from flow360.examples import Airplane

# Activate the pre-production environment because of beta testing status
fl.Env.preprod.active()

# Step 1: Create a new project from a predefined geometry file in the Airplane example
# This initializes a project with the specified geometry and assigns it a name.
# project = fl.Project.from_file(Airplane.geometry, name="Python Project (Geometry, from file)")
project = fl.Project.from_cloud("prj-cb6035a5-7238-45cc-bd4a-d9dc5766c672")
# geo = project.geometry  # Access the geometry of the project
from flow360.component.geometry import Geometry

geo = Geometry.from_cloud("geo-81bc8b2f-ccd2-4f52-8688-296ab9f7410d")
# Step 2: Display available groupings in the geometry (helpful for identifying group names)
# geo.show_available_groupings(verbose_mode=True)

# Step 3: Group faces by a specific tag for easier reference in defining `Surface` objects
geo.group_faces_by_tag("groupName")

# Step 4: Define simulation parameters within a specific unit system
with SI_unit_system:
    # Define an automated far-field boundary condition for the simulation
    far_field_zone = fl.AutomatedFarfield()

    # Set up the main simulation parameters
    params = fl.SimulationParams(
        # Meshing parameters, including boundary layer and maximum edge length
        meshing=fl.MeshingParams(
            defaults=fl.MeshingDefaults(
                boundary_layer_first_layer_thickness=0.001,  # Boundary layer thickness
                surface_max_edge_length=1,  # Maximum edge length on surfaces
            ),
            volume_zones=[far_field_zone],  # Apply the automated far-field boundary condition
        )
    )

# Step 5: Run the simulation case with the specified parameters
project.generate_volume_mesh(params=params, name="Case of Simple Airplane from Python")
