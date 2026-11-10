import flow360 as fl
from flow360.examples import Airplane

project = fl.Project.from_cloud("PROJECT_ID_HERE")

with fl.SI_unit_system:
    params = fl.SimulationParams(...)

# Run the mesher once
project.generate_surface_mesh(params=params, name="Surface mesh 1")
surface_mesh_1 = project.surface_mesh

# Tweak some parameter in the params
params.meshing.defaults.surface_max_edge_length = 2 * fl.u.m

# Run the mesher again
project.generate_surface_mesh(params=params, name="Surface mesh 2")
surface_mesh_2 = project.surface_mesh

# Check available surface mesh IDs in the project
ids = project.get_surface_mesh_ids()
print(ids)
