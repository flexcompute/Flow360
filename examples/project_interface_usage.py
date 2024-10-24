from flow360.component.project import Project
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.environment import dev
from flow360.component.simulation.unit_system import SI_unit_system

dev.active()

project = Project.from_cloud("prj-9c73a84b-6938-45c2-a8a3-5e3ffb2291b0")

# This could be nicer...
simulation_json = project.get_simulation_json()
with SI_unit_system:
    params = SimulationParams(**simulation_json)

geometry = project.geometry
geometry.show_available_groupings(verbose_mode=True)
geometry.group_faces_by_tag("faceId")

project.set_default_params(params)

project.run_surface_mesher()
project.run_volume_mesher()
project.run_case()