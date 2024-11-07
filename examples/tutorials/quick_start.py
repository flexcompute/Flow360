import flow360 as fl
from flow360.component.geometry import Geometry
from flow360.component.project import Project
from flow360.component.simulation.meshing_param.params import (
    MeshingDefaults,
    MeshingParams,
)
from flow360.component.simulation.meshing_param.volume_params import AutomatedFarfield
from flow360.component.simulation.models.surface_models import Freestream, Wall
from flow360.component.simulation.operating_condition.operating_condition import (
    AerospaceCondition,
)
from flow360.component.simulation.outputs.outputs import SurfaceOutput
from flow360.component.simulation.primitives import ReferenceGeometry
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.time_stepping.time_stepping import Steady
from flow360.component.simulation.unit_system import SI_unit_system, u
from flow360.examples import Airplane

fl.Env.preprod.active()


project = Project.from_cloud("prj-d3bcbd86-af18-43cc-8dc6-54ad1e19ce36")
geo = project.geometry

geo.show_available_groupings(verbose_mode=True)
geo.group_faces_by_tag("groupName")


def half_run(params: SimulationParams):
    from flow360.component.project import _set_up_param_entity_info
    from flow360.component.simulation.unit_system import LengthType
    from flow360.component.simulation.utils import model_attribute_unlock

    defaults = project._root_simulation_json

    cache_key = "private_attribute_asset_cache"
    length_key = "project_length_unit"

    length_unit = defaults[cache_key][length_key]

    with model_attribute_unlock(params.private_attribute_asset_cache, length_key):
        params.private_attribute_asset_cache.project_length_unit = LengthType.validate(length_unit)

    root_asset = project._root_asset

    # Check if there are any new draft entities that have been added in the params by the user
    entity_info = _set_up_param_entity_info(root_asset.entity_info, params)

    with model_attribute_unlock(params.private_attribute_asset_cache, "project_entity_info"):
        params.private_attribute_asset_cache.project_entity_info = entity_info
    return params


with SI_unit_system:
    params = SimulationParams(
        meshing=MeshingParams(
            defaults=MeshingDefaults(
                boundary_layer_first_layer_thickness=0.001, surface_max_edge_length=1
            ),
            volume_zones=[AutomatedFarfield()],
        ),
        reference_geometry=ReferenceGeometry(),
        operating_condition=AerospaceCondition(velocity_magnitude=100, alpha=5 * u.deg),
        time_stepping=Steady(max_steps=1000),
        models=[
            Wall(
                surfaces=[geo["*"]],
                name="Wall",
            ),
            Freestream(surfaces=[AutomatedFarfield().farfield], name="Freestream"),
        ],
        outputs=[SurfaceOutput(surfaces=geo["*"], output_fields=["Cp", "Cf", "yPlus", "CfVec"])],
    )


params_new = half_run(params)
print(params_new.model_dump_json())
from flow360.component.simulation.services import simulation_to_case_json

case_json, _ = simulation_to_case_json(params_new, "m")
import json

with open("python.json", "w") as f:
    json.dump(params_new.model_dump(), f, indent=4, sort_keys=True)

with open("webUI.json", "r") as f:
    webUI_json = json.load(f)

with open("webUI.json", "w") as f:
    json.dump(webUI_json, f, indent=4, sort_keys=True)
