from pylab import show

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

project = Project.from_file(Airplane.geometry, name="Python Project (Geometry, from file)")
geo = project.geometry

geo.show_available_groupings(verbose_mode=True)
geo.group_faces_by_tag("groupName")


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

project.run_case(params=params, name="Case of Simple Airplane from Python")

case = project.case


cd_curve = case.results.x_slicing_force_distribution
# wait for postprocessing to finish
cd_curve.wait()

print(cd_curve)
print(cd_curve.entities)

# plot CD vs X curve using all entities
cd_curve.as_dataframe().plot(x="X", y="totalCumulative_CD_Curve")
show()


# plot CD vs X curve using "include" filter
cd_curve.filter(include="*Wing*")
print(cd_curve)

cd_curve.as_dataframe().plot(x="X", y="totalCumulative_CD_Curve")
show()


# plot CD vs X curve using "exclude" filter
cd_curve.filter(exclude="*fuselage*")
cd_curve.as_dataframe().plot(x="X", y="totalCumulative_CD_Curve")
show()


# plot CD vs X curve using explicit names
cd_curve.filter(include=["fluid/leftWing", "fluid/rightWing"])
cd_curve.as_dataframe().plot(x="X", y="totalCumulative_CD_Curve")
show()
