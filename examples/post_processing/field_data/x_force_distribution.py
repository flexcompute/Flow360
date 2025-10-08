from pylab import show

import flow360 as fl
from flow360.examples import Airplane

project = fl.Project.from_geometry(Airplane.geometry, name="X force distribution from Python")
geo = project.geometry

geo.show_available_groupings(verbose_mode=True)
geo.group_faces_by_tag("groupName")


with fl.SI_unit_system:
    params = fl.SimulationParams(
        meshing=fl.MeshingParams(
            defaults=fl.MeshingDefaults(
                boundary_layer_first_layer_thickness=0.001, surface_max_edge_length=1
            ),
            volume_zones=[fl.AutomatedFarfield()],
        ),
        reference_geometry=fl.ReferenceGeometry(),
        operating_condition=fl.AerospaceCondition(velocity_magnitude=100, alpha=5 * fl.u.deg),
        time_stepping=fl.Steady(max_steps=1000),
        models=[
            fl.Wall(
                surfaces=[geo["*"]],
            ),
            fl.Freestream(surfaces=[fl.AutomatedFarfield().farfield]),
        ],
        outputs=[fl.SurfaceOutput(surfaces=geo["*"], output_fields=["Cp", "Cf", "yPlus", "CfVec"])],
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
