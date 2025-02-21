import flow360 as fl
import flow360.v1 as fl_v1
from flow360.examples import OM6wing

# create folder in ROOT level
folder_A = fl_v1.Folder.create("folder-python-level-A").submit()
print(folder_A)

# create folder inside the above folder
folder_B = fl_v1.Folder.create("folder-python-level-B", parent_folder=folder_A).submit()
print(folder_B)

# create folder in ROOT level and move inside folder_B
folder_C = fl_v1.Folder.create("folder-python-level-C").submit()
folder_C = folder_C.move_to_folder(folder_B)
print(folder_C)


OM6wing.get_files()

project = fl.Project.from_file(
    files=fl.VolumeMeshFile(OM6wing.mesh_filename),
    name="Forking cases from Python"
)
vm = project.volume_mesh

with fl.SI_unit_system:
    params = fl.SimulationParams(
        reference_geometry=fl.ReferenceGeometry(
            area=1.15315084119231,
            moment_center=[0, 0, 0],
            moment_length=[1.47602, 0.801672958512342, 1.47602]
        ),
        operating_condition=fl.AerospaceCondition(
            velocity_magnitude=286,
            alpha=3.06 * fl.u.deg
        ),
        time_stepping=fl.Steady(
            max_steps=500
        ),
        models=[
            fl.Wall(surfaces=vm["1"]),
            fl.SlipWall(surfaces=vm["2"]),
            fl.Freestream(surfaces=vm["3"])
        ],
        outputs=[
            fl.SurfaceOutput(
                output_fields=["primitiveVars", "Cp", "Cf"], surfaces=[vm["1"]]
            ),
            fl.VolumeOutput(output_fields=["primitiveVars", "Mach"]),
        ]
    )

case = project.run_case(params, "OM6Wing-default-0")

case = fl_v1.Case(case.id)

# move case to folder_C
case = case.move_to_folder(folder_C)
print(case.info)
