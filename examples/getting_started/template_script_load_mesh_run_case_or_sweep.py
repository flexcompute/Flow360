"""
Sample Flow 360 API scripts.
Requires a mesh that you are ready to upload and run cases on.
"""

import os

import flow360 as fl
from flow360.log import log

# Variables we want to export in our volume solution files. Many more are available
vol_fields = ["Mach", "Cp", "mut", "mutRatio", "primitiveVars", "qcriterion"]

# Variables we want to export in our surface solution files. Many more are available
surf_fields = ["Cp", "yPlus", "Cf", "CfVec", "primitiveVars", "wallDistance"]


######################################################################################################################
def upload_mesh(file_path, project_name):
    """
    Given a file path and name of the project, this function creates a project and uploads a mesh.
    """
    # length_unit should be 'm', 'mm', 'cm', 'inch' or 'ft'
    project = fl.Project.from_volume_mesh(file_path, length_unit="m", name=project_name)
    log.info(f"The project id is {project.id}")

    return project


######################################################################################################################
def make_params(mesh_object):
    """
    Create the params object that contains all the run parameters.
    Needs the mesh_object to get the list of surfaces.
    """
    with fl.SI_unit_system:
        params = fl.SimulationParams(
            # Dimensions can  be either in inches, or m or mm or many other units
            reference_geometry=fl.ReferenceGeometry(
                moment_center=(0, 0, 0) * fl.u.m, moment_length=1 * fl.u.m, area=1 * fl.u.m * fl.u.m
            ),
            operating_condition=fl.AerospaceCondition(
                velocity_magnitude=100 * fl.u.m / fl.u.s, alpha=0 * fl.u.deg
            ),
            time_stepping=fl.Steady(max_steps=5000, CFL=fl.AdaptiveCFL()),
            models=[
                # These boundary names can be taken from the vm.boundary_names printout
                fl.Wall(
                    surfaces=[
                        mesh_object["fluid/leftWing"],
                        mesh_object["fluid/rightWing"],
                        mesh_object["fluid/fuselage"],
                    ],
                ),
                fl.Freestream(surfaces=mesh_object["fluid/farfield"]),  # For far field boundaries
                # Define what sort of physical model of a fluid we will use
                fl.Fluid(
                    navier_stokes_solver=fl.NavierStokesSolver(),
                    turbulence_model_solver=fl.SpalartAllmaras(),
                ),
            ],
            outputs=[
                fl.VolumeOutput(output_format="tecplot", output_fields=vol_fields),
                # This mesh_object['*'] will select all the boundaries in the mesh and export the surface results.
                # Regular expressions can be used to filter for certain boundaries
                fl.SurfaceOutput(
                    surfaces=[mesh_object["*"]], output_fields=surf_fields, output_format="tecplot"
                ),
            ],
        )
    return params


######################################################################################################################
def launch_sweep(params, project):
    """
    Launch a sweep of cases.
    """

    # for example let's vary alpha:
    alphas = [-10, -5, 0, 5, 10, 11, 12]

    for alpha_angle in alphas:
        # modify the alpha
        params.operating_condition.alpha = alpha_angle * fl.u.deg

        # launch the case
        project.run_case(params=params, name=f"{alpha_angle}_case ")
        log.info(f"The case ID is: {project.case.id} with {alpha_angle=} ")


######################################################################################################################
def main():
    """
    Main function that drives the mesh upload and case launching functions.
    """

    # if you want to upload a new mesh and create a new project
    mesh_file_path = os.path.join(os.getcwd(), "mesh_name.cgns")  # mesh could also be ugrid format
    project_name = "project_name"
    project = upload_mesh(mesh_file_path, project_name)

    # Or as an alternative, if you want to run from an existing project:
    # project = fl.Project.from_cloud(
    #     'prj-XXXXXXXXXX')  # where prj-XXXXXXXXXX is an ID that can be saved from a previously created project or read off the WEBUI

    vm = project.volume_mesh  # get the volume mesh entity associated with that project.
    log.info(f"The volume mesh contains the following boundaries:{vm.boundary_names}")
    log.info(f"The volume mesh ID is: {vm.id}")

    params = make_params(vm)  # define the run params used to launch the run

    # launch_sweep(params, project)  # if you want to launch a sweep

    # or if you want to simply launch the case\
    project.run_case(params=params, name=f"case_name")
    log.info(f"case id is {project.case.id}")


######################################################################################################################

if __name__ == "__main__":
    main()
