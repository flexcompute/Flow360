'''
Sample Flow 360 API scripts

Requires that you have a mesh that you are ready to upload and run cases on.

'''

import flow360 as fl
from flow360 import u  # used to give us access to dimensional units
import os

# VARIABLES WE WANT TO EXPORT IN OUR VOLUME SOLUTION FILES. MANY MORE ARE AVAILABLE
vol_fields = ['Mach', "Cp", "mut", "mutRatio", "primitiveVars", "qcriterion"]

# VARIABLES WE WANT TO EXPORT IN OUR SURFACE SOLUTION FILES. MANY MORE ARE AVAILABLE
surf_fields = ["Cp", "yPlus", "Cf", "CfVec", "primitiveVars", "wallDistance"]


######################################################################################################################
def upload_mesh(file_path, project_name):
    '''
    given a file path it will uplaod a mesh
    Returns
    -------
    project id associated with that mesh
    '''

    project = fl.Project.from_file(file_path, length_unit='inch', name=project_name)
    print(f"The project id is {project.id}")

    return project


######################################################################################################################
def make_params(mesh_object):
    with fl.imperial_unit_system:
        params = fl.SimulationParams(
            # dimensions can  be either in inches, or m or mm or many other units
            reference_geometry=fl.ReferenceGeometry(moment_center=(0, 0, 0) * u.inch,
                                                    moment_length=1 * u.inch, area=1 * u.inch * u.inch),
            operating_condition=fl.FreestreamFromVelocity(velocity=100 * u.kt, alpha=10 * u.deg),
            time_stepping=fl.Steady(max_steps=5000, CFL=fl.AdaptiveCFL()),
            models=[
                fl.Wall(surfaces=mesh_object['BOUNDARY1'], name="Boundary1"),
                # these boundary names can be taken from the vm.boundary_names printout
                fl.Wall(surfaces=mesh_object['BOUNDARY2'], name="Boundary2"),
                fl.SlipWall(surfaces=mesh_object['BOUNDARY3'], name="Boundary3"),  # slip wall boundary
                fl.Freestream(surfaces=mesh_object['BOUNDARY4'], name="Boundary4"),  # for far field boundaries

                # define what sort of physical model of a fluid we will use
                fl.Fluid(navier_stokes_solver=fl.NavierStokesSolver(),
                         turbulence_model_solver=fl.SpalartAllmaras())
            ],
            outputs=[
                fl.VolumeOutput(output_format="tecplot",
                                output_fields=vol_fields),

                # This mesh_object['*'] will select all the boundaries in the mesh and export the surface results.
                # Regular expressions can be used to fileter for certain boundaries
                fl.SurfaceOutput(surfaces=[mesh_object['*']],
                                 output_fields=surf_fields, output_format="tecplot")]
        )
    return params


######################################################################################################################
def launch_sweep(params, project):
    '''

    launch a sweep of cases


    Parameters
    ----------
    params
    project

    Returns
    -------

    '''

    # for example let's vary alpha:
    alphas = [-10, -5, 0, 5, 10, 11, 12]

    for alpha_angle in alphas:
        # modify the alpha
        params.operating_condition.alpha = alpha_angle * u.deg

        # launch the case
        project.run_case(params=params,
                         name=f'{alpha_angle}_case ')
        print(f'case id is {project.case.id} with {alpha_angle=} ')


######################################################################################################################
def main():
    # if you want to upload a new mesh and create a new project
    mesh_file_path = os.path.join(os.getcwd(), 'MESHNAME.cgns')  # mesh could also be ugrid format
    project_name = 'SOME_PROJECT_NAME'
    project = upload_mesh(mesh_file_path, project_name)

    # Or as an alternative, if you want to run from an existing project:
    # project = fl.Project.from_cloud(
    #     'prj-XXXXXXXXXX')  # where the prj-XXXXXX ID can be saved from a previously created project or read off the WEBUI

    vm = project.volume_mesh  # get the volume mesh entity associated with that project.
    print("The volume mesh contains the following boundaries:\n", vm.boundary_names)
    print(f"The mesh id is {vm.id}")

    params = make_params(vm)  # define the run params used to launch the run

    # launch_sweep(params, project)  # if you want to launch a sweep

    # or if you want to simply launch the case
    project.run_case(params=params,
                     name=f'name_of_case ')
    print(f'case id is {project.case.id}')



######################################################################################################################

if __name__ == '__main__':
    main()
