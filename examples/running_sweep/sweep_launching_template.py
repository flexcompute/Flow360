"""
Sample Flow 360 API scripts.
Requires a volume mesh or CAD file  that you are ready to upload and run cases on.

This script will:
1:
    Either:
        1.a- create a project and upload a volume mesh
        1.b-create a project and upload a CAD. Mesh parameters will need to be given and mesh will cration will need to be launched
        1.b- connect to an existing project and associate with a volume mesh from that project.

2:
    Create the directory structure on the local machine to organize the data that will be generated

3:
    -Launch all cases required for the sweep
    -Save all the caseID and other relevant data in the directory created in step 2: above
"""

import os
import click
import flow360 as fl
from flow360.log import log

# Variables we want to export in our volume solution files. Many more are available
vol_fields = ["Mach", "Cp", "mut", "mutRatio", "primitiveVars", "qcriterion"]

# Variables we want to export in our surface solution files. Many more are available
surf_fields = ["Cp", "yPlus", "Cf", "CfVec", "primitiveVars", "wallDistance"]

VEL_MAG = 100 * fl.u.m / fl.u.s  # flow velocity magnitude


######################################################################################################################
def make_run_params(mesh_object):
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
                velocity_magnitude=VEL_MAG, alpha=0 * fl.u.deg
            ),
            time_stepping=fl.Steady(max_steps=5000, CFL=fl.AdaptiveCFL()),
            models=[
                # assign the various surfaces to the various airplane subcomponents.
                fl.Wall(surfaces=mesh_object['fluid/fuselage'], name="fuselage"),
                fl.Wall(surfaces=mesh_object['fluid/*_pylon'], name="pylons"),
                # assign both left and right pylon to the pylons subcomponent
                fl.Wall(surfaces=mesh_object['fluid/left_wing'], name="left_wing"),
                fl.Wall(surfaces=mesh_object['fluid/right_wing'], name="right_wing"),
                fl.Wall(surfaces=mesh_object['fluid/h_tail'], name="h_tail"),
                fl.Wall(surfaces=mesh_object['fluid/v_tail'], name="v_tail"),
                fl.Freestream(surfaces=mesh_object['fluid/farfield'], name="farfield"),
                # For far field boundaries
                # Define what sort of physical model of a fluid we will use
                fl.Fluid(
                    navier_stokes_solver=fl.NavierStokesSolver(),
                    turbulence_model_solver=fl.SpalartAllmaras(),
                ),
            ],
            outputs=[
                # output format could be 'paraview' or 'tecplot' or 'both'
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
def launch_sweep(params, project, project_name, mesh_object, dir_path):
    """
    Launch a sweep of cases.
    """

    # create the csv file we will store all relevant sweep data into
    csv_file = open(os.path.join(dir_path, 'sweep_saved_data.csv'),'w')
    csv_file.write(f' Sweep name:, {os.path.basename(dir_path)}\n')  # include the sweep name (aka directory name)
    csv_file.write(f' Project name:, {project_name}\n')  # include the project name
    csv_file.write(f' Poject id:, {project.id}\n')  # include the project id
    csv_file.write(f' volume mesh id:, {mesh_object.id}\n')  # include the volume mesh id
    csv_file.write(f' Velocity magnitude:, {VEL_MAG}\n')  # include flow speed ( as a unyt entity)
    csv_file.write('\n')  # Add an empty line separator
    csv_file.write('alpha(deg), beta(deg), case_id\n')

    # for example let's vary alpha:
    alphas = [-10, 12]

    for alpha_angle in alphas:
        # modify the alpha
        params.operating_condition.alpha = alpha_angle * fl.u.deg

        # launch the case
        project.run_case(params=params, name=f"{alpha_angle}_case ")
        csv_file.write(f'{params.operating_condition.alpha.value},{params.operating_condition.beta.value},{project.case.id}\n')
        log.info(f"The case ID is: {project.case.id} with {alpha_angle=} ")

    csv_file.close()

######################################################################################################################
def create_directory(dir_name):
    '''

    Create on the local machine the required directories if not already created

    Parameters
    ----------
    dir_name

    Returns
    -------

    '''

    dir_path = os.path.join(os.getcwd(), dir_name)
    if os.path.isdir(dir_path):  # if directory already exists.
        overwrite_bool = click.confirm(
            f"Directory '{dir_path}' already exists, continuing might overwrite some of its content, do you want to continue?",
            default=True,
            abort=True,
        )
    else:
        os.makedirs(dir_path)  # make that dir
    return dir_path

######################################################################################################################
def create_mesh_params(project):
    '''

    Create the parameters object containing all the information required by the solver to generate the mesh

    Parameters
    ----------
    project

    Returns
    -------
        params object
    '''

    geometry = project.geometry
    geometry.group_faces_by_tag("faceName")
    geometry.group_edges_by_tag("edgeName")
    with fl.SI_unit_system:
        params = fl.SimulationParams(
            # set the meshing parameters.
            meshing=fl.MeshingParams(
                defaults=fl.MeshingDefaults(
                    boundary_layer_first_layer_thickness=1e-5, surface_max_edge_length=1
                ),
                volume_zones=[fl.AutomatedFarfield()],
                refinements=[
                    fl.SurfaceEdgeRefinement(name="leading_edges", edges=[geometry["leadingEdge"]],
                                             method=fl.AngleBasedRefinement(value=2 * fl.u.deg))
                ]
            )
        )
    return params


######################################################################################################################
def main():
    """
    Main function that drives the functions to achieve all the steps mentioned at top.
    """

    # Step1: Let's connect to an existing project and volume mesh
    # Chose one of the three options below

    # Option 1a: if you want to upload a new mesh and create a new project
    # mesh_file_path = os.path.join(os.getcwd(), "evtol_quickstart_volume_mesh.cgns")  # mesh could also be ugrid format
    # project_name = "sweep_evtol_from_mesh"
    # project = fl.Project.from_file(mesh_file_path, name=project_name,length_unit = "m")     # length_unit should be 'm', 'mm', 'cm', 'inch' or 'ft'
    # vm = project.volume_mesh  # get the volume mesh entity associated with that project.

    # Option 1b: if you want to upload a CAD geometry and create a new project
    # mesh_file_path = os.path.join(os.getcwd(), "evtol_quickstart_grouped.csm")  # mesh could also be ugrid format
    # project_name = "sweep_evtol_from_cad"
    # project = fl.Project.from_file(mesh_file_path, name=project_name)
    # mesh_params=create_mesh_params(project)
    # project.generate_volume_mesh(params=mesh_params, name='mesh_name') # generate the volume mesh
    # vm = project.volume_mesh  # get the volume mesh entity we just created

    # Option 1c:  if you want to run from an existing project:
    project = fl.Project.from_cloud(
        'prj-e63c3822-98e5-445e-bfaa-8608993addd7')  # where prj-XXXXXXXXXX is an ID that can be saved from a previously created project or read off the WEBUI
    project_name = project.metadata.name
    vm = project.volume_mesh  # get the volume mesh entity in that project.

    # if the project has more then one mesh then use this line below instead.
    # vm = project.get_volume_mesh(asset_id='vm-XXXXXXXXXXXXXXX')  # get the specific volume mesh entity in that project we want to use.

    #  simple printouts of useful information now that we have a project and a volume mesh.
    log.info(f"The project id is {project.id}")
    log.info(f"The volume mesh contains the following boundaries:{vm.boundary_names}")
    log.info(f"The volume mesh ID is: {vm.id}")

    # step 2: create the directories to locally store relevant data
    dir_name = 'evtol_alpha_sweep'
    dir_path = create_directory(dir_name)

    # step3: launch the cases and save the relevant data
    params = make_run_params(vm)  # define the run params used to launch the run
    launch_sweep(params, project, project_name, vm, dir_path)  # launch a sweep


######################################################################################################################

if __name__ == "__main__":
    main()
