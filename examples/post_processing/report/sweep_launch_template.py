"""
Sample Flow 360 API scripts.
Requires a volume mesh or CAD file  that you are ready to upload and run cases on.

This script will:
1:
    Either:
        1.a- create a project and upload a volume mesh
        1.b-create a project and upload a CAD. Mesh parameters will need to be given and mesh will creation will need to be launched
        1.c- connect to an existing project and associate with a volume mesh from that project.

2:
    Create the directory structure on the local machine to organize the data that will be generated

3:
    -Launch all cases required for the sweep
    -Save all the caseID and other relevant data in the directory created in step 2: above
"""

import os

import click
from sweep_launch_report import generate_report

import flow360 as fl
from flow360.examples import EVTOL
from flow360.log import log

# Variables we want to export in our volume and surface solution files. Many more are available

vol_fields = ["Mach", "Cp", "mut", "mutRatio", "primitiveVars", "qcriterion"]

# Variables we want to export in our surface solution files. Many more are available
surf_fields = ["Cp", "yPlus", "Cf", "CfVec", "primitiveVars", "wallDistance"]

VEL_MAG = 100 * fl.u.m / fl.u.s  # flow velocity magnitude


######################################################################################################################
def make_run_params(mesh_object, models):
    """
    Create the params object that contains all the run parameters.
    Needs the mesh_object to get the list of surfaces.
    """
    with fl.SI_unit_system:
        params = fl.SimulationParams(
            # Dimensions can  be either in inches, m, mm or many other units
            reference_geometry=fl.ReferenceGeometry(
                moment_center=(0, 0, 0) * fl.u.m, moment_length=1 * fl.u.m, area=1 * fl.u.m * fl.u.m
            ),
            operating_condition=fl.AerospaceCondition(
                velocity_magnitude=VEL_MAG, alpha=0 * fl.u.deg
            ),
            time_stepping=fl.Steady(max_steps=5000, CFL=fl.AdaptiveCFL()),
            models=[
                *models,
                # Define what sort of physical model of a fluid we will use
                fl.Fluid(
                    navier_stokes_solver=fl.NavierStokesSolver(),
                    turbulence_model_solver=fl.SpalartAllmaras(),
                ),
            ],
            outputs=[
                # output format could be 'paraview' or 'tecplot' or 'both'
                fl.VolumeOutput(output_format="tecplot", output_fields=vol_fields),
                # mesh_object['*'] will select all the boundaries in the mesh and export the surface results.
                # Regular expressions can be used to filter for certain boundaries
                fl.SurfaceOutput(
                    surfaces=[mesh_object["*"]], output_fields=surf_fields, output_format="tecplot"
                ),
            ],
        )

    # Add meshing params in case the project starts from geometry
    params.meshing = mesh_object.params.meshing

    return params


######################################################################################################################
def launch_sweep(params, project, project_name, mesh_object, dir_path):
    """
    Launch a sweep of cases.
    """

    case_list = []
    cases_params = []

    # create the csv file where we will store all relevant sweep data
    csv_path = os.path.join(dir_path, "sweep_saved_data.csv")
    with open(csv_path, "w") as csv_file:
        csv_file.write(
            f" Sweep name:, {os.path.basename(dir_path)}\n"
        )  # include the sweep name (directory name)
        csv_file.write(f" Project name:, {project_name}\n")
        csv_file.write(f" Project id:, {project.id}\n")
        csv_file.write(f" volume mesh id:, {mesh_object.id}\n")
        csv_file.write(f" Velocity magnitude:, {VEL_MAG}\n")
        csv_file.write("\n")
        csv_file.write(
            "case_id, alpha(deg), beta(deg), CL(avg), CD(avg), CFx(avg), CFy(avg), CFz(avg), CMx(avg), CMy(avg), CMz(avg)\n"
        )

        # for example let's vary alpha:
        alphas = [-10, 12]

        for alpha_angle in alphas:
            # modify the alpha
            params.operating_condition.alpha = alpha_angle * fl.u.deg

            # launch the case
            case = project.run_case(params=params, name=f"{alpha_angle}_case ")
            case_params = f"{case.id}, {params.operating_condition.alpha.value}, {params.operating_condition.beta.value}, "
            log.info(f"The case ID is: {case.id} with {alpha_angle=} ")
            case_list.append(case)
            cases_params.append(case_params)

        # fraction used for averaging over the last % of iterations, 0.1 -> last 10%
        fraction = 0.1

        for case, case_params in zip(case_list, cases_params):
            case.wait()
            results = case.results
            avg_total_forces = results.total_forces.get_averages(fraction).to_dict()
            csv_file.write(
                case_params
                + f'{avg_total_forces["CL"]}, {avg_total_forces["CD"]}, {avg_total_forces["CFx"]}, {avg_total_forces["CFy"]}, {avg_total_forces["CFz"]}, {avg_total_forces["CMx"]}, {avg_total_forces["CMy"]}, {avg_total_forces["CMz"]}\n'
            )

    return case_list


######################################################################################################################
def create_directory(dir_name):
    """

    Create on the local machine the required directories if not already created

    Parameters
    ----------
    dir_name

    Returns
    -------

    """

    dir_path = os.path.join(os.getcwd(), dir_name)
    if os.path.isdir(dir_path):  # if directory already exists.
        overwrite_bool = click.confirm(
            f"Directory '{dir_path}' already exists, continuing might overwrite some of its content, do you want to continue?",
            default=True,
            abort=True,
        )
    else:
        os.makedirs(dir_path)
    return dir_path


######################################################################################################################
def create_mesh_params(project):
    """

    Create the parameters object containing all the information required by the solver to generate the mesh

    Parameters
    ----------
    project

    Returns
    -------
        params object
    """

    geometry = project.geometry
    geometry.group_faces_by_tag("faceName")
    geometry.group_edges_by_tag("edgeName")
    with fl.SI_unit_system:
        params = fl.SimulationParams(
            meshing=fl.MeshingParams(
                defaults=fl.MeshingDefaults(
                    boundary_layer_first_layer_thickness=1e-5, surface_max_edge_length=1
                ),
                volume_zones=[fl.AutomatedFarfield()],
                refinements=[
                    fl.SurfaceEdgeRefinement(
                        name="leading_edges",
                        edges=[geometry["leadingEdge"]],
                        method=fl.AngleBasedRefinement(value=2 * fl.u.deg),
                    )
                ],
            )
        )
    return params


def assign_wall(project):
    if project.project_tree.root.asset_type == "Geometry":
        geo = project.geometry
        geo.group_faces_by_tag("faceName")
        models = [
            fl.Wall(
                name="NoSlipWall",
                surfaces=[
                    geo["fuselage"],
                    # *_pylon will select all boundaries ending with _pylon
                    geo["*_pylon"],
                    geo["left_wing"],
                    geo["right_wing"],
                    geo["h_tail"],
                    geo["v_tail"],
                ],
            ),
            fl.Freestream(surfaces=fl.AutomatedFarfield().farfield, name="farfield"),
        ]
    else:
        vm = project.volume_mesh
        models = [
            fl.Wall(
                name="NoSlipWall",
                surfaces=[
                    vm["fluid/fuselage"],
                    # *_pylon will select all boundaries ending with _pylon
                    vm["fluid/*_pylon"],
                    vm["fluid/left_wing"],
                    vm["fluid/right_wing"],
                    vm["fluid/h_tail"],
                    vm["fluid/v_tail"],
                ],
            ),
            fl.Freestream(surfaces=vm["fluid/farfield"], name="farfield"),
        ]
    return models


def project_from_volume_mesh():
    EVTOL.get_files()
    project_name = "sweep_evtol_from_mesh"
    project = fl.Project.from_volume_mesh(
        EVTOL.mesh_filename,  # mesh could also be ugrid format
        name=project_name,
        length_unit="m",  # length_unit should be 'm', 'mm', 'cm', 'inch' or 'ft'
    )

    return project


def project_from_geometry():
    EVTOL.get_files()
    project_name = "sweep_evtol_from_geometry"
    project = fl.Project.from_geometry(EVTOL.geometry, name=project_name)
    mesh_params = create_mesh_params(project)
    project.generate_volume_mesh(params=mesh_params, name="mesh_name")  # generate the volume mesh

    return project


######################################################################################################################
def main():
    """
    Main function that drives the functions to achieve all the steps mentioned at the top.
    """

    # Step1: Connect to an existing project and volume mesh.
    # Chose one of the three options below

    # Option 1a: if you want to upload a new mesh and create a new project.
    project = project_from_volume_mesh()

    # Option 1b: if you want to upload a CAD geometry and create a new project.
    # project = project_from_geometry()

    # Option 1c: if you want to run from an existing project.
    # project = fl.Project.from_cloud(
    #     'prj-XXXXXXXXXX')  # where prj-XXXXXXXXXX is an ID that can be saved from a previously created project or read off the WEBUI

    project_name = project.metadata.name

    vm = project.volume_mesh  # get the volume mesh entity associated with that project.
    # if the project has more then one mesh then use this line below instead.
    # vm = project.get_volume_mesh(asset_id='vm-XXXXXXXXXXXXXXX')  # get the specific volume mesh entity in that project we want to use.

    log.info(f"The project id is {project.id}")
    log.info(f"The volume mesh contains the following boundaries:{vm.boundary_names}")
    log.info(f"The volume mesh ID is: {vm.id}")

    # step 2: create the directories to locally store relevant data
    dir_name = "evtol_alpha_sweep"
    dir_path = create_directory(dir_name)

    # step3: launch the cases and save the relevant data
    models = assign_wall(project)
    params = make_run_params(vm, models)  # define the run params used to launch the run
    cases = launch_sweep(params, project, project_name, vm, dir_path)  # launch a sweep

    generate_report(
        cases,
        params,
        include_geometry=True,
        include_general_tables=True,
        include_residuals=True,
        include_cfl=True,
        include_forces_moments_table=True,
        include_forces_moments_charts=True,
        include_cf_vec=True,
        include_cp=True,
        include_yplus=True,
        include_qcriterion=True,
    )


######################################################################################################################

if __name__ == "__main__":
    main()
