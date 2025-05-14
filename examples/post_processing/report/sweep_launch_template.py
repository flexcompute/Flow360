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

import pandas as pd
from sweep_launch_report import csv_reader, generate_report

import flow360 as fl
from flow360 import u
from flow360.examples import EVTOL

# Variables we want to export in our volume and surface solution files. Many more are available.
vol_fields = ["Mach", "Cp", "mut", "mutRatio", "primitiveVars", "qcriterion"]
surf_fields = ["Cp", "yPlus", "Cf", "CfVec", "primitiveVars", "wallDistance"]

velocity_magnitude = 100 * u.m / u.s


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
                moment_center=(0, 0, 0) * u.m, moment_length=1 * u.m, area=1 * u.m * u.m
            ),
            operating_condition=fl.AerospaceCondition(
                velocity_magnitude=velocity_magnitude, alpha=0 * u.deg
            ),
            time_stepping=fl.Steady(max_steps=5000, CFL=fl.AdaptiveCFL()),
            models=[
                *models,
                fl.Fluid(
                    navier_stokes_solver=fl.NavierStokesSolver(),
                    turbulence_model_solver=fl.SpalartAllmaras(),
                ),
            ],
            outputs=[
                fl.VolumeOutput(output_format="tecplot", output_fields=vol_fields),
                # mesh_object['*'] will select all the boundaries in the mesh and export the surface results.
                # Regular expressions can be used to filter for certain boundaries.
                fl.SurfaceOutput(
                    surfaces=[mesh_object["*"]], output_fields=surf_fields, output_format="tecplot"
                ),
            ],
        )

    # Add meshing params in case the project starts from geometry.
    params.meshing = mesh_object.params.meshing

    return params


######################################################################################################################
def launch_sweep(params, project, mesh_object, dir_path):
    """
    Launch a sweep of cases.
    """

    case_list = []

    # Create the csv file where we will store all relevant sweep data.
    os.makedirs(dir_path, exist_ok=True)
    csv_path = os.path.join(dir_path, "sweep_saved_data.csv")
    general_info = {
        "Sweep name": os.path.basename(dir_path),
        "Project name": project.metadata.name,
        "Project ID": project.id,
        "Volume mesh ID": mesh_object.id,
        "Velocity magnitude": velocity_magnitude,
    }
    df = pd.DataFrame.from_dict([general_info])
    df.to_csv(csv_path, index=False)

    # For example let's vary alpha:
    alphas = [-10, -5, 0, 5, 10, 12, 14] * u.deg

    cases_params = []
    for i, alpha_angle in enumerate(alphas):
        # modify the alpha
        params.operating_condition.alpha = alpha_angle

        case = project.run_case(params=params, name=f"alpha_{alpha_angle.value}_case")
        data = {
            "case_id": case.id,
            "alpha(deg)": params.operating_condition.alpha.value,
            "beta(deg)": params.operating_condition.beta.value,
        }

        print(f"The case ID is: {case.id} with {alpha_angle=} ")
        case_list.append(case)
        cases_params.append(data)

    # fraction used for averaging over the last % of iterations, 0.1 -> last 10%
    fraction = 0.1

    cases_forces = []
    for i, case in enumerate(case_list):
        case.wait()
        results = case.results
        avg_total_forces = results.total_forces.get_averages(fraction).to_dict()
        forces = {
            "CL(avg)": avg_total_forces["CL"],
            "CD(avg)": avg_total_forces["CD"],
            "CFx(avg)": avg_total_forces["CFx"],
            "CFy(avg)": avg_total_forces["CFy"],
            "CFz(avg)": avg_total_forces["CFz"],
            "CMx(avg)": avg_total_forces["CMx"],
            "CMy(avg)": avg_total_forces["CMy"],
            "CMz(avg)": avg_total_forces["CMz"],
        }
        cases_forces.append(forces)

    df_data = pd.DataFrame.from_dict(cases_params)
    df_forces = pd.DataFrame.from_dict(cases_forces)
    df = df_data.join(df_forces)
    df.to_csv(csv_path, index=False, mode="a")

    return csv_path


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
                        method=fl.AngleBasedRefinement(value=2 * u.deg),
                    )
                ],
            )
        )
    return params


def assign_boundary_conditions(project):

    if project.project_tree.root.asset_type == "Geometry":
        geo = project.geometry
        geo.group_faces_by_tag("faceName")
        models = [
            fl.Wall(
                surfaces=[
                    # "*" will select all geometry boundaries
                    geo["*"],
                ],
            ),
            fl.Freestream(surfaces=fl.AutomatedFarfield().farfield),
        ]
    else:
        vm = project.volume_mesh
        models = [
            fl.Wall(
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
            fl.Freestream(surfaces=vm["fluid/farfield"]),
        ]
    return models


def project_from_volume_mesh():
    EVTOL.get_files()
    project_name = "sweep_evtol_from_mesh"
    project = fl.Project.from_volume_mesh(
        EVTOL.mesh_filename,  # mesh could also be in ugrid format
        name=project_name,
        length_unit="m",  # length_unit should be 'm', 'mm', 'cm', 'inch' or 'ft'
    )

    return project


def project_from_geometry():
    EVTOL.get_files()
    project_name = "sweep_evtol_from_geometry"
    project = fl.Project.from_geometry(EVTOL.geometry, name=project_name)
    mesh_params = create_mesh_params(project)
    project.generate_volume_mesh(params=mesh_params, name="mesh_name")

    return project


######################################################################################################################
def main():
    """
    Main function that drives the functions to achieve all the steps mentioned at the top.
    """

    # Step1: Connect to an existing project and volume mesh.
    # Chose one of two options below

    # Option 1a: If you want to upload a new mesh and create a new project.
    project = project_from_volume_mesh()

    # Option 1b: If you want to upload a CAD geometry and create a new project.
    # project = project_from_geometry()

    # Option 1c: if you want to run from an existing project.
    # project = fl.Project.from_cloud(
    #     'prj-XXXXXXXXXXX')  # where prj-XXXXXXXXXX is an ID that can be saved from a previously created project or read off the WEBUI

    vm = project.volume_mesh
    # If the project has more then one mesh then use hte line below to choose a specific mesh instead.
    # vm = project.get_volume_mesh(asset_id='vm-XXXXXXXXXXXXXXX')

    print(f"The project id is {project.id}")
    print(f"The volume mesh contains the following boundaries:{vm.boundary_names}")
    print(f"The volume mesh ID is: {vm.id}")

    # Step 2: Create the directories to locally store relevant data.
    dir_name = "evtol_alpha_sweep"

    # Step3: Launch the cases and save the relevant data.
    models = assign_boundary_conditions(project)
    params = make_run_params(vm, models)
    csv_path = launch_sweep(params, project, vm, dir_name)

    generate_report(
        *csv_reader(csv_path),
        include_geometry=True,
        include_general_tables=True,
        include_residuals=True,
        include_cfl=True,
        include_forces_moments_table=True,
        include_forces_moments_charts=True,
        include_forces_moments_alpha_charts=True,
        include_forces_moments_beta_charts=True,
        include_cf_vec=True,
        include_cp=True,
        include_yplus=True,
        include_qcriterion=True,
    )


######################################################################################################################
if __name__ == "__main__":
    main()
