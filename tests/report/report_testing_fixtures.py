from flow360.component.case import CaseMeta
from flow360.component.resource_base import local_metadata_builder
from flow360.component.utils import LocalResourceCache
from flow360.component.volume_mesh import VolumeMeshMetaV2, VolumeMeshV2
import numpy as np
import shutil
import pandas as pd
from flow360 import Case
import pytest
import os

from flow360.plugins.report.report_items import PlotModel


@pytest.fixture
def here():
    return os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def cases(here):

    case_ids = [
        "case-11111111-1111-1111-1111-111111111111",
        "case-2222222222-2222-2222-2222-2222222222",
        "case-333333333-333333-3333333333-33333333",
    ]
    vm_id = "vm-11111111-1111-1111-1111-111111111111"

    cache = LocalResourceCache()

    cases = []
    for case_id in case_ids:
        case_meta = CaseMeta(
            caseId=case_id,
            name=f"{case_id}-name",
            status="completed",
            userId="user-id",
            caseMeshId=vm_id,
            cloud_path_prefix="s3://flow360cases-v1/users/user-id",
        )
        case = Case.from_local_storage(os.path.join(here, "..", "data", case_id), case_meta)
        cases.append(case)

    vm = VolumeMeshV2.from_local_storage(
        mesh_id=vm_id,
        local_storage_path=os.path.join(here, "..", "data", vm_id),
        meta_data=VolumeMeshMetaV2(
            **local_metadata_builder(
                id=vm_id,
                name="DrivAer mesh",
                cloud_path_prefix="s3://flow360meshes-v1/users/user-id",
            )
        ),
    )
    cache.add(vm)

    return cases


def get_cumulative_pseudo_time_step(pseudo_time_step):
    cumulative = []
    last = 0
    for step in pseudo_time_step:
        if (step == 0) and cumulative:
            last = cumulative[-1] + 1
        cumulative.append(step + last)

    return cumulative


def get_last_time_step_values(pseudo_time_step, value_array):
    last_array = []
    for idx, step in enumerate(pseudo_time_step[1:]):
        if step == 0:
            last_array.append(float(value_array[idx]))
    last_array.append(float(value_array[idx + 1]))
    return last_array


@pytest.fixture
def cases_transient(here):

    case_ids = [
        "case-444444444-444444-4444444444-44444444",
        "case-5555-5555555-5555555555-555555555555",
    ]

    vm_id = "vm-22222222-22222222-2222-2222-22222222"

    cache = LocalResourceCache()

    cases = []
    for case_id in case_ids:
        case_meta = CaseMeta(
            caseId=case_id,
            name=f"{case_id}-name",
            status="completed",
            userId="user-id",
            caseMeshId=vm_id,
            cloud_path_prefix="s3://flow360cases-v1/users/user-id",
        )
        case = Case.from_local_storage(os.path.join(here, "..", "data", case_id), case_meta)
        cases.append(case)

    vm = VolumeMeshV2.from_local_storage(
        mesh_id=vm_id,
        local_storage_path=os.path.join(here, "..", "data", vm_id),
        meta_data=VolumeMeshMetaV2(
            **local_metadata_builder(
                id=vm_id,
                name="Cylinder mesh",
                cloud_path_prefix="s3://flow360meshes-v1/users/user-id",
            )
        ),
    )
    cache.add(vm)

    return cases


@pytest.fixture
def residual_plot_model_SA(here):
    residuals_sa = ["0_cont", "1_momx", "2_momy", "3_momz", "4_energ", "5_nuHat"]
    residual_data = pd.read_csv(
        os.path.join(
            here,
            "..",
            "data",
            "case-11111111-1111-1111-1111-111111111111",
            "results",
            "nonlinear_residual_v2.csv",
        ),
        skipinitialspace=True,
    )

    x_data = [list(residual_data["pseudo_step"]) for _ in residuals_sa]
    y_data = [list(residual_data[res]) for res in residuals_sa]

    x_label = "pseudo_step"

    return PlotModel(x_data=x_data, y_data=y_data, x_label=x_label, y_label="none")


@pytest.fixture
def residual_plot_model_SST(here):
    residuals_sst = ["0_cont", "1_momx", "2_momy", "3_momz", "4_energ", "5_k", "6_omega"]
    residual_data = pd.read_csv(
        os.path.join(
            here,
            "..",
            "data",
            "case-333333333-333333-3333333333-33333333",
            "results",
            "nonlinear_residual_v2.csv",
        ),
        skipinitialspace=True,
    )

    x_data = [list(residual_data["pseudo_step"]) for _ in residuals_sst]
    y_data = [list(residual_data[res]) for res in residuals_sst]

    x_label = "pseudo_step"

    return PlotModel(x_data=x_data, y_data=y_data, x_label=x_label, y_label="none")


@pytest.fixture
def two_var_two_cases_plot_model(here, cases):
    loads = ["CL", "CD"]

    x_data = []
    y_data = []
    for case in cases:
        load_data = pd.read_csv(
            os.path.join(here, "..", "data", case.info.id, "results", "total_forces_v2.csv"),
            skipinitialspace=True,
        )

        for load in loads:
            x_data.append(list(load_data["pseudo_step"]))
            y_data.append(list(load_data[load]))

    y_label = "value"
    x_label = "pseudo_step"

    return PlotModel(x_data=x_data, y_data=y_data, x_label=x_label, y_label=y_label)


@pytest.fixture
def cases_beta_sweep(cases, here):
    betas = [0, 2, 4, 6]
    turbulence_models = ["SpalartAllmaras", "kOmegaSST"]
    tags = ["a", "b", "c"]

    tag_multiplier = {
        "a": 1.3,
        "b": 0.8,
        "c": 2.0
    }

    base_case_ids = {"SpalartAllmaras": cases[0].id, "kOmegaSST": cases[2].id}
    vm_id = "vm-11111111-1111-1111-1111-111111111111"

    new_cases = []
    new_case_ids = []

    for beta in betas:
        for turbulence_model in turbulence_models:
            for tag in tags:
                new_case_id = f"case-{'SA1' if turbulence_model=='SpalartAllmaras' else 'SST'}11111-beta{beta}-111-tag{tag}-111111111111"
                new_case_ids.append(new_case_id)
                case_local_storage = os.path.join(here, "..", "data", new_case_id)
                os.mkdir(case_local_storage)
                os.mkdir(os.path.join(case_local_storage, "results"))
                shutil.copyfile(os.path.join(here, "..", "data", base_case_ids[turbulence_model], "results", "nonlinear_residual_v2.csv"), 
                                os.path.join(case_local_storage, "results", "nonlinear_residual_v2.csv"))
                shutil.copyfile(os.path.join(here, "..", "data", base_case_ids[turbulence_model], "simulation.json"), 
                                os.path.join(case_local_storage, "simulation.json"))
                surface_forces = pd.read_csv(os.path.join(here, "..", "data", base_case_ids[turbulence_model], "results", "surface_forces_v2.csv"), skipinitialspace=True)
                total_forces = pd.read_csv(os.path.join(here, "..", "data", base_case_ids[turbulence_model], "results", "total_forces_v2.csv"), skipinitialspace=True)

                surface_forces_to_change = list({*surface_forces} - {"physical_step", "pseudo_step"})
                total_forces_to_change = list({*total_forces} - {"physical_step", "pseudo_step"})

                surface_forces[surface_forces_to_change] = surface_forces[surface_forces_to_change] * (1+(beta/10)) * tag_multiplier[tag]
                total_forces[total_forces_to_change] = total_forces[total_forces_to_change] * (1+(beta/10)) * tag_multiplier[tag]

                surface_forces.to_csv(os.path.join(case_local_storage, "results", "surface_forces_v2.csv"), index=False)
                total_forces.to_csv(os.path.join(case_local_storage, "results", "total_forces_v2.csv"), index=False)

                case_meta = CaseMeta(
                    caseId=new_case_id,
                    name=f"{new_case_id}-name",
                    status="completed",
                    userId="user-id",
                    caseMeshId=vm_id,
                    cloud_path_prefix="s3://flow360cases-v1/users/user-id",
                    tags=[tag]
                )
                case = Case.from_local_storage(os.path.join(here, "..", "data", new_case_id), case_meta)
                new_cases.append(case)

    yield new_cases

    for case_id in new_case_ids:
        shutil.rmtree(os.path.join(here, "..", "data", case_id))

@pytest.fixture
def cases_beta_sweep_example_expected_values(cases_beta_sweep):
    expected_values = pd.DataFrame(columns=["case_id", "CLtotal_avg_0.1", "CDtotal_avg_0.1", "CLtotal_avg_0.2", "CDtotal_avg_0.2"])

    for case in cases_beta_sweep:
        expected_values["case_id"] = case.id
        expected_values["CLtotal_avg_0.1"] = case.results.surface_forces.get_averages(0.1)["totalCL"]
        expected_values["CLtotal_avg_0.2"] = case.results.surface_forces.get_averages(0.2)["totalCL"]
        expected_values["CDtotal_avg_0.1"] = case.results.surface_forces.get_averages(0.1)["totalCD"]
        expected_values["CDtotal_avg_0.2"] = case.results.surface_forces.get_averages(0.2)["totalCD"]

    expected_values.set_index("case_id")

    return expected_values

@pytest.fixture
def expected_y_data(cases_beta_sweep_example_expected_values):
    betas = [0, 2, 4, 6]
    turbulence_models = ["SpalartAllmaras", "kOmegaSST"]
    tags = ["a", "b", "c"]

    cl_data = [[]] * 4
    cd_data = [[]] * 4

    k = 0
    for beta in betas:
        for j, turbulence_model in enumerate(turbulence_models):
            for tag in tags:
                case_id = f"case-{'SA1' if turbulence_model=='SpalartAllmaras' else 'SST'}11111-beta{beta}-111-tag{tag}-111111111111"
                if tag in ["a", "b"]:
                    k = 0
                else:
                    k = 1

            cl_data[j*2+ k] = cases_beta_sweep_example_expected_values.loc[case_id, "totalCL"]
            cd_data[j*2+ k] = cases_beta_sweep_example_expected_values.loc[case_id, "totalCD"]

    return [*cl_data, *cd_data]