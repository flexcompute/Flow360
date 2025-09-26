import os
import re

import pytest

from flow360.plugins.report.report import ReportTemplate
from flow360.plugins.report.report_items import (
    Chart2D,
    Inputs,
    NonlinearResiduals,
    Summary,
    Table,
)
from flow360.plugins.report.utils import (
    Average,
    DataItem,
    Delta,
    RequirementItem,
    get_root_path,
)


@pytest.fixture
def here():
    return os.path.dirname(os.path.abspath(__file__))


def test_get_root_path():
    dataitem = DataItem(data="total_forces/CL")
    delta = Delta(data="total_forces/CL")
    delta_di = Delta(data=DataItem(data="total_forces/CL", operations=[Average(fraction=0.5)]))

    assert get_root_path(dataitem) == "total_forces/CL"
    assert get_root_path(delta) == "total_forces/CL"
    assert get_root_path(delta_di) == "total_forces/CL"


def test_reporttemplate_requirements():
    template = ReportTemplate(
        items=[
            Summary(),  # no requirements
            Inputs(),  # has params requirements
            Table(data=["total_forces/CL", "volume_mesh/stats"], section_title="misc"),
            Chart2D(
                x="params/version", y="y_slicing_force_distribution/Y"
            ),  # y_slicing_force_distribution, total_forces
        ]
    )
    reqs = template.get_requirements()
    expected_keys = [
        "params",
        "y_slicing_force_distribution",
        "total_forces",
        "volume_mesh",
        "surface_mesh",
        "geometry",
        "volume_mesh/stats",
    ]
    expected_reqs = {RequirementItem.from_data_key(data_key=k) for k in expected_keys}
    assert set(reqs) == expected_reqs

    template_advanced = ReportTemplate(
        items=[
            Inputs(),  # has params requirements
            Table(data=["total_forces/CL"], section_title="Forces"),  # total_forces,
            NonlinearResiduals(),
            Chart2D(
                x="results/monitors/massFluxExhaust/pseudo_step",
                y="results/monitors/massFluxExhaust/MassFlux",
            ),
            Chart2D(
                x="results/user_defined_dynamics/alpha-controller/pseudo_step",
                y="results/user_defined_dynamics/alpha-controller/alpha",
            ),
            Chart2D(x="bet_forces/pseudo_step", y="bet_forces/Disk0_Force_y"),
            Chart2D(x="bet_forces/pseudo_step", y="bet_forces/Disk0_Force_y"),
            Chart2D(
                x="bet_forces_radial_distribution/Disk0_All_Radius",
                y="bet_forces_radial_distribution/Disk0_Blade0_All_TorqueCoeff",
            ),
            Chart2D(x="actuator_disks/physical_step", y="actuator_disks/Disk0_Power"),
            Chart2D(x="aeroacoustics/physical_step", y="aeroacoustics/var1"),
            Chart2D(x="surface_heat_transfer/physical_step", y="surface_heat_transfer/var1"),
        ]
    )
    reqs = template_advanced.get_requirements()
    expected_filepaths = [
        "results/total_forces_v2.csv",
        "results/nonlinear_residual_v2.csv",
        "simulation.json",
        "results/bet_forces_radial_distribution_v2.csv",
        "results/bet_forces_v2.csv",
        "results/actuatorDisk_output_v2.csv",
        "results/total_acoustics_v3.csv",
        "results/surface_heat_transfer_v2.csv",
        "results/monitor_massFluxExhaust_v2.csv",
        "results/udd_alpha-controller_v2.csv",
    ]
    expected_reqs = {
        RequirementItem(filename=fp, resource_type="case") for fp in expected_filepaths
    }
    expected_reqs.add(RequirementItem(filename="simulation.json", resource_type="surface_mesh"))
    expected_reqs.add(RequirementItem(filename="simulation.json", resource_type="geometry"))
    expected_reqs.add(RequirementItem(filename="simulation.json", resource_type="volume_mesh"))

    assert set(reqs) == expected_reqs
