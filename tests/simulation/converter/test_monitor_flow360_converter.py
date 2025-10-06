import json
import os
import re
from tempfile import NamedTemporaryFile

import pytest

import flow360.component.simulation.units as u
from flow360.component.simulation.framework.updater_utils import compare_values
from flow360.component.simulation.migration.ProbeOutput import read_all_v0_monitors
from flow360.component.simulation.outputs.outputs import ProbeOutput
from flow360.component.simulation.simulation_params import SimulationParams


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


def test_flow360_monitor_convert():

    monitor_list = read_all_v0_monitors(
        file_path="./data/monitor_flow360.json",
        mesh_unit=u.m,
    )
    assert isinstance(monitor_list, list)
    assert len(monitor_list) == 2
    assert len(monitor_list[0].entities.stored_entities) == 1
    assert len(monitor_list[1].entities.stored_entities) == 3
    assert monitor_list[0].output_fields.items == ["primitiveVars"]
    assert monitor_list[1].output_fields.items == ["mut"]
    assert all([isinstance(item, ProbeOutput) for item in monitor_list])

    with u.SI_unit_system:
        params = SimulationParams(outputs=[*monitor_list])

    params_dict = params.model_dump(
        mode="json",
        exclude={
            "type_name",
            "private_attribute_constructor",
            "private_attribute_input_cache",
            "private_attribute_dict",
        },
    )
    with open("./ref/ref_monitor.json", mode="r") as fp:
        ref_dict = json.load(fp=fp)
    assert compare_values(params_dict, ref_dict, ignore_keys=["private_attribute_id", "version"])

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Invalid monitor settings: R1 monitor group does not specify any `outputFields`."
        ),
    ):
        read_all_v0_monitors(
            file_path="./data/monitor_flow360_incomplete1.json",
            mesh_unit=u.m,
        )

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Invalid monitor settings: R1 monitor group does not specify any `monitorLocations`."
        ),
    ):
        read_all_v0_monitors(
            file_path="./data/monitor_flow360_incomplete2.json",
            mesh_unit=u.m,
        )
