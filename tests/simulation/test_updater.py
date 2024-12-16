import pytest

import flow360 as fl


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


def test_updater_to_24_11_1():
    files = ["simulation_24_11_0.json"]

    for file in files:
        params = fl.SimulationParams(f"../data/simulation/{file}")
        assert params
