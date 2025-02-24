import pytest

from flow360.component.project_utils import _replace_ghost_surfaces
from flow360.component.simulation.primitives import GhostSphere
from flow360.component.simulation.simulation_params import SimulationParams


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


def test_replace_ghost_surfaces():
    params = SimulationParams.from_file("./data/simulation_with_old_ghost_surface.json")
    new_params = _replace_ghost_surfaces(params)
    entity = new_params.models[1].entities.stored_entities[0]
    assert isinstance(entity, GhostSphere)  # This used to be GhostSurface by python
    assert entity.center == [5.0007498695, 0.0, 0.0]
    assert entity.max_radius == 504.16453591327473
