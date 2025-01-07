import re

import pytest

from flow360.component.simulation.meshing_param.params import MeshingParams
from flow360.component.simulation.meshing_param.volume_params import AutomatedFarfield


def test_automated_farfield_import_export():

    my_farfield = AutomatedFarfield(name="my_farfield")
    model_as_dict = my_farfield.model_dump()

    model_as_dict = {"name": "my_farfield", "method": "auto"}
    my_farfield = AutomatedFarfield(**model_as_dict)

    model_as_dict = {"name": "my_farfield"}
    my_farfield = AutomatedFarfield(**model_as_dict)

    with pytest.raises(
        ValueError,
        match=re.escape("Unable to extract tag using discriminator 'type'"),
    ):
        MeshingParams(**{"volume_zones": [model_as_dict]})

    model_as_dict = {"name": "my_farfield", "type": "AutomatedFarfield"}
    meshing = MeshingParams(**{"volume_zones": [model_as_dict]})
    assert isinstance(meshing.volume_zones[0], AutomatedFarfield)
