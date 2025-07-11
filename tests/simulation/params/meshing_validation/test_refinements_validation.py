import pydantic as pd
import pytest
import re

from flow360.component.simulation.meshing_param.params import ModularMeshingWorkflow
from flow360.component.simulation.meshing_param.surface_mesh_refinements import (
    SnappyBodyRefinement,
    SnappySurfaceEdgeRefinement,
    SnappyRegionRefinement
)
from flow360.component.simulation.primitives import Surface
from flow360.component.simulation.unit_system import SI_unit_system
import flow360.component.simulation.units as u



def test_snappy_refinements_validators():
    message = "Minimum spacing must be lower than maximum spacing."
    with SI_unit_system, pytest.raises(ValueError, match=re.escape(message)):
        SnappyRegionRefinement(
            min_spacing=4.3 * u.mm, 
            max_spacing=2.1 * u.mm,
            regions=[Surface(name="test")]
        )
    