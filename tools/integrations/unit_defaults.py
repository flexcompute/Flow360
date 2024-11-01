import json
from typing import Dict

import pydantic.v1 as pd

from flow360.component.v1.params_base import Flow360BaseModel

from flow360.component.v1.unit_system import (
    CGS_unit_system,
    SI_unit_system,
    flow360_unit_system,
    imperial_unit_system,
)

class UnitDefaults(Flow360BaseModel):
    defaults_SI: Dict = pd.Field(alias="defaultsSI", default=SI_unit_system.defaults())
    defaults_CGS: Dict = pd.Field(alias="defaultsCGS", default=CGS_unit_system.defaults())
    defaults_imperial: Dict = pd.Field(
        alias="defaultsImperial", default=imperial_unit_system.defaults()
    )
    defaults_flow360: Dict = pd.Field(
        alias="defaultsFlow360", default=flow360_unit_system.defaults()
    )


with open("./data/UnitDefaults.json", "w") as outfile:
    outfile.write(json.dumps(UnitDefaults().dict(), indent=2))
