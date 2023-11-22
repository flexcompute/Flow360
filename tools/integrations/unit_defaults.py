import json
from typing import Dict

import pydantic as pd
import flow360 as fl

from flow360.component.flow360_params.params_base import Flow360BaseModel


class UnitDefaults(Flow360BaseModel):
    defaults_SI: Dict = pd.Field(alias="defaultsSI", default=fl.SI_unit_system.defaults())
    defaults_CGS: Dict = pd.Field(alias="defaultsCGS", default=fl.CGS_unit_system.defaults())
    defaults_imperial: Dict = pd.Field(alias="defaultsImperial", default=fl.imperial_unit_system.defaults())


with open("./data/UnitDefaults.json", "w") as outfile:
    outfile.write(json.dumps(UnitDefaults().dict(), indent=2))


