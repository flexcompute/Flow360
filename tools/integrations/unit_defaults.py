import json
from typing import Dict

import pydantic.v1 as pd

import flow360.component.v1 as fl
from flow360.component.v1.params_base import Flow360BaseModel


class UnitDefaults(Flow360BaseModel):
    defaults_SI: Dict = pd.Field(alias="defaultsSI", default=fl.SI_unit_system.defaults())
    defaults_CGS: Dict = pd.Field(alias="defaultsCGS", default=fl.CGS_unit_system.defaults())
    defaults_imperial: Dict = pd.Field(
        alias="defaultsImperial", default=fl.imperial_unit_system.defaults()
    )
    defaults_flow360: Dict = pd.Field(
        alias="defaultsFlow360", default=fl.flow360_unit_system.defaults()
    )


with open("./data/UnitDefaults.json", "w") as outfile:
    outfile.write(json.dumps(UnitDefaults().dict(), indent=2))
