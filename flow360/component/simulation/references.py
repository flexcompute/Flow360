from typing import List, Literal, Optional, Tuple, Union

import pydantic as pd


class ReferenceGeometry:
    "Contains all geometrical related refrence values"
    # Only affects output values? Maybe each surface/volume should also have a copy to enable custom moment axis.
    mrc: Tuple[float, float, float] = pd.Field()
    chord = pd.Field()
    span = pd.Field()
    area = pd.Field()
