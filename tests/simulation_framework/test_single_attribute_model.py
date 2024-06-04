from typing import Union

import pydantic as pd

from flow360.component.simulation.framework.single_attribute_base import (
    SingleAttributeModel,
)


class MyTestClass(SingleAttributeModel):
    value: Union[pd.StrictFloat, pd.StrictStr] = pd.Field()


def test_single_attribute_model():
    a = MyTestClass(1.0)
    assert a.value == 1.0

    a = MyTestClass(value=2.0)
    assert a.value == 2.0
