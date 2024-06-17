from __future__ import annotations


from flow360.component.simulation.framework.cached_model_base import MultiConstructorModelBase
from flow360.component.simulation.framework.base_model import Flow360BaseModel
import pydantic as pd
from typing import Optional, Literal






def test_multi_constructor_framework():

    class TempModelB(Flow360BaseModel):
        b: Optional[float] = None


    class MyModelA(MultiConstructorModelBase):
        type: Literal['MyModelA'] = 'MyModelA'
        private_attribute_constructor: Literal['default', 'from_b'] = 'default'

        a: Optional[float]
        private_attribute_input_cache : TempModelB = TempModelB()


        # computed field cannot be used as discriminator
        # @pd.computed_field
        # def type(self) -> str:
        #     return self.__class__.__name__

        @MultiConstructorModelBase.model_constructor
        def from_b(cls, b: float=None, **kwargs) -> MyModelA:
            # conversion ...
            a = b + 2
            return cls(a=a, **kwargs)

        @pd.computed_field
        def b(self) -> float:
            return self.private_attribute_input_cache.b


    ma = MyModelA(a=2)
    mb = MyModelA.from_b(b=0)

    ma_dict = ma.model_dump()
    mb_dict = mb.model_dump()

    print(ma_dict)
    print(mb_dict)
