from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.validation import validation_context

from typing import Optional
import pydantic as pd



    

class Model(Flow360BaseModel):
    a: str
    b: Optional[int] = validation_context.ConditionallyRequiredField(required_for=validation_context.SURFACE_MESH)
    c: Optional[str] = validation_context.ConditionallyRequiredField(required_for=validation_context.VOLUME_MESH)
    d: Optional[float] = validation_context.ConditionallyRequiredField(required_for=validation_context.CASE)
    e: Optional[float] = validation_context.ConditionalField(relevant_for=validation_context.CASE)
    
class BaseModel(Flow360BaseModel):
    m: Model
    c: Optional[Model] = validation_context.ConditionallyRequiredField(required_for=validation_context.CASE)


test_data1 = dict(m=dict())
test_data2 = dict(m=dict(a="f", b=1, c="d", d=1.2), c=dict(a="f", b=1, c="d", d=1.2))



try:
   BaseModel(**test_data1)
except pd.ValidationError as err:
    errors = err.errors()
    print(errors)
    assert len(errors) == 1


try:
    with validation_context.ValidationLevelContext(validation_context.SURFACE_MESH):
        BaseModel(**test_data1)
except pd.ValidationError as err:
    errors = err.errors()
    print(errors)
    assert len(errors) == 2


try:
    with validation_context.ValidationLevelContext(validation_context.VOLUME_MESH):
        BaseModel(**test_data1)
except pd.ValidationError as err:
    errors = err.errors()
    print(errors)
    assert len(errors) == 2



try:
    with validation_context.ValidationLevelContext(validation_context.CASE):
        BaseModel(**test_data1)
except pd.ValidationError as err:
    errors = err.errors()
    print(errors)
    assert len(errors) == 3



try:
    with validation_context.ValidationLevelContext(validation_context.ALL):
        BaseModel(**test_data1)
except pd.ValidationError as err:
    errors = err.errors()
    print(errors)
    assert len(errors) == 5


BaseModel(**test_data2)
with validation_context.ValidationLevelContext(validation_context.SURFACE_MESH):
    BaseModel(**test_data2)

with validation_context.ValidationLevelContext(validation_context.VOLUME_MESH):
    BaseModel(**test_data2)

with validation_context.ValidationLevelContext(validation_context.CASE):
    BaseModel(**test_data2)

with validation_context.ValidationLevelContext(validation_context.ALL):
    BaseModel(**test_data2)

