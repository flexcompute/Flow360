from __future__ import annotations
from contextlib import contextmanager

import pydantic as pd
from functools import wraps

from typing import Optional, Callable, Type, Literal, Union, List




# requirements for data models with custom constructors:
# 1. data model can be saved to JSON and read back to pydantic model without problems
# 2. data model can return data provided to custom constructor
# 3. data model can be created from JSON that contains only custom constructor inputs - incomplete JSON
# 4. incomplete JSON is not in a conflict with complete JSON (from point 1), such that there is no need for 2 parsers 





@contextmanager
def _model_attribute_unlock(model, attr: str):
    try:
        model.model_fields[attr].frozen = False
        yield
    finally:
        model.model_fields[attr].frozen = True


def model_constructor(func: Callable) -> Callable:
    @classmethod
    @wraps(func)
    def wrapper(cls: Type[pd.BaseModel], *args, **kwargs):
        result = func(cls, *args, **kwargs)
        result.input_kwargs = kwargs
        with _model_attribute_unlock(result, 'constructor'):
            result.constructor = func.__name__
        return result
    return wrapper





class MyModelA(pd.BaseModel):
    type: Literal['MyModelA'] = 'MyModelA'
    constructor: Literal['default', 'from_b'] = pd.Field('default', frozen=True)

    a: Optional[float]
    input_kwargs : dict = dict()


    # computed field cannot be used as discriminator
    # @pd.computed_field
    # def type(self) -> str:
    #     return self.__class__.__name__

    @model_constructor
    def from_b(cls, b, **kwargs) -> MyModelA:
        # conversion ...
        a = b + 2
        return cls(a=a, **kwargs)

    @pd.computed_field
    def b(self) -> float:
        return self.input_kwargs.get('b', None)




class MyModelB(pd.BaseModel):
    type: Literal['MyModelB'] = 'MyModelB'
    constructor: Literal['default', 'from_b'] = pd.Field('default', frozen=True)

    a: Optional[float]
    input_kwargs : dict = dict()

    @model_constructor
    def from_b(cls, b, **kwargs) -> MyModelB:
        # conversion ...
        a = b + 2
        return cls(a=a, **kwargs)

    @pd.computed_field
    def b(self) -> float:
        return self.input_kwargs.get('b', None)



class MyModel(pd.BaseModel):
    type: Literal['MyModel'] = 'MyModel'    
    constructor: Literal['default', 'from_c'] = pd.Field('default', frozen=True)
    a_or_b: Union[MyModelA, MyModelB] = pd.Field(discriminator='type')
    list_a_or_b: List[Union[MyModelA, MyModelB]] = []

    input_kwargs : dict = dict()
    
    @model_constructor
    def from_c(cls, c, **kwargs):
        return cls(a_or_b=c, **kwargs)




ma = MyModelA(a=2)
mb = MyModelA.from_b(b=0)

ma_dict = ma.model_dump()
mb_dict = mb.model_dump()


print(ma_dict)
# >>> {'type': 'MyModelA', 'constructor': 'default', 'a': 2.0, 'input_kwargs': {}, 'b': None}
print(mb_dict)
# >>> {'type': 'MyModelA', 'constructor': 'from_b', 'a': 2.0, 'input_kwargs': {'b': 0}, 'b': 0}


ma_reread = MyModelA(**ma_dict)
print(ma_reread.model_dump())
# >>> {'type': 'MyModelA', 'constructor': 'default', 'a': 2.0, 'input_kwargs': {}, 'b': None}

mb_reread = MyModelA(**mb_dict)
print(mb_reread.model_dump())
# >>> {'type': 'MyModelA', 'constructor': 'from_b', 'a': 2.0, 'input_kwargs': {'b': 0}, 'b': 0}





m = MyModel(a_or_b=mb_reread, list_a_or_b=[MyModelA(a=3), MyModelB.from_b(b=4)])

m_dict = m.model_dump()

print(repr(m))
# >>> MyModel(type='MyModel', a_or_b=MyModelA(type='MyModelA', constructor='from_b', a=2.0, input_kwargs={'b': 0}, b=0), list_a_or_b=[MyModelA(type='MyModelA', constructor='default', a=3.0, input_kwargs={}, b=None), MyModelB(type='MyModelB', constructor='from_b', a=6.0, input_kwargs={'b': 4}, b=4)])
print(m_dict)
# >>> {'type': 'MyModel', 'a_or_b': {'type': 'MyModelA', 'constructor': 'from_b', 'a': 2.0, 'input_kwargs': {'b': 0}, 'b': 0}, 'list_a_or_b': [{'type': 'MyModelA', 'constructor': 'default', 'a': 3.0, 'input_kwargs': {}, 'b': None}, {'type': 'MyModelB', 'constructor': 'from_b', 'a': 6.0, 'input_kwargs': {'b': 4}, 'b': 4}]}



m = MyModel.from_c(c=mb_reread)

m_dict = m.model_dump()

print(repr(m))
# >>> MyModel(type='MyModel', constructor='from_c', a_or_b=MyModelA(type='MyModelA', constructor='from_b', a=2.0, input_kwargs={'b': 0}, b=0), list_a_or_b=[], input_kwargs={'c': MyModelA(type='MyModelA', constructor='from_b', a=2.0, input_kwargs={'b': 0}, b=0)})
print(m_dict)
# >>> {'type': 'MyModel', 'constructor': 'from_c', 'a_or_b': {'type': 'MyModelA', 'constructor': 'from_b', 'a': 2.0, 'input_kwargs': {'b': 0}, 'b': 0}, 'list_a_or_b': [], 'input_kwargs': {'c': {'type': 'MyModelA', 'constructor': 'from_b', 'a': 2.0, 'input_kwargs': {'b': 0}, 'b': 0}}}

# the above examples satisfies points 1 and 2





def get_class_method(cls, method_name):
    """
    Retrieve a class method by its name.

    Parameters
    ----------
    cls : type
        The class containing the method.
    method_name : str
        The name of the method as a string.

    Returns
    -------
    method : callable
        The class method corresponding to the method name.

    Raises
    ------
    AttributeError
        If the method_name is not a callable method of the class.

    Examples
    --------
    >>> class MyClass:
    ...     @classmethod
    ...     def my_class_method(cls):
    ...         return "Hello from class method!"
    ...
    >>> method = get_class_method(MyClass, "my_class_method")
    >>> method()
    'Hello from class method!'
    """
    method = getattr(cls, method_name)
    if not callable(method):
        raise AttributeError(f"{method_name} is not a callable method of {cls.__name__}")
    return method




def get_class_by_name(class_name):
    """
    Retrieve a class by its name from the global scope.

    Parameters
    ----------
    class_name : str
        The name of the class as a string.

    Returns
    -------
    cls : type
        The class corresponding to the class name.

    Raises
    ------
    NameError
        If the class_name is not found in the global scope.
    TypeError
        If the found object is not a class.

    Examples
    --------
    >>> class MyClass:
    ...     pass
    ...
    >>> cls = get_class_by_name("MyClass")
    >>> cls
    <class '__main__.MyClass'>
    """
    cls = globals().get(class_name)
    if cls is None:
        raise NameError(f"Class '{class_name}' not found in the global scope.")
    if not isinstance(cls, type):
        raise TypeError(f"'{class_name}' found in global scope, but it is not a class.")
    return cls





def model_custom_constructor_parser(model_as_dict):
    constructor_name = model_as_dict.get('constructor', None)
    if constructor_name is not None:
        if constructor_name != 'default':
            model_cls = get_class_by_name(model_as_dict.get('type'))
            constructor = get_class_method(model_cls, constructor_name)
            return constructor(**model_as_dict.get('input_kwargs')).model_dump()
    return model_as_dict






def parse_model_dict(model_as_dict) -> dict:
    if isinstance(model_as_dict, dict):
        for key, value in model_as_dict.items():
            model_as_dict[key] = parse_model_dict(value)
        model_as_dict = model_custom_constructor_parser(model_as_dict)
    elif isinstance(model_as_dict, list):
        model_as_dict = [parse_model_dict(item) for item in model_as_dict]
    return model_as_dict






# examples:

# 1. Full model:

data = {'type': 'MyModel', 'a_or_b': {'type': 'MyModelA', 'constructor': 'from_b', 'a': 2.0, 'input_kwargs': {'b': 0}}}

data_parsed = parse_model_dict(data)
print(data_parsed)
model = MyModel(**data_parsed)
print(model.model_dump())
assert model.a_or_b.a == 2



# 2. Incomplete model (from webUI):

data = {'type': 'MyModel', 'a_or_b': {'type': 'MyModelA', 'constructor': 'from_b', 'a': None, 'input_kwargs': {'b': 0}}}

data_parsed = parse_model_dict(data)
print(data_parsed)
model = MyModel(**data_parsed)
print(model.model_dump())
assert model.a_or_b.a == 2


# 3. Incomplete model with list (from webUI):

data = {'type': 'MyModel', 'a_or_b': {'type': 'MyModelA', 'constructor': 'from_b', 'a': 2.0, 'input_kwargs': {'b': 0}, 'b': 0}, 'list_a_or_b': [{'type': 'MyModelA', 'constructor': 'default', 'a': 3.0, 'input_kwargs': {}, 'b': None}, {'type': 'MyModelB', 'constructor': 'from_b', 'a': None, 'input_kwargs': {'b': 4}, 'b': 4}]}
data_parsed = parse_model_dict(data)
print(data_parsed)
model = MyModel(**data_parsed)
print(model.model_dump())
assert model.list_a_or_b[1].a == 6

# 4. Incomplete nested model (from webUI):


data = {'type': 'MyModel', 'constructor': 'from_c', 'a_or_b': None, 'list_a_or_b': [], 'input_kwargs': {'c': {'type': 'MyModelA', 'constructor': 'from_b', 'a': 2.0, 'input_kwargs': {'b': 0}, 'b': 0}}}
data_parsed = parse_model_dict(data)
print(data_parsed)
model = MyModel(**data_parsed)
print(model.model_dump())
assert model.a_or_b.a == 2