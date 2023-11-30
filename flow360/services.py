import pydantic as pd
from .component.flow360_params.flow360_params import Flow360Params, Geometry, NavierStokesSolver, TurbulenceModelSolverSA, FreestreamFromVelocity



def get_default_params(unit_system_context):

    """
    example of generating default case settings. 
    - Use Model() if all fields has defaults or there are no required fields
    - Use Model.construct() to disable validation - when there are required fields without value

    """


    with unit_system_context:
        params = Flow360Params(
            geometry=Geometry(),
            freestream=FreestreamFromVelocity.construct(),
            navier_stokes_solver=NavierStokesSolver(),
            turbulence_model_solver=TurbulenceModelSolverSA()
            )

    return params



def get_default_retry(params_as_dict):
    params = Flow360Params(**params_as_dict)
    return params


def get_default_fork(params_as_dict):
    params = Flow360Params(**params_as_dict)
    return params



def validate_flow360_params_model(params_as_dict):

    values, fields_set, validation_errors = pd.validate_model(Flow360Params, params_as_dict)
    print(f'{values=}')
    print(f'{fields_set=}')
    print(f'{validation_errors=}')

    # TODO:
    # when validating freestream, errors from all Union options will be returned. Need to reduce number of validation errors:
    # example when provided temperature -1
    # validation_errors=ValidationError(model='Flow360Params', errors=[
    # {'loc': ('freestream', 'Temperature'), 'msg': 'ensure this value is greater than 0', 'type': 'value_error.number.not_gt', 'ctx': {'limit_value': 0}}, 
    # {'loc': ('freestream', 'Reynolds'), 'msg': 'field required', 'type': 'value_error.missing'}, 
    # {'loc': ('freestream', 'Temperature'), 'msg': 'ensure this value is greater than 0', 'type': 'value_error.number.not_gt', 'ctx': {'limit_value': 0}}, 
    # {'loc': ('freestream', 'mu_ref'), 'msg': 'extra fields not permitted', 'type': 'value_error.extra'}, 
    # {'loc': ('freestream', 'velocity'), 'msg': 'field required', 'type': 'value_error.missing'}, 
    # {'loc': ('freestream', 'Mach'), 'msg': 'extra fields not permitted', 'type': 'value_error.extra'}, 
    # {'loc': ('freestream', 'mu_ref'), 'msg': 'extra fields not permitted', 'type': 'value_error.extra'}, 
    # {'loc': ('freestream', 'temperature'), 'msg': 'extra fields not permitted', 'type': 'value_error.extra'}, 
    # {'loc': ('freestream', 'Mach'), 'msg': 'unexpected value; permitted: 0', 'type': 'value_error.const', 'ctx': {'given': 0.5, 'permitted': (0,)}}, 
    # {'loc': ('freestream', 'Mach_ref'), 'msg': 'field required', 'type': 'value_error.missing'}, 
    # {'loc': ('freestream', 'Temperature'), 'msg': 'ensure this value is greater than 0', 'type': 'value_error.number.not_gt', 'ctx': {'limit_value': 0}}, 
    # {'loc': ('freestream', 'velocity_ref'), 'msg': 'field required', 'type': 'value_error.missing'}, {'loc': ('freestream', 'Mach'), 'msg': 'extra fields not permitted', 'type': 'value_error.extra'}, 
    # {'loc': ('freestream', 'mu_ref'), 'msg': 'extra fields not permitted', 'type': 'value_error.extra'}, {'loc': ('freestream', 'temperature'), 'msg': 'extra fields not permitted', 'type': 'value_error.extra'}])
    

    validation_warnings = []

    if validation_errors is not None:
        return validation_errors.errors(), validation_warnings

    return None, validation_warnings