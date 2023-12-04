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



