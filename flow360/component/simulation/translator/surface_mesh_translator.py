from flow360.component.simulation.translator.utils import preprocess_input


@preprocess_input
def get_surface_mesh_json(input_params):
    """
    Get the surface mesh json from the simulation parameters.
    """
    print(input_params)
