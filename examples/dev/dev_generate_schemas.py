import json

import flow360 as fl


def remove_key_from_nested_dict(dictionary, key_to_remove):
    if not isinstance(dictionary, dict):
        raise ValueError("Input must be a dictionary")

    for key, value in list(dictionary.items()):
        if key == key_to_remove:
            del dictionary[key]
        elif isinstance(value, dict):
            remove_key_from_nested_dict(value, key_to_remove)

    return dictionary


def clean_schema(schema):
    remove_key_from_nested_dict(schema, "description")
    remove_key_from_nested_dict(schema, "_type")
    remove_key_from_nested_dict(schema, "comments")


def generate_schema(model):
    schema = model.schema()
    clean_schema(schema)
    json_str = json.dumps(schema, indent=2)

    with open(f"{model.__name__}.json", "w") as outfile:
        outfile.write(json_str)


generate_schema(fl.Geometry)
generate_schema(fl.Freestream)
generate_schema(fl.Boundaries)
generate_schema(fl.SlidingInterface)
generate_schema(fl.NavierStokesSolver)
generate_schema(fl.TurbulenceModelSolverSA)
generate_schema(fl.TurbulenceModelSolverSST)
generate_schema(fl.NoneSolver)
generate_schema(fl.TransitionModelSolver)
generate_schema(fl.HeatEquationSolver)
generate_schema(fl.PorousMedium)
generate_schema(fl.TimeStepping)
generate_schema(fl.ActuatorDisk)
generate_schema(fl.BETDisk)
generate_schema(fl.SurfaceOutput)
generate_schema(fl.SliceOutput)
generate_schema(fl.VolumeOutput)
generate_schema(fl.AeroacousticOutput)
generate_schema(fl.MonitorOutput)
generate_schema(fl.IsoSurfaceOutput)
