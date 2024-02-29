"""
Flow360 solver parameters' utility functions
"""


def get_all_output_fields(output_obj):
    """
    Presents all the output fields requested in this output type

    Args:
        output_obj (Union[VolumeOutput, SurfaceOutput, SliceOutput, IsoSurfaceOutput, MonitorOutput]):
            The output object. Can be any of the above.

    Returns:
        set: All the output fields used (at root level/shared and for each output item)
    """
    output_name = output_obj.__class__.__name__
    output_name = (
        output_name[: output_name.find("Output")].lower()
        if output_name.find("Output") != -1
        else output_name
    )
    if output_name == "isosurface":
        output_name = "iso_surface"
    sortable_item_name = output_name + "s"

    all_output_fields = set()
    if output_obj is None:
        return all_output_fields
    shared_output = output_obj.output_fields
    if shared_output is not None:
        all_output_fields.update(shared_output)
    sortable_items = getattr(output_obj, sortable_item_name, None)
    if sortable_items is None:
        return all_output_fields
    for name in sortable_items.names():
        item_output = sortable_items[name].output_fields
        if item_output is None:
            continue
        all_output_fields.update(item_output)
    return all_output_fields
