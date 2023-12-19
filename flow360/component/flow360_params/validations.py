"""
validation logic
"""
from ...log import log


def _check_tri_quad_boundaries(values):
    boundaries = values.get("boundaries")
    boundary_names = []
    if boundaries is not None:
        boundary_names = list(boundaries.dict().keys())
    for boundary_name in boundary_names:
        if "/tri_" in boundary_name:
            patch = boundary_name[boundary_name.find("/tri_") + len("/tri_") :]
            quad = boundary_name[0 : boundary_name.find("/tri_")] + "/quad_" + patch
            if quad in boundary_names:
                suggested = boundary_name[0 : boundary_name.find("/tri_") + 1] + patch
                log.warning(
                    f"<{boundary_name}> and <{quad}> found. These may not be valid boundaries and \
                    maybe <{suggested}> should be used instead."
                )
    return values


def _check_duplicate_boundary_name(values):
    boundaries = values.get("boundaries")
    boundary_names = set()
    if boundaries is not None:
        for patch_name, patch_obj in boundaries.dict().items():
            if patch_obj["name"] is not None:
                boundary_name_curr = patch_obj["name"]
            else:
                boundary_name_curr = patch_name
            if boundary_name_curr in boundary_names:
                raise ValueError(
                    f"Boundary name <{boundary_name_curr}> under patch <{patch_name}> appears multiple times."
                )
            boundary_names.add(boundary_name_curr)
    return values
