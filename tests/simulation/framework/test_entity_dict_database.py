"""Client-side parity tests for get_registry_from_params()."""

import copy

from flow360.component.simulation.framework.entity_expansion_utils import (
    get_entity_info_and_registry_from_dict,
    get_registry_from_params,
)
from flow360.component.simulation.framework.entity_registry import EntityRegistry
from flow360.component.simulation.primitives import (
    Edge,
    GenericVolume,
    GeometryBodyGroup,
    Surface,
)


class _AssetCache:
    """Simple object to hold asset cache data."""

    def __init__(self, project_entity_info, selectors, mirror_status=None):
        self.project_entity_info = project_entity_info
        self.selectors = selectors
        self.mirror_status = mirror_status


class _DummyParams:
    """Minimal SimulationParams-like object for database helpers."""

    def __init__(self, params_dict: dict, entity_info_obj=None):
        self._params_dict = copy.deepcopy(params_dict)
        asset_cache_dict = self._params_dict.get("private_attribute_asset_cache", {})
        if entity_info_obj is None:
            entity_info_dict = asset_cache_dict.get("project_entity_info", {})
            # Deserialize entity_info_dict to actual entity_info object
            from flow360.component.simulation.entity_info import parse_entity_info_model

            entity_info_obj = parse_entity_info_model(entity_info_dict)

        selectors = asset_cache_dict.get("selectors")
        mirror_status = None
        mirror_status_dict = asset_cache_dict.get("mirror_status")
        if isinstance(mirror_status_dict, dict) and mirror_status_dict:
            # pylint: disable=import-outside-toplevel
            from flow360.component.simulation.draft_context.mirror import MirrorStatus

            mirror_status = MirrorStatus.deserialize(mirror_status_dict)
        self.private_attribute_asset_cache = _AssetCache(
            project_entity_info=entity_info_obj, selectors=selectors, mirror_status=mirror_status
        )

    def model_dump(self, **kwargs):
        return copy.deepcopy(self._params_dict)


def _entity_names(entries):
    return [entry["name"] if isinstance(entry, dict) else entry.name for entry in entries]


def _build_simple_params_dict():
    return {
        "private_attribute_asset_cache": {
            "project_entity_info": {
                "type_name": "VolumeMeshEntityInfo",
                "boundaries": [
                    {
                        "name": "wall",
                        "private_attribute_entity_type_name": "Surface",
                        "private_attribute_is_interface": False,
                    },
                    {
                        "name": "sym",
                        "private_attribute_entity_type_name": "Surface",
                        "private_attribute_is_interface": False,
                    },
                ],
                "zones": [
                    {"name": "zone-1", "private_attribute_entity_type_name": "GenericVolume"}
                ],
            }
        }
    }


def _build_simple_params_dict_with_mirror_status():
    params_as_dict = _build_simple_params_dict()
    params_as_dict["private_attribute_asset_cache"]["mirror_status"] = {
        "mirror_planes": [
            {
                "name": "plane-1",
                "normal": [0, 1, 0],
                "center": {"value": [0, 0, 0], "units": "m"},
                "private_attribute_entity_type_name": "MirrorPlane",
                "private_attribute_id": "mirror-plane-1",
            }
        ],
        "mirrored_geometry_body_groups": [
            {
                "name": "body-1_<mirror>",
                "geometry_body_group_id": "body-1",
                "mirror_plane_id": "mirror-plane-1",
                "private_attribute_entity_type_name": "MirroredGeometryBodyGroup",
                "private_attribute_id": "mirrored-body-1",
            }
        ],
        "mirrored_surfaces": [
            {
                "name": "wall_<mirror>",
                "surface_id": "wall",
                "mirror_plane_id": "mirror-plane-1",
                "private_attribute_entity_type_name": "MirroredSurface",
                "private_attribute_id": "mirrored-surface-1",
            }
        ],
    }
    return params_as_dict


def test_get_registry_from_params_matches_dict_with_mirror_status():
    params_as_dict = _build_simple_params_dict_with_mirror_status()
    dummy_params = _DummyParams(params_as_dict)

    _, dict_registry = get_entity_info_and_registry_from_dict(params_as_dict)
    instance_registry = get_registry_from_params(dummy_params)

    dict_mirrored_surfaces = dict_registry.find_by_type_name("MirroredSurface")
    instance_mirrored_surfaces = instance_registry.find_by_type_name("MirroredSurface")
    assert _entity_names(dict_mirrored_surfaces) == _entity_names(instance_mirrored_surfaces)

    dict_planes = dict_registry.find_by_type_name("MirrorPlane")
    instance_planes = instance_registry.find_by_type_name("MirrorPlane")
    assert _entity_names(dict_planes) == _entity_names(instance_planes)


def test_get_registry_from_params_matches_dict():
    params_as_dict = _build_simple_params_dict()
    dummy_params = _DummyParams(params_as_dict)

    _, dict_registry = get_entity_info_and_registry_from_dict(params_as_dict)
    instance_registry = get_registry_from_params(dummy_params)

    assert isinstance(instance_registry, EntityRegistry)

    dict_surfaces = dict_registry.find_by_type(Surface)
    instance_surfaces = instance_registry.find_by_type(Surface)
    assert _entity_names(dict_surfaces) == _entity_names(instance_surfaces)

    dict_edges = dict_registry.find_by_type(Edge)
    instance_edges = instance_registry.find_by_type(Edge)
    assert _entity_names(dict_edges) == _entity_names(instance_edges)

    dict_body_groups = dict_registry.find_by_type(GeometryBodyGroup)
    instance_body_groups = instance_registry.find_by_type(GeometryBodyGroup)
    assert _entity_names(dict_body_groups) == _entity_names(instance_body_groups)

    dict_volumes = dict_registry.find_by_type(GenericVolume)
    instance_volumes = instance_registry.find_by_type(GenericVolume)
    assert _entity_names(dict_volumes) == _entity_names(instance_volumes)
