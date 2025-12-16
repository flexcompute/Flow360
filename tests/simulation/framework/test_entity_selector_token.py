import copy

import pytest

from flow360.component.simulation.entity_info import GeometryEntityInfo
from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.framework.entity_expansion_utils import (
    expand_all_entity_lists_in_place,
)
from flow360.component.simulation.framework.entity_selector import (
    EntitySelector,
    collect_and_tokenize_selectors_in_place,
)
from flow360.component.simulation.framework.param_utils import AssetCache
from flow360.component.simulation.models.surface_models import Wall
from flow360.component.simulation.primitives import Surface
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.units import SI_unit_system


def test_entity_selector_token_flow():
    # 1. Setup input dictionary with repeated selectors
    with SI_unit_system:
        selector = Surface.matches("wing*", name="sel1")
        params = SimulationParams(
            models=[
                Wall(name="m1", entities=[selector]),
                Wall(name="m2", entities=[selector]),
            ],
            private_attribute_asset_cache=AssetCache(
                project_entity_info=GeometryEntityInfo(boundaries=[])
            ),
        )
    # params = {
    #     "private_attribute_asset_cache": {
    #         # Mock entity info for database
    #         "project_entity_info": {"type_name": "GeometryEntityInfo", "boundaries": []}
    #     },
    #     "models": [
    #         {
    #             "name": "m1",
    #             "selectors": [
    #                 {
    #                     "selector_id": "sel1-token",
    #                     "target_class": "Surface",
    #                     "name": "sel1",
    #                     "children": [
    #                         {"attribute": "name", "operator": "matches", "value": "wing*"}
    #                     ],
    #                 }
    #             ],
    #         },
    #         {
    #             "name": "m2",
    #             "selectors": [
    #                 {
    #                     "selector_id": "sel1-token",
    #                     "target_class": "Surface",
    #                     "name": "sel1",
    #                     "children": [
    #                         {"attribute": "name", "operator": "matches", "value": "wing*"}
    #                     ],
    #                 }
    #             ],
    #         },
    #     ],
    # }

    # 2. Run tokenization
    tokenized_params = collect_and_tokenize_selectors_in_place(params.model_dump(mode="json"))

    # 3. Verify AssetCache and tokens
    asset_cache = tokenized_params["private_attribute_asset_cache"]
    assert "used_selectors" in asset_cache
    assert len(asset_cache["used_selectors"]) == 1
    assert asset_cache["used_selectors"][0]["selector_id"] == "sel1-token"
    assert asset_cache["used_selectors"][0]["name"] == "sel1"

    # Check that selectors are replaced by tokens
    assert tokenized_params["models"][0]["selectors"] == ["sel1-token"]
    assert tokenized_params["models"][1]["selectors"] == ["sel1-token"]

    # 4. Resolve selector tokens back to full dicts
    resolved_params = expand_all_entity_lists_in_place(tokenized_params)

    # 5. Verify selectors are resolved from tokens to full dicts (not strings)
    sel1 = resolved_params["models"][0]["selectors"]
    sel2 = resolved_params["models"][1]["selectors"]

    assert len(sel1) == 1
    assert isinstance(sel1[0], dict), "Selector token should be expanded to dict"
    assert sel1[0]["selector_id"] == "sel1-token"
    assert sel1[0]["name"] == "sel1"

    assert len(sel2) == 1
    assert isinstance(sel2[0], dict), "Selector token should be expanded to dict"

    # 7. Verify expanded selectors can be validated as EntitySelector
    EntitySelector.model_validate(sel1[0])
    EntitySelector.model_validate(sel2[0])


def test_entity_selector_token_round_trip_validation():
    params = {
        "private_attribute_asset_cache": {},
        "models": [
            {
                "name": "m1",
                "selectors": [
                    {
                        "selector_id": "sel1-token",
                        "target_class": "Surface",
                        "name": "sel1",
                        "children": [
                            {"attribute": "name", "operator": "matches", "value": "wing*"}
                        ],
                    }
                ],
            },
            {
                "name": "m2",
                "selectors": [
                    {
                        "selector_id": "sel1-token",
                        "target_class": "Surface",
                        "name": "sel1",
                        "children": [
                            {"attribute": "name", "operator": "matches", "value": "wing*"}
                        ],
                    }
                ],
            },
        ],
    }

    tokenized_params = collect_and_tokenize_selectors_in_place(copy.deepcopy(params))

    class _ModelWithSelectors(Flow360BaseModel):
        name: str
        selectors: list

    class _ParamsWithAssetCache(Flow360BaseModel):
        private_attribute_asset_cache: AssetCache
        models: list[_ModelWithSelectors]

    validated = _ParamsWithAssetCache.model_validate(tokenized_params)

    cache = validated.private_attribute_asset_cache
    assert cache.used_selectors is not None
    assert cache.used_selectors[0].selector_id == "sel1-token"
    assert validated.models[0].selectors == ["sel1-token"]
    assert validated.models[1].selectors == ["sel1-token"]


def test_entity_selector_mixed_token_and_dict():
    # Test mixing tokens and dicts
    params = {
        "private_attribute_asset_cache": {
            "used_selectors": [
                {
                    "selector_id": "sel-cache-id",
                    "target_class": "Surface",
                    "name": "sel_cache",
                    "children": [{"attribute": "name", "operator": "matches", "value": "wing*"}],
                }
            ]
        },
        "model": {
            "selectors": [
                "sel-cache-id",  # Token
                {  # Inline dict
                    "selector_id": "sel-inline-id",
                    "target_class": "Surface",
                    "name": "sel_inline",
                    "children": [{"attribute": "name", "operator": "matches", "value": "fuselage"}],
                },
            ]
        },
    }

    resolved = expand_all_entity_lists_in_place(params)

    # Verify selectors are all dicts after resolution (token resolved, inline kept)
    selectors = resolved["model"]["selectors"]
    assert len(selectors) == 2
    assert all(isinstance(s, dict) for s in selectors), "All selectors should be dicts"
    assert selectors[0]["selector_id"] == "sel-cache-id"  # Token was expanded
    assert selectors[1]["selector_id"] == "sel-inline-id"  # Inline kept as-is

    # Verify both can be validated as EntitySelector
    for sel in selectors:
        EntitySelector.model_validate(sel)


def test_entity_selector_unknown_token_raises_error():
    """Test that referencing an unknown selector token raises a ValueError."""
    params = {
        "private_attribute_asset_cache": {
            "used_selectors": [
                {
                    "selector_id": "known-selector-id",
                    "target_class": "Surface",
                    "name": "known_selector",
                    "children": [{"attribute": "name", "operator": "matches", "value": "wing*"}],
                }
            ]
        },
        "model": {
            "selectors": [
                "unknown-selector-id",  # This token does not exist in used_selectors
            ]
        },
    }

    with pytest.raises(ValueError, match="Selector token 'unknown-selector-id' not found"):
        expand_all_entity_lists_in_place(params)
