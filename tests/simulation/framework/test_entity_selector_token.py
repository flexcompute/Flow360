import copy

from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.framework.entity_selector import (
    EntityDictDatabase,
    collect_and_tokenize_selectors_in_place,
    expand_entity_selectors_in_place,
)
from flow360.component.simulation.framework.param_utils import AssetCache


def test_entity_selector_token_flow():
    # 1. Setup input dictionary with repeated selectors
    params = {
        "private_attribute_asset_cache": {
            # Mock entity info for database
            "project_entity_info": {"type_name": "GeometryEntityInfo", "boundaries": []}
        },
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

    # Mock database
    db = EntityDictDatabase(
        surfaces=[
            {"name": "wing_left", "private_attribute_entity_type_name": "Surface"},
            {"name": "wing_right", "private_attribute_entity_type_name": "Surface"},
            {"name": "fuselage", "private_attribute_entity_type_name": "Surface"},
        ]
    )

    # 2. Run tokenization
    tokenized_params = collect_and_tokenize_selectors_in_place(copy.deepcopy(params))

    # 3. Verify AssetCache and tokens
    asset_cache = tokenized_params["private_attribute_asset_cache"]
    assert "used_selectors" in asset_cache
    assert len(asset_cache["used_selectors"]) == 1
    assert asset_cache["used_selectors"][0]["selector_id"] == "sel1-token"
    assert asset_cache["used_selectors"][0]["name"] == "sel1"

    # Check that selectors are replaced by tokens
    assert tokenized_params["models"][0]["selectors"] == ["sel1-token"]
    assert tokenized_params["models"][1]["selectors"] == ["sel1-token"]

    # 4. Run expansion
    expanded_params = expand_entity_selectors_in_place(db, tokenized_params)

    # 5. Verify expansion results
    s1 = expanded_params["models"][0].get("stored_entities", [])
    s2 = expanded_params["models"][1].get("stored_entities", [])

    # Only 2 surfaces match "wing*"
    assert len(s1) == 2
    assert {e["name"] for e in s1} == {"wing_left", "wing_right"}

    assert len(s2) == 2
    assert {e["name"] for e in s2} == {"wing_left", "wing_right"}


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
    assert cache.used_selectors[0]["selector_id"] == "sel1-token"
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

    db = EntityDictDatabase(
        surfaces=[
            {"name": "wing_left", "private_attribute_entity_type_name": "Surface"},
            {"name": "fuselage", "private_attribute_entity_type_name": "Surface"},
        ]
    )

    expanded = expand_entity_selectors_in_place(db, params)
    stored = expanded["model"].get("stored_entities", [])

    names = {e["name"] for e in stored}
    assert "wing_left" in names
    assert "fuselage" in names
    assert len(names) == 2
