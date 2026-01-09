import copy

import pytest

from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.framework.entity_materializer import (
    materialize_entities_and_selectors_in_place,
)
from flow360.component.simulation.framework.entity_selector import (
    EntitySelector,
    collect_and_tokenize_selectors_in_place,
)
from flow360.component.simulation.framework.param_utils import AssetCache


def test_entity_selector_token_flow():
    # 1. Setup input dictionary with repeated selectors (same definition repeated in two places)
    selector_dict = {
        "selector_id": "sel1-token",
        "target_class": "Surface",
        "name": "sel1",
        "logic": "AND",
        "children": [{"attribute": "name", "operator": "matches", "value": "wing*"}],
    }
    params_as_dict = {
        "private_attribute_asset_cache": AssetCache().model_dump(mode="json", exclude_none=True),
        "models": [
            {"name": "m1", "selectors": [copy.deepcopy(selector_dict)]},
            {"name": "m2", "selectors": [copy.deepcopy(selector_dict)]},
        ],
    }

    # 2. Run tokenization (dict -> tokens + used_selectors list)
    tokenized_params = collect_and_tokenize_selectors_in_place(copy.deepcopy(params_as_dict))

    # 3. Verify AssetCache and tokens
    asset_cache = tokenized_params["private_attribute_asset_cache"]
    assert "used_selectors" in asset_cache
    assert len(asset_cache["used_selectors"]) == 1
    assert asset_cache["used_selectors"][0]["selector_id"] == selector_dict["selector_id"]
    assert asset_cache["used_selectors"][0]["name"] == selector_dict["name"]

    # Check that selectors are replaced by tokens
    assert tokenized_params["models"][0]["selectors"] == [selector_dict["selector_id"]]
    assert tokenized_params["models"][1]["selectors"] == [selector_dict["selector_id"]]

    # 4. Resolve selector tokens back to shared EntitySelector objects
    resolved_params = materialize_entities_and_selectors_in_place(tokenized_params)

    # 5. Verify selectors are resolved from tokens to EntitySelector objects (not strings)
    sel1 = resolved_params["models"][0]["selectors"]
    sel2 = resolved_params["models"][1]["selectors"]

    assert len(sel1) == 1
    assert isinstance(
        sel1[0], EntitySelector
    ), "Selector token should be materialized to EntitySelector"
    assert sel1[0].selector_id == selector_dict["selector_id"]
    assert sel1[0].name == selector_dict["name"]

    assert len(sel2) == 1
    assert isinstance(
        sel2[0], EntitySelector
    ), "Selector token should be materialized to EntitySelector"

    # 6. Verify shared instance linkage across references
    assert sel1[0] is sel2[0]
    assert resolved_params["private_attribute_asset_cache"]["used_selectors"][0] is sel1[0]


def test_entity_selector_token_round_trip_validation():
    params_as_dict = {
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

    tokenized_params = collect_and_tokenize_selectors_in_place(copy.deepcopy(params_as_dict))

    class _ModelWithSelectors(Flow360BaseModel):
        name: str
        selectors: list[EntitySelector]

    class _ParamsWithAssetCache(Flow360BaseModel):
        private_attribute_asset_cache: AssetCache
        models: list[_ModelWithSelectors]

    # Materialize selector tokens before validation, matching validate_model() preprocessing behavior.
    materialize_entities_and_selectors_in_place(tokenized_params)
    validated = _ParamsWithAssetCache.model_validate(tokenized_params)

    cache = validated.private_attribute_asset_cache
    assert cache.used_selectors is not None
    assert cache.used_selectors[0].selector_id == "sel1-token"
    assert validated.models[0].selectors[0].selector_id == "sel1-token"
    assert validated.models[1].selectors[0].selector_id == "sel1-token"


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

    with pytest.raises(ValueError, match=r"Selector token not found.*unknown-selector-id"):
        materialize_entities_and_selectors_in_place(params)
