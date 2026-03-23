# Fix: selectors 残留导致 meshing config hash 不一致

## 问题

两次相同输入的 GAI 翻译会产生语义完全相同但 hash 不同的 meshing config JSON。

**根因**：翻译流程中 `@preprocess_input` 已经通过 `expand_selectors_for_translation()` 将 selectors 展开到 `stored_entities` 中。但 `model_dump()` 后 `selectors` 仍然残留在输出 JSON 中（位于 `refinements[*].entities.selectors`），其中包含的 `selector_id`（随机 UUID）导致 hash 不一致。

**本质**：`selectors` 在展开后就是冗余信息 —— mesher 只消费 `stored_entities`。无论用户用什么 selector 规则来选择实体，只要最终选出的 `stored_entities` 相同，mesh 结果就相同。selectors 不应影响 meshing config 的 hash。

**不受影响的部分**：
- `AssetCache.used_selectors`（selector 池）已被 GAI whitelist 排除，不在翻译输出中
- Box 的 `private_attribute_id`（随机 UUID）已被 `_calculate_hash` strip

**受影响的字段**（来自两个示例 JSON 的 diff）：
- `refinements[*].entities.selectors` 中 3 个 selector 的 `selector_id`

## 方案

在 `filter_simulation_json()` 的翻译输出 dict 中，递归移除所有 `selectors` key（整个 selectors 数组，而非仅 `selector_id`）。

**理由**：selectors 展开后是冗余的，mesher 不需要，且包含随机 UUID 影响 hash 确定性。

**不改** `_calculate_hash`。

## 实现步骤

### Step 1: 在 `filter_simulation_json()` 返回前移除 `selectors`

**文件**：`flow360/component/simulation/translator/surface_meshing_translator.py`

在 `filter_simulation_json()` 函数中，`_traverse_and_filter` 之后、return 之前，递归遍历 `filtered_json` 移除所有 `selectors` key。

```python
def _remove_selectors(obj):
    if isinstance(obj, dict):
        return {k: _remove_selectors(v) for k, v in obj.items() if k != "selectors"}
    if isinstance(obj, list):
        return [_remove_selectors(item) for item in obj]
    return obj

filtered_json = _remove_selectors(filtered_json)
```

### Step 2: 添加测试——翻译两次，hash 一致

**文件**：`tests/simulation/translator/test_surface_meshing_translator.py`

新增测试，模仿用户场景：
1. 构造一个带 selectors 的 SimulationParams（包含 SurfaceRefinement + EntitySelector）
2. 调用 GAI 翻译两次
3. 断言两次的 hash 完全相同
4. 断言翻译输出中不包含 `selectors` key
