# Entity Management Refactoring Plan

## Overview
Refactor entity management to establish entity_info as the single source of truth, eliminate entity_bucket concept, enhance EntityRegistry as a reference-only interface, and properly handle DraftContext entity isolation with clean merging semantics.

## Problem Statement
Current issues:
1. **Multiple copies**: `asset_base.py:entity_info` property returns new copy each time → no single source of truth
2. **Registry as storage**: EntityRegistry stores entity data instead of just referencing entity_info
3. **Bucket abstraction**: entity_bucket concept adds complexity with minimal value
4. **Merge confusion**: update_persistent_entities() tries to merge entity_info ↔ registry (should be one-way)
5. **Draft isolation**: DraftContext doesn't have independent entity_info copy

## Design Principles
1. **Single Source of Truth**: entity_info stores all entity data
2. **Registry as Reference**: EntityRegistry provides lookup/query interface over entity_info data
3. **Type-Based Organization**: Replace bucket with direct type-based access
4. **Draft Isolation**: DraftContext gets deep copy of entity_info via model_dump() + parse
5. **Clean Merging**: Collect draft entities from params.used_entity_registry on upload

---

## Stage 1: Foundation - COMPLETED ✅

### Overview
Eliminate entity_bucket concept and refactor EntityRegistry to type-based storage.

### Tasks

#### Task 1.1: Remove entity_bucket from EntityBase and all primitives
**Files**:
- `flow360/component/simulation/framework/entity_base.py`
- `flow360/component/simulation/primitives.py`
- `flow360/component/simulation/outputs/output_entities.py`

**Changes**:
- Remove `entity_bucket: ClassVar[str]` from EntityBase
- Remove entity_bucket validation in `__init_subclass__`
- Remove entity_bucket property and setter
- Remove entity_bucket from all entity classes

#### Task 1.2: Refactor EntityRegistry to type-based storage
**File**: `flow360/component/simulation/framework/entity_registry.py`

**Changes**:
- Change `internal_registry: Dict[str, List]` → `Dict[type[EntityBase], List]`
- Update `register()` and `fast_register()` to use `type(entity)` as key
- Update `find_by_type()` for efficient direct type lookup
- Update `clear()`, `contains()`, `replace_existing_with()` to use type-based keys
- Update `__str__()` to display type names
- Add deprecation warning to `get_bucket()` for backward compatibility

#### Task 1.3: Add registry.view() method
**File**: `flow360/component/simulation/framework/entity_registry.py`

**Changes**:
- Create `EntityRegistryView` class with glob pattern support
- Add `view(entity_type)` method to EntityRegistry
- Support `__iter__`, `__len__`, `__getitem__` with glob patterns

### Testing
- Update test files to use `registry.view()` instead of `get_bucket()`
- Verify 117/119 framework tests pass

---

## Stage 2: EntityRegistry.from_entity_info() for DraftContext - COMPLETED ✅

### Overview
Add `EntityRegistry.from_entity_info()` method specifically for the new DraftContext workflow. This is completely separate from the legacy `get_persistent_entity_registry()` used by assets.

**Key Principle**: Two isolated code paths:
- **Legacy (assets)**: Uses `entity_info.get_persistent_entity_registry()` - DO NOT MODIFY
- **New (DraftContext)**: Uses `EntityRegistry.from_entity_info()` - implement this

### Tasks

#### Task 2.1: Add EntityRegistry.from_entity_info() method
**File**: `flow360/component/simulation/framework/entity_registry.py`

**Add**:
```python
@classmethod
def from_entity_info(cls, entity_info) -> 'EntityRegistry':
    """Build registry by referencing entities from entity_info.

    This is for the DraftContext workflow only. Legacy asset code
    continues to use entity_info.get_persistent_entity_registry().
    """
    registry = cls()
    registry._register_from_entity_info(entity_info)
    return registry

def _register_from_entity_info(self, entity_info):
    """Populate internal_registry with references to entity_info entities."""
    # Import here to avoid circular imports
    from flow360.component.simulation.entity_info import (
        GeometryEntityInfo, VolumeMeshEntityInfo, SurfaceMeshEntityInfo
    )

    if isinstance(entity_info, GeometryEntityInfo):
        for surface_list in entity_info.grouped_faces:
            for surface in surface_list:
                self.register(surface)
        for edge_list in entity_info.grouped_edges:
            for edge in edge_list:
                self.register(edge)
        for body_list in entity_info.grouped_bodies:
            for body in body_list:
                self.register(body)

    elif isinstance(entity_info, VolumeMeshEntityInfo):
        for boundary in entity_info.boundaries:
            self.register(boundary)
        for zone in entity_info.zones:
            self.register(zone)

    elif isinstance(entity_info, SurfaceMeshEntityInfo):
        for boundary in entity_info.boundaries:
            self.register(boundary)

    # Common to all: draft_entities, ghost_entities
    for entity in entity_info.draft_entities:
        self.register(entity)
    for entity in entity_info.ghost_entities:
        self.register(entity)
```

#### Task 2.2: Future - Replace SelectorEntityPool with EntityRegistry
**Note**: SelectorEntityPool should eventually be replaced by EntityRegistry for entity materialization. This will be addressed in a later stage.

### Testing
- Test EntityRegistry.from_entity_info() creates correct registry for each entity_info type
- Verify entities are registered by type (not bucket)
- Test that registry references same entity objects (no copies)
- Verify legacy get_persistent_entity_registry() still works unchanged

---

## Stage 3: DraftContext Entity Isolation - COMPLETED ✅

### Overview
Implement proper entity_info deep copying in DraftContext and integrate registry.view().

### Tasks

#### Task 3.1: Deep copy entity_info in create_draft() ✅
**File**: `flow360/component/project.py`

**Implemented**: `create_draft()` now deep copies entity_info via model_dump + model_validate:
```python
def _deep_copy_entity_info(entity_info):
    entity_info_dict = entity_info.model_dump(mode="json")
    return type(entity_info).model_validate(entity_info_dict)

# In create_draft():
entity_info_copy = _deep_copy_entity_info(new_run_from.entity_info)
return DraftContext(entity_info=entity_info_copy)
```

#### Task 3.2: Update DraftContext to use registry.view() ✅
**File**: `flow360/component/simulation/draft_context/context.py`

**Changes made**:
- Removed `_SingleTypeEntityRegistry` class
- Updated `__init__` to use `EntityRegistry.from_entity_info()`
- Replaced property implementations with `registry.view()`:

```python
@property
def body_groups(self) -> EntityRegistryView:
    return self._entity_registry.view(GeometryBodyGroup)

@property
def surfaces(self) -> EntityRegistryView:
    return self._entity_registry.view(Surface)
```

#### Task 3.3: Test draft isolation ✅
**Test file**: `tests/simulation/draft_context/test_draft_context.py`

**Added tests**:
- `test_draft_entity_info_is_deep_copy` - Verifies entity_info is deep copied
- `test_draft_entity_modifications_are_isolated` - Verifies draft modifications don't affect original
- `test_draft_entity_info_is_independent_for_geometry` - Tests isolation for geometry assets
- `test_draft_entities_reference_copied_entity_info` - Verifies registry references copied entity_info
- `test_multiple_drafts_are_isolated_from_each_other` - Tests multiple drafts are independent
- `test_draft_uses_entity_registry_from_entity_info` - Verifies EntityRegistry.from_entity_info() usage

### Testing
- All 8 draft context tests pass ✅
- All 112 framework tests pass ✅

---

## Stage 4: Clean Entity Merging on Upload - COMPLETED ✅

### Overview
Implement proper entity collection from params.used_entity_registry during upload.

### Tasks

#### Task 4.1: Implement _merge_draft_entities_from_params()
**File**: `flow360/component/project_utils.py`

**Add function**:
```python
def _merge_draft_entities_from_params(
    entity_info: EntityInfoModel,
    params: SimulationParams
) -> EntityInfoModel:
    """Collect draft entities from params.used_entity_registry and merge into entity_info."""

    used_registry = params.used_entity_registry
    draft_type_union = get_args(DraftEntityTypes)[0]
    draft_type_list = get_args(draft_type_union)

    # Clear existing draft_entities to rebuild from scratch
    entity_info.draft_entities.clear()

    for draft_type in draft_type_list:
        draft_entities_used = used_registry.find_by_type(draft_type)
        for draft_entity in draft_entities_used:
            # Check if entity already in entity_info (from DraftContext)
            existing = _find_entity_in_entity_info(entity_info, draft_entity)
            if existing is not None:
                entity_info.draft_entities.append(existing)  # Use entity_info version
            else:
                entity_info.draft_entities.append(draft_entity)  # Use params version

    return entity_info
```

#### Task 4.2: Refactor set_up_params_for_uploading()
**File**: `flow360/component/project_utils.py`

**Update function (lines 409-459)**:
```python
def set_up_params_for_uploading(
    root_asset,
    length_unit: LengthType,
    params: SimulationParams,
    use_beta_mesher: bool,
    use_geometry_AI: bool,
    draft_entity_info: Optional[EntityInfoModel] = None,  # NEW parameter
) -> SimulationParams:
    """Set up params before submitting draft."""

    # 1. Update asset cache fields
    with model_attribute_unlock(params.private_attribute_asset_cache, "project_length_unit"):
        params.private_attribute_asset_cache.project_length_unit = length_unit

    # 2. Use draft_entity_info if provided, otherwise use root_asset.entity_info
    entity_info = draft_entity_info if draft_entity_info else root_asset.entity_info

    # 3. Collect draft entities from params.used_entity_registry
    entity_info = _merge_draft_entities_from_params(
        entity_info=entity_info,
        params=params
    )

    # 4. Update params.private_attribute_asset_cache.project_entity_info
    with model_attribute_unlock(params.private_attribute_asset_cache, "project_entity_info"):
        params.private_attribute_asset_cache.project_entity_info = entity_info

    # ... (rest of existing logic)
    return params
```

#### Task 4.3: Update entity_info.get_persistent_entity_registry()
**File**: `flow360/component/simulation/entity_info.py`

**Refactor (line 410-445)**:
```python
def get_persistent_entity_registry(self, internal_registry=None, **_) -> EntityRegistry:
    """Return EntityRegistry referencing this entity_info."""
    if internal_registry is None:
        return EntityRegistry.from_entity_info(self)
    # Reuse existing registry but ensure it references this entity_info
    internal_registry.entity_info = self
    internal_registry._register_persistent_entities()
    return internal_registry
```

Apply same pattern to:
- `VolumeMeshEntityInfo.get_persistent_entity_registry()` (line 588-605)
- `SurfaceMeshEntityInfo.get_persistent_entity_registry()` (line 628-638)

#### Task 4.4: Remove old merge logic
**File**: `flow360/component/project_utils.py`

**Remove**: `_set_up_params_non_persistent_entity_info()` (lines 255-271)

### Testing
- Test draft entity collection from params
- Test entity_info as source of truth during upload
- Verify no duplicate entities

---

## Stage 5: Replace SelectorEntityPool with EntityRegistry - COMPLETED ✅

### Overview
Replace SelectorEntityPool with EntityRegistry for entity materialization/expansion. This unifies entity lookup under a single interface.

**Current State**:
- `SelectorEntityPool` (entity_selector.py:128) - used for entity expansion during materialization
- `build_entity_pool_from_entity_info()` - builds a dict for reference identity
- Two separate mechanisms doing similar things

**Target State**:
- EntityRegistry handles all entity lookup needs
- SelectorEntityPool deprecated (SelectorPoolAdapter provides backward-compatible interface)
- Use `EntityRegistry.from_entity_info()` with `SelectorPoolAdapter` for entity selection

### Tasks

#### Task 5.1: Analyze SelectorEntityPool usage ✅
**Files**:
- `flow360/component/simulation/framework/entity_selector.py` (class definition at line 128)
- `flow360/component/simulation/framework/entity_expansion_utils.py` (builds pool from entity_info)

**Understand**:
- How SelectorEntityPool is structured (stores entities by type/name)
- How it's used in `expand_selector_to_entities()` and related functions
- What methods EntityRegistry needs to support the same use cases

#### Task 5.2: Add required methods to EntityRegistry ✅
**File**: `flow360/component/simulation/framework/entity_registry.py`

**Added methods**:
```python
def find_by_name(self, name: str) -> Optional[EntityBase]:
    """Find entity by exact name match."""

def find_by_type(self, entity_class: type[EntityBase]) -> list[EntityBase]:
    """Find all registered entities of a given type (including subclasses)."""

def find_by_type_name(self, type_name: str) -> list[EntityBase]:
    """Find entities by their serialized type name (e.g., 'Surface', 'Edge')."""

def get_all_entities(self) -> list[EntityBase]:
    """Return all registered entities."""
```

#### Task 5.3: Update entity materialization to use EntityRegistry ✅
**File**: `flow360/component/simulation/framework/entity_expansion_utils.py`

**Changes**:
- Added `get_selector_pool_adapter_from_registry()` - creates adapter from EntityRegistry
- Added `get_selector_pool_adapter_from_entity_info()` - creates adapter using EntityRegistry.from_entity_info()
- Legacy functions still work unchanged for backward compatibility

#### Task 5.4: Update entity_selector.py ✅
**File**: `flow360/component/simulation/framework/entity_selector.py`

**Changes**:
- Created `SelectorPoolAdapter` class that wraps EntityRegistry with SelectorEntityPool-compatible interface
- Added `SelectorPoolType` type alias for Union[SelectorEntityPool, SelectorPoolAdapter]
- Updated all function signatures to accept `SelectorPoolType`:
  - `_get_entity_pool()`
  - `_process_selectors()`
  - `_expand_node_selectors()`
  - `expand_entity_selectors_in_place()`
- Marked `SelectorEntityPool` as DEPRECATED

#### Task 5.5: Update validate_model() to use EntityRegistry
**File**: `flow360/component/simulation/services.py`

**Status**: Deferred to Stage 6/7 - not needed for current functionality since SelectorPoolAdapter provides backward compatibility.

### Testing
- All 672 simulation tests pass ✅
- All 112 framework tests pass ✅
- All 19 draft context tests pass ✅

---

## Stage 6: Update All get_bucket() Call Sites

### Overview
Find and update all remaining get_bucket() calls to use new API.

### Tasks

#### Task 6.1: Search and replace get_bucket() calls
**Search pattern**: `\.get_bucket\(`

**Files to check**:
- `flow360/component/simulation/framework/param_utils.py`
- `flow360/component/volume_mesh.py`
- Any other files in codebase

**Replacement strategy**:
- `registry.get_bucket(by_type=Type).entities` → `registry.find_by_type(Type)` or `list(registry.view(Type))`
- Choose based on context: use `find_by_type()` for subclass matching, `view()` for exact type

#### Task 6.2: Update param_utils.py
**File**: `flow360/component/simulation/framework/param_utils.py`

**Line 211**:
```python
# OLD:
for volume_entity in registry.get_bucket(by_type=_VolumeEntityBase).entities:

# NEW:
for volume_entity in registry.find_by_type(_VolumeEntityBase):
```

#### Task 6.3: Update volume_mesh.py
**File**: `flow360/component/volume_mesh.py`

**Lines 1218, 1223, 1242**:
```python
# OLD:
for surface in self.internal_registry.get_bucket(by_type=Surface).entities:

# NEW:
for surface in self.internal_registry.find_by_type(Surface):
```

### Testing
- Run full test suite
- Verify no deprecation warnings in production code
- Check that all tests pass

---

## Stage 7: Integration Testing and Polish

### Overview
Comprehensive testing, documentation updates, and final polish.

### Tasks

#### Task 7.1: Write comprehensive tests
**Test files to create/update**:

1. **tests/simulation/framework/test_entity_registry.py**
   - Test type-based storage
   - Test registry.view() method
   - Test EntityRegistryView glob patterns
   - Test find_by_type() with subclasses

2. **tests/simulation/draft_context/test_draft_context.py**
   - Test entity_info deep copy isolation
   - Test modifications in draft don't affect original
   - Test draft entity collection from params

3. **tests/simulation/test_entity_info.py**
   - Test EntityRegistry.from_entity_info()
   - Test reference identity between entity_info and registry

4. **tests/simulation/test_services.py**
   - Test validate_model() with entity_pool parameter
   - Test entity reference identity in deserialized params

5. **tests/simulation/test_project_utils.py**
   - Test _merge_draft_entities_from_params()
   - Test entity_info as source of truth during upload

#### Task 7.2: Update Project._run() integration
**File**: `flow360/component/project.py`

**Update (line 1470-1476)**:
```python
params = set_up_params_for_uploading(
    params=params,
    root_asset=self._root_asset,
    draft_entity_info=draft.entity_info if draft else None,  # Pass draft entity_info
    length_unit=self.length_unit,
    use_beta_mesher=use_beta_mesher,
    use_geometry_AI=use_geometry_AI,
)
```

#### Task 7.3: Run full test suite
- `poetry run pytest tests/simulation/ -v`
- Fix any remaining failures
- Ensure backward compatibility for legacy code

### Testing
- All tests pass
- No deprecation warnings in production code
- Legacy code still works with warnings

---

## Critical Files Modified

### Core Framework (6 files)
1. `flow360/component/simulation/framework/entity_base.py` - Remove entity_bucket ✅
2. `flow360/component/simulation/framework/entity_registry.py` - Type-based storage, view(), find_by_type(), find_by_name() ✅
3. `flow360/component/simulation/framework/entity_selector.py` - SelectorPoolAdapter, SelectorPoolType ✅
4. `flow360/component/simulation/framework/entity_expansion_utils.py` - get_selector_pool_adapter_from_*() ✅
5. `flow360/component/simulation/framework/param_utils.py` - Update get_bucket() calls
6. `flow360/component/simulation/entity_info.py` - Update get_persistent_entity_registry()

### Draft Context (1 file)
7. `flow360/component/simulation/draft_context/context.py` - Deep copy, use registry.view()

### Project/Asset Integration (3 files)
8. `flow360/component/simulation/web/asset_base.py` - entity_info direct reference, entity_pool
9. `flow360/component/project.py` - create_draft() deep copy, pass draft.entity_info
10. `flow360/component/project_utils.py` - Refactor set_up_params_for_uploading()

### Services/Validation (1 file)
11. `flow360/component/simulation/services.py` - Add entity_pool parameter

### Primitives (1 file)
12. `flow360/component/simulation/primitives.py` - Remove all entity_bucket definitions ✅

---

## Migration Guide for Users

### Breaking Changes

1. **Direct entity_info access**: `asset.entity_info` now returns the same object reference (not a copy)
   - **Old behavior**: Each call returned a new copy
   - **New behavior**: Returns direct reference to internal entity_info
   - **Migration**: If you need a copy, use `asset.entity_info.model_dump()` and re-parse

2. **EntityRegistry.get_bucket()**: Deprecated, use `.view(type)` instead
   - **Old**: `registry.get_bucket(by_type=Surface).entities`
   - **New**: `registry.view(Surface)` or `registry.find_by_type(Surface)`

3. **entity_bucket removed**: If user code references `entity.entity_bucket`, update to `type(entity).__name__`

### For Developers (Internal Changes)

1. **EntityRegistry storage**: `internal_registry` now keyed by type, not string bucket name
2. **entity_info lifecycle**: entity_info is deep copied in DraftContext, referenced elsewhere
3. **Draft entity merging**: Automatic collection from params.used_entity_registry

---

## Success Criteria

✅ **Stage 1 Complete**: entity_bucket removed, type-based storage, registry.view() added
✅ **Stage 2 Complete**: EntityRegistry.from_entity_info() added for DraftContext workflow
✅ **Stage 3 Complete**: DraftContext has deep copied entity_info, uses registry.view()
✅ **Stage 4 Complete**: Clean entity merging on upload
✅ **Stage 5 Complete**: SelectorPoolAdapter bridges EntityRegistry with SelectorEntityPool interface
⬜ **Stage 6**: All get_bucket() calls updated
⬜ **Stage 7**: All tests pass, documentation complete

**Final Success Metrics**:
- ✅ Single Source of Truth: entity_info stores all entities, registry references
- ✅ Draft Isolation: Modifications in DraftContext don't affect original asset
- ⬜ Reference Identity: Deserialized params share entity references with entity_info
- ✅ Clean Merging: Draft entities collected from params.used_entity_registry
- ✅ No Buckets: Type-based access throughout, entity_bucket removed
- ✅ Backward Compatible: Legacy asset.internal_registry still works with warnings
- ⬜ Tests Pass: All tests pass + new tests for refactored behavior

---

## Implementation Notes

### Stage 1 Completion Summary (Current Status)

**Completed**:
- ✅ Removed entity_bucket from EntityBase and all entity classes
- ✅ Refactored EntityRegistry to type-based storage (Dict[type[EntityBase], List])
- ✅ Added EntityRegistryView class with glob pattern support
- ✅ Added registry.view() method
- ✅ Updated test files to use new API
- ✅ ~~Implemented build_entity_pool_from_entity_info() (for Stage 5)~

**Next Steps**:
- Begin Stage 2: Implement EntityRegistry.from_entity_info()
- Note: build_entity_pool_from_entity_info() will eventually be replaced by EntityRegistry

### Stage 4 Completion Summary

**Completed**:
- ✅ Implemented `_merge_draft_entities_from_params()` in project_utils.py
  - Collects draft entities from params.used_entity_registry
  - Preserves entity_info as source of truth for existing draft entities
  - Adds new draft entities from params that aren't already in entity_info
- ✅ Refactored `set_up_params_for_uploading()` to accept optional `draft_entity_info` parameter
  - When `draft_entity_info` is provided (DraftContext workflow), uses it as source of truth
  - When not provided (legacy workflow), uses root_asset.entity_info
- ✅ Updated `Project._run()` to check for active DraftContext via `get_active_draft()`
  - Passes `draft.entity_info` to `set_up_params_for_uploading()` when DraftContext is active
- ✅ Added `entity_info` property to DraftContext for public access

**Testing**:
- All 661 simulation tests pass ✅
- All 8 draft context tests pass ✅
- All 112 framework tests pass ✅

### Key Design Decisions

1. **Why model_dump() + parse for deep copy?**
   - Cleaner separation than copy.deepcopy()
   - Avoids copying unwanted references
   - Ensures all Pydantic validators run

2. **Why EntityRegistryView instead of modifying _SingleTypeEntityRegistry?**
   - Integrated into EntityRegistry as first-class feature
   - Consistent API across codebase
   - Easier to maintain

3. **Why search by name in replace_existing_with()?**
   - Supports replacing GenericVolume with Cylinder (same name, different types)
   - Matches old bucket-based behavior where types shared buckets
   - More flexible for user overrides

4. **Why keep legacy asset.entity_info copy behavior?**
   - Changing it would break backward compatibility
   - DraftContext doesn't depend on asset.entity_info anyway
   - The new DraftContext path has its own entity_info deep copy

5. **Why EntityRegistry should replace SelectorEntityPool?**
   - EntityRegistry already provides type-based lookup (find_by_type, view)
   - SelectorEntityPool duplicates functionality
   - Single unified interface is cleaner
   - Will be addressed when tackling entity materialization

### Common Pitfalls to Avoid

1. **Don't use type(entity).__name__ for entity_pool keys**
   - Use `entity.private_attribute_entity_type_name` instead
   - This matches the _stable_entity_key_from_obj() implementation

2. **Don't forget to update both find_by_type() uses**
   - Some places need exact type match (use view())
   - Some places need subclass matching (use find_by_type())

3. **Don't modify entity_info without unlocking frozen fields**
   - Use `model_attribute_unlock()` context manager
   - Example: `with model_attribute_unlock(entity, "field_name"):`

### Stage 5 Completion Summary

**Completed**:
- ✅ Added `find_by_name()`, `find_by_type()`, `find_by_type_name()`, `get_all_entities()` to EntityRegistry
- ✅ Created `SelectorPoolAdapter` class in entity_selector.py that wraps EntityRegistry
  - Provides lazy property access to entity lists (surfaces, edges, generic_volumes, geometry_body_groups)
  - Uses EntityRegistry.find_by_type() internally for proper subclass matching
- ✅ Added `SelectorPoolType` type alias for functions accepting either pool type
- ✅ Updated all function signatures to accept `SelectorPoolType` instead of just `SelectorEntityPool`
  - `_get_entity_pool()`
  - `_process_selectors()`
  - `_expand_node_selectors()`
  - `expand_entity_selectors_in_place()`
- ✅ Added helper functions in entity_expansion_utils.py:
  - `get_selector_pool_adapter_from_registry()`: Create adapter from existing EntityRegistry
  - `get_selector_pool_adapter_from_entity_info()`: Create adapter using EntityRegistry.from_entity_info()
- ✅ Marked `SelectorEntityPool` as DEPRECATED in its docstring

**Testing**:
- All 672 simulation tests pass ✅
- All 112 framework tests pass ✅
- All 19 draft context tests pass ✅

**Usage**:
```python
# Modern approach using EntityRegistry and SelectorPoolAdapter
from flow360.component.simulation.framework.entity_registry import EntityRegistry
from flow360.component.simulation.framework.entity_selector import SelectorPoolAdapter

registry = EntityRegistry.from_entity_info(entity_info)
adapter = SelectorPoolAdapter(registry)
expand_entity_selectors_in_place(adapter, params_as_dict)

# Or use the helper function
from flow360.component.simulation.framework.entity_expansion_utils import (
    get_selector_pool_adapter_from_entity_info
)
adapter = get_selector_pool_adapter_from_entity_info(entity_info)
expand_entity_selectors_in_place(adapter, params_as_dict)
```
