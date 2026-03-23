# 移除 OctreeSpacing 非标准构造接口

## 背景

`OctreeSpacing` 类目前有一个 `_project_spacing_to_object` model_validator，允许两种非标准用法：
1. `OctreeSpacing(1*u.cm)` — 不带关键字参数的构造（其他 Pydantic 模型不支持这种写法）
2. `octree_spacing=3*u.mm` — 在父模型字段中直接传入 dimensional value（Pydantic 会尝试用该值构造 `OctreeSpacing`，触发 validator 转换）

这两种用法与项目中其他 Pydantic 模型的接口不一致，属于反模式。

## 根因

所有问题都来自 `OctreeSpacing._project_spacing_to_object` validator (meshing_specs.py:38-43)：
```python
@pd.model_validator(mode="before")
@classmethod
def _project_spacing_to_object(cls, input_data):
    if isinstance(input_data, u.unyt.unyt_quantity):
        return {"base_spacing": input_data}
    return input_data
```

## 实施步骤

### Step 1: 修改 `_project_spacing_to_object` — 改为发出 deprecation warning 并仍然转换

**文件**: `flow360/component/simulation/meshing_param/meshing_specs.py`

将 validator 改为：当检测到传入的是 `unyt_quantity` 时，发出 deprecation warning，然后仍然转换（保持功能但提醒用户迁移）。

```python
@pd.model_validator(mode="before")
@classmethod
def _project_spacing_to_object(cls, input_data):
    if isinstance(input_data, u.unyt.unyt_quantity):
        import warnings
        warnings.warn(
            "Passing a plain dimensional value to OctreeSpacing is deprecated. "
            "Use OctreeSpacing(base_spacing=<value>) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return {"base_spacing": input_data}
    return input_data
```

> **注意**: 这里选择 warn + 仍然转换，而不是直接报错。这样现有代码不会立即崩溃，但用户会收到迁移提示。

### Step 2: 更新所有测试代码，使用显式构造方式

**文件**: `tests/simulation/params/meshing_validation/test_meshing_param_validation.py`

以下行中的 `octree_spacing=X * u.unit` 需要改为 `octree_spacing=OctreeSpacing(base_spacing=X * u.unit)`:
- Line 1174: `octree_spacing=3 * u.mm` → `octree_spacing=OctreeSpacing(base_spacing=3 * u.mm)`
- Line 1975: `octree_spacing=2 * u.m` → `octree_spacing=OctreeSpacing(base_spacing=2 * u.m)`
- Line 2083: `octree_spacing=1 * u.mm` → `octree_spacing=OctreeSpacing(base_spacing=1 * u.mm)`
- Line 2111: `octree_spacing=1 * u.mm` → `octree_spacing=OctreeSpacing(base_spacing=1 * u.mm)`
- Line 2169: `octree_spacing=1 * u.mm` → `octree_spacing=OctreeSpacing(base_spacing=1 * u.mm)`
- Line 2186: `octree_spacing=1 * u.mm` → `octree_spacing=OctreeSpacing(base_spacing=1 * u.mm)`

**文件**: `tests/simulation/translator/test_surface_meshing_translator.py`
- Line 614: `octree_spacing=3.5 * u.mm` → `octree_spacing=OctreeSpacing(base_spacing=3.5 * u.mm)`
- Line 713: `octree_spacing=5 * u.mm` → `octree_spacing=OctreeSpacing(base_spacing=5 * u.mm)`
- Line 776: `octree_spacing=3 * u.mm` → `octree_spacing=OctreeSpacing(base_spacing=3 * u.mm)`

需确保这两个测试文件中已 import `OctreeSpacing`。

## 影响范围

- `OctreeSpacing` 类本身（1处 validator 修改）
- 测试文件（9处用法更新）
- 内部代码中所有 `OctreeSpacing(base_spacing=...)` 的显式构造**不受影响**
- `SurfaceMeshingParams` 的 `validation_alias="base_spacing"` 是**独立的 field alias 机制**（字段名向后兼容），与本次修改无关，不需要改动
