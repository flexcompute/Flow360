# MVP Refactor Plan - Unit Primitives

## 目标
将当前实现精简为 MVP，只保留最基本的功能来验证架构设计。

## 变更概览

### 1. 文件结构重构
- **删除** `primitives/types/` 文件夹
- **创建** `primitives/types.py`（从 `types/__init__.py` 转换）
- 只保留 `Length` 维度类

### 2. 重命名 DimensionMeta → PhysicalDimensionMeta
避免 "Dimension" 一词的歧义（shape dimension vs physical dimension）

**涉及文件：**
- `dimensions.py` - 类定义
- `composers.py` - 引用
- `dimension_base.py` - docstring
- `types.py` - 使用

### 3. 精简 MVP 范围

**保留的 Physical Dimensions:**
- `Length` only

**保留的 Data Types (data_types.py):**
- `ScalarFloat64` - 无约束标量
- `PositiveScalar` - 正数标量 (> 0)
- `Vector3Type` - 3D向量
- `NonNullVector3Type` - 非零3D向量

**保留的 DimensionBase 属性 (dimension_base.py):**
- `Float64`
- `PositiveFloat64`
- `Vector3`
- `NonNullVector3`

**保留的 Schema Generators (schema_generators.py):**
- `scalar_schema()`
- `vector3_schema()`

**删除:**
- `vector2_schema()` - out of scope
- `array_schema()` - out of scope
- `matrix3_schema()` - out of scope
- `matrix_schema()` - out of scope
- 所有 Range types
- 所有 Array types
- NonNegative* types
- Positive* vector/array types

**保留的 Validators (validators.py):**
- `positive` - 用于 PositiveScalar
- `vector3_shape` - 用于 Vector3
- `non_null_vector` - 用于 NonNullVector3

**删除的 Validators:**
- `non_negative`
- `positive_components`
- `non_negative_components`
- `array_length`
- `strictly_increasing`

**Serializers (serializers.py):**
- 保留 `serialize_scalar`, `serialize_vector3`
- 删除 `serialize_array`

**unyt_adapter.py:**
- 保留 `to_unyt_scalar`, `to_unyt_array` (vector3 uses array)
- 保留 `check_dimension`, `is_unyt_quantity`

---

## 实施步骤

### Step 1: 重命名 DimensionMeta → PhysicalDimensionMeta
- 修改 `dimensions.py`
- 更新所有引用文件

### Step 2: 精简 validators.py
- 只保留: `positive`, `vector3_shape`, `non_null_vector`

### Step 3: 精简 schema_generators.py
- 只保留: `scalar_schema`, `vector3_schema`
- 删除: `vector2_schema`, `array_schema`, `matrix3_schema`, `matrix_schema`

### Step 4: 精简 serializers.py
- 只保留: `serialize_scalar`, `serialize_vector3`

### Step 5: 精简 data_types.py
- 只保留: `ScalarFloat64`, `PositiveScalar`, `Vector3Type`, `NonNullVector3Type`

### Step 6: 精简 dimension_base.py
- 只保留: `Float64`, `PositiveFloat64`, `Vector3`, `NonNullVector3`

### Step 7: 精简 composers.py
- 更新 `_from_unyt` 移除 ARRAY case
- 更新引用名称

### Step 8: 重构 types/ → types.py
- 删除 `types/` 文件夹
- 创建 `types.py`，只包含 `Length` 类

### Step 9: 更新 primitives/__init__.py
- 只导出 `Length`

---

## 最终文件结构
```
primitives/
├── __init__.py          # 导出 Length
├── composers.py         # _ComposedTypeBase, _compose_type
├── data_types.py        # 4个 DataTypeDescriptor
├── dimension_base.py    # DimensionBase with 4 @classproperty
├── dimensions.py        # PhysicalDimensionMeta
├── schema_generators.py # scalar_schema, vector3_schema
├── serializers.py       # serialize_scalar, serialize_vector3
├── types.py             # Length class only
├── unyt_adapter.py      # unyt 转换函数
└── validators.py        # positive, vector3_shape, non_null_vector
```

## 预期使用方式
```python
from flow360.flow360_schema.primitives import Length

class MyModel(BaseModel):
    distance: Length.PositiveFloat64
    direction: Length.NonNullVector3
```
