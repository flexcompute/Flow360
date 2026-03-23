# 单位原语迁移计划

## 目标
将单位原语从 `flow360/component/simulation/unit_system.py` 迁移到新的 `flow360/flow360_schema/` 包，使其：
1. JSON Schema 与 common-schema 兼容
2. 验证函数从集中式改为组合式
3. 序列化只存 value（SI单位），unit 作为 schema 元数据
4. unyt 依赖封装（懒加载），JSON schema 生成不依赖 unyt
5. 允许零值输入（除温度外）—— 纯数字0应被接受并转换为 `0 * si_unit`

---

## 已确认的设计决策

### 1. 命名约定
使用 `Dimension.Constraint` 形式，维度与数据类型分离：
```python
Length.Float64           # 基础长度标量
Length.PositiveFloat64   # 正值长度标量
Length.Vector3           # 长度向量
Length.PositiveVector3   # 正分量长度向量
```

### 2. 零值处理
- **非温度类型**：允许纯数字 `0` 输入，自动转换为 `0 * si_unit`
  - 原因：0 在任何单位下等价（0m = 0ft = 0）
- **温度类型**：**不允许**纯数字 `0` 输入（0K ≠ 0°C，有歧义）
  - 必须带单位或使用非零值

### 3. 向量约束
两种约束都需要：
- `NonNullVector3`：范数 > 0（向量不为零向量）
- `PositiveVector3`：所有分量 > 0
- **TODO**: 需要在 common-schema 中添加 `PositiveVector3` 组件

### 4. 温度处理
- `AbsoluteTemperature`：必须 > 0K
- `DeltaTemperature`：可以是任意值（正负均可）

---

## 建议的核心原语列表

### 第一优先级（最常用）

| 访问方式 | 约束 | 形状 | SI单位 |
|----------|------|------|--------|
| `Length.Float64` | 无 | scalar | m |
| `Length.PositiveFloat64` | >0 | scalar | m |
| `Length.NonNegativeFloat64` | >=0 | scalar | m |
| `Length.Vector3` | 无 | vector3 | m |
| `Length.PositiveVector3` | 分量>0 | vector3 | m |
| `Length.NonNullVector3` | 范数>0 | vector3 | m |
| `Angle.Float64` | 无 | scalar | rad |
| `Velocity.Float64` | 无 | scalar | m/s |
| `Area.PositiveFloat64` | >0 | scalar | m² |
| `Pressure.PositiveFloat64` | >0 | scalar | Pa |
| `Temperature.Float64` | >0K | scalar | K |
| `Time.PositiveFloat64` | >0 | scalar | s |

### 第二优先级

| 访问方式 | 约束 | 形状 | SI单位 |
|----------|------|------|--------|
| `Density.PositiveFloat64` | >0 | scalar | kg/m³ |
| `Viscosity.PositiveFloat64` | >0 | scalar | Pa·s |
| `Force.Vector3` | 无 | vector3 | N |
| `Moment.Vector3` | 无 | vector3 | N·m |
| `AngularVelocity.Float64` | 无 | scalar | rad/s |
| `Frequency.PositiveFloat64` | >0 | scalar | Hz |
| `Length.Range` | 递增 | [2] | m |
| `Length.CoordinateGroup` | 无 | matrix(N,3) | m |

---

## 目录结构

```
flow360_schema/
├── __init__.py
└── primitives/
    ├── __init__.py
    ├── base.py              # 基类定义
    ├── validators.py        # 可组合的验证函数
    ├── serializers.py       # 序列化逻辑
    ├── schema_generators.py # JSON Schema 生成（无 unyt 依赖）
    ├── unyt_adapter.py      # unyt 封装（懒加载）
    ├── dimensions.py        # 维度定义（SI单位映射）
    └── types/
        ├── __init__.py
        ├── length.py        # Length 命名空间类
        ├── velocity.py
        ├── temperature.py
        └── ...
```

---

## 关键设计

### 1. 命名空间类结构

```python
# types/length.py
class Length:
    """Length 维度的类型命名空间"""
    Float64 = _LengthFloat64
    PositiveFloat64 = _LengthPositiveFloat64
    NonNegativeFloat64 = _LengthNonNegativeFloat64
    Vector3 = _LengthVector3
    PositiveVector3 = _LengthPositiveVector3
    NonNullVector3 = _LengthNonNullVector3
    Range = _LengthRange
    # ... 其他变体
```

### 2. 验证函数组合化

```python
# validators.py - 独立的验证函数
def positive(value): ...           # value > 0
def non_negative(value): ...       # value >= 0
def vector3_shape(value): ...      # len == 3
def non_null_vector(value): ...    # magnitude > 0
def positive_components(value): ... # all components > 0
def allow_zero_scalar(value, si_unit): ...  # 0 -> 0*si_unit

# 类型定义 - 组合验证函数
class _LengthPositiveFloat64(UnitPrimitiveBase):
    validators = [positive]

class _LengthPositiveVector3(UnitPrimitiveBase):
    validators = [vector3_shape, positive_components]
```

### 3. 零值特殊处理

```python
def validate(cls, value):
    # 特殊处理：纯数字 0 转换为 0 * si_unit
    if value == 0 and cls.allow_zero:
        return 0 * cls.si_unit
    # 温度不允许此特殊处理
    if value == 0 and not cls.allow_zero:
        raise ValueError("Temperature requires explicit unit for zero value")
    # ... 正常验证流程
```

### 4. 序列化格式变化

**旧格式：**
```json
{"value": 1.5, "units": "m"}
```

**新格式：**
```json
1.5
```

### 5. JSON Schema 生成（重要变更）

**必须使用 `$ref` 引用 common-schema 中的类型**，而不是内联定义约束。

Common-schema 白名单类型（见 `property-composition-schema.json`）：
- 数值: `Float64`, `PositiveFloat64`, `NonNegativeFloat64`, `NegativeFloat64`, `NonPositiveFloat64`
- 向量: `Vector3Json`, `NonNullVector3Json`, `Vector2Json`, `Vector4Json`
- 矩阵: `Matrix3Json`, `Matrix4Json`
- 等等...

**Schema 格式：**
```json
{
  "$ref": "https://flexcompute.com/schemas/1.0.0/PositiveFloat64.json",
  "$units": "m"
}
```

**向量示例：**
```json
{
  "$ref": "https://flexcompute.com/schemas/1.0.0/Vector3Json.json",
  "$units": "m"
}
```

**注意**：`PositiveVector3`（分量>0）在 common-schema 中不存在，需要后续添加。

### 6. Unyt 封装

- JSON Schema 生成：**不依赖** unyt
- 验证/反序列化：**懒加载** unyt
- 内存存储：unyt 对象
- 序列化：转换为 SI 数值

---

## 分步实现计划

### 阶段 1：基础架构
- [ ] 1.1 创建目录结构
- [ ] 1.2 实现 `dimensions.py` - 维度元数据（不依赖 unyt）
- [ ] 1.3 实现 `validators.py` - 可组合验证函数（含零值处理）
- [ ] 1.4 实现 `schema_generators.py` - JSON Schema 生成
- [ ] 1.5 实现 `unyt_adapter.py` - unyt 懒加载封装
- [ ] 1.6 实现 `base.py` - 基类

### 阶段 2：核心标量类型
- [ ] 2.1 Length 命名空间类（Float64, Positive, NonNegative）
- [ ] 2.2 其他高频标量（Angle, Time, Velocity, Area, Pressure）
- [ ] 2.3 Temperature（特殊处理，不允许零）

### 阶段 3：向量类型
- [ ] 3.1 Length 向量族（Vector3, NonNull, Positive）
- [ ] 3.2 其他向量（Velocity, Force）

### 阶段 4：数组和矩阵类型
- [ ] 4.1 Range 类型
- [ ] 4.2 Array 类型
- [ ] 4.3 CoordinateGroup 类型

### 阶段 5：测试和文档
- [ ] 5.1 单元测试
- [ ] 5.2 与 common-schema 兼容性测试

---

## 后续工作（不在本计划范围内）

- [ ] 在 common-schema 中添加 `PositiveVector3` 组件
- [ ] 迁移现有代码使用新原语

---

## 关键文件参考

- 现有实现：`flow360/component/simulation/unit_system.py`
- Common-schema：`/disk2/ben/flex/frontend/workspace/packages/common-schema/`
- 目标位置：`flow360/flow360_schema/primitives/`
