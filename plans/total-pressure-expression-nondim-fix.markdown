# TotalPressure 表达式语义修正：从 ratio 改为 Flow360 无量纲压力值

## 问题背景

当前 `TotalPressure.value` 接受两种输入：
1. **带量纲值**（如 `1.04e6 * fl.u.Pa`）：translator 会自动乘以 `ρa²/P`（即 γ）转换为 `totalPressureRatio`
2. **字符串表达式**（如 `"pow(1.0+0.2*pow(0.1*(1.0-y*y),2.0),1.4/0.4)"`）：translator **直接原样**传给 solver 的 `totalPressureRatio`

**问题**：用户写的字符串表达式实际上必须是 total pressure **ratio**（`P_total / P_static_∞`），但 class 名字叫 `TotalPressure`，docstring 说的是 "nondimensionalized by operating condition pressure"。实际上 `totalPressureRatio` 是 `P_total / P_∞`（无量纲化方式是除以来流静压），而不是除以 Flow360 内部压力单位 `ρ∞a∞²`。两者差一个 γ 倍。这对用户非常 confusing。

**目标**：让用户的字符串表达式语义 = **total pressure 除以 Flow360 压力单位 `ρ∞a∞²`**（与带量纲值的无量纲化方式一致），translator 侧做反向转换生成 solver 需要的 `totalPressureRatio`。

### 物理关系

```
op_acoustic_to_static_pressure_ratio = ρ∞ * a∞² / P∞ = γ  (对于 calorically perfect gas, γ=1.4)

totalPressureRatio = P_total / P∞
                   = (P_total / (ρ∞a∞²)) * (ρ∞a∞² / P∞)
                   = P_total_flow360_nondim * op_acoustic_to_static_pressure_ratio
```

所以：
- **新语义**：用户表达式 = `P_total / (ρ∞a∞²)` = `P_total_flow360_nondim`
- **Translator 转换**：`totalPressureRatio = "({expression}) * {op_acoustic_to_static_pressure_ratio}"`

### 关于 ThermallyPerfectGas（develop 分支新特性）

develop 分支已引入 `ThermallyPerfectGas`，`Air` 的 γ 不再固定为 1.4，而是通过 NASA 9 系数多项式根据温度计算。

**对本方案的影响**：
- **Translator 侧无影响**：`op_acoustic_to_static_pressure_ratio` 已经是从 `ρ*a²/P` 动态计算的，`a` 内部已使用 `get_specific_heat_ratio(temperature)` 获取温度相关的 γ。所以 translator 的转换自动适应。
- **Updater 侧有影响**：不能硬编码 `γ = 1.4`，需要从序列化的 operating_condition 中获取实际 γ。但由于 ThermallyPerfectGas 的 γ 依赖温度，而 updater 只能访问 serialized dict，提取精确 γ 比较复杂。
  - **方案**：在 updater 中重建 `Air` material 对象 + `ThermalState`，调用 `get_specific_heat_ratio(temperature)` 获取运行时 γ。或者，由于 updater 在 model 构建之前运行，可以利用 `thermal_state` 中的 temperature 和 material 信息重新计算。
  - **简化方案**：由于 ThermallyPerfectGas 和 TotalPressure string expression 都是新特性，不太可能有用户同时使用旧版 string expression + ThermallyPerfectGas。updater 迁移只需处理旧版标准 Air（γ=1.4）的情况。对于使用了 ThermallyPerfectGas 的旧数据（理论上不存在），可以跳过或 warn。

---

## 实施步骤

### Step 1: 修改 Translator — 字符串表达式乘以 `op_acoustic_to_static_pressure_ratio`

**文件**: `flow360/component/simulation/translator/solver_translator.py` (L1537-1542)

当前代码：
```python
if isinstance(model.spec, TotalPressure):
    boundary["type"] = "SubsonicInflow"
    total_pressure_ratio = model_dict["spec"]["value"]
    if not isinstance(model.spec.value, str):
        total_pressure_ratio *= op_acoustic_to_static_pressure_ratio
    boundary["totalPressureRatio"] = total_pressure_ratio
```

修改为：
```python
if isinstance(model.spec, TotalPressure):
    boundary["type"] = "SubsonicInflow"
    total_pressure_ratio = model_dict["spec"]["value"]
    if isinstance(model.spec.value, str):
        # Expression specifies total pressure in Flow360 nondim units (P/(ρa²)),
        # convert to ratio (P/P∞) by multiplying by ρa²/P∞
        total_pressure_ratio = f"({total_pressure_ratio}) * {op_acoustic_to_static_pressure_ratio}"
    else:
        total_pressure_ratio *= op_acoustic_to_static_pressure_ratio
    boundary["totalPressureRatio"] = total_pressure_ratio
```

**效果**：字符串表达式和带量纲值现在走**相同的无量纲化路径**（都乘以 `ρa²/P`），语义统一。对于 ThermallyPerfectGas，`op_acoustic_to_static_pressure_ratio` 已经是动态计算的正确值。

### Step 2: 更新 TotalPressure 的 docstring 和 description

**文件**: `flow360/component/simulation/models/surface_models.py` (L126-154)

- 修改 docstring 中的说明：表达式代表 total pressure 以 Flow360 压力单位（`ρ∞a∞²`）无量纲化后的值
- 修改 `value` field 的 description 以匹配新语义
- 更新示例表达式（当前的示例 `pow(1.0+0.2*pow(0.1*(1.0-y*y),2.0),1.4/0.4)` 是 ratio 语义，需要除以 γ 来反映新语义）

### Step 3: 在 Updater 中增加向后兼容迁移

**文件**: `flow360/component/simulation/framework/updater.py`

新增一个 updater 函数（版本号待定），处理旧格式的 TotalPressure 字符串表达式：

**迁移逻辑**：
1. 遍历 `params_as_dict["models"]`，找到 `type == "Inflow"` 且 `spec.type_name == "TotalPressure"` 的项
2. 如果 `spec.value` 是字符串（即表达式），说明是旧格式（ratio 语义）
3. 计算 γ：
   - 检查 `operating_condition` 类型：如果是 `LiquidOperatingCondition`，γ = 1.0，跳过
   - 对于 gas：从序列化的 `thermal_state.material` 中提取 NASA 9 系数和 temperature，重新计算 γ
   - **简化**：由于旧版数据必然使用标准 Air（ThermallyPerfectGas 与 string expression 同为新特性，不可能共存于旧数据），直接使用 γ = 1.4
4. 转换：`new_value = f"({old_value}) / {gamma}"`

```python
def _to_VERSION(params_as_dict):
    """Convert TotalPressure string expressions from ratio to Flow360 nondim pressure."""
    operating_condition = params_as_dict.get("operating_condition", {})

    # Liquid operating conditions have ratio = 1.0, no conversion needed
    if operating_condition.get("type") == "LiquidOperatingCondition":
        return params_as_dict

    # For gas: old data always uses standard Air (γ=1.4)
    # ThermallyPerfectGas with non-1.4 gamma cannot coexist with old string expressions
    gamma = 1.4

    for model in params_as_dict.get("models", []):
        if model.get("type") != "Inflow":
            continue
        spec = model.get("spec")
        if spec and spec.get("type_name") == "TotalPressure" and isinstance(spec.get("value"), str):
            old_expr = spec["value"]
            spec["value"] = f"({old_expr}) / {gamma}"
    return params_as_dict
```

### Step 4: 更新测试

**文件**: `tests/simulation/translator/test_solver_translator.py`

- 修改现有的 TotalPressure 表达式测试，验证新行为：表达式会被 translator 包裹 `* {ratio}`
- 新增测试验证：带量纲值和等价的表达式值产生相同的 `totalPressureRatio` 输出

**文件**: `tests/simulation/framework/test_updater.py` (或相关 updater 测试)

- 新增测试验证旧格式表达式被正确迁移（除以 γ）

---

## 待确认事项

1. **版本号**：新的 updater 版本号应该是什么？
2. **operating_condition 序列化结构**：需要确认 `operating_condition.type` 在 dict 中的实际 key 是什么（可能是 `"type"` 或别的字段名），以判断是否为 `LiquidOperatingCondition`。
3. **Liquid 场景确认**：`LiquidOperatingCondition` 的 `op_acoustic_to_static_pressure_ratio = 1.0`，这意味着对 liquid 而言旧表达式（ratio）和新表达式（nondim）在数值上相同，updater 可以跳过，是否正确？
