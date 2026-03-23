# Flow360 Schema 迁移与重构计划 (Migration Plan)

## 目标 (Objectives)
将 `Unit` 原语迁移至独立的 `flow360_schema` 包中，实现以下关键改进：
1.  **解耦 (Decoupling):** 移除 JSON Schema 生成过程对 `unyt` 和 `numpy` 的硬依赖（采用 Lazy Import）。
2.  **通用 Schema 兼容 (Common-Schema Compatibility):** 生成的 JSON Schema 仅包含数值（Value-only），不再包含 Python 特有类型或 Unit 字符串字段。
3.  **组合式验证 (Composable Validation):** 将原本臃肿的中心化验证函数拆解为独立、可组合的原子验证器。
4.  **无单位零值 (Unitless Zero):** 允许 `0` 或 `0.0` 作为大部分物理量的有效输入，无需指定单位。
5.  **SI 单位序列化 (SI Serialization):** 序列化时自动转换为 SI 标准单位数值。

---

## 详细实施步骤 (Detailed Implementation Steps)

### 步骤 1: 基础设施构建 - 验证器与工具 (Infrastructure - Validators & Utils)
**文件位置:** `flow360/flow360_schema/primitives/unit_utils.py` (或新建 `validators.py`)

**任务:**
建立无需立即导入 `unyt` 的验证基础设施。

*   **`LazyUnyt` 类:**
    *   封装 `unyt` 和 `numpy` 的导入逻辑。
    *   仅在运行时（Runtime）进行单位换算或数值检查时加载。
*   **原子验证器 (Atomic Validators):**
    *   `allow_unitless_zero(value)`: 检查输入是否为 `0`，如果是，则跳过后续单位解析，直接返回带有基本单位的零值对象。
    *   `parse_unit_string(value, default_unit=None)`: 将字符串输入解析为 `unyt` 数量对象 (Quantity)。
    *   `validate_dimension(value, expected_dim)`: 检查物理量维度是否匹配（例如 Length vs Mass）。
    *   `validate_positivity(value)`: 检查数值正负性。
    *   `validate_vector_shape(value, length)`: 检查数组/向量长度。
*   **序列化器 (Serializer):**
    *   `serialize_to_si(value)`: 将对象转换为 SI 单位的纯数值（float 或 list[float]）。

### 步骤 2: 基础单位类型定义 (Base Unit Definition)
**文件位置:** `flow360/flow360_schema/primitives/base.py`

**任务:**
定义所有物理量的基类，处理 Pydantic 集成。

*   **`Flow360UnitBase` 类:**
    *   利用 `Annotated` 和 `pydantic.GetJsonSchemaHandler`。
    *   **Schema 生成 (`__get_pydantic_json_schema__`):** 
        *   输出类型定义为 `number` (标量) 或 `array` (向量)。
        *   添加元数据（metadata），如 `original_unit_dimension`，但不破坏通用 Schema 结构。
    *   **核心验证逻辑 (`__get_pydantic_core_schema__`):**
        *   按顺序调用验证链：`Unitless Zero` -> `Parse String` -> `Check Dimension` -> `Check Constraints` -> `Convert to SI` (可选，视存储需求而定，通常在序列化时做)。

### 步骤 3: 核心原语实现 (Core Primitives Implementation)
**文件位置:** `flow360/flow360_schema/primitives/unit_objects.py`

**任务:**
基于基类实现具体的物理量类型。此阶段不修改旧代码，只在 `flow360_schema` 中创建新类型。

*   **基础类型:**
    *   `Scalar` (无量纲标量)
    *   `Vector3` (无量纲 3D 向量)
    *   `Direction` (模不为 0 的向量)
*   **关键物理量 (Critical Dimensions):**
    *   `Length` / `PositiveLength`
    *   `Angle`
    *   `Mass` / `PositiveMass`
    *   `Time` / `PositiveTime`
    *   `Temperature` (绝对温度，需特殊处理零值逻辑，不允许无单位 0)
    *   `TemperatureDifference` (温差，允许无单位 0)
    *   `Velocity`
    *   `Pressure`
    *   `Density`

**命名规范:** 使用 PascalCase，例如 `Length` 而非 `LengthType`，以保持与 Common Schema 的风格一致。

### 步骤 4: 验证与测试 (Verification)
**任务:**
编写独立的测试脚本验证新架构。

*   **验证点 1: 依赖隔离**
    *   测试: `import flow360.flow360_schema` 时，系统模块列表 (`sys.modules`) 中不应包含 `unyt`。
*   **验证点 2: Schema 输出**
    *   测试: `Length.model_json_schema()` 应返回 `{"type": "number"}` 结构，无自定义复杂类型。
*   **验证点 3: 零值处理**
    *   测试: 输入 `0` 给 `Length` 应通过验证；输入 `0` 给 `Temperature` 应报错（或根据业务逻辑决定）。
*   **验证点 4: 序列化**
    *   测试: `Length(1, "km")` 序列化为 JSON 后应为 `1000.0`。

---

## 风险与注意事项 (Risks & Notes)
1.  **Temperature 歧义:** 绝对温度（K/C/F）与温差（Delta K）在 `unyt` 中处理较复杂。新 Schema 中必须明确区分 `Temperature` 和 `TemperatureDifference`，且 `Temperature` 不应接受无单位的 0（物理意义不明）。
2.  **向后兼容性:** 新包仅供新 Schema 使用，旧的 `flow360/component/simulation/unit_system.py` 暂时保留不动，直到完成完全迁移。
3.  **Frontend 同步:** 需确认前端 Common Schema 对 metadata 的解析方式，确保生成的 Schema 能被前端表单生成器正确识别为带有单位选择器的输入框（虽然传值是 SI，但前端可能需要显示单位）。
