# Flow360 schema 拆分：unit primitives 迁移计划（中文）

## 背景与目标
- SimulationParams schema 迁入独立包 `flow360/flow360_schema/`，与 Python client 其他部分解耦
- Pydantic 导出的 JSON Schema 必须兼容 common-schema，用于前端自动表单生成
- unit primitives 新实现，避免改动现有 `flow360/component/simulation/unit_system.py`

## 关键约束/设计原则
- schema 生成不依赖 unyt/numpy；运行时验证可 lazy import unyt
- 序列化仅输出 value（SI 单位）；unit 作为编译期元数据
- 反序列化先解析数值，再做单位/维度验证
- 验证按“单一准则函数 + 组合”的方式拆分
- 零值策略：除温度外，0（无单位）可接受

## 首批建议映射的 common-schema 原语（关键/常用）
- 数字类：Float64、PositiveFloat64、NonNegativeFloat64、NegativeFloat64、NonPositiveFloat64、UnitIntervalFloat64、CenteredUnitIntervalFloat64、UnitBallIntervalFloat64
- 向量类：Vector2Json、Vector3Json、Vector4Json
- 非零向量：NonNullVector2Json、NonNullVector3Json、NonNullVector4Json
- 单位向量：UnitVector2Json、UnitVector3Json、UnitVector4Json
- （可选）矩阵：Matrix2Json、Matrix3Json、Matrix4Json、Matrix2RowMajorJson、Matrix3RowMajorJson、Matrix4RowMajorJson
- （可选）四元数：QuaternionJson、UnitQuaternionJson、NonNullQuaternionJson

> 说明：若需要“PositiveVector3”一类逐分量正值向量，需要确认 common-schema 是否新增相应原语，或在 Flow360 侧以 JSON Schema 组合（allOf + item minExclusive）实现。

## 计划步骤（分阶段）
1. 盘点 & 对齐
   - 列出现有 Flow360 unit primitives 的维度类型、向量/矩阵约束、默认单位
   - 在 common-schema 中确认对应原语的 brand/version/units 约定
   - 形成“形状原语（common-schema 名称）+ 维度元数据（SI units）”的映射表
2. 新 primitives 结构设计
   - 在 `flow360/flow360_schema/primitives/` 下新增模块（如 `validators.py`, `numbers.py`, `vectors.py`, `units.py`）
   - 提供单一准则验证函数（长度、正/负、非零范数、单位向量、范围等）
   - 每个派生类只组合所需准则，避免单一巨型 validate
3. 反序列化/校验实现
   - 输入：纯数值/数组 或 带单位对象
   - Lazy import unyt，仅在 validate/convert 时触发
   - 允许大多数维度的 0（无单位）直通；温度类单独处理（绝对温度 vs 温差）
4. 序列化实现
   - 统一转换到目标 SI 单位
   - 输出仅 value（数值/数组），不输出 units
5. JSON Schema 对齐
   - 覆写 `__get_pydantic_json_schema__` 或 `json_schema_extra`
   - 注入 common-schema 的 `brand` / `version` / `units` 元数据
   - 确保 schema 生成时不触发 unyt 导入
6. 验收与回归
   - 对比导出的 JSON Schema 与 common-schema 示例（关键字段一致）
   - 典型输入的 round-trip（serialize/deserialize）测试
   - 零值与温度特例测试

## 待确认问题
1. 新 plan.md 放在仓库根目录可以吗？还是你希望放到别的路径？
2. “单位元数据”在 JSON schema 中的字段名与格式（`units`/`target_dimension`/其它）是否已有约定？
3. 对于 “PositiveVector3” 等逐分量正值向量：希望在 common-schema 侧新增原语，还是 Flow360 侧组合实现？
4. 温度零值策略：绝对温度允许 0 吗？温差单位是否允许 0？
5. 首批原语清单是否需要收缩（仅标量 + Vector3）或扩展（矩阵/四元数）？
