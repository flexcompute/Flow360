# RotationVolume/RotationSphere 拆分与 25.9.2 升级计划

## 目标
1. 将当前 `RotationVolume` 拆分为两个 schema：
- `RotationVolume`：仅支持非球体实体（`Cylinder`、`AxisymmetricBody`）
- `RotationSphere`：仅支持球体实体（`Sphere`）
2. 增加 `25.9.2` updater：把旧 JSON 中 `type="RotationVolume"` 且实体为 `Sphere` 的 zone 迁移为 `type="RotationSphere"`。
3. 更新测试到新语法与新升级路径。

## 反模式提醒（建议避免）
- 直接复制一份 `RotationVolume` 代码再改名为 `RotationSphere` 会造成重复校验逻辑，后续维护易漂移（典型 duplication anti-pattern）。
- 更稳妥方案：引入一个私有公共基类（例如 `_RotationVolumeBase`）承载公共字段和公共校验，然后两个子类只保留各自差异字段与校验。

## 设计决策（建议方案）
1. `volume_params.py` 结构
- 新增 `_RotationVolumeBase`：
  - 公共字段：`name`、`enclosed_entities`、`stationary_enclosed_entities`
  - 公共校验：单实体限制、名称长度限制（Cylinder）、`enclosed_entities` beta 约束、`stationary_enclosed_entities` 约束、subset 约束、surface existence
- `RotationVolume(_RotationVolumeBase)`：
  - `type="RotationVolume"`
  - `entities: EntityList[Cylinder, AxisymmetricBody]`
  - `spacing_axial/radial/circumferential` 均必填（可改为非 Optional）
  - 去掉 Sphere 分支相关校验与文档
- `RotationSphere(_RotationVolumeBase)`：
  - `type="RotationSphere"`
  - `entities: EntityList[Sphere]`
  - `spacing_circumferential` 必填
  - 不暴露 `spacing_axial/radial`（避免 schema 继续接受旧字段）
  - 保留 sphere 的 beta mesher 约束
- `RotationCylinder` 继续继承 `RotationVolume`（如果你确认仍保留该 deprecated 类）

2. 参数联合类型与导出
- 在 `meshing_param/params.py` 的 `VolumeZonesTypes`、`ZoneTypesModular` 增加 `RotationSphere`
- 更新 `flow360/__init__.py` 导出 `RotationSphere`
- 其他 import/type hint 同步补齐（translator/simulation_params 等）

3. translator 适配
- `volume_meshing_translator.py`
  - `spherical_refinement_translator` 的类型改为 `RotationSphere`
  - `rotation_volume_translator` 接受 `Union[RotationVolume, RotationSphere]`
  - sliding interface 收集时单独加入 `RotationSphere`
- `surface_meshing_translator.py`
  - `_get_volume_zones` 白名单加入 `"RotationSphere"`
  - `has_rotation_zones` 判定加入 `RotationSphere`
  - 过滤 `stationary_enclosed_entities` 的逻辑覆盖 `RotationSphere`

4. updater（25.9.2）
- 在 `framework/updater.py` 新增 `_to_25_9_2`
- 处理路径：
  - `meshing.volume_zones`
  - `meshing.zones`（modular）
- 迁移规则：
  - 若 `zone["type"] == "RotationVolume"` 且 `entities.stored_entities[0].private_attribute_entity_type_name == "Sphere"`，改为 `zone["type"] = "RotationSphere"`
  - 删除 `spacing_axial`、`spacing_radial`（如果存在）
  - 保留 `spacing_circumferential`、`enclosed_entities`、`stationary_enclosed_entities` 等
- 将 `(Flow360Version("25.9.2"), _to_25_9_2)` 加入 `VERSION_MILESTONES`

5. 测试更新
- `tests/simulation/params/meshing_validation/test_meshing_param_validation.py`
  - Sphere 相关正/反例迁移到 `RotationSphere`
  - 新增：`RotationVolume` 传入 `Sphere` 应报错
- `tests/simulation/translator/test_volume_meshing_translator.py`
  - Sphere sliding interface 测试改用 `RotationSphere`
- `tests/simulation/translator/test_surface_meshing_translator.py`
  - 白名单相关断言覆盖 `RotationSphere`
- `tests/simulation/params/test_draft_entities_setup.py`
  - sphere case 改用 `RotationSphere`
- `tests/simulation/test_updater.py`
  - 新增 `_to_25_9_2` 单测 + `updater("25.9.1" -> "25.9.2")` 路径测试

## 分步实施（每步完成后等待你 review）
1. 仅改 `volume_params.py`：引入 `RotationSphere` 与模型拆分（不改 translator/updater/tests）。
2. 改参数联合类型与全局导出（`params.py`、`__init__.py`、必要 import）。
3. 改 translator（volume + surface）使新类型可翻译。
4. 加入 `25.9.2` updater 与里程碑注册。
5. 修改/新增测试到新语法与新 updater 行为。

## 本计划中的关键待确认点（interview）
1. 对于旧 JSON 中 `RotationVolume + Sphere` 且同时带了 `spacing_axial/radial` 的情况：
- 方案 A（推荐）：updater 直接删除这两个字段并迁移为 `RotationSphere`
- 方案 B：检测到即报错（fail loudly）
2. `RotationCylinder` 是否继续保留 deprecated 兼容？
3. `RotationSphere` 是否需要支持 `stationary_enclosed_entities`（我当前方案是支持，保持与现有旋转区能力一致）？
4. updater 是否只覆盖 `meshing.volume_zones`，还是同时覆盖 modular 的 `meshing.zones`（我建议两者都覆盖）？
