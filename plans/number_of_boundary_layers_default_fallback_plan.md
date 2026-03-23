# edge_split_layers=0 视为 deactivated - 实施计划

## 目标
统一 `edge_split_layers` 的行为语义：
- `0` 表示 deactivated。
- `>0` 表示用户启用并设置层数。

## 规则（你刚确认的版本）
1. 字段类型：`int`，约束 `ge=0`。
2. 非 beta mesher：
- `edge_split_layers == 0`：不输出 warning。
- `edge_split_layers > 0`：输出 warning（该功能仅 beta 支持，将被忽略）。
3. beta mesher translator：`numEdgeSplitLayers` 直接输出该整数值。

## 步骤
1. 修改 validator 触发条件（只做这一步）。
2. 修改 volume translator 的 `numEdgeSplitLayers` 映射为整数直传。
3. 同步更新相关测试与 reference JSON。

## 执行约定
按你的要求，一次只改一步，每步改完停下等你 review。
