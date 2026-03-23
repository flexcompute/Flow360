# billing_method 重构计划

## 背景
根据 Review 反馈，需要：
1. 撤销 `exclude_none=True`，改为手动排除 `job_type`
2. 将 `billing_method` 收拢到 Case 专属逻辑
3. 添加 VGPU 账户校验

## 核心设计思路

将 **用户概念** (`billing_method`) 和 **API 概念** (`job_type`) 分层：
- `billing_method`（用户接口层）仅存在于 `run_case()` 中
- `job_type`（API 层）通过 `_run()` → `run_up_to_target_asset()` → `DraftRunRequest` 传递
- VGPU 账户检查在 `run_case()` 中完成（fail fast）

这样做的好处：
- `generate_surface_mesh()` / `generate_volume_mesh()` 完全不感知 billing 概念
- `_run()` 只传递一个 Optional `job_type`，与 `fork_from` / `interpolate_to_mesh` 等 Case-only 参数保持一致的模式
- 所有用户侧校验逻辑（VGPU 开关检查、draft_only 冲突等）集中在 `run_case()`

---

## 实现步骤

### Step 1: 手动排除 `job_type`（而非 `exclude_none=True`）

**文件**: `flow360/component/simulation/web/draft.py`

- 撤回 `model_dump(by_alias=True, exclude_none=True)` → 恢复 `model_dump(by_alias=True)`
- 在 `model_dump()` 之后，手动 pop 掉值为 None 的 `job_type`：
  ```python
  request_body = run_request.model_dump(by_alias=True)
  if request_body.get("job_type") is None:
      request_body.pop("job_type", None)
  ```

### Step 2: 将 `billing_method` → `job_type` 映射移到 `run_case()`

**文件**: `flow360/component/simulation/web/draft.py`
- `run_up_to_target_asset()`: 将参数 `billing_method` 改为 `job_type: Optional[Literal["TIME_SHARED_VGPU", "FLEX_CREDIT"]]`
- 删除 `run_up_to_target_asset()` 中的 `billing_method → job_type` 映射逻辑和相关 log.info

**文件**: `flow360/component/project.py`
- `run_case()`: 添加 `billing_method → job_type` 的映射逻辑（含 log.info 消息）
- `_run()`: 参数从 `billing_method` 改为 `job_type: Optional[Literal["TIME_SHARED_VGPU", "FLEX_CREDIT"]] = None`
- `_run()` 中传给 `draft.run_up_to_target_asset()` 的参数也从 `billing_method` 改为 `job_type`

### Step 3: 添加 VGPU 账户校验

**文件**: `flow360/component/project.py`（或新建 helper）
- 在 `run_case()` 中，当 `billing_method == "VirtualGPU"` 时：
  1. 调用 `http.get("flow360/account")` 获取账户信息
  2. 检查响应中 `timeSharedVGpuEnabled` 字段是否为 True
  3. 如果不是 True，抛出清晰的错误信息，阻止提交
- 细化 `draft.py` 中的 `# TODO: Exception capture` 注释

### Step 4: 清理

- 确认 `print(">> billing_method = ", billing_method)` 已删除（用户表示已做）
