# PyPI 发布工作流修复计划（tag commit 绑定 + CI gate 收紧）

## 背景与问题
- 问题 1（高优先级）：`workflow_dispatch` 重跑时，`tag_commit_sha` 取的是分支触发点 `GITHUB_SHA`，不是原始 tag 对应 commit，存在“发版代码与 tag 不一致”的风险。
- 问题 2（中优先级）：CI gate 把 `neutral/skipped` 当成可通过信号，并且“只要有任意 CI 信号就 success”，可能在没有真实成功测试信号时放行发布。

## 需要先明确的设计挑战（建议）
- 我建议把手动重跑入口从“输入 version + 当前分支”改成“输入 tag（例如 `v1.2.3`）并强制以该 tag commit 发布”。
- 理由：`version` 输入是弱约束，容易和源码 commit 脱钩；发布系统的单一真实来源应当是 tag -> commit。
- 这属于设计层修正，不只是打补丁，否则仍会残留“逻辑校验与实际 checkout 源不一致”的反模式。

## 分步实施（按步执行，每步完成后等你 review）
1. 修复发布源解析：为 `workflow_dispatch` 增加 `tag` 输入并解析真实 tag commit。
   - 在 `validate-release-source` 中：
   - `push tag`：沿用 `GITHUB_REF/GITHUB_SHA`。
   - `workflow_dispatch`：基于输入 tag 显式解析 `tag_commit_sha`（不是 `GITHUB_SHA`），并复用 release-candidate 分支归属校验。
   - 输出新增 `release_tag`，统一后续作业使用。

2. 绑定实际构建源码到 `tag_commit_sha`（高风险修复闭环）。
   - 在 `publish` job 的 `actions/checkout` 增加 `ref: needs.validate-release-source.outputs.tag_commit_sha`。
   - `RELEASE_VERSION` 从 `release_tag` 推导，避免手工 version 与源码 commit 失配。

3. 收紧 CI gate 判定口径，禁止“只有 skipped/neutral 也通过”。
   - 在 `collect-approval-context` 与 `ci-gate` 两处脚本统一规则：
   - 失败：任意 failed/error。
   - 待定：任意 pending/in_progress，或没有成功信号。
   - 成功：必须存在“至少一个成功信号”（`combined status == success` 或至少一个 `check_run conclusion == success`），且无失败、无 pending。
   - `neutral/skipped` 仅统计展示，不计入成功。

4. 增强审批摘要可观测性。
   - 在 summary 中新增成功/跳过/中性计数，明确 gate 判定依据，降低审批误判。

5. 最小化自检（不跑单测）。
   - 仅做 workflow 语义检查与 diff 自查，确认表达式与输出引用一致，不触发测试流水线。

## 预期结果
- 手动重跑与 tag 触发都绑定到同一个 tag commit。
- 没有成功 CI 信号（只有 skipped/neutral）时，`ci-gate` 必须阻断发布。
- 审批人可直接看到 gate 的成功信号是否真实存在。
