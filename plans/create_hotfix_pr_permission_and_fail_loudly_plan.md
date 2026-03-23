# Create Hotfix PR workflow 修复计划（Fail Loudly + Workflow 改动时创建人工跟踪 PR + Guard）

## 背景
- 当前 `create-hotfix-pr` workflow 在 cherry-pick 后若改动包含 `.github/workflows/*`，使用默认 `GITHUB_TOKEN` 推送会被 GitHub 拒绝（workflow 文件写权限限制）。
- 同时，`git push` 失败后脚本仅 `echo`，没有 `exit 1`，因此 job UI 仍显示成功，误导排障。
- 用户希望即使自动 hotfix 失败，也能自动创建一个“显眼”的 PR 作为待办，避免失败邮件被淹没。

## 目标（按最新决定）
- 保留 `git push` 失败时的 fail-loudly（`::error::` + `exit 1`）。
- 当检测到 hotfix 改动包含 `.github/workflows/*` 时，不再尝试推送真实 cherry-pick 分支，而是创建一个“人工处理跟踪 Draft PR”。
- 该跟踪 PR 通过“占位文件 + CI guard”实现阻止 merge（不依赖额外标题/PR body 约束）。
- 为避免跟踪 PR 被误合并，增加 CI guard：仅对带 `auto-hotfix` 标签的 PR 检查占位文件路径并阻止 merge。
- 不使用 `GITHUB_TOKEN` 权限升级或 GitHub App/PAT 方案（本次不做）。

## 分步实施（一次只做一步）
1. 修改 `.github/workflows/create-hotfix-pr.yml`
   - 删除之前尝试添加的 `permissions.workflows: write`（对 `GITHUB_TOKEN` 不适用）
   - 保留并确认 `git push` 失败分支里的 `::error::...` + `exit 1`

2. 在 `Create and Process Hotfix Branches` 中增加 workflow 改动检测分支
   - 在 `CHANGED_FILES` 计算后检测是否命中 `.github/workflows/`
   - 命中时走“人工处理跟踪 PR”流程，而不是直接 `git push` 真正 hotfix 分支

3. 实现“人工处理跟踪 Draft PR”流程（仅针对 workflow 改动）
   - 基于目标分支创建一个 tracking 分支（建议命名：`manual-hotfix-<target>-pr<原PR号>`）
   - 新增一个标记文件（例如 `.github/hotfix-manual/PR-<原PR号>-to-<target>.md`）
   - 文件内容使用最小必要信息即可（作为“不可 merge 的占位信号”）
   - 提交并 push tracking 分支（该改动不涉及 `.github/workflows/*`）
   - 创建 Draft PR，并打 `auto-hotfix` 标签（如已有）

4. 新增 CI guard（阻止跟踪 PR 被误合并）
   - 新增一个轻量 workflow（建议名称：`hotfix-tracking-guard.yml`）
   - `pull_request` 触发（至少包含 `opened`, `reopened`, `synchronize`, `labeled`, `unlabeled`）
   - 先判断 PR 是否带 `auto-hotfix` 标签；没有则直接成功退出
   - 若带 `auto-hotfix`，检查 PR diff 是否包含占位文件路径（例如 `.github/hotfix-manual/`）
   - 命中则 `::error::` + `exit 1`，提示这是人工跟踪 PR，不可 merge
   - 后续需在分支保护中把该检查设为 required（仓库设置动作，不在本次代码改动中）

5. 信号策略（让 UI 更明显）
   - 创建 tracking PR 成功后，当前 job 仍使用 `::error::` 标红并最终失败（确保 Actions UI 显眼）
   - 若 tracking PR 创建失败，也明确 `::error::` 并失败

6. 本地静态复查（不跑单测）
   - 检查 YAML 缩进、shell 分支与变量引用
   - 检查多目标分支循环下的失败行为（是否需要累计错误后统一退出）

## 设计备注（按当前决定）
- 我不建议“故意做无效内容 PR”或随便改文件；这属于流程反模式，容易被误合并。
- 更稳妥的是“跟踪型 Draft PR”：
  - 专用标记文件路径
  - 同时让 workflow run 失败（红色信号）
  - 通过 CI guard（`auto-hotfix` + 占位文件路径）阻止 merge
- 为减少误伤，guard 不会对所有 PR 生效，而是使用 `auto-hotfix` 标签做预过滤，再结合占位文件路径做最终判定。
- 如果你同意这个策略，我会按步骤执行；并且仍遵守“一次只做一步”。 
