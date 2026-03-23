# PyPI 发布保护改造计划

## 目标
- 防止 tag 打在错误分支导致误发版。
- 在真正执行 `poetry publish` 前增加人工审批闸门。
- 避免把“单测重跑”塞进发布流程（与你这次 incident 无直接关系）。
- 在审批前向审批人展示可判断信息：commit、分支、分支 HEAD、HEAD CI 状态。

## 设计决策
- 推荐方案：使用 GitHub Actions `environment` 的 Required reviewers 做手动确认。
- 不推荐方案：用 `workflow_dispatch` 增加 `confirm=true` 输入参数。
  - 原因：输入参数不是强审批，容易被误点，审计性也弱。
- 审批信息展示方式：
  - 在 `approval` 前增加 `collect-approval-context` job，写入 `GITHUB_STEP_SUMMARY`。
  - `approval` job 的 `environment.url` 指向当前 run 页面，审批人可一键看到 summary。
- CI 状态口径：
  - 默认读取“目标 release 分支当前 HEAD commit”的 GitHub Check/Status 聚合结果（success/failure/pending）。
  - 同时展示“tag 对应 commit”的 SHA，避免审批人误解是同一个 commit。
- 分支归属判定规则（避免歧义）：
  - 若 tag commit 不在任何 `release-candidate/*`：直接失败。
  - 若在多个 `release-candidate/*`：直接失败并提示人工处理（这是防误发的保护，不做隐式猜测）。
  - 若仅在一个 `release-candidate/*`：作为目标分支继续。

## 分步实施
1. 在 `.github/workflows/pypi-publish.yml` 增加 `validate-release-source` job。
   - `push tag` 场景：校验 `GITHUB_SHA` 必须且仅能属于一个 `origin/release-candidate/*`。
   - `workflow_dispatch` 场景：校验当前分支名必须匹配 `release-candidate/*`。
   - 校验失败直接 `exit 1`。

2. 在同一 workflow 增加 `collect-approval-context` job。
   - `needs: validate-release-source`。
   - 产出并展示以下信息（写入 Step Summary）：
     - tag commit hash（`GITHUB_SHA`）。
     - 命中的 `release-candidate/*` 分支名。
     - 该分支当前 HEAD SHA。
     - 该分支当前 HEAD 的 CI 聚合状态（success/failure/pending）。
   - 使用 `actions/github-script` 或 `gh api` 读取 commit checks/status。

3. 增加 `approval` job。
   - `needs: [validate-release-source, collect-approval-context]`。
   - 设置 `environment: pypi-release`（名称可调整）。
   - 设置 `environment.url` 指向当前 workflow run 页面，便于审批时查看 summary。
   - `runs-on: ubuntu-latest`，仅放一个占位 step（echo），用于触发环境审批。

4. 调整 `publish` job 依赖。
   - `needs: [validate-release-source, collect-approval-context, approval]`。
   - 删除发布前跑 `test.yml` 的依赖（如果存在）。
   - 保持现有版本检查逻辑与发布逻辑不变。

5. 在仓库 GitHub 设置中配置环境审批规则（仓库侧操作，不在代码仓库内）。
   - Settings -> Environments -> `pypi-release`。
   - 打开 Required reviewers，指定审批人。
   - 可选：禁止 self-review。

6. 验证策略。
   - 用错误分支 tag 触发，预期在 `validate-release-source` 失败。
   - 用同一 commit 同时属于多个 `release-candidate/*` 的场景，预期在 `validate-release-source` 失败（避免歧义）。
   - 用正确 `release-candidate/*` tag 触发，预期卡在 `approval` 等待人工批准。
   - 审批前可在 run summary 中看到 4 个判断依据。
   - 批准后才进入 `publish`。

## 风险与注意事项
- 仅改 workflow 文件不会自动创建环境审批规则；必须同时完成仓库 Settings 配置，否则不会真正阻断。
- 分支包含判断基于 `git branch -r --contains`，依赖远端分支拉取完整性，因此 checkout 需 `fetch-depth: 0`。
- “分支 HEAD 的 CI 状态”是分支当前状态，可能与 tag commit 不同步；因此必须同时展示 tag commit SHA。
