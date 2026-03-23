# 修复 pypi-publish.yml 的三个 Review Issues

## Issue 1: Manual tag validation 过严，拒绝合法的 release tags（中等严重性）

**问题：** `on.push.tags` 使用 `v*.*.*` 模式（较宽松），但 `workflow_dispatch` 路径的验证使用 `^v[0-9]+\.[0-9]+\.[0-9]+$`（严格）。例如 `v1.2.3-beta` 或 `v1.2.3rc1` 可以通过 push trigger 触发，但在 workflow_dispatch 手动重跑时会被拒绝。

**修复方案：** 统一两边的验证标准。将 `on.push.tags` 收紧为严格的 semver 格式 `v[0-9]+.[0-9]+.[0-9]+`（因为我们只发布严格 semver 的版本）。同时把 workflow_dispatch 的 regex 保持一致。

**变更位置：** Line 14 (`on.push.tags` pattern)

## Issue 2: collect-approval-context 的 job outputs 无人消费（低严重性）

**问题：** `collect-approval-context` job 声明了 `tag_commit_ci_state` 和 `ci_signal_available` 作为 job outputs（Lines 96-97），但下游没有任何 job 通过 `needs.collect-approval-context.outputs.*` 引用它们。这是 dead code，可能导致维护混乱。

**修复方案：** 删除 `collect-approval-context` job 的 `outputs` 声明（Lines 95-97）。step-level outputs 仍然存在，在 job summary 中使用，只是不需要作为 job-level outputs 暴露给下游。

**变更位置：** Lines 95-97

## Issue 3: Annotated tags 可能解析到错误的 SHA（中等严重性）

**问题：** 对于 tag-triggered runs（Line 42），`tag_commit_sha` 直接从 `GITHUB_SHA` 获取。但对于 annotated tags，`GITHUB_SHA` 可能是 tag object 的 SHA，而不是底层 commit 的 SHA。这会导致后续的 checkout 和 CI 状态检查使用错误的对象。

注意 workflow_dispatch 路径（Line 77）正确使用了 `git rev-list -n 1` 来解析到实际 commit。

**修复方案：** 将 Line 42 从 `tag_commit_sha="$GITHUB_SHA"` 改为 `tag_commit_sha="$(git rev-list -n 1 "refs/tags/${release_tag}")"` ，与 workflow_dispatch 路径保持一致。

**变更位置：** Line 42

同时 Line 379-380 的 `actions/checkout` 使用 `ref: ${{ needs.validate-release-source.outputs.tag_commit_sha }}`，在修复后会正确得到 commit SHA。

---

## 实施步骤

1. **Step 1:** 修复 Issue 3 — annotated tag SHA 解析
2. **Step 2:** 修复 Issue 2 — 删除 dead job outputs
3. **Step 3:** 修复 Issue 1 — 统一 tag pattern 验证
