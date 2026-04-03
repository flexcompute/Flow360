# Plan: Reference JSON Key Sorting Script

## 目标
写一个 Python 脚本，递归排序 reference JSON 文件的 keys，方便 code review 时看 diff。

## 脚本功能
- 路径: `scripts/sort_ref_json.py`
- 零外部依赖（只用标准库 `json`, `pathlib`, `sys`）
- 两种模式:
  - **修复模式** (默认): `python scripts/sort_ref_json.py` — 原地重写所有未排序的 JSON 文件
  - **检查模式**: `python scripts/sort_ref_json.py --check` — 只检查，不修改，未排序则 exit 1（用于 CI）
- 扫描范围: `tests/` 目录下所有 `*.json` 文件
- 递归排序所有嵌套 dict 的 keys（list 内的 dict 也排序 keys，但 list 元素顺序不变）
- 保持 4 空格缩进 + trailing newline（与现有格式一致）

## 步骤
1. 创建 `scripts/sort_ref_json.py`
2. (可选) 在 CI 配置里加 `python scripts/sort_ref_json.py --check`

## 不做的事
- 不引入 pre-commit framework
- 不修改非 tests/ 目录的 JSON 文件
