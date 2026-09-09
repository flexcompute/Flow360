#!/usr/bin/env python
"""Validate the Python in the docs against the installed Flow360 client.

The grab-and-go snippets and example-library notebooks embed Python that calls
the Flow360 client API. Nothing in the Sphinx build imports ``flow360`` or
checks that code, so when the client renames a symbol, moves an attribute, or
drops a member, the docs drift silently. This script closes that gap.

It is deliberately *static*: it never executes a snippet or submits anything to
the cloud (snippets carry placeholder IDs and reference cloud assets, so they
cannot run end-to-end). Two checks run against every target:

1. **Syntax** -- ``compile()`` each target; a SyntaxError fails the run.
2. **Symbol resolution** -- parse the AST, find the names bound to ``flow360``
   (``import flow360 as fl``, ``import flow360``, ``from flow360 import u``),
   collect every attribute chain rooted at one of those names, and resolve it
   with ``getattr`` against the *live* imported module. Any dotted path that no
   longer resolves (e.g. ``fl.Project.frmo_cloud``) fails the run.

The resolver only follows chains rooted in the flow360 alias, so third-party
calls (pandas, matplotlib, numpy) are invisible to it and cannot raise false
positives -- crucial for notebooks, which pull in many libraries that are not
installed in the docs build environment.

The client is sourced exactly as ``docs/source/conf.py`` sources it: from the
``Flow360`` submodule by default, or from a local ``compute`` checkout when
``local_paths.toml`` is present. So this validates against the *same* client
version the rendered API reference is built from -- the committed submodule
pointer in CI.

Optionally (``--pyright``) it also runs pyright over the snippet ``.py`` files
for instance-level type checking (e.g. drift on ``case.results...``); this is
skipped silently when pyright is not installed.

Exit code is 0 when everything resolves, 1 otherwise.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import importlib
import io
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_SOURCE = REPO_ROOT / "docs" / "source"

SNIPPET_DIR = DOCS_SOURCE / "python_api" / "grab_and_go_snippets" / "_snippets"
NOTEBOOK_DIRS = [
    DOCS_SOURCE / "python_api" / "example_library" / "notebooks",
    DOCS_SOURCE / "quick_start" / "API_quickstart" / "notebooks",
]


# --------------------------------------------------------------------------- #
# Client resolution (mirrors docs/source/conf.py::_flow360_source_paths)
# --------------------------------------------------------------------------- #
def flow360_source_paths() -> list[str]:
    """Return sys.path entries for the client, sourced from the in-repo compute files.

    Mirrors ``conf.py::_flow360_source_paths``: the flow360 package lives beside the
    docs at ``<root>/flex/public/Flow360`` and the schema uses a ``src/`` layout at
    ``<root>/flex/share/flow360-schema/src``. Derived relative to this file so it
    resolves identically in compute and in the flex Copybara mirror.
    """
    # REPO_ROOT is <root>/flex/public/Flow360/docs (this file is docs/tools/...).
    flow360_root = REPO_ROOT.parent  # <root>/flex/public/Flow360 (contains flow360/)
    flex_root = REPO_ROOT.parents[2]  # <root>/flex
    return [
        str(flow360_root),
        str(flex_root / "share" / "flow360-schema" / "src"),
    ]


def import_flow360():
    """Import the flow360 module, silencing its noisy beta version banner.

    Prerelease client builds print a multi-line ASCII banner at import (and on
    first use) via their own logger. The client honours an env var to suppress
    it; set it before import so the validator's report stays readable.
    """
    os.environ.setdefault("FLOW360_SUPPRESS_BETA_WARNING", "1")
    for path in reversed(flow360_source_paths()):
        sys.path.insert(0, path)
    devnull = io.StringIO()
    with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
        module = importlib.import_module("flow360")
    return module


# --------------------------------------------------------------------------- #
# Extracting code from notebooks (mirrors conf.py::generate_notebook_scripts)
# --------------------------------------------------------------------------- #
def notebook_source(nb_path: Path) -> str:
    """Concatenate a notebook's code cells into one importable script string."""
    from nbconvert.exporters import ScriptExporter

    body, _ = ScriptExporter().from_filename(str(nb_path))
    cleaned: list[str] = []
    for line in body.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("# In[") and stripped.endswith("]:"):
            continue
        if stripped.startswith("#"):
            continue
        if stripped == "" and cleaned and cleaned[-1] == "":
            continue
        cleaned.append(line if stripped != "" else "")
    return "\n".join(cleaned)


# --------------------------------------------------------------------------- #
# The symbol resolver
# --------------------------------------------------------------------------- #
@dataclass
class Finding:
    label: str
    kind: str  # "syntax" | "symbol"
    message: str
    lineno: int | None = None


def _flow360_roots(tree: ast.AST, module) -> dict[str, object]:
    """Map each in-scope name bound to flow360 (or a member) to its object.

    Handles:
      import flow360            -> {"flow360": module}
      import flow360 as fl                  -> {"fl": module}
      import flow360.v1 as fl                -> {"fl": flow360.v1}
      from flow360 import u, X               -> {"u": module.u, "X": module.X}
      from flow360.examples import OM6wing   -> {"OM6wing": flow360.examples.OM6wing}
    """
    roots: dict[str, object] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name != "flow360" and not alias.name.startswith("flow360."):
                    continue
                if alias.asname:
                    # `import flow360.v1 as fl` -> fl is the *full* submodule.
                    try:
                        roots[alias.asname] = importlib.import_module(alias.name)
                    except ImportError:
                        roots[alias.asname] = _Missing(alias.name)
                else:
                    # `import flow360` or `import flow360.v1` -> bound name is
                    # `flow360`; dotted chains resolve via getattr from there.
                    roots["flow360"] = module
        elif isinstance(node, ast.ImportFrom):
            # Handle `from flow360 import X` *and* `from flow360.<sub> import X`
            # (e.g. `from flow360.examples import OM6wing`). Without the latter,
            # those names and their attribute uses would be silently unchecked.
            if node.level == 0 and node.module and (
                node.module == "flow360" or node.module.startswith("flow360.")
            ):
                try:
                    base_mod = importlib.import_module(node.module)
                except ImportError:
                    base_mod = None
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    bound = alias.asname or alias.name
                    qualified = f"{node.module}.{alias.name}"
                    # Resolve the imported member now; an unknown member is
                    # itself drift, reported via a sentinel that always fails
                    # to resolve further.
                    if base_mod is None:
                        roots[bound] = _Missing(qualified)
                    else:
                        roots[bound] = getattr(base_mod, alias.name, _Missing(qualified))
    return roots


class _Missing:
    """Sentinel for an imported flow360 name that no longer exists.

    `name` is the fully-qualified dotted path (e.g. `flow360.examples.OM6wing`).
    """

    def __init__(self, name: str):
        self.name = name


def _chain_from_attribute(node: ast.Attribute) -> tuple[list[str], ast.AST] | None:
    """Flatten an Attribute node into (root_name, attr, attr, ...) if rooted in a Name."""
    parts: list[str] = []
    cur: ast.AST = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
        parts.reverse()
        return parts, node
    return None


def _outermost_attribute_nodes(tree: ast.AST) -> list[ast.Attribute]:
    """Attribute nodes that are not themselves the .value of another Attribute."""
    inner: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Attribute):
            inner.add(id(node.value))
    return [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Attribute) and id(n) not in inner
    ]


def resolve_symbols(src: str, label: str, module) -> list[Finding]:
    """Resolve every flow360-rooted attribute chain against the live module."""
    findings: list[Finding] = []
    try:
        tree = ast.parse(src, filename=label)
    except SyntaxError as exc:
        return [Finding(label, "syntax", f"{exc.msg}", exc.lineno)]

    roots = _flow360_roots(tree, module)
    if not roots:
        return findings  # nothing imports flow360 here; nothing to check

    seen: set[tuple[str, int]] = set()
    for attr_node in _outermost_attribute_nodes(tree):
        flat = _chain_from_attribute(attr_node)
        if flat is None:
            continue
        chain, node = flat
        root_name, *tail = chain
        if root_name not in roots:
            continue

        base = roots[root_name]
        dotted = ".".join(chain)
        key = (dotted, getattr(node, "lineno", -1))
        if key in seen:
            continue
        seen.add(key)

        if isinstance(base, _Missing):
            findings.append(
                Finding(
                    label,
                    "symbol",
                    f"`{base.name}` (imported name `{root_name}`) does not exist in the client",
                    getattr(node, "lineno", None),
                )
            )
            continue

        obj = base
        resolved = root_name
        for attr in tail:
            try:
                obj = getattr(obj, attr)
            except AttributeError:
                # The symbol genuinely does not exist on the resolved object:
                # this is the API drift we are looking for.
                findings.append(
                    Finding(
                        label,
                        "symbol",
                        f"`{dotted}` -> `{resolved}` has no attribute `{attr}`",
                        getattr(node, "lineno", None),
                    )
                )
                break
            except Exception:  # noqa: BLE001
                # The attribute exists but its getter has runtime side effects
                # that fail in this context (e.g. an example object's
                # `.geometry` property reads files that were never downloaded).
                # The name resolves, so this is not drift; we simply cannot
                # descend past it.
                break
            resolved = f"{resolved}.{attr}"
    return findings


# --------------------------------------------------------------------------- #
# pyright (optional, snippet dir only)
# --------------------------------------------------------------------------- #
def run_pyright(extra_path: str) -> int:
    """Run pyright over the snippet dir if available. Returns its exit code, or 0 if absent."""
    if not SNIPPET_DIR.exists() or not any(SNIPPET_DIR.glob("*.py")):
        return 0
    # Resolve how to invoke pyright; treat genuine absence as a skip (return 0)
    # rather than a failure. The `python -m pyright` fallback would otherwise
    # exit 1 ("No module named pyright") when the module is not installed.
    import importlib.util

    exe = shutil.which("pyright")
    if exe:
        cmd = [exe]
    elif importlib.util.find_spec("pyright") is not None:
        cmd = [sys.executable, "-m", "pyright"]
    else:
        print("  (pyright not installed; skipping type check)")
        return 0
    env = dict(os.environ)
    # Let pyright import the client the same way we do.
    env["PYTHONPATH"] = os.pathsep.join(
        [extra_path, env.get("PYTHONPATH", "")]
    ).strip(os.pathsep)
    try:
        proc = subprocess.run(
            cmd + [str(SNIPPET_DIR)], env=env, capture_output=True, text=True
        )
    except FileNotFoundError:
        print("  (pyright not installed; skipping type check)")
        return 0
    if proc.stdout:
        print(proc.stdout)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)
    return proc.returncode


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
@dataclass
class Target:
    label: str
    source: str


def collect_targets(include_notebooks: bool) -> list[Target]:
    targets: list[Target] = []
    if SNIPPET_DIR.exists():
        for py in sorted(SNIPPET_DIR.glob("*.py")):
            targets.append(
                Target(str(py.relative_to(REPO_ROOT)), py.read_text(encoding="utf-8"))
            )
    if include_notebooks:
        for nb_dir in NOTEBOOK_DIRS:
            if not nb_dir.exists():
                continue
            for nb in sorted(nb_dir.glob("*.ipynb")):
                targets.append(
                    Target(str(nb.relative_to(REPO_ROOT)), notebook_source(nb))
                )
    return targets


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-notebooks",
        action="store_true",
        help="validate snippets only, skip example/quickstart notebooks",
    )
    parser.add_argument(
        "--pyright",
        action="store_true",
        help="additionally run pyright over the snippet .py files (if installed)",
    )
    args = parser.parse_args()

    print("Validating doc Python against the Flow360 submodule client\n")
    module = import_flow360()
    version = getattr(module, "__version__", "unknown")
    print(f"  client: flow360 {version}")
    print(f"  source: {flow360_source_paths()[0]}\n")

    targets = collect_targets(include_notebooks=not args.no_notebooks)
    if not targets:
        print(
            "ERROR: no targets found (no _snippets/*.py and no notebooks). "
            "Check the snippet/notebook paths; an empty set means validation "
            "was silently skipped."
        )
        return 1

    all_findings: list[Finding] = []
    for target in targets:
        findings = resolve_symbols(target.source, target.label, module)
        status = "FAIL" if findings else "ok"
        print(f"  [{status:>4}] {target.label}")
        for f in findings:
            loc = f":{f.lineno}" if f.lineno else ""
            print(f"           {f.kind}{loc}: {f.message}")
        all_findings.extend(findings)

    rc = 0
    if all_findings:
        n_syntax = sum(1 for f in all_findings if f.kind == "syntax")
        n_symbol = sum(1 for f in all_findings if f.kind == "symbol")
        print(
            f"\nFAILED: {len(all_findings)} issue(s) "
            f"({n_syntax} syntax, {n_symbol} unresolved symbol) "
            f"across {len({f.label for f in all_findings})} file(s)."
        )
        rc = 1
    else:
        print(f"\nPASSED: {len(targets)} file(s) validated, no API drift found.")

    if args.pyright:
        print("\nRunning pyright over snippets...")
        if run_pyright(flow360_source_paths()[0]) != 0:
            rc = 1

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
