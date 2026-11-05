#!/usr/bin/env python
"""Validate the Python in the docs and examples against the in-tree Flow360 client.

Doc code drifts silently: when the client renames a field or drops a symbol,
nothing in the Sphinx build imports ``flow360`` or checks the samples, so the
page keeps rendering the old API. This script closes that gap across **every**
surface that carries Python:

1. ``grab_and_go_snippets/_snippets/*.py``
2. example-library and quickstart notebooks
3. ``examples/**`` -- the client's own example corpus, published to the public
   Flow360 repo next to the docs, and previously seen only by ``black``
4. inline ``.. code-block:: python`` in rst and ```` ```python ```` fences in md
   -- the largest surface by count, and previously unchecked entirely

It is deliberately *static*: it never executes a sample or submits anything to
the cloud. Snippets carry placeholder IDs and elided ``...`` bodies by design,
so they cannot run end-to-end. Three checks run:

* **Syntax** -- ``compile()`` each target. For inline prose blocks, which are
  often deliberately elided, a parse failure is reported as a non-fatal
  ``fragment`` rather than an error (``--strict-fragments`` gates them).
* **Symbol resolution** -- resolve every attribute chain rooted at a name bound
  to ``flow360`` against the live module, so ``fl.Project.frmo_cloud`` fails.
  Inline blocks omit the import (the reader is assumed to have it), so the
  conventional ``fl``/``flow360`` aliases are assumed there unless the block
  binds those names itself.
* **Call keywords** -- bind each call's keyword arguments against the callee.
  This is the drift symbol resolution structurally cannot see: after a field
  rename, ``fl.Fluid`` still resolves while ``fl.Fluid(turbulence_model=...)``
  is dead. The grab-and-go snippets stay static-only by design, so this is the
  check that carries them.

The keyword check runs in two stages, because a pydantic model's accepted
keywords are not knowable from its signature -- ``model_validator(mode="before")``
hooks and field aliases both widen them invisibly. The signature proposes a
candidate; pydantic itself confirms it via a construction probe. Without that
confirmation the check reported false positives on correct pages (verified
against ``fl.UserVariable``, ``fl.NavierStokesSolver`` and ``fl.CustomVolume``).

The resolver only follows chains rooted in a flow360 binding, so third-party
calls (pandas, matplotlib, numpy) are invisible to it and cannot raise false
positives -- crucial for notebooks, which import many libraries that are not
installed in the docs build environment.

The client is sourced exactly as ``docs/source/conf.py`` sources it, so this
validates against the *same* client the rendered API reference is built from.

Optionally (``--pyright``) it also runs pyright over the snippet ``.py`` files
for instance-level type checking; skipped silently when pyright is absent.

Exit code is 0 when everything resolves, 1 otherwise.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import importlib
import inspect
import io
import logging
import os
import shutil
import re
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_SOURCE = REPO_ROOT / "docs" / "source"

SNIPPET_DIR = DOCS_SOURCE / "python_api" / "grab_and_go_snippets" / "_snippets"
# Snippets of the Reports API page. The page and its snippets are fetched from
# flex by tools/fetch_report_docs.py; the directory is absent until that ran.
REPORT_SNIPPET_DIR = DOCS_SOURCE / "python_api" / "API_reference" / "reports_snippets"
SNIPPET_DIRS = [SNIPPET_DIR, REPORT_SNIPPET_DIR]
# Every notebook directory that lives in the docs tree. Both are real committed
# sources today; if the example-library corpus is ever single-sourced elsewhere
# and merely materialized here at build time, its entry moves out of this list
# (a materialized path finds nothing in CI, where no Sphinx build has run).
NOTEBOOK_DIRS = [
    DOCS_SOURCE / "quick_start" / "API_quickstart" / "notebooks",
    DOCS_SOURCE / "python_api" / "example_library" / "notebooks",
]

# The client's own example scripts. They are published to the public Flow360
# repo alongside the docs, so they are doc-facing content in every sense that
# matters -- but until now only `black` ever looked at them.
EXAMPLES_DIR = REPO_ROOT.parent / "examples"

# Prose that embeds Python inline rather than including a file. These carry the
# bulk of the user-facing code and were previously unchecked entirely.
INLINE_ROOTS = [DOCS_SOURCE]

# Names the docs use that the client intends to expose but currently does not.
# These are *client* gaps, not doc drift: fixing the page would document around
# the bug and leave the client's own error messages pointing at a name nobody
# can import. Each entry must name the tracking ticket, and must be deleted in
# the same PR that lands the export.
#
# `SurfaceSelector` is defined in flow360_schema/framework/entity/entity_selector.py
# alongside EdgeSelector, VolumeSelector and BodyGroupSelector, but only the other
# three are re-exported in flow360/_public_namespace.py -- while the client's own
# preview_selector() error tells users to "Use fl.SurfaceSelector, fl.EdgeSelector,
# fl.VolumeSelector, or fl.BodyGroupSelector".
KNOWN_CLIENT_GAPS = {
    "fl.SurfaceSelector": "FXC-10751: not re-exported in flow360/_public_namespace.py",
    "flow360.SurfaceSelector": "FXC-10751: not re-exported in flow360/_public_namespace.py",
}

# Inline blocks are fragments: they are written assuming the reader already has
# `import flow360 as fl` in scope, so the alias is almost never bound in-block.
# Without an assumed binding the resolver finds no flow360 root and skips them,
# which is exactly how the surface stayed invisible.
ASSUMED_ALIASES = ("fl", "flow360")


# --------------------------------------------------------------------------- #
# Client resolution (mirrors docs/source/conf.py::_flow360_source_paths)
# --------------------------------------------------------------------------- #
# Packages whose attribute chains are resolved. `flow360` comes from the in-tree
# client; `flexcompute.flow_report` (the Reports companion package documented in
# python_api/API_reference/reports.rst) is installed from PyPI at the version
# pinned in pyproject.toml's `docs` extra, so its snippets are checked against
# exactly the release the API page autodocs.
ROOT_PACKAGES = ("flow360", "flexcompute.flow_report")


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


def import_root_packages() -> dict[str, object]:
    """Import every package in ROOT_PACKAGES; fail loudly if one is missing.

    A missing package would silently skip every snippet rooted in it, which is
    the failure mode this validator exists to prevent.
    """
    modules = {"flow360": import_flow360()}
    for name in ROOT_PACKAGES[1:]:
        try:
            modules[name] = importlib.import_module(name)
        except ImportError as exc:
            raise SystemExit(
                f"ERROR: cannot import `{name}` ({exc}). Install the docs extra "
                "(`uvx poetry install --extras docs`) so the report snippets are checked."
            ) from exc
    return modules


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
# Extracting inline code blocks from prose (.rst directives, .md fences)
# --------------------------------------------------------------------------- #
# The directive may sit in a list-table cell, where a "- "/"* " bullet precedes
# it on the same line. `indent` therefore captures everything up to the "..",
# so the body-indent comparison uses the directive's real column.
RST_BLOCK_RE = re.compile(
    r"^(?P<indent>[ \t]*(?:[-*+][ \t]+)?)\.\.[ \t]+code-block::[ \t]*(?:python|py)[ \t]*$"
)
RST_OPTION_RE = re.compile(r"^[ \t]*:[\w-]+:")
# `indent` also takes blockquote markers (`  > ```python`), which the GUI guide
# uses for FAQ answers; body lines carry the same prefix and lose it on extraction.
MD_FENCE_RE = re.compile(
    r"^(?P<indent>[ \t]*(?:>[ \t]?)*)(?P<fence>```+|~~~+)[ \t]*(?:python|py)[ \t]*$"
)


def _indent_width(line: str) -> int:
    return len(line) - len(line.lstrip())


def extract_rst_python_blocks(text: str) -> list[tuple[str, int]]:
    """Return (code, first_body_line) for every ``code-block:: python`` in an rst file.

    Directives nest (a code-block inside a ``.. note::`` is indented), so the
    body is everything indented deeper than the directive itself, up to the first
    non-blank line that dedents back to or past it. Options (``:linenos:``) and
    the blank separator are consumed before the body starts.
    """
    lines = text.splitlines()
    blocks: list[tuple[str, int]] = []
    i = 0
    while i < len(lines):
        match = RST_BLOCK_RE.match(lines[i])
        if not match:
            i += 1
            continue
        directive_indent = len(match.group("indent").expandtabs())
        j = i + 1
        # Consume directive options and the blank line that follows them.
        while j < len(lines) and (RST_OPTION_RE.match(lines[j]) or not lines[j].strip()):
            j += 1
        body_start = j
        body: list[str] = []
        while j < len(lines):
            line = lines[j]
            if not line.strip():
                body.append("")
                j += 1
                continue
            if _indent_width(line.expandtabs()) <= directive_indent:
                break
            body.append(line)
            j += 1
        if any(line.strip() for line in body):
            blocks.append((textwrap.dedent("\n".join(body)), body_start + 1))
        i = j
    return blocks


def extract_md_python_blocks(text: str) -> list[tuple[str, int]]:
    """Return (code, first_body_line) for every fenced python block in a md file."""
    lines = text.splitlines()
    blocks: list[tuple[str, int]] = []
    i = 0
    while i < len(lines):
        match = MD_FENCE_RE.match(lines[i])
        if not match:
            i += 1
            continue
        fence = match.group("fence")
        indent = match.group("indent")
        quote = re.compile(r"^[ \t]*" + r"(?:>[ \t]?)" * indent.count(">"))
        body_start = i + 1
        body: list[str] = []
        j = body_start
        while j < len(lines):
            line = quote.sub("", lines[j], count=1) if ">" in indent else lines[j]
            stripped = line.strip()
            # A closing fence is the same marker (or longer) and nothing else.
            if stripped.startswith(fence[0] * len(fence)) and set(stripped) == {fence[0]}:
                break
            if ">" not in indent and line.startswith(indent):
                line = line[len(indent) :]
            body.append(line)
            j += 1
        if any(line.strip() for line in body):
            blocks.append((textwrap.dedent("\n".join(body)), body_start + 1))
        i = j + 1
    return blocks


# --------------------------------------------------------------------------- #
# The symbol resolver
# --------------------------------------------------------------------------- #
@dataclass
class Finding:
    label: str
    kind: str  # syntax | symbol | keyword | fragment | client-gap | stale-gap
    message: str
    lineno: int | None = None


def _is_rooted(dotted: str) -> bool:
    """True if `dotted` is one of ROOT_PACKAGES or a submodule of one."""
    return any(dotted == pkg or dotted.startswith(pkg + ".") for pkg in ROOT_PACKAGES)


def _import_from_member(base_mod, name: str, qualified: str):
    """Resolve one name from ``from <base_mod> import <name>``.

    ``getattr`` alone is not the whole of Python's import semantics. When a
    package does not re-export a submodule in its ``__init__``, the attribute
    is absent until something imports it, yet ``from pkg import sub`` still
    succeeds -- the import system falls back to importing ``pkg.sub``.

    ``flow360.component.simulation.migration`` is exactly that shape: its
    ``__init__`` exports only ``BETDisk``, so a getattr-only resolver reported
    the working ``from ... import ProbeOutput`` as drift. Mirror the fallback
    so the checker agrees with the interpreter.
    """
    member = getattr(base_mod, name, None)
    if member is not None:
        return member
    try:
        return importlib.import_module(f"{base_mod.__name__}.{name}")
    except ImportError:
        return _Missing(qualified)


def _flow360_roots(tree: ast.AST, module, assume_aliases: bool = False) -> dict[str, object]:
    """Map each in-scope name bound to a root package (or a member) to its object.

    Handles, for every package in ROOT_PACKAGES:
      import flow360            -> {"flow360": module}
      import flow360 as fl                  -> {"fl": module}
      import flow360.v1 as fl                -> {"fl": flow360.v1}
      from flow360 import u, X               -> {"u": module.u, "X": module.X}
      from flow360.examples import OM6wing   -> {"OM6wing": flow360.examples.OM6wing}
      from flexcompute.flow_report import Report -> {"Report": flexcompute.flow_report.Report}
    """
    roots: dict[str, object] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if not _is_rooted(alias.name):
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
                    # Likewise `import flexcompute.flow_report` binds `flexcompute`.
                    top = alias.name.split(".")[0]
                    if top == "flow360":
                        roots[top] = module
                    else:
                        try:
                            roots[top] = importlib.import_module(top)
                        except ImportError:
                            roots[top] = _Missing(top)
        elif isinstance(node, ast.ImportFrom):
            # Handle `from flow360 import X` *and* `from flow360.<sub> import X`
            # (e.g. `from flow360.examples import OM6wing`). Without the latter,
            # those names and their attribute uses would be silently unchecked.
            if node.level == 0 and node.module and _is_rooted(node.module):
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
                        roots[bound] = _import_from_member(base_mod, alias.name, qualified)

    if assume_aliases:
        # Inline prose blocks are fragments written for a reader who already has
        # `import flow360 as fl` above; the import is not repeated in the block.
        # Bind the conventional aliases so those chains are checked -- but never
        # shadow a name the block binds itself (a local `fl = ...` or a loop
        # variable would otherwise be resolved against the client and misreported).
        bound_locally = _locally_bound_names(tree)
        for alias in ASSUMED_ALIASES:
            if alias not in roots and alias not in bound_locally:
                roots[alias] = module
    return roots


def _locally_bound_names(tree: ast.AST) -> set[str]:
    """Names the block assigns, imports, or otherwise binds itself."""
    bound: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                bound.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name != "*":
                    bound.add(alias.asname or alias.name)
    return bound


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
    return [n for n in ast.walk(tree) if isinstance(n, ast.Attribute) and id(n) not in inner]


def _resolve_chain(chain: list[str], roots: dict[str, object]):
    """Walk a dotted chain from a flow360 root. Returns the object or None."""
    root_name, *tail = chain
    if root_name not in roots:
        return None
    obj = roots[root_name]
    if isinstance(obj, _Missing):
        return None
    for attr in tail:
        try:
            obj = getattr(obj, attr)
        except AttributeError:
            return None
        except Exception:  # noqa: BLE001 -- property with side effects; stop here
            return None
    return obj


_PROBE_CACHE: dict[tuple[int, str], bool | None] = {}


@contextlib.contextmanager
def _muted():
    """Silence stdout, stderr and logging for the duration of a probe."""
    sink = io.StringIO()
    previous = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            yield
    finally:
        logging.disable(previous)


def _is_pydantic_model(func) -> bool:
    return isinstance(func, type) and isinstance(getattr(func, "model_fields", None), dict)


def _signature_keywords(func) -> set[str] | None:
    """Keyword names ``func`` accepts by signature, or None when undecidable.

    None means "do not check": ``**kwargs``, or no introspectable signature.
    """
    if not callable(func):
        return None
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return None
    names: set[str] = set()
    for name, param in signature.parameters.items():
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            return None
        if param.kind is not inspect.Parameter.VAR_POSITIONAL:
            names.add(name)
    return names


def _rejects(err: dict, keyword: str) -> bool:
    """True when one pydantic error says ``keyword`` itself is not accepted.

    Two shapes count. ``extra_forbidden`` at ``(keyword,)`` is pydantic's own
    verdict. A model-level ``value_error`` that quotes the keyword is the
    client's idiom for a removed field: a ``mode="before"`` validator raises a
    targeted message (``Air ... no longer accepts 'thermally_perfect_gas'``)
    before extra-field checking ever runs.
    """
    loc = tuple(err.get("loc", ()))
    if err.get("type") == "extra_forbidden":
        return loc == (keyword,)
    if err.get("type") == "value_error" and loc == ():
        msg = str(err.get("msg", ""))
        return f"'{keyword}'" in msg or f"`{keyword}`" in msg
    return False


def _probe_rejects_keyword(model: type, keyword: str) -> bool | None:
    """Ask pydantic whether ``keyword`` is genuinely unknown to ``model``.

    A model's accepted keywords are NOT knowable from its signature. Two client
    idioms widen them invisibly:

    * ``model_validator(mode="before")`` hooks that accept extra keys -- e.g.
      ``UserVariable(value=...)`` and the legacy-rename shim that still accepts
      ``CustomVolume(boundaries=...)``. Both are valid; both are absent from
      ``model_fields`` and from ``inspect.signature``.
    * field aliases, where signature and ``model_fields`` disagree.

    So instead of reasoning about the schema, construct with the single keyword
    and read pydantic's own verdict. Only an error that names exactly this
    keyword as the problem (see ``_rejects``) proves it is unknown; any other
    outcome (a missing required field, a type error, success) means it was
    accepted. Returns None
    when the probe cannot decide, and the caller then stays silent.
    """
    cache_key = (id(model), keyword)
    if cache_key in _PROBE_CACHE:
        return _PROBE_CACHE[cache_key]

    verdict: bool | None
    try:
        from pydantic import ValidationError
    except ImportError:
        return None
    # Probing constructs real models, and the client logs deprecation notices to
    # stdout while doing so. Those would interleave with the report, so mute both
    # logging and the streams for the duration of the probe.
    try:
        with _muted():
            model(**{keyword: None})
        verdict = False
    except ValidationError as exc:
        try:
            verdict = any(_rejects(err, keyword) for err in exc.errors())
        except Exception:  # noqa: BLE001 -- malformed error payload
            verdict = None
    except Exception:  # noqa: BLE001 -- validator blew up on the sentinel
        verdict = None

    _PROBE_CACHE[cache_key] = verdict
    return verdict


def unknown_keywords(func, keywords: list[str]) -> list[str]:
    """Return the subset of ``keywords`` that ``func`` genuinely does not accept.

    Two stages, deliberately: the cheap signature check proposes a candidate,
    and for pydantic models the runtime probe confirms it. Reporting on the
    signature alone produced false positives on correct pages (verified against
    ``fl.UserVariable``, ``fl.NavierStokesSolver`` and ``fl.CustomVolume``), and
    a check that cries wolf gets ignored.
    """
    accepted = _signature_keywords(func)
    if accepted is None:
        return []
    candidates = [kw for kw in keywords if kw not in accepted]
    if not candidates:
        return []

    if not _is_pydantic_model(func):
        # Plain functions and classmethods have no schema magic; the signature
        # is the whole truth.
        return candidates

    confirmed: list[str] = []
    for keyword in candidates:
        if _probe_rejects_keyword(func, keyword) is True:
            confirmed.append(keyword)
    return confirmed


def check_call_keywords(tree: ast.AST, roots: dict[str, object], label: str) -> list[Finding]:
    """Flag keyword arguments the callee does not accept.

    This is the drift the attribute resolver structurally cannot see: when a
    client field is renamed, ``fl.Wall`` still resolves, so symbol resolution
    passes while ``fl.Wall(use_wall_funcion=...)`` is dead on arrival. For the
    grab-and-go snippets -- which stay static-only by design -- this is the
    check that carries them.
    """
    findings: list[Finding] = []
    seen: set[tuple[str, str, int]] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        flat = None
        if isinstance(node.func, ast.Name):
            flat = ([node.func.id], node.func)
        elif isinstance(node.func, ast.Attribute):
            flat = _chain_from_attribute(node.func)
        if flat is None:
            continue
        chain, _ = flat
        if chain[0] not in roots:
            continue
        func = _resolve_chain(chain, roots)
        if func is None:
            continue  # unresolved chains are the symbol resolver's business
        supplied = [kw.arg for kw in node.keywords if kw.arg is not None]
        if not supplied:
            continue
        rejected = set(unknown_keywords(func, supplied))
        if not rejected:
            continue
        dotted = ".".join(chain)
        for keyword in node.keywords:
            if keyword.arg is None or keyword.arg not in rejected:
                continue
            lineno = getattr(keyword.value, "lineno", getattr(node, "lineno", -1))
            key = (dotted, keyword.arg, lineno)
            if key in seen:
                continue
            seen.add(key)
            findings.append(
                Finding(
                    label,
                    "keyword",
                    f"`{dotted}()` does not accept keyword `{keyword.arg}`",
                    lineno,
                )
            )
    return findings


def _known_gap(dotted: str) -> str | None:
    """The KNOWN_CLIENT_GAPS key that ``dotted`` is, or is an attribute of."""
    for gap in KNOWN_CLIENT_GAPS:
        if dotted == gap or dotted.startswith(gap + "."):
            return gap
    return None


def _gap_finding(label: str, gap: str, lineno: int | None) -> Finding:
    return Finding(
        label,
        "client-gap",
        f"`{gap}` is missing from the client's public namespace ({KNOWN_CLIENT_GAPS[gap]})",
        lineno,
    )


def check_imports(tree: ast.AST, roots: dict[str, object], label: str) -> list[Finding]:
    """Report each imported flow360 name that does not exist, once, at its import.

    Uses of a missing name are not reported again: a bare ``Name(...)`` call is
    not an attribute chain, so reporting only at uses missed ``from flow360
    import Renamed`` whenever the name was called directly.
    """
    findings: list[Finding] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            bound = [(a.asname or a.name.split(".")[0], a.name) for a in node.names if a.asname]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            bound = [(a.asname or a.name, f"{node.module}.{a.name}") for a in node.names]
        else:
            continue
        for name, qualified in bound:
            missing = roots.get(name)
            if not isinstance(missing, _Missing) or missing.name != qualified:
                continue
            gap = _known_gap(qualified)
            if gap:
                findings.append(_gap_finding(label, gap, node.lineno))
            else:
                findings.append(
                    Finding(
                        label,
                        "symbol",
                        f"`{qualified}` (imported as `{name}`) does not exist in the client",
                        node.lineno,
                    )
                )
    return findings


ELISION_LINE_RE = re.compile(r"^(?P<indent>[ \t]*)\.\.\.[ \t]*,?[ \t]*$")


def _parse_elided(src: str, label: str) -> ast.AST | None:
    """Parse a prose block whose only syntax error is a ``...`` elision in a call.

    ``fl.SimulationParams(reference_geometry=..., ...)`` with the ``...`` on its
    own line is the docs' standard way to say "other arguments here", but a
    positional after keywords does not parse, which would leave the whole block
    unchecked. Rewriting each such line to ``**{},`` keeps the line numbers and
    is ignored by the keyword check. Returns None when the block still fails.
    """
    lines = [
        ELISION_LINE_RE.sub(r"\g<indent>**{},", line) if ELISION_LINE_RE.match(line) else line
        for line in src.splitlines()
    ]
    try:
        return ast.parse("\n".join(lines), filename=label)
    except SyntaxError:
        return None


def resolve_symbols(
    src: str,
    label: str,
    module,
    assume_aliases: bool = False,
    tolerate_fragments: bool = False,
) -> list[Finding]:
    """Resolve flow360-rooted chains and bind call keywords against the live module.

    ``assume_aliases`` binds the conventional ``fl``/``flow360`` names for inline
    prose blocks that omit the import. ``tolerate_fragments`` downgrades a
    SyntaxError to a non-fatal ``fragment`` finding, for prose blocks that are
    deliberately elided rather than runnable.
    """
    findings: list[Finding] = []
    try:
        tree = ast.parse(src, filename=label)
    except SyntaxError as exc:
        tree = _parse_elided(src, label) if tolerate_fragments else None
        if tree is None:
            kind = "fragment" if tolerate_fragments else "syntax"
            return [Finding(label, kind, f"{exc.msg}", exc.lineno)]

    roots = _flow360_roots(tree, module, assume_aliases=assume_aliases)
    if not roots:
        return findings  # nothing imports flow360 here; nothing to check

    findings.extend(check_imports(tree, roots, label))
    findings.extend(check_call_keywords(tree, roots, label))

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
            continue  # already reported at its import by check_imports

        gap = _known_gap(dotted)
        if gap:
            findings.append(_gap_finding(label, gap, getattr(node, "lineno", None)))
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
    env["PYTHONPATH"] = os.pathsep.join([extra_path, env.get("PYTHONPATH", "")]).strip(os.pathsep)
    try:
        proc = subprocess.run(cmd + [str(SNIPPET_DIR)], env=env, capture_output=True, text=True)
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
    surface: str = "snippet"
    # Inline blocks are a slice of a bigger file; findings must report the line
    # in that file, not in the extracted slice.
    line_offset: int = 0
    assume_aliases: bool = False
    tolerate_fragments: bool = False


def _rel(path: Path) -> str:
    """Repo-relative label, falling back to the absolute path when outside."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path.relative_to(REPO_ROOT.parent))


def collect_targets(
    include_notebooks: bool = True,
    include_inline: bool = True,
    include_examples: bool = True,
) -> list[Target]:
    targets: list[Target] = []
    for snippet_dir in SNIPPET_DIRS:
        if not snippet_dir.exists():
            continue
        for py in sorted(snippet_dir.glob("*.py")):
            targets.append(Target(_rel(py), py.read_text(encoding="utf-8"), surface="snippet"))

    if include_notebooks:
        for nb_dir in NOTEBOOK_DIRS:
            if not nb_dir.exists():
                continue
            for nb in sorted(nb_dir.glob("*.ipynb")):
                targets.append(Target(_rel(nb), notebook_source(nb), surface="notebook"))

    if include_examples and EXAMPLES_DIR.exists():
        for py in sorted(EXAMPLES_DIR.rglob("*.py")):
            targets.append(Target(_rel(py), py.read_text(encoding="utf-8"), surface="example"))
        for nb in sorted(EXAMPLES_DIR.rglob("*.ipynb")):
            targets.append(Target(_rel(nb), notebook_source(nb), surface="example"))

    if include_inline:
        for root in INLINE_ROOTS:
            if not root.exists():
                continue
            for path in sorted(root.rglob("*.rst")):
                text = path.read_text(encoding="utf-8")
                for code, lineno in extract_rst_python_blocks(text):
                    targets.append(
                        Target(
                            f"{_rel(path)}:{lineno}",
                            code,
                            surface="inline",
                            line_offset=lineno - 1,
                            assume_aliases=True,
                            tolerate_fragments=True,
                        )
                    )
            for path in sorted(root.rglob("*.md")):
                text = path.read_text(encoding="utf-8")
                for code, lineno in extract_md_python_blocks(text):
                    targets.append(
                        Target(
                            f"{_rel(path)}:{lineno}",
                            code,
                            surface="inline",
                            line_offset=lineno - 1,
                            assume_aliases=True,
                            tolerate_fragments=True,
                        )
                    )

    return targets


def stale_client_gaps(module) -> list[str]:
    """KNOWN_CLIENT_GAPS entries that now resolve, so the waiver must be deleted.

    Without this, an entry outlives the export it waits for and keeps waiving the
    name silently, which is exactly the drift the validator exists to catch.
    """
    stale = []
    for dotted in KNOWN_CLIENT_GAPS:
        obj = module
        try:
            for attr in dotted.split(".")[1:]:
                obj = getattr(obj, attr)
        except AttributeError:
            continue
        stale.append(dotted)
    return stale


def _is_waived(finding, args) -> bool:
    """True when a finding is reported but must not fail the run.

    Two kinds are non-fatal: deliberately elided inline prose (``fragment``,
    gated by ``--strict-fragments``) and known client gaps, which are tracked
    in ``KNOWN_CLIENT_GAPS`` and are the client's bug to fix, not the page's.
    """
    if finding.kind == "client-gap":
        return True
    return finding.kind == "fragment" and not args.strict_fragments


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-notebooks",
        action="store_true",
        help="skip example-library and quickstart notebooks",
    )
    parser.add_argument(
        "--no-inline",
        action="store_true",
        help="skip inline code-block/fence samples embedded in rst and md prose",
    )
    parser.add_argument(
        "--no-examples",
        action="store_true",
        help="skip the client example corpus under examples/",
    )
    parser.add_argument(
        "--strict-fragments",
        action="store_true",
        help="fail on inline blocks that do not parse (default: report, do not fail)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="print only failures and the summary, not every passing target",
    )
    parser.add_argument(
        "--pyright",
        action="store_true",
        help="additionally run pyright over the snippet .py files (if installed)",
    )
    args = parser.parse_args()

    print("Validating doc Python against the in-tree Flow360 client\n")
    modules = import_root_packages()
    module = modules["flow360"]
    from importlib.metadata import version as dist_version

    print(f"  client: flow360 {dist_version('flow360')}")
    print(f"  source: {flow360_source_paths()[0]}")
    print(f"  report package: flexcompute-flow-report {dist_version('flexcompute-flow-report')}")
    if not REPORT_SNIPPET_DIR.exists():
        print(
            f"  WARNING: {REPORT_SNIPPET_DIR.relative_to(REPO_ROOT)} is absent; run "
            "tools/fetch_report_docs.py first so the Reports snippets are checked."
        )
    print()

    targets = collect_targets(
        include_notebooks=not args.no_notebooks,
        include_inline=not args.no_inline,
        include_examples=not args.no_examples,
    )
    if not targets:
        print(
            "ERROR: no targets found. Check the snippet/notebook/example/inline "
            "paths; an empty set means validation was silently skipped."
        )
        return 1
    # A surface that yields nothing (a moved directory, a renamed root) would
    # otherwise just drop out of the coverage table while the run says PASSED.
    requested = {"snippet"}
    requested |= {"notebook"} if not args.no_notebooks else set()
    requested |= {"example"} if not args.no_examples else set()
    requested |= {"inline"} if not args.no_inline else set()
    empty = sorted(requested - {t.surface for t in targets})
    if empty:
        print(f"ERROR: no targets found for surface(s): {', '.join(empty)}. Check their paths.")
        return 1

    all_findings: list[Finding] = []
    per_surface: dict[str, list[int]] = {}
    for target in targets:
        findings = resolve_symbols(
            target.source,
            target.label,
            module,
            assume_aliases=target.assume_aliases,
            tolerate_fragments=target.tolerate_fragments,
        )
        # Map slice-local line numbers back onto the file they came from.
        if target.line_offset:
            for finding in findings:
                if finding.lineno is not None:
                    finding.lineno += target.line_offset

        fatal = [f for f in findings if not _is_waived(f, args)]
        counts = per_surface.setdefault(target.surface, [0, 0])
        counts[0] += 1
        if fatal:
            counts[1] += 1

        if fatal:
            status = "FAIL"
        elif any(f.kind == "client-gap" for f in findings):
            status = "gap"
        elif findings:
            status = "frag"
        else:
            status = "ok"
        if not args.quiet or findings:
            print(f"  [{status:>4}] {target.label}")
            for finding in findings:
                loc = f":{finding.lineno}" if finding.lineno else ""
                print(f"           {finding.kind}{loc}: {finding.message}")
        all_findings.extend(findings)

    for dotted in stale_client_gaps(module):
        all_findings.append(
            Finding(
                "KNOWN_CLIENT_GAPS",
                "stale-gap",
                f"`{dotted}` now resolves in the client; delete its KNOWN_CLIENT_GAPS entry",
            )
        )
        print(f"  [FAIL] KNOWN_CLIENT_GAPS\n           stale-gap: `{dotted}` now resolves")

    fatal_findings = [f for f in all_findings if not _is_waived(f, args)]
    fragments = [f for f in all_findings if f.kind == "fragment"]

    print("\n  surface coverage:")
    for surface in ("snippet", "notebook", "example", "inline"):
        if surface not in per_surface:
            continue
        checked, failed = per_surface[surface]
        print(f"    {surface:<9} {checked:>4} checked, {failed} failing")

    if fragments and not args.strict_fragments:
        print(
            f"\n  {len(fragments)} inline block(s) did not parse (deliberately elided "
            "prose); reported above, not failing. Use --strict-fragments to gate them."
        )

    rc = 0
    if fatal_findings:
        by_kind: dict[str, int] = {}
        for finding in fatal_findings:
            by_kind[finding.kind] = by_kind.get(finding.kind, 0) + 1
        breakdown = ", ".join(f"{n} {k}" for k, n in sorted(by_kind.items()))
        print(
            f"\nFAILED: {len(fatal_findings)} issue(s) ({breakdown}) across "
            f"{len({f.label for f in fatal_findings})} target(s)."
        )
        rc = 1
    else:
        print(f"\nPASSED: {len(targets)} target(s) validated, no API drift found.")

    if args.pyright:
        print("\nRunning pyright over snippets...")
        if run_pyright(flow360_source_paths()[0]) != 0:
            rc = 1

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
