#!/usr/bin/env python3
"""Materialize the Flow360 report documentation pages from the flex repo.

The pages documenting the interactive report feature (GUI guide, user guide,
Python API reference) live in flex at
``frontend/flow360-python-plugin/flow360-report-docs/``, next to the
``flexcompute-flow-report`` package they describe, and are reviewed there. This
script copies that folder's ``source/`` tree over the Sphinx source root so the
pages render inside the Flow360 docs. Nothing it writes is tracked by git: every
destination must be gitignored, and the script refuses to touch a tracked file.

Modes::

    fetch_report_docs.py                    # sparse-fetch flex at the ref in report_docs.ref
    fetch_report_docs.py --ref <branch|sha> # same, overriding the ref file
    fetch_report_docs.py --from-local PATH  # copy from a local flex checkout
    fetch_report_docs.py --clean            # remove what a previous run copied

The default mode authenticates with ``FLOW360_REPORT_DOCS_TOKEN`` (a GitHub
token with read access to flexcompute/flex) or, when unset, with the token of
the logged-in ``gh`` CLI. Only the standard library is used so the script also
runs in jobs that never install the docs environment.
"""

from __future__ import annotations

import argparse
import base64
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

FLEX_URL = "https://github.com/flexcompute/flex.git"
FLEX_FOLDER = "frontend/flow360-python-plugin/flow360-report-docs"
REQUIRED_PAGES = (
    "gui_guide/03.analysis/07.reports.md",
    "user_guide/Report/Report.rst",
    "python_api/API_reference/reports.rst",
)

DOCS_DIR = Path(__file__).resolve().parents[1]  # <root>/flex/public/Flow360/docs
SOURCE_DIR = DOCS_DIR / "docs" / "source"
REF_FILE = DOCS_DIR / "report_docs.ref"
MANIFEST = SOURCE_DIR / ".report_docs_manifest"
CODESPELL_IGNORE_DST = DOCS_DIR / ".report_docs_codespell_ignore.txt"
LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/"


def fail(message: str) -> None:
    print(f"fetch_report_docs: error: {message}", file=sys.stderr)
    raise SystemExit(1)


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, text=True, capture_output=True, **kwargs)


def read_ref(override: str | None) -> str:
    if override:
        return override
    if not REF_FILE.exists():
        fail(f"ref file {REF_FILE} is missing")
    lines = [ln.strip() for ln in REF_FILE.read_text().splitlines()]
    refs = [ln for ln in lines if ln and not ln.startswith("#")]
    if len(refs) != 1:
        fail(f"{REF_FILE} must contain exactly one non-comment line (the flex ref to fetch)")
    return refs[0]


def github_token() -> str | None:
    token = os.environ.get("FLOW360_REPORT_DOCS_TOKEN")
    if token:
        return token
    if shutil.which("gh"):
        try:
            return run(["gh", "auth", "token"]).stdout.strip() or None
        except subprocess.CalledProcessError:
            return None
    return None


def fetch_from_github(ref: str, workdir: Path) -> tuple[Path, str]:
    """Sparse-fetch FLEX_FOLDER at ``ref`` into ``workdir``; return (folder, sha)."""
    env = os.environ.copy()
    token = github_token()
    if token:
        # Same mechanism actions/checkout uses: the credential travels in the
        # environment, never on a command line.
        basic = base64.b64encode(f"x-access-token:{token}".encode()).decode()
        env.update(
            GIT_CONFIG_COUNT="1",
            GIT_CONFIG_KEY_0="http.https://github.com/.extraheader",
            GIT_CONFIG_VALUE_0=f"AUTHORIZATION: basic {basic}",
        )
    else:
        print("fetch_report_docs: no token found, fetching anonymously", file=sys.stderr)
    git = ["git", "-C", str(workdir)]
    try:
        run(["git", "init", "-q", str(workdir)], env=env)
        run(git + ["remote", "add", "origin", FLEX_URL], env=env)
        run(git + ["sparse-checkout", "set", "--no-cone", f"/{FLEX_FOLDER}/"], env=env)
        # Nothing in the folder is LFS-tracked in flex; if that ever changes the
        # pointer check in materialize() fails the run instead of shipping pointers.
        run(git + ["-c", "filter.lfs.smudge=", "-c", "filter.lfs.process=", "-c", "filter.lfs.required=false",
                   "fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", ref], env=env)
        run(git + ["-c", "filter.lfs.smudge=", "-c", "filter.lfs.process=", "-c", "filter.lfs.required=false",
                   "checkout", "-q", "FETCH_HEAD"], env=env)
        sha = run(git + ["rev-parse", "HEAD"], env=env).stdout.strip()
    except subprocess.CalledProcessError as exc:
        fail(f"git failed while fetching flex@{ref}:\n{exc.stderr.strip()}")
    return workdir / FLEX_FOLDER, sha


def folder_from_local(path: str) -> tuple[Path, str]:
    root = Path(path).expanduser().resolve()
    folder = root / FLEX_FOLDER
    if not folder.is_dir():
        fail(f"{folder} does not exist; --from-local expects the root of a flex checkout")
    try:
        sha = run(["git", "-C", str(root), "rev-parse", "HEAD"]).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        sha = "unknown"
    return folder, f"local checkout {root} @ {sha}"


def previous_files() -> list[Path]:
    if not MANIFEST.exists():
        return []
    rels = [ln for ln in MANIFEST.read_text().splitlines() if ln and not ln.startswith("#")]
    return [SOURCE_DIR / rel for rel in rels]


def clean() -> int:
    removed = 0
    for path in previous_files():
        if path.exists():
            path.unlink()
            removed += 1
    for path in (MANIFEST, CODESPELL_IGNORE_DST):
        if path.exists():
            path.unlink()
    print(f"fetch_report_docs: removed {removed} previously fetched file(s)")
    return 0


def git_tracked(paths: list[Path]) -> list[str]:
    rels = [str(p.relative_to(DOCS_DIR)) for p in paths]
    out = run(["git", "-C", str(DOCS_DIR), "ls-files", "-z", "--"] + rels).stdout
    return [p for p in out.split("\0") if p]


def git_not_ignored(paths: list[Path]) -> list[str]:
    rels = [str(p.relative_to(DOCS_DIR)) for p in paths]
    proc = subprocess.run(
        ["git", "-C", str(DOCS_DIR), "check-ignore", "-z", "--stdin"],
        input="\0".join(rels), text=True, capture_output=True,
    )
    if proc.returncode not in (0, 1):
        fail(f"git check-ignore failed:\n{proc.stderr.strip()}")
    ignored = {p for p in proc.stdout.split("\0") if p}
    return [r for r in rels if r not in ignored]


def materialize(folder: Path, origin: str) -> int:
    if not folder.is_dir():
        fail(f"{FLEX_FOLDER} does not exist in {origin}")
    source = folder / "source"
    if not source.is_dir():
        fail(f"{FLEX_FOLDER} in {origin} has no source/ directory")
    files = sorted(p for p in source.rglob("*") if p.is_file())
    rels = [p.relative_to(source) for p in files]
    missing = [page for page in REQUIRED_PAGES if Path(page) not in rels]
    if missing:
        fail("required page(s) missing from the flex folder: " + ", ".join(missing))
    pointers = [str(r) for p, r in zip(files, rels) if p.open("rb").read(len(LFS_POINTER_PREFIX)) == LFS_POINTER_PREFIX]
    if pointers:
        fail("these files are git-lfs pointers, not content (install git-lfs or pull them): " + ", ".join(pointers))

    dests = [SOURCE_DIR / r for r in rels]
    tracked = git_tracked(dests)
    if tracked:
        fail("refusing to overwrite git-tracked file(s); report pages are never edited in this repo:\n  " + "\n  ".join(tracked))
    unignored = git_not_ignored(dests + [MANIFEST, CODESPELL_IGNORE_DST])
    if unignored:
        fail("destination(s) not covered by docs/.gitignore; add a pattern before fetching:\n  " + "\n  ".join(unignored))

    clean()
    for src, dst in zip(files, dests):
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    MANIFEST.write_text(
        f"# Files copied by docs/tools/fetch_report_docs.py from {origin}. Do not edit or commit.\n"
        + "".join(f"{r.as_posix()}\n" for r in rels)
    )
    ignore_src = folder / "codespell-ignore-words.txt"
    CODESPELL_IGNORE_DST.write_text(ignore_src.read_text() if ignore_src.exists() else "")
    print(f"fetch_report_docs: copied {len(files)} file(s) from {origin} into {SOURCE_DIR}")
    for r in rels:
        print(f"  {r.as_posix()}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ref", help=f"flex ref to fetch (default: the one in {REF_FILE.name})")
    parser.add_argument("--from-local", metavar="PATH", help="root of a local flex checkout to copy from")
    parser.add_argument("--clean", action="store_true", help="remove previously fetched files and exit")
    args = parser.parse_args()
    if args.clean:
        return clean()
    if args.from_local:
        folder, origin = folder_from_local(args.from_local)
        return materialize(folder, origin)
    ref = read_ref(args.ref)
    with tempfile.TemporaryDirectory(prefix="flex-report-docs-") as tmp:
        folder, sha = fetch_from_github(ref, Path(tmp))
        return materialize(folder, f"flex@{ref} ({sha[:12]})")


if __name__ == "__main__":
    raise SystemExit(main())
