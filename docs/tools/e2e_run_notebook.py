#!/usr/bin/env python3
"""Submit-only executor for the E2E example-notebook workflow.

Executes a notebook against the Flow360 cloud but stops at the first
``case.wait()`` (or any ``AssetBase.wait()``). Every submission before that
point (geometry upload, mesh generation, ``run_case``) still runs and is
validated by the cloud, so genuine submission errors surface; the blocking
wait and all downstream postprocessing are skipped.

This is intentionally notebook-agnostic: notebooks are not edited. We inject a
single setup cell that monkeypatches ``AssetBase.wait`` to raise a unique
sentinel, then treat that one sentinel as success and any other error as a
real failure.

Driven entirely by environment variables (set by the workflow):
  NB_PATH       path to the notebook to execute (relative to cwd, i.e. Flow360/)
  KERNEL_NAME   ipykernel spec name registered earlier in the job

Note: submissions are async by default in the client (run_async=True), so they
return immediately at their call site; wait() is the first thing that blocks.
For notebooks structured as submit -> wait -> postprocess -> submit -> wait,
only submissions before the first wait() are exercised.
"""

import os
import sys

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

# Unique enough that it can never collide with a real notebook exception name.
SENTINEL = "_Flow360E2ESubmitOnlyStop"

SETUP_CELL = (
    "import flow360.component.simulation.web.asset_base as _ab\n"
    f"class {SENTINEL}(Exception):\n"
    "    pass\n"
    "def _e2e_submit_only_wait(self, *args, **kwargs):\n"
    f"    raise {SENTINEL}(\n"
    "        'E2E submit-only: skipping wait() on ' + type(self).__name__\n"
    "    )\n"
    "_ab.AssetBase.wait = _e2e_submit_only_wait\n"
    "print('E2E submit-only: AssetBase.wait patched to stop after submission')\n"
)


def main() -> int:
    nb_path = os.environ["NB_PATH"]
    kernel = os.environ["KERNEL_NAME"]
    out_dir = os.path.join(os.path.dirname(__file__), "..", "_e2e_out")
    os.makedirs(out_dir, exist_ok=True)

    nb = nbformat.read(nb_path, as_version=4)
    nb.cells.insert(0, nbformat.v4.new_code_cell(SETUP_CELL))

    # Run with the notebook's own directory as cwd so relative file references
    # (geometry/mesh inputs) resolve exactly as they do under `jupyter nbconvert`.
    run_path = os.path.dirname(os.path.abspath(nb_path))
    client = NotebookClient(
        nb,
        kernel_name=kernel,
        timeout=-1,  # no per-cell timeout; the job-level timeout is the backstop
        resources={"metadata": {"path": run_path}},
    )

    try:
        client.execute()
    except CellExecutionError as err:
        if SENTINEL in str(err):
            print("E2E submit-only: reached the first wait() with no submission " "errors -> PASS")
            _write_executed(nb, nb_path, out_dir)
            return 0
        # A real submission/validation error: re-raise so the step fails.
        _write_executed(nb, nb_path, out_dir)
        raise

    # No wait() was ever called (e.g. a notebook that only submits, or reuses an
    # already-finished cloud project). It ran clean, so it is a pass.
    print("E2E submit-only: notebook completed without hitting wait() -> PASS")
    _write_executed(nb, nb_path, out_dir)
    return 0


def _write_executed(nb, nb_path, out_dir) -> None:
    """Best-effort: persist the executed notebook for the failure artifact."""
    try:
        dest = os.path.join(out_dir, os.path.basename(nb_path))
        nbformat.write(nb, dest)
    except Exception as exc:  # pragma: no cover - artifact write is non-critical
        print(f"(could not write executed notebook: {exc})")


if __name__ == "__main__":
    sys.exit(main())
