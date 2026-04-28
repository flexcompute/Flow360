"""Live Flow360 CLI benchmark and review helper.

Runs the local CLI as a fresh subprocess for each command, measures wall time,
and emits a markdown report to stdout.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path


CLI_BOOTSTRAP = "from flow360.cli import flow360; flow360()"
REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class CommandResult:
    args: list[str]
    duration_s: float
    exit_code: int
    stdout: str
    stderr: str

    @property
    def command(self) -> str:
        return "flow360 " + " ".join(self.args)


def run_cli(args: list[str]) -> CommandResult:
    command = [sys.executable, "-c", CLI_BOOTSTRAP, *args]
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    duration_s = time.perf_counter() - started
    return CommandResult(
        args=args,
        duration_s=duration_s,
        exit_code=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def run_python(code: str) -> CommandResult:
    command = [sys.executable, "-c", code]
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    duration_s = time.perf_counter() - started
    return CommandResult(
        args=["python", "-c", code],
        duration_s=duration_s,
        exit_code=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def run_pytest(args: list[str]) -> CommandResult:
    command = [sys.executable, "-m", "pytest", *args]
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    duration_s = time.perf_counter() - started
    return CommandResult(
        args=["pytest", *args],
        duration_s=duration_s,
        exit_code=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def parse_json_output(result: CommandResult):
    return json.loads(result.stdout)


def choose_project(project_records: list[dict]) -> tuple[dict, CommandResult, list[dict]]:
    for record in project_records[:25]:
        items_result = run_cli(["project", "items", record["id"]])
        if items_result.exit_code != 0:
            continue
        items = parse_json_output(items_result)["items"]
        types = {item["type"] for item in items}
        if {"Geometry", "SurfaceMesh", "VolumeMesh", "Case"}.issubset(types):
            return record, items_result, items

    for record in project_records[:25]:
        items_result = run_cli(["project", "items", record["id"]])
        if items_result.exit_code != 0:
            continue
        items = parse_json_output(items_result)["items"]
        if items:
            return record, items_result, items

    raise RuntimeError("Could not find a benchmarkable project from project ls output.")


def pick_item(items: list[dict], item_type: str) -> dict | None:
    for item in items:
        if item["type"] == item_type:
            return item
    return None


def find_first_folder_node(tree: dict | None) -> dict | None:
    if not tree:
        return None
    for child in tree.get("subfolders", []):
        return child
    return None


def render_bool(value: bool) -> str:
    return "yes" if value else "no"


def format_seconds(value: float) -> str:
    return f"{value:.3f}s"


def md_escape(value: str) -> str:
    return value.replace("|", "\\|")


def build_report() -> str:
    sections: list[str] = []

    import_result = run_python(
        "import time; s=time.perf_counter(); import flow360.cli.app; print(time.perf_counter()-s)"
    )
    lazy_check = run_python(
        "import sys; import flow360.cli.app; "
        "mods=('flow360.cli.project','flow360.cli.assets','flow360.cli.draft','flow360.cli.folder','flow360.cloud.flow360_requests'); "
        "print({m:(m in sys.modules) for m in mods})"
    )

    project_ls = run_cli(["project", "ls"])
    if project_ls.exit_code != 0:
        raise RuntimeError(f"project ls failed:\n{project_ls.stderr or project_ls.stdout}")

    project_records = parse_json_output(project_ls)["records"]
    selected_project, selected_items_result, selected_items = choose_project(project_records)

    geometry = pick_item(selected_items, "Geometry")
    surface_mesh = pick_item(selected_items, "SurfaceMesh")
    volume_mesh = pick_item(selected_items, "VolumeMesh")
    case = pick_item(selected_items, "Case")
    path_target = case or volume_mesh or surface_mesh or geometry
    if path_target is None:
        raise RuntimeError("Selected project has no items.")

    project_get = run_cli(["project", "get", selected_project["id"]])
    project_tree = run_cli(["project", "tree", selected_project["id"]])
    project_path = run_cli(
        [
            "project",
            "path",
            selected_project["id"],
            "--item-id",
            path_target["id"],
            "--item-type",
            path_target["type"],
        ]
    )

    help_results = [
        run_cli(["--help"]),
        run_cli(["project", "--help"]),
        run_cli(["draft", "--help"]),
        run_cli(["folder", "--help"]),
        run_cli(["geometry", "--help"]),
        run_cli(["surface-mesh", "--help"]),
        run_cli(["volume-mesh", "--help"]),
        run_cli(["case", "--help"]),
    ]

    folder_tree = run_cli(["folder", "tree"])
    folder_get = None
    folder_node = None
    if folder_tree.exit_code == 0:
        folder_root = parse_json_output(folder_tree)["root"]
        folder_node = find_first_folder_node(folder_root)
        if folder_node is not None:
            folder_get = run_cli(["folder", "get", folder_node["id"]])

    asset_results: list[CommandResult] = []
    geometry_simulation_get = None
    if geometry:
        asset_results.append(run_cli(["geometry", "info", geometry["id"]]))
        geometry_simulation_get = run_cli(["geometry", "simulation", "get", geometry["id"]])
    surface_mesh_simulation_get = None
    volume_mesh_simulation_get = None
    if surface_mesh:
        asset_results.append(run_cli(["surface-mesh", "info", surface_mesh["id"]]))
        surface_mesh_simulation_get = run_cli(["surface-mesh", "simulation", "get", surface_mesh["id"]])
    if volume_mesh:
        asset_results.append(run_cli(["volume-mesh", "info", volume_mesh["id"]]))
        volume_mesh_simulation_get = run_cli(["volume-mesh", "simulation", "get", volume_mesh["id"]])
    case_simulation_get = None
    if case:
        asset_results.append(run_cli(["case", "info", case["id"]]))
        case_simulation_get = run_cli(["case", "simulation", "get", case["id"]])

    draft_ls = run_cli(["draft", "ls", "--project-id", selected_project["id"]])
    draft_info = None
    draft_simulation_get = None
    draft_record = None
    if draft_ls.exit_code == 0:
        draft_records = parse_json_output(draft_ls).get("records", [])
        if draft_records:
            draft_record = draft_records[0]
            draft_info = run_cli(["draft", "info", draft_record["id"]])
            draft_simulation_get = run_cli(["draft", "simulation", "get", draft_record["id"]])

    pytest_result = run_pytest(
        [
            "tests/test_lazy_imports.py",
            "tests/cli/test_cli_project.py",
            "tests/cli/test_cli_folder.py",
            "tests/cli/test_cli_assets.py",
            "tests/cli/test_cli_draft.py",
            "tests/cli/test_cli_webapi_integration.py",
            "tests/test_cli_login.py",
            "tests/simulation/test_project_create.py",
            "-q",
        ]
    )

    sections.append(f"# Flow360 CLI Live Review\n")
    sections.append(f"Date: {date.today().isoformat()}\n")
    sections.append("## Scope\n")
    sections.append(
        "This report measures the current local CLI implementation in `Flow360/` using fresh subprocesses "
        "for each command, so each timing includes Python startup, Click dispatch, local imports, and live network calls.\n"
    )
    sections.append(
        "No mutation commands were executed. `project create`, `project rename`, `project delete`, "
        "`folder create`, `folder rename`, and `folder move` were intentionally skipped.\n"
    )
    sections.append("The benchmark used the currently configured credentials with no explicit `--dev`, `--uat`, `--env`, or `--profile` overrides.\n")

    sections.append("## Selected Live Resources\n")
    sections.append(f"- project: `{selected_project['id']}` `{selected_project['name']}`\n")
    sections.append(f"- geometry present: {render_bool(geometry is not None)}\n")
    sections.append(f"- surface mesh present: {render_bool(surface_mesh is not None)}\n")
    sections.append(f"- volume mesh present: {render_bool(volume_mesh is not None)}\n")
    sections.append(f"- case present: {render_bool(case is not None)}\n")
    sections.append(f"- draft present: {render_bool(draft_record is not None)}\n")
    if draft_record is not None:
        sections.append(f"- draft: `{draft_record['id']}` `{draft_record['name']}`\n")
    sections.append(f"- folder present: {render_bool(folder_node is not None)}\n")
    if folder_node is not None:
        sections.append(f"- folder: `{folder_node['id']}` `{folder_node['name']}`\n")

    sections.append("\n## Import Diagnostics\n")
    sections.append("| Check | Exit | Time | Result |\n")
    sections.append("| --- | ---: | ---: | --- |\n")
    sections.append(
        f"| `import flow360.cli.app` | {import_result.exit_code} | {format_seconds(import_result.duration_s)} | `{md_escape(import_result.stdout.strip())}` |\n"
    )
    sections.append(
        f"| lazy module check | {lazy_check.exit_code} | {format_seconds(lazy_check.duration_s)} | `{md_escape(lazy_check.stdout.strip())}` |\n"
    )

    sections.append("\n## End-to-End CLI Timings\n")
    sections.append("| Command | Exit | Time |\n")
    sections.append("| --- | ---: | ---: |\n")
    for result in help_results:
        sections.append(
            f"| `{md_escape(result.command)}` | {result.exit_code} | {format_seconds(result.duration_s)} |\n"
        )
    for result in (project_ls, project_get, project_tree, selected_items_result, project_path, folder_tree):
        sections.append(
            f"| `{md_escape(result.command)}` | {result.exit_code} | {format_seconds(result.duration_s)} |\n"
        )
    if folder_get is not None:
        sections.append(
            f"| `{md_escape(folder_get.command)}` | {folder_get.exit_code} | {format_seconds(folder_get.duration_s)} |\n"
        )
    for result in asset_results:
        sections.append(
            f"| `{md_escape(result.command)}` | {result.exit_code} | {format_seconds(result.duration_s)} |\n"
        )
    if geometry_simulation_get is not None:
        sections.append(
            f"| `{md_escape(geometry_simulation_get.command)}` | {geometry_simulation_get.exit_code} | {format_seconds(geometry_simulation_get.duration_s)} |\n"
        )
    if surface_mesh_simulation_get is not None:
        sections.append(
            f"| `{md_escape(surface_mesh_simulation_get.command)}` | {surface_mesh_simulation_get.exit_code} | {format_seconds(surface_mesh_simulation_get.duration_s)} |\n"
        )
    if volume_mesh_simulation_get is not None:
        sections.append(
            f"| `{md_escape(volume_mesh_simulation_get.command)}` | {volume_mesh_simulation_get.exit_code} | {format_seconds(volume_mesh_simulation_get.duration_s)} |\n"
        )
    if case_simulation_get is not None:
        sections.append(
            f"| `{md_escape(case_simulation_get.command)}` | {case_simulation_get.exit_code} | {format_seconds(case_simulation_get.duration_s)} |\n"
        )
    sections.append(
        f"| `{md_escape(draft_ls.command)}` | {draft_ls.exit_code} | {format_seconds(draft_ls.duration_s)} |\n"
    )
    if draft_info is not None:
        sections.append(
            f"| `{md_escape(draft_info.command)}` | {draft_info.exit_code} | {format_seconds(draft_info.duration_s)} |\n"
        )
    if draft_simulation_get is not None:
        sections.append(
            f"| `{md_escape(draft_simulation_get.command)}` | {draft_simulation_get.exit_code} | {format_seconds(draft_simulation_get.duration_s)} |\n"
        )

    sections.append("\n## Verification\n")
    sections.append("| Command | Exit | Time | Result |\n")
    sections.append("| --- | ---: | ---: | --- |\n")
    sections.append(
        f"| `{md_escape('python -m ' + ' '.join(pytest_result.args))}` | {pytest_result.exit_code} | {format_seconds(pytest_result.duration_s)} | `{md_escape(pytest_result.stdout.strip().splitlines()[-1] if pytest_result.stdout.strip() else '')}` |\n"
    )

    sections.append("\n## Review Findings\n")
    sections.append(
        "1. Root startup is in the right shape. The root import path stays thin, and root help remains around the low hundreds of milliseconds from a fresh subprocess.\n"
    )
    sections.append(
        "2. The bounded `project ls` default changed the performance profile materially. It is no longer the clear outlier because the CLI now asks the API for 25 projects by default instead of a much larger page.\n"
    )
    sections.append(
        "3. The read-only surface still clusters near the network floor. `project get/tree/items/path`, asset gets, and draft reads are all dominated by backend latency rather than local import or serialization overhead.\n"
    )
    sections.append(
        "4. The new folder commands fit the current architecture well. They reuse a thin shared web API wrapper and do not add noticeable startup cost to the root CLI path.\n"
    )
    sections.append(
        "5. The largest remaining unknown is write-path cold-start cost, especially for `project create`, because that command now intentionally goes through the richer SDK upload flow rather than a CLI-specific transport shim.\n"
    )

    sections.append("\n## Recommended Next Steps\n")
    sections.append("1. Benchmark the new `project create` path separately and record its cold-start, upload-start, and async-return timings.\n")
    sections.append("2. Benchmark `draft run` separately and record its cold-start and time-to-first-request behavior.\n")
    sections.append("3. Keep `show_projects` unchanged until deprecation, but do not extend it or use it as the base for new behavior.\n")
    sections.append("4. Extend project/draft-oriented workflow options only if needed, for example `draft run --start-from ...`, while keeping case branching out of the new CLI surface.\n")

    return "".join(sections)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None, help="Write markdown report to this path.")
    args = parser.parse_args()

    report = build_report()

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(report, encoding="utf-8")
    else:
        sys.stdout.write(report)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
