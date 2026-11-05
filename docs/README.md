# The documentation of Flow360: https://flexcompute-flow360documentation.readthedocs-hosted.com/

## Documentation structure

Every path given in this section is assumed to be a child of docs/source.

### Configuration folders and files

- **_ext/**: custom autodoc extension modules (`flow360_autodoc.py`, `flow360_documenters.py`, `flow360_type_resolver.py`, etc.) that drive the Python API reference generation.
- **_static/**: site assets — logos (`Flow360-logo.svg`), stylesheets (`custom.css`, `theme_overrides.css`, `bugfix.css`, `justify.css`, `example-filter.css`), fonts (`inter_var.woff2`, `ornitons_bold.ttf`), and the JS for the example library filter (`example-filter.js`).
- **_templates/**: Jinja templates that override Sphinx defaults — page chrome (`layout.html`, `page.html`) and autosummary stubs for classes, attributes, functions, and properties (`class.rst`, `attribute.rst`, `function.rst`, `property.rst`). `rst.py` is a helper used by the templates.
- **conf.py**: Sphinx configuration — extensions, theme, intersphinx, autodoc options, and registration of the custom `_ext/` modules.
- **docutils.conf**: docutils-level settings (e.g. table handling) applied to every build.
- **Figures/**: figures referenced directly from the top-level `index.rst` landing page.

### Main index page

- **index.rst**: the landing page. Renders the Flow360 intro grid, the "Useful pages" link cards (Web App, WebUI Quickstart/Examples, Python API Getting Started/Quickstart/Examples), the top-level `Contents` cards, and the hidden root `toctree` that wires every main section into the navigation.

### gui_guide/

Guide for the Flow360 web interface. Authored in **Markdown** (MyST) rather than rST. Sub-sections are numbered to control ordering and each has its own `README.md` index:

- `01.introduction/` — dashboard, starting a project, workbench layout, general workflow, project tree, shortcuts, project settings
- `02.simulation-setup/` — flow conditions, mesh, flow solver, output
- `03.analysis/` — dashboard, convergence, monitor, visualization, aeroacoustic, case comparison, reports (`07.reports.md` is fetched from flex, see "Report docs fetched from flex" below)
- `04.entities-browser/` — geometry, surface/volume mesh, volumes, points, slices, sample surfaces
- `05.tools/`, `06.account-settings/`
- `README.md` — the section's toctree-equivalent index page

### introduction/

Short marketing/overview pages: `introduction.rst` (entry point linked from the root `toctree`) and `capabilities.rst`.

### quick_start/

Tutorial walkthroughs for new users. `quick_start.rst` is the section index; subfolders cover each first-run path:

- `WebUI_Geometry/`, `WebUI_AutomatedMeshing/`, `WebUI_CaseLaunching/`, `WebUI_CasePostprocessing/` — WebUI tutorial chain
- `API_quickstart/` — the Python API "run your first simulation" notebook (`notebooks/quickstart_API`)

### user_guide/

In-depth reference for solver setup and workflows. `user_guide.rst` is the section index; subfolders are topic-grouped:

- `HowDoesItWork/`, `Meshing/`, `AssetDrafts/`, `WorkflowsInterfaces/`
- `NonDimensionalization/`, `UnitsAndExpressions/`, `OutputConfiguration/`
- `RunControl/`, `UserDefinedDynamics/`, `Troubleshooting/`

That section is supposed to go over different functionalities of the software and explain how to use different features. It can contain recommendations of the setup of different parameters, best practices, some initial workflows etc. It should be relatively interface-independent, except of the features that are available only through one of them. 

### python_api/

Everything related to the Python client. `python_api.rst` is the section index:

- **API_reference/** — auto-generated reference, organized by domain (`meshing/`, `outputs/`, `surface_models/`, `volume_models/`, `solver_configuration/`, `operating_condition/`, `time_stepping/`, `run_control/`, plus top-level `setup.rst`, `entities.rst`, `results.rst`, etc.). Stubs land in `_autosummary/`. `reports.rst` autodocs the `flexcompute-flow-report` companion package (installed from PyPI through the `docs` extra) and is fetched from flex, see "Report docs fetched from flex" below.
- **getting_started/** — installation and API-key setup (`installation_setup.rst`).
- **example_library/** — end-to-end Jupyter examples. `notebooks/` holds the executed `.ipynb` files, `_notebooks_py/` the paired `.py` sources (those are generated automatically when running sphinx build, do NOT edit!), `thumbnails/` the gallery images.
- **grab_and_go_snippets/** — short, single-task recipes (download results, fork, list cases, plot convergence, etc.). `index.rst` lists them all. Each recipe's code lives in a real `.py` file under `_snippets/` and is pulled into the page with `.. literalinclude:: _snippets/<name>.py`, never pasted inline. This keeps the code importable so CI can check it against the client (see "Validating doc code against the client" below). A page with two code blocks splits into two files (e.g. `calculate_dimensional_forces_total.py` and `..._excluded.py`).
- **migration_guide/** — version-to-version migration notes (e.g. `bet_migration.rst`, `monitor_conversion.rst`).

### knowledge_base/

Deeper-dive material. `knowledge_base.rst` is the section index; content is split into `PreProcessing/`, `Simulation/`, `validationStudies/`, and `papers/`.

### release_notes/

`release_notes.rst` plus one `release-X.Y[.Z].rst` per shipped version.


## Repo guide

### How do branches work:

- **`master`** branch is responsible for development of the docs for the next release, it is rendered to the `latest` version of the docs which is hidden by default, can be found [here](https://docs.flexcompute.com/projects/flow360/en/latest/).
- **`release-candidate`** branches get created during the release procedure from the `master` branch.

### Release procedure:

1. Branch out `release-candidate` from `master`.
2. Create a **release** on GitHub, it is important that the name of the release and the tag follow the semantic versioning (`vX.Y.Z` -> for example `v25.7.1`) the minor versions do not have to correspond to Flow360. 
3. The **stable** version will automatically get created from the latest tag created.

### Hotfix procedure:

1. Create a feature branch from `release-candidate` branch 
2. Go on with the normal PR process, review, approval etc. 
3. After merging to `release-candidate` create a **release** in GitHub with the new minor version, so the stable documentation gets updated. 
4. Get approval and merge the hotfix to master.
5. Hotfix PRs which are cherry-picks of the merged commit to the relevant branches should be created by an automatic action, if not, see logs.
6. Look over the auto generated PRs, resolve conflicts if needed, approve if looks ok and merge.

### Local development against a `compute` checkout

The Python code that drives the API reference normally comes from the `Flow360/` git submodule. That submodule is just a public mirror of code that lives in the `compute` repo:

- `compute/flex/public/Flow360`
- `compute/flex/share/flow360-schema/src`

For local work it is often more convenient to point the docs build at your local `compute` clone so you can iterate on both at once. To do that:

1. Copy `local_paths.toml.example` (at the repo root) to `local_paths.toml` (also at the repo root).
2. Set `compute_root` to the absolute path of your local `compute` checkout.

`local_paths.toml` is gitignored, so the override never reaches CI — the published build always uses the submodule. `conf.py` prints which mode it is using at build time.

**Caveat — `literalinclude` and figure paths.** The override only redirects `sys.path` (which is what autodoc / the API reference uses). Any `.. literalinclude::` directives or figure references that hardcode `../../../../Flow360/...` (e.g. [docs/source/python_api/migration_guide/monitor_conversion.rst](docs/source/python_api/migration_guide/monitor_conversion.rst)) still resolve against the submodule directory on disk. For those to render locally you need to keep the `Flow360/` submodule initialized; otherwise Sphinx will warn that the file cannot be found. A more invasive fix (e.g. junction-linking `Flow360/` to the compute path) is possible but is intentionally out of scope of this override.

### Validating doc code against the client

`tools/validate_doc_code.py` checks the Python in the docs against the **same client the docs build from** (the in-tree `flex/public/Flow360` client, as `docs/source/conf.py` sources it), without executing anything or touching the cloud. It covers the grab-and-go snippets, the example-library and quickstart notebooks, the client examples under `examples/`, and the inline `code-block:: python` / ```` ```python ```` samples in rst and md pages. For each one it:

1. parses the source (syntax check),
2. resolves every `flow360`-rooted and `flexcompute.flow_report`-rooted attribute chain (e.g. `fl.Project.from_cloud`, `fl.u.deg`, `Report.create`) against the live client and the installed report package, failing on any symbol that no longer exists, and
3. checks the keyword arguments of each call against the callee, confirming a rejected keyword with a pydantic construction probe, so `fl.Fluid(turbulence_model=...)` fails after that field is renamed.

It ignores anything not rooted in the `flow360` import, so third-party calls (pandas, numpy) never cause false positives. Names the client is known not to export yet are listed in `KNOWN_CLIENT_GAPS` and reported without failing; the run fails once such a name resolves, so delete its entry then.

The CI `validate-code` job runs it with `--strict-fragments` on every PR that changes the docs, the Flow360 client or the schema, so a renamed or removed client API fails on the PR that renames it. An inline block that is deliberately not Python (a placeholder, a shell command) should be tagged `text` or `bash`, not `python`.

Run it locally in the docs environment:

```
python tools/validate_doc_code.py                     # every surface
python tools/validate_doc_code.py --strict-fragments  # as CI runs it
python tools/validate_doc_code.py --no-notebooks      # also --no-inline, --no-examples
python tools/validate_doc_code.py --pyright           # also type-check snippets (needs pyright)
```

### Report docs fetched from flex

The pages of the interactive report feature are not stored in this tree. They live in the `flex` repo, in `frontend/flow360-python-plugin/flow360-report-docs/`, next to the `flexcompute-flow-report` package they describe, so the package owners review them where they work (they have no access to `compute`). The folder's `source/` mirrors this Sphinx source root:

| Fetched into | From flex |
| :--- | :--- |
| `gui_guide/03.analysis/07.reports.md` (+ `Figures/reports-*`) | `source/gui_guide/03.analysis/...` |
| `user_guide/Report/Report.rst` | `source/user_guide/Report/Report.rst` |
| `python_api/API_reference/reports.rst` (+ `reports_snippets/`) | `source/python_api/API_reference/...` |
| `docs/.report_docs_codespell_ignore.txt` | `codespell-ignore-words.txt` |

`tools/fetch_report_docs.py` copies them in before every docs job (`sphinx-build`, `validate-code`, `spellcheck`, `linkcheck` in `.github/workflows/flow360-docs-checks.yml`, authenticated with a flex read token) and writes the list of copied files to `docs/source/.report_docs_manifest`. Every destination is gitignored and the script refuses to overwrite a tracked file, so the pages can never be edited here: **change them in flex.** Only the navigation slots that point at them (toctree entries, index cards, options-table rows) belong to this tree; `conf.py` stops with a clear error when the pages are missing.

Which flex commit is fetched is written in `docs/report_docs.ref`:

- `master` and the release candidate under development point at `main`: every merge in flex reaches the next build here without any action.
- When a release candidate is retired, a one-off PR on that branch pins the file to a flex commit SHA. The pin is never forward-ported.

The API page imports the package from PyPI at the exact version pinned in `pyproject.toml` (`docs` extra, keep `poetry.lock` in sync), so `sphinx-build` autodocs and `validate_doc_code.py` checks the report snippets against the same release. A page change that needs a newer package is a two-step landing: release the package, then bump the pin here.

Locally:

```
cd flex/public/Flow360
python3 docs/tools/fetch_report_docs.py                              # flex main (or the pinned SHA), uses `gh auth token`
python3 docs/tools/fetch_report_docs.py --from-local ~/repos/flex    # a local flex checkout, e.g. a flex PR branch
python3 docs/tools/fetch_report_docs.py --clean                      # remove the copied files
```

Flex PRs touching the folder run the same Sphinx build against the mirrored tree (`public_flow360-self-hosted-docs.yml`, source=flex), so a broken cross-reference fails there before it can reach a build here.

### Expected behavior for new pages

- The AI creates the page (`.rst` by default; `.md` if it belongs in the GUI guide) with a stable label.
- The page is added to navigation (nearest `toctree`).
- The page cross-references relevant:
  - Python API reference page(s)
  - GUI guide page(s)
  - Example(s) (prefer `docs/source/python_api/example_library/`, fallback to `Flow360/examples/`)
- Any assets are placed in child `Figures/`, `Tables/`, `Files/` folders and referenced only within the page subtree.

### End-to-end notebook execution

The example-library and quickstart notebooks submit real cloud cases. The `E2E Example Notebooks` workflow (`.github/workflows/e2e-notebooks.yml`) executes them end-to-end against the Flow360 cloud to catch notebooks that stop working against the current client/solver (something the static doc-code check cannot see, since it never runs anything).

- **Trigger:** manual only for now (run it from the Actions tab via *Run workflow*). The optional `notebook` input runs a single notebook (e.g. `DARPA_SUBOFF_AD.ipynb`) for a quick smoke test; leave it as `all` to run every notebook. No cron is configured yet.
- **Account / secret:** runs against `demo@flexcompute.com`. The API key must be stored as the repository secret **`FLOW360_DEMO_APIKEY`** (Settings → Secrets and variables → Actions). It is read from the environment and passed to `fl.configure(...)`; it is never committed or placed on a command line.
- **Execution:** one job per notebook (matrix) so they run in parallel with independent timeouts. Each runs `jupyter nbconvert --execute` in the same poetry env the docs build uses, against the committed `Flow360` submodule client.
- **Reporting:** failures are collected into a single tracking GitHub issue labelled `e2e-notebooks` (opened/updated per run, closed automatically when everything passes), each with the failing cell's traceback.
- **Cost:** every run submits real cloud cases and consumes credits; several unsteady/DDES notebooks are long-running. This is why it is manual-only initially. It does **not** commit refreshed notebook outputs.