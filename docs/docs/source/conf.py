# Configuration file for the Sphinx documentation builder.
import datetime
import os
import sys
from pathlib import Path
from nbconvert.exporters import ScriptExporter
import logging

logger = logging.getLogger(__name__)

# -- Project information

project = "Flow360 Documentation"
author = "Flexcompute Inc"
year = datetime.date.today().strftime("%Y")
copyright = f"Flexcompute 2020-{year}"
master_doc = "index"  # The master toctree document.s

# release = 'release-21.3.3.0'

sys.path.insert(0, os.path.abspath(""))
sys.path.insert(0, os.path.abspath("_ext"))
sys.path.insert(0, os.path.abspath("_static"))


def _flow360_source_paths() -> list[str]:
    """Resolve where to import `flow360` and its schema from.

    The docs now live inside the compute monorepo (under
    `flex/public/Flow360/docs/`), so the client is sourced directly from the
    in-repo files instead of the old `Flow360/` git submodule. Paths are
    derived relative to this file, so they resolve identically in `compute`
    and in the `flex` Copybara mirror (both keep the `flex/public/Flow360`
    and `flex/share/flow360-schema` layout).
    """

    # conf.py lives at <root>/flex/public/Flow360/docs/docs/source/conf.py
    flow360_root = Path(__file__).resolve().parents[3]  # <root>/flex/public/Flow360 (contains flow360/)
    flex_root = Path(__file__).resolve().parents[5]  # <root>/flex
    # flow360-schema uses a src/ layout, so the importable root is src/.
    schema_src = flex_root / "share" / "flow360-schema" / "src"

    paths = [str(flow360_root), str(schema_src)]
    for path in paths:
        if not Path(path).exists():
            logger.warning("conf.py: expected Flow360 client path does not exist: %s", path)
    logger.info("conf.py: sourcing Flow360 client from in-repo files: %s", paths)
    return paths


for _flow360_path in _flow360_source_paths():
    sys.path.insert(0, _flow360_path)

# -- General configuration
add_module_names = False  # Remove namespaces from class/method signatures
autosummary_generate = True  # Turn on sphinx.ext.autosummary
autodoc_class_signature = "separated"
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "undoc-members": True,
    "exclude-members": "SchemaConfig,__init__,Config,model_fields_set",
}
autodoc_typehints = "description"
copybutton_selector = "div.highlight pre, code.inline-code .pre"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]
extensions = [
    "IPython.sphinxext.ipython_directive",
    "IPython.sphinxext.ipython_console_highlighting",
    "nbsphinx",  # Integrate Jupyter Notebooks and Sphinx
    "notfound.extension",
    "myst_parser",
    # "sphinxcontrib.divparams", # TODO FIX
    "sphinx.ext.autodoc",  # Core Sphinx library for auto html doc generation from docstrings
    "sphinx.ext.autosummary",  # Create neat summary tables for modules/classes/methods etc
    "sphinx.ext.coverage",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",  # Link to other project's documentation (see mapping below)
    "sphinx.ext.imgconverter",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",  # Add a link to the Python source code for classes, functions etc.
    "sphinx_copybutton",
    "sphinx_favicon",
    "sphinx_tabs.tabs",
    "sphinxemoji.sphinxemoji",
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinxcontrib.autodoc_pydantic",
    "sphinxcontrib.cairosvgconverter",
    "sphinx_prompt",
    "sphinx_design",
    "sphinx_toolbox.collapse",
    "sphinxcontrib.mermaid",
    "flow360_autodoc",
]
extlinks = {}
epub_show_urls = "footnote"
favicons = [
    {
        "sizes": "16x16",
        "href": "logo.svg",
    }
]
language = "en"

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# MyST parser configuration
myst_enable_extensions = [
    "dollarmath",  # Enables $...$ and $$...$$ syntax for math
]

# Generate header anchors so Markdown `#fragment` links resolve.
myst_heading_anchors = 4

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]

nbsphinx_prolog = r"""
.. only:: html

   .. container:: ex-download-buttons

      :download:`Download notebook (.ipynb) <{{ env.docname.split('/')[-1] }}.ipynb>`

      :download:`Download script (.py) <../_notebooks_py/{{ env.docname.split('/')[-1] }}.py>`
"""

html_theme = "sphinx_book_theme"
html_title = "Flow360 Computational Fluid Dynamics Solver"
html_logo = "_static/Flow360-logo.svg"
html_logo_dark = "_static/Flow360-logo-dark.svg"
html_favicon = "_static/logo.svg"
html_show_sourcelink = False
html_static_path = ["_static"]
html_css_files = [
    "theme_overrides.css",
    "bugfix.css",
    "custom.css",
    "justify.css",
    "example-filter.css",
]  # TODO: Fine-tune CSS style
html_js_files = ["example-filter.js"]

# -- Docs version switcher ----------------------------------------------------
# Product-local switcher: base JS/CSS and the per-product config all ship in
# this package's _static/ (see the base file's header for the config contract).
# The config sets window.FLEX_DOCS_VERSIONS and MUST load before the base.
html_js_files += [
    "js/version-switcher-config.js",
    "js/version-switcher.js",
]
html_css_files.append("version-switcher.css")

html_theme_options = {
    "path_to_docs": "docs",
    "logo": {
        "image_light": "_static/Flow360-logo.svg",
        "image_dark": "_static/Flow360-logo-dark.svg",
    },
    "repository_url": "https://github.com/flexcompute/flow360",
    "repository_branch": "latest",
    "use_edit_page_button": False,
    "use_issues_button": True,
    "use_repository_button": True,
    "use_download_button": True,
    "use_fullscreen_button": False,
    "pygments_light_style": "default",
    "pygments_dark_style": "material",
    "max_navbar_depth": 6,
    "navbar_persistent": [],
}

# Self-hosted docs build overrides (env-driven). When these env vars are
# unset the configuration is unchanged, so the in-repo flow360-docs-checks
# CI build behaves exactly as before. They are set by the flex/main
# self-hosted docs build to control canonical URL, indexing, and version.
_docs_base_url = os.environ.get("FLOW360_DOCS_BASE_URL", "")
if _docs_base_url:
    html_baseurl = _docs_base_url

if os.environ.get("FLOW360_DOCS_NOINDEX") == "1":
    html_meta = {**globals().get("html_meta", {}), "robots": "noindex, nofollow"}

_docs_version = os.environ.get("FLOW360_DOCS_VERSION", "")
if _docs_version:
    version = _docs_version
    release = _docs_version

latex_engine = "xelatex"
math_number_all = True
math_eqref_format = "Eq.({number})"
# Serve MathJax from our own site rather than the cdn.jsdelivr.net default, so
# that readers who cannot reach that host still get typeset equations instead
# of raw LaTeX. Resolved against _static; see _static/mathjax/README.txt.
mathjax_path = "mathjax/tex-mml-chtml.js"
numfig = False
# numfig_secnum_depth = 2  # Not used when numfig is False
templates_path = ["_templates"]
rst_prolog = """
.. role:: inline-code(code)

.. |deg| replace:: °

"""

# TODO: Adjust autodoc_pydantic settings, clean up the above general settings.
autodoc_pydantic_model_show_json = False
autodoc_pydantic_model_show_config_summary = False
autodoc_pydantic_model_show_validator_summary = False
autodoc_pydantic_model_show_validator_members = False
autodoc_pydantic_model_show_field_summary = False
autodoc_pydantic_model_signature_prefix = "class"
autodoc_pydantic_model_member_order = "bysource"

autodoc_pydantic_field_list_validators = False
autodoc_pydantic_field_show_type_annotation = True
autodoc_pydantic_field_signature_prefix = ""
autodoc_pydantic_field_doc_policy = "description"
# to support Unicode in our doc, pdfLatex is not good enough

# Linkcheck configuration
linkcheck_ignore = [
    # Academic publishers with aggressive bot blocking (links work in browsers)
    r"https://arc\.aiaa\.org/.*",  # AIAA - blocks all automated requests
    r"https://doi\.org/10\.2514/.*",  # DOI links that redirect to AIAA
    r"https://www\.sciencedirect\.com/.*",  # Elsevier - blocks automated requests
    r"https://www\.cfd-online\.com/.*",  # CFD Online - blocks automated requests (403)
    # NASA NTRS frequently hits read timeouts from GitHub Actions (often OK in a browser)
    r"https://ntrs\.nasa\.gov/.*",
    # NASA hosts that refuse connections from GitHub Actions runners
    # ("[Errno 101] Network is unreachable"). All verified reachable and
    # serving 200 from an ordinary network, so these are CI egress blocks
    # rather than dead links.
    r"https://transitionmodeling\.larc\.nasa\.gov/.*",
    r"https://hiliftpw-ftp\.larc\.nasa\.gov/.*",
    r"https://www\.grc\.nasa\.gov/.*",
]
linkcheck_timeout = 120  # Seconds per link; NTRS ignored above so CI stays bounded
linkcheck_retries = 2  # Number of retries for failed links
linkcheck_anchors = True  # Check anchors in URLs


def skip_private_attribute_members(app, what, name, obj, skip, options):
    if name.startswith("private_attribute"):
        return True
    return None  # Returning None will process the member as usual


def setup(app):
    app.connect("autodoc-skip-member", skip_private_attribute_members)
    app.connect("builder-inited", generate_notebook_scripts)


def generate_notebook_scripts(app):
    """
    Convert notebooks under python_api/example_library/notebooks/
    into .py scripts in python_api/example_library/notebooks_py/,
    with comments stripped for compact scripts.
    """
    srcdir = Path(app.confdir)  # docs/source
    nb_dir = srcdir / "python_api" / "example_library" / "notebooks"
    out_dir = srcdir / "python_api" / "example_library" / "_notebooks_py"
    out_dir.mkdir(exist_ok=True)

    exporter = ScriptExporter()

    for nb_path in nb_dir.glob("*.ipynb"):
        body, _ = exporter.from_filename(str(nb_path))

        cleaned_lines = []
        for line in body.splitlines():
            stripped = line.lstrip()

            # Drop nbconvert cell markers like "# In[1]:"
            if stripped.startswith("# In[") and stripped.endswith("]:"):
                continue

            # Drop any *full-line* comments (including encoding / metadata)
            if stripped.startswith("#"):
                continue

            # Optionally drop extra blank lines from removed comments
            if stripped == "":
                # keep at most one blank line in a row
                if cleaned_lines and cleaned_lines[-1] == "":
                    continue

            cleaned_lines.append(line if stripped != "" else "")

        body_clean = "\n".join(cleaned_lines)

        py_path = out_dir / f"{nb_path.stem}.py"
        py_path.write_text(body_clean, encoding="utf-8")
        logger.info("Wrote %s", py_path.relative_to(srcdir))

    quickstart_dir = srcdir / "quick_start" / "API_quickstart" / "notebooks"
    quickstart_out_dir = srcdir / "quick_start" / "API_quickstart" / "_notebooks_py"

    quickstart_out_dir.mkdir(exist_ok=True)

    for nb_path in quickstart_dir.glob("*.ipynb"):
        body, _ = exporter.from_filename(str(nb_path))

        cleaned_lines = []
        for line in body.splitlines():
            stripped = line.lstrip()

            # Drop nbconvert cell markers like "# In[1]:"
            if stripped.startswith("# In[") and stripped.endswith("]:"):
                continue

            # Drop any *full-line* comments (including encoding / metadata)
            if stripped.startswith("#"):
                continue

            # Optionally drop extra blank lines from removed comments
            if stripped == "":
                # keep at most one blank line in a row
                if cleaned_lines and cleaned_lines[-1] == "":
                    continue

            cleaned_lines.append(line if stripped != "" else "")

        body_clean = "\n".join(cleaned_lines)

        py_path = quickstart_out_dir / f"{nb_path.stem}.py"
        py_path.write_text(body_clean, encoding="utf-8")
        logger.info("Wrote %s", py_path.relative_to(srcdir))
