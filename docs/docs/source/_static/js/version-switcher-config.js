/*
 * Flow360 config for the docs version switcher (product-local, ships
 * alongside conf.py in this docs tree). MUST load BEFORE the base
 * `version-switcher.js` (conf.py orders html_js_files: config, then base).
 * See the base file's header for the full contract and optional overload
 * fields (mounts, showHidden, contextLabel, renderLabel, onNavigate).
 *
 * Single sidebar mount for ALL breakpoints: in sphinx-book-theme the primary
 * sidebar (`.bd-sidebar-primary`) IS the mobile drawer, so one block instance
 * mounted there serves desktop and mobile alike. Prepended into the sidebar's
 * nav tree (`nav.bd-links`) so it lands directly under the sidebar search
 * field, immediately before the toctree. The old navbar (`inline`) mount
 * died with compute PR #6863, which empties `navbar_persistent`, so
 * `.bd-header` never renders and that mount target never exists.
 */
window.FLEX_DOCS_VERSIONS = {
  manifestUrl: "/projects/flow360/versions.json",
  docsRoot: "/projects/flow360/en/",
  mounts: [
    {
      selectors: [".bd-sidebar-primary nav.bd-links", ".bd-sidebar-primary .sidebar-primary-items__start", ".bd-sidebar-primary"],
      insert: "prepend",
      variant: "block"
    }
  ]
};
