/*
 * Flex self-hosted docs version switcher (product-agnostic base).
 * ---------------------------------------------------------------------------
 * Product-local: ships in Flow360's own docs tree (`_static/js/`), synced
 * one-way to flex via Copybara under the existing `flex/public/Flow360/**`
 * allowlist; edit in compute, not flex. Written to be reusable as-is by other
 * self-hosted Sphinx products (a per-product config file + a few conf.py
 * lines is all that's required — see the CONFIG CONTRACT below).
 *
 * Injects a "Version" <select> into the theme sidebar for self-hosted product
 * docs (where RTD's version flyout is absent), populated from the per-product
 * manifest { project, default_version, generated_at, versions:[{name,label,
 * path,hidden?}] }. Zero dependencies, browser-only, conservative ES5; the one
 * modern API is fetch(), typeof-guarded. Any missing config / fetch / JSON
 * error is a console.warn + no-op — the docs page must never break.
 *
 * CONFIG CONTRACT — a per-product file MUST set this BEFORE loading this script:
 *   window.FLEX_DOCS_VERSIONS = {
 *     manifestUrl: "/projects/<product>/versions.json",  // REQUIRED
 *     docsRoot:    "/projects/<product>/en/",            // REQUIRED, trailing /
 *     mounts: [                    // OPTIONAL, one switcher instance per entry
 *       { selectors: [".a", ".b"], // first matching selector wins; entry skipped if none
 *         insert: "append",        // "append" (default) | "prepend"
 *         variant: "inline" }      // "block" (default, sidebar) | "inline" (navbar)
 *     ],
 *     mountSelectors: [".sel-a"],  // OPTIONAL legacy single mount (= one prepend/block entry)
 *     showHidden:   false,         // OPTIONAL also list `hidden` manifest entries
 *     contextLabel: "current build", // OPTIONAL label for off-docsRoot pages
 *     renderLabel:  function (entry) {...},              // OPTIONAL -> option text
 *     onNavigate:   function (entry, currentPath) {...}  // OPTIONAL, see below
 *   };
 *
 * OVERLOAD POINTS
 *   - mounts: render into several containers at once (e.g. desktop navbar +
 *     mobile drawer). Each gets its own instance with a `--<variant>` modifier
 *     class; CSS owns which is visible per breakpoint. `mountSelectors` (or the
 *     built-in defaults) is the legacy single-mount shorthand.
 *   - showHidden: the CURRENT version is always shown even when hidden, marked
 *     "<name> (unlisted)".
 *   - contextLabel: shown (disabled) when the page is NOT under docsRoot (e.g. a
 *     preview build off a non-canonical path); listed versions stay selectable
 *     and navigate to the published site.
 *   - onNavigate(entry, currentPath): return a URL string to have the base
 *     location.assign it, false to suppress (you navigated), or undefined to
 *     fall back to entry.path.
 */
(function () {
  "use strict";

  var WRAPPER_CLASS = "flex-docs-version-switcher";
  var SELECT_ID = "flex-docs-version-switcher-select"; // suffixed per instance

  // Legacy single-mount default candidates (used only when neither `mounts` nor
  // `mountSelectors` is set), tried in order. Covers sphinx_book_theme / pydata.
  var DEFAULT_MOUNT_SELECTORS = [
    ".bd-sidebar-primary .sidebar-primary-items__start",
    ".sidebar-primary-items__start",
    ".bd-sidebar-primary .sidebar-primary-items",
    ".bd-sidebar-primary",
    ".bd-sidebar"
  ];

  function config() {
    return window.FLEX_DOCS_VERSIONS;
  }

  function normalizePath(path) {
    if (!path) {
      return "/";
    }
    return path.charAt(path.length - 1) === "/" ? path : path + "/";
  }

  function validateConfig(cfg) {
    if (!cfg || typeof cfg !== "object") {
      console.warn(
        "[flex-docs-version-switcher] window.FLEX_DOCS_VERSIONS is not set; " +
          "load the per-product config file before version-switcher.js. No switcher rendered."
      );
      return false;
    }
    if (!cfg.manifestUrl || !cfg.docsRoot) {
      console.warn(
        "[flex-docs-version-switcher] FLEX_DOCS_VERSIONS requires both " +
          "`manifestUrl` and `docsRoot`. No switcher rendered."
      );
      return false;
    }
    return true;
  }

  // Current version = first path segment after docsRoot in location.pathname.
  // Returns null when the page is not under docsRoot (e.g. a local preview).
  function currentVersion(cfg) {
    var root = normalizePath(cfg.docsRoot);
    var here = normalizePath(window.location.pathname);
    if (here.indexOf(root) !== 0) {
      return null;
    }
    var rest = here.slice(root.length);
    var slash = rest.indexOf("/");
    var segment = slash === -1 ? rest : rest.slice(0, slash);
    return segment || null;
  }

  function entryPath(cfg, entry) {
    var path = entry && entry.path;
    if (!path) {
      return normalizePath(cfg.docsRoot) + entry.name + "/";
    }
    // Absolute paths and full URLs pass through; relative manifest paths
    // resolve against docsRoot, never against the current page directory.
    if (path.charAt(0) === "/" || /^https?:\/\//.test(path)) {
      return path;
    }
    return normalizePath(cfg.docsRoot) + path;
  }

  function labelFor(cfg, entry) {
    if (typeof cfg.renderLabel === "function") {
      try {
        var custom = cfg.renderLabel(entry);
        if (custom) {
          return String(custom);
        }
      } catch (err) {
        console.warn("[flex-docs-version-switcher] renderLabel threw; using default label.", err);
      }
    }
    return entry.label || entry.name;
  }

  function findEntry(versions, name) {
    for (var i = 0; i < versions.length; i++) {
      if (versions[i] && versions[i].name === name) {
        return versions[i];
      }
    }
    return null;
  }

  // Normalize config into a list of mount specs {selectors, insert, variant}.
  // Prefer the multi-mount `mounts`; else the legacy single `mountSelectors`
  // (or built-in defaults) as one prepend/block entry.
  function resolveMounts(cfg) {
    if (cfg.mounts && cfg.mounts.length) {
      var out = [];
      for (var i = 0; i < cfg.mounts.length; i++) {
        var m = cfg.mounts[i] || {};
        if (!m.selectors || !m.selectors.length) {
          continue;
        }
        out.push({
          selectors: m.selectors,
          insert: m.insert === "prepend" ? "prepend" : "append",
          variant: m.variant === "inline" ? "inline" : "block"
        });
      }
      return out;
    }
    var legacy =
      cfg.mountSelectors && cfg.mountSelectors.length
        ? cfg.mountSelectors
        : DEFAULT_MOUNT_SELECTORS;
    return [{ selectors: legacy, insert: "prepend", variant: "block" }];
  }

  function firstMatch(selectors) {
    for (var i = 0; i < selectors.length; i++) {
      var node = document.querySelector(selectors[i]);
      if (node) {
        return node;
      }
    }
    return null;
  }

  function navigateTo(cfg, entry, target) {
    if (typeof cfg.onNavigate === "function") {
      var current = window.location.pathname;
      var result = cfg.onNavigate(entry, current);
      if (result === false) {
        return; // override handled navigation itself
      }
      if (typeof result === "string" && result) {
        target = result;
      }
    }
    window.location.assign(target);
  }

  // Build a populated <select> for one instance. `selectId` must be unique so
  // multiple instances don't collide on id / label htmlFor.
  function buildSelect(cfg, allVersions, current, listed, currentIsListed, selectId) {
    var select = document.createElement("select");
    select.id = selectId;
    select.className = WRAPPER_CLASS + "__select";

    // A disabled, selected context option marks "where you are" when the page
    // is not one of the listed, switchable versions:
    //   - current === null: page not under docsRoot (e.g. a preview build off a
    //     non-canonical path). Label from cfg.contextLabel ("current build").
    //   - current hidden / absent from the manifest: mark it "<name> (unlisted)".
    var contextText = null;
    var contextValue = "";
    if (!current) {
      contextText = cfg.contextLabel || "current build";
    } else if (!currentIsListed) {
      var contextEntry = findEntry(allVersions, current) || { name: current };
      contextText = labelFor(cfg, contextEntry) + " (unlisted)";
      contextValue = entryPath(cfg, contextEntry);
    }
    if (contextText !== null) {
      var contextOption = document.createElement("option");
      contextOption.value = contextValue;
      contextOption.textContent = contextText;
      contextOption.selected = true;
      contextOption.disabled = true;
      select.appendChild(contextOption);
    }

    for (var m = 0; m < listed.length; m++) {
      var listedEntry = listed[m];
      var option = document.createElement("option");
      option.value = entryPath(cfg, listedEntry);
      option.textContent = labelFor(cfg, listedEntry);
      if (listedEntry.name === current) {
        option.selected = true;
      }
      // Stash the name so the change handler can recover the full entry.
      option.setAttribute("data-version-name", listedEntry.name);
      select.appendChild(option);
    }

    select.addEventListener("change", function () {
      var chosenName = select.options[select.selectedIndex].getAttribute("data-version-name");
      var chosen = findEntry(allVersions, chosenName) || { name: chosenName, path: select.value };
      navigateTo(cfg, chosen, select.value);
    });

    return select;
  }

  function render(cfg, manifest) {
    var allVersions = [];
    if (manifest && Object.prototype.toString.call(manifest.versions) === "[object Array]") {
      for (var i = 0; i < manifest.versions.length; i++) {
        var v = manifest.versions[i];
        if (v && v.name) {
          allVersions.push(v);
        }
      }
    }
    if (!allVersions.length) {
      return;
    }

    var current = currentVersion(cfg);
    var showHidden = cfg.showHidden === true;

    // Listed entries: all non-hidden ones (or everything when showHidden).
    var listed = [];
    for (var j = 0; j < allVersions.length; j++) {
      var entry = allVersions[j];
      if (showHidden || entry.hidden !== true) {
        listed.push(entry);
      }
    }

    var currentIsListed = false;
    for (var k = 0; k < listed.length; k++) {
      if (listed[k].name === current) {
        currentIsListed = true;
        break;
      }
    }

    // One instance per mount spec. CSS owns which instance is visible.
    var mounts = resolveMounts(cfg);
    for (var s = 0; s < mounts.length; s++) {
      var spec = mounts[s];
      var mountPoint = firstMatch(spec.selectors);
      if (!mountPoint) {
        continue;
      }
      // Idempotency guard is PER CONTAINER (not document-wide) so each mount is
      // filled once even across repeated init / dynamic sidebar re-renders.
      if (mountPoint.querySelector("." + WRAPPER_CLASS)) {
        continue;
      }

      var selectId = SELECT_ID + "-" + s;
      var wrapper = document.createElement("div");
      wrapper.className = WRAPPER_CLASS + " " + WRAPPER_CLASS + "--" + spec.variant;

      var label = document.createElement("label");
      label.className = WRAPPER_CLASS + "__label";
      label.htmlFor = selectId;
      label.textContent = "Version";

      wrapper.appendChild(label);
      wrapper.appendChild(buildSelect(cfg, allVersions, current, listed, currentIsListed, selectId));

      if (spec.insert === "prepend") {
        mountPoint.insertBefore(wrapper, mountPoint.firstChild);
      } else {
        mountPoint.appendChild(wrapper);
      }
    }
  }

  function init() {
    var cfg = config();
    if (!validateConfig(cfg)) {
      return;
    }
    if (typeof window.fetch !== "function") {
      return; // no fetch -> silently skip; docs must not break
    }
    window
      .fetch(cfg.manifestUrl, { cache: "no-store" })
      .then(function (response) {
        if (!response || !response.ok) {
          return null;
        }
        return response.json();
      })
      .then(function (manifest) {
        if (!manifest) {
          return;
        }
        render(cfg, manifest);
      })
      ["catch"](function (error) {
        console.warn("[flex-docs-version-switcher] Unable to load the docs version manifest.", error);
      });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
