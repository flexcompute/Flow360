// example-filter.js
document.addEventListener("DOMContentLoaded", () => {
  const bar  = document.querySelector(".ex-filterbar");
  const grid = document.querySelector(".ex-grid");
  if (!bar || !grid) return;

  const row   = grid.querySelector(".sd-row") || grid;
  const items = Array.from(row.querySelectorAll(".sd-col"));
  if (!items.length) return;

  const cardOf = (item) => item.querySelector(".sd-card");

  // --- Utils: slug + prettify ---
  const slug = (s) => (s || "").toString().trim().toLowerCase()
    .replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");

  const ACRONYMS = new Set(["ai", "rans", "bet", "dns", "les", "ddes", "mrf", "udd", "cht"]); // add more if needed
  function prettify(tok) {
    if (!tok) return "";

    const cap = (w) => (w ? (w.charAt(0).toUpperCase() + w.slice(1)) : "");
    return tok
      .split("-")
      .map((w) => (ACRONYMS.has(w) ? w.toUpperCase() : cap(w)))
      .join(" ");
  }

  // --- Build groups + normalize buttons ---
  const groups = Array.from(bar.querySelectorAll(".ex-group"));
  const groupButtons = new Map();
  groups.forEach(g => {
    const name = g.dataset.group || "default";
    const btns = Array.from(g.querySelectorAll(".ex-filter"));
    // normalize each button: prefer data-filter if present
    btns.forEach(b => {
      const tok = slug(b.dataset.filter || b.textContent);
      b.dataset.filter = tok;        // ensure it's a slug
      b.textContent   = prettify(tok); // pretty label in UI
    });
    groupButtons.set(name, btns);
  });

  // --- Read tags from cards as tokens ---
  function cardTags(item) {
    const cls = Array.from(cardOf(item).classList);
    return new Set(cls.filter(c => c.startsWith("tag-")).map(c => slug(c.slice(4))));
  }

  // --- Tag pills inside each card ---
  function renderTags() {
    items.forEach(item => {
      const card = cardOf(item); if (!card) return;
      const body = card.querySelector(".sd-card-body") || card;

      const tokens = Array.from(card.classList)
        .filter(c => c.startsWith("tag-"))
        .map(c => slug(c.slice(4)))
        .sort();

      let wrap = body.querySelector(".ex-tags");
      if (!wrap) {
        wrap = document.createElement("div");
        wrap.className = "ex-tags";
        body.appendChild(wrap);
      }
      wrap.innerHTML = "";
      tokens.forEach(tok => {
        const pill = document.createElement("span");
        pill.className = "ex-tag";
        pill.textContent = prettify(tok);
        wrap.appendChild(pill);
      });
    });
  }

  // --- Selection logic ---
  function selectedByGroup() {
    const need = new Map();
    groupButtons.forEach((btns, groupName) => {
      const picks = btns
        .filter(b => b.dataset.filter !== "all" && b.classList.contains("is-active"))
        .map(b => b.dataset.filter); // already slugged
      need.set(groupName, picks);
    });
    return need;
  }

  // OR within a group; AND across groups with picks
  function isMatchGrouped(needMap, have) {
    for (const [, picks] of needMap.entries()) {
      if (!picks.length) continue;
      if (!picks.some(t => have.has(t))) return false;
    }
    return true;
  }

  // "No results" message, shown only when there are no matching cards
  const noResults = document.createElement("div");
  noResults.className = "ex-no-results";
  noResults.textContent = "No examples match the selected filters.";

  function applyFilter() {
    const need = selectedByGroup();
    const matched = [];

    // Decide which cards match and collect them in original DOM order
    items.forEach(item => {
      const ok = isMatchGrouped(need, cardTags(item));
      item.classList.toggle("is-matched", ok);
      if (ok) {
        matched.push(item);
      }
    });

    // Remove any existing cards and message from the row
    Array.from(row.querySelectorAll(".sd-col")).forEach(col => {
      row.removeChild(col);
    });
    if (noResults.parentNode) {
      noResults.parentNode.removeChild(noResults);
    }

    // If nothing matches, show the message inside the grid
    if (!matched.length) {
      row.appendChild(noResults);
    } else {
      // Otherwise show only the matched cards, in a consistent order
      const frag = document.createDocumentFragment();
      matched.forEach(el => {
        el.style.display = ""; // ensure visible
        frag.appendChild(el);
      });
      row.appendChild(frag);
    }
  }

  // --- Wire up buttons (per group) ---
  groupButtons.forEach((btns) => {
    btns.forEach(btn => {
      btn.setAttribute("type", "button");
      btn.addEventListener("click", () => {
        if (btn.dataset.filter === "all") {
          btns.forEach(b => { b.classList.remove("is-active"); b.setAttribute("aria-pressed", "false"); });
          btn.classList.add("is-active");
          btn.setAttribute("aria-pressed", "true");
        } else {
          btn.classList.toggle("is-active");
          btn.setAttribute("aria-pressed", btn.classList.contains("is-active") ? "true" : "false");
          const allBtn = btns.find(b => b.dataset.filter === "all");
          const anySpecific = btns.some(b => b.dataset.filter !== "all" && b.classList.contains("is-active"));
          if (allBtn) {
            allBtn.classList.toggle("is-active", !anySpecific);
            allBtn.setAttribute("aria-pressed", !anySpecific ? "true" : "false");
          }
        }
        applyFilter();
      });
    });
    // init group "All"
    const allBtn = btns.find(b => b.dataset.filter === "all");
    const anySpecific = btns.some(b => b.dataset.filter !== "all" && b.classList.contains("is-active"));
    if (allBtn) {
      allBtn.classList.toggle("is-active", !anySpecific);
      allBtn.setAttribute("aria-pressed", !anySpecific ? "true" : "false");
    }
  });

  renderTags();
  applyFilter();
});
