import { get } from "../api.js";
import { rememberTranslations, refreshTranslations } from "../translations.js";
import { EXAMPLE_QUERIES, MODES, empty, escapeHtml, failure, loading, reviewCard } from "../ui.js";

const LIMIT = 8;

export async function render(root, context) {
  const initialQuery = context.params.get("q") ?? EXAMPLE_QUERIES[0];

  root.innerHTML = `
    <form class="searchbox" role="search" data-form>
      <svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="11" cy="11" r="7" /><path d="m20 20-3.5-3.5" /></svg>
      <input type="search" data-primary-input value="${escapeHtml(initialQuery)}" placeholder="One query, three retrieval strategies" aria-label="Query to compare" autocomplete="off" />
      <button class="button primary" type="submit">Compare</button>
    </form>
    <div class="examples"><span class="muted">Examples</span>${EXAMPLE_QUERIES.map(
      (query) => `<button type="button" class="chip" data-example="${escapeHtml(query)}">${escapeHtml(query)}</button>`,
    ).join("")}</div>
    <p class="explainer">Badges such as <span class="badge rank semantic">S#2</span> show where the same review ranks in the other modes. Hybrid rewards reviews that both keyword and semantic search rank highly.</p>
    <div class="result-bar" data-summary></div>
    <div class="compare-grid" data-grid></div>`;

  const input = root.querySelector("[data-primary-input]");
  const grid = root.querySelector("[data-grid]");
  const summary = root.querySelector("[data-summary]");
  let runToken = 0;

  async function run() {
    const query = input.value.trim();
    context.setParams({ q: query });
    if (!query) {
      grid.innerHTML = empty("Type a query to compare");
      summary.textContent = "";
      return;
    }
    const token = ++runToken;
    grid.innerHTML = loading("Running keyword, semantic and hybrid search…");
    summary.textContent = "";
    try {
      const response = await get(`/api/apps/${context.app}/compare`, { q: query, limit: LIMIT });
      if (token !== runToken || !context.isCurrent()) return;

      const ranks = Object.fromEntries(
        response.modes.map((entry) => [
          entry.mode,
          new Map(entry.results.map((hit, index) => [hit.review_id, index + 1])),
        ]),
      );
      const lexicalIds = new Set(ranks.lexical?.keys() ?? []);
      const overlap = [...(ranks.semantic?.keys() ?? [])].filter((id) => lexicalIds.has(id)).length;
      summary.innerHTML = `<span>${response.modes
        .map((entry) => `${MODES[entry.mode].label}: ${entry.results.length} in ${entry.took_ms} ms`)
        .join(" · ")}</span><span>Found by both keyword and semantic: ${overlap}</span>`;

      grid.innerHTML = response.modes.map((entry) => column(entry, ranks)).join("");
      rememberTranslations(response.modes.flatMap((entry) => entry.results));
      refreshTranslations(grid);
    } catch (error) {
      if (token === runToken) grid.innerHTML = failure(error);
    }
  }

  root.querySelector("[data-form]").addEventListener("submit", (event) => {
    event.preventDefault();
    run();
  });
  root.querySelectorAll("[data-example]").forEach((chip) => {
    chip.addEventListener("click", () => {
      input.value = chip.dataset.example;
      run();
    });
  });

  await run();
}

function column(entry, ranks) {
  const mode = MODES[entry.mode];
  const cards = entry.results.map((hit) => {
    const badges = Object.entries(ranks)
      .filter(([other]) => other !== entry.mode)
      .map(([other, positions]) => {
        const rank = positions.get(hit.review_id);
        return rank
          ? `<span class="badge rank ${other}" title="Rank ${rank} in ${MODES[other].label} search">${MODES[other].short}#${rank}</span>`
          : "";
      })
      .join("");
    return reviewCard(hit, { score: hit.score, scoreLabel: mode.scoreLabel, extraBadges: badges, compact: true });
  });
  return `<section class="column" aria-label="${mode.label} results">
    <div class="column-head">
      <div class="column-title"><span>${mode.label}</span><span class="badge accent">${entry.results.length} · ${entry.took_ms} ms</span></div>
      <p class="muted small" style="margin:0">${mode.explain.replace(/<[^>]+>/g, "")}</p>
    </div>
    ${cards.length ? cards.join("") : empty("No results", entry.mode === "lexical" ? "Not every query word appears in any review." : "")}
  </section>`;
}
