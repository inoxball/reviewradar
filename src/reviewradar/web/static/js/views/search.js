import { get } from "../api.js";
import { rememberTranslations, refreshTranslations } from "../translations.js";
import {
  EXAMPLE_QUERIES,
  MODES,
  RATING_PRESETS,
  empty,
  escapeHtml,
  failure,
  languageOptions,
  loading,
  onSegmentChange,
  reviewCard,
  onSelectChange,
  segmented,
  selectControl,
} from "../ui.js";

const STORE_OPTIONS = [
  { id: "", label: "All stores" },
  { id: "app_store", label: "App Store" },
  { id: "google_play", label: "Google Play" },
];
const LIMITS = ["20", "50", "100"].map((id) => ({ id, label: `${id} results` }));

export async function render(root, context) {
  const params = context.params;
  const state = {
    q: params.get("q") ?? EXAMPLE_QUERIES[0],
    mode: MODES[params.get("mode")] ? params.get("mode") : "hybrid",
    store: params.get("store") ?? "",
    language: params.get("language") ?? "",
    rating: params.get("rating") ?? "",
    limit: LIMITS.some((option) => option.id === params.get("limit")) ? params.get("limit") : "20",
  };
  const overview = await context.overview();
  if (!context.isCurrent()) return;

  root.innerHTML = `
    <form class="searchbox" role="search" data-form>
      <svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="11" cy="11" r="7" /><path d="m20 20-3.5-3.5" /></svg>
      <input type="search" data-primary-input value="${escapeHtml(state.q)}" placeholder="Search reviews by meaning or keywords, in any language" aria-label="Search reviews" autocomplete="off" />
      <span class="kbd" title="Press / to focus">/</span>
      <button class="button primary" type="submit">Search</button>
    </form>
    <div class="controls">
      ${segmented("mode", Object.entries(MODES).map(([id, mode]) => ({ id, label: mode.label })), state.mode)}
      ${selectControl("store", STORE_OPTIONS, state.store, "Store")}
      ${selectControl("rating", RATING_PRESETS, state.rating, "Rating")}
      <select data-language aria-label="Language">${languageOptions(overview, state.language)}</select>
      ${selectControl("limit", LIMITS, state.limit, "Number of results")}
    </div>
    <div class="result-bar" data-result-bar></div>
    <div class="review-list" data-results></div>`;

  const input = root.querySelector("[data-primary-input]");
  const results = root.querySelector("[data-results]");
  const resultBar = root.querySelector("[data-result-bar]");
  let runToken = 0;

  async function run() {
    state.q = input.value.trim();
    context.setParams(state);
    if (!state.q) {
      resultBar.textContent = "";
      results.innerHTML = empty("Type a query to search", "Or pick one of the examples above.");
      return;
    }
    const token = ++runToken;
    results.innerHTML = loading("Searching…");
    resultBar.textContent = "";
    const rating = RATING_PRESETS.find((preset) => preset.id === state.rating);
    try {
      const response = await get(`/api/apps/${context.app}/search`, {
        q: state.q,
        mode: state.mode,
        limit: Number(state.limit),
        store: state.store,
        language: state.language,
        min_rating: rating?.min,
        max_rating: rating?.max,
      });
      if (token !== runToken || !context.isCurrent()) return;
      resultBar.innerHTML = `
        <span>${response.results.length} results in ${Math.round(response.took_ms)} ms</span>
        <a class="text-link" href="#/compare?q=${encodeURIComponent(state.q)}">Compare search modes</a>`;
      results.innerHTML = response.results.length
        ? response.results
            .map((hit) => reviewCard(hit, { score: hit.score, scoreLabel: MODES[response.mode].scoreLabel }))
            .join("")
        : empty(
            "No matching reviews",
            response.mode === "lexical"
              ? "Keyword search needs every word to appear. Try <strong>Semantic</strong> or <strong>Hybrid</strong>."
              : "Try fewer filters or a different phrasing.",
          );
      rememberTranslations(response.results);
      refreshTranslations(results);
    } catch (error) {
      if (token === runToken) results.innerHTML = failure(error);
    }
  }

  root.querySelector("[data-form]").addEventListener("submit", (event) => {
    event.preventDefault();
    run();
  });
  onSegmentChange(root, "mode", (mode) => {
    state.mode = mode;
    run();
  });
  onSelectChange(root, "store", (store) => {
    state.store = store;
    run();
  });
  onSelectChange(root, "rating", (rating) => {
    state.rating = rating;
    run();
  });
  onSelectChange(root, "limit", (limit) => {
    state.limit = limit;
    run();
  });
  root.querySelector("[data-language]").addEventListener("change", (event) => {
    state.language = event.target.value;
    run();
  });

  await run();
}
