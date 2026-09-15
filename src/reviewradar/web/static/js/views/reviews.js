import { get } from "../api.js";
import { rememberTranslations, refreshTranslations } from "../translations.js";
import {
  RATING_PRESETS,
  empty,
  failure,
  languageOptions,
  loading,
  onSegmentChange,
  pagination,
  reviewCard,
  onSelectChange,
  segmented,
  selectControl,
} from "../ui.js";

const PAGE_SIZE = 20;
const STORE_OPTIONS = [
  { id: "", label: "All stores" },
  { id: "app_store", label: "App Store" },
  { id: "google_play", label: "Google Play" },
];
const SORTS = [
  ["newest", "Newest first"],
  ["oldest", "Oldest first"],
  ["lowest_rating", "Lowest rating"],
  ["highest_rating", "Highest rating"],
];

export async function render(root, context) {
  const params = context.params;
  const state = {
    store: params.get("store") ?? "",
    rating: params.get("rating") ?? "",
    language: params.get("language") ?? "",
    sort: params.get("sort") ?? "newest",
    page: Math.max(1, Number(params.get("page")) || 1),
  };
  const overview = await context.overview();
  if (!context.isCurrent()) return;

  root.innerHTML = `
    <div class="controls">
      ${selectControl("store", STORE_OPTIONS, state.store, "Store")}
      ${selectControl("rating", RATING_PRESETS, state.rating, "Rating")}
      <select data-language aria-label="Language">${languageOptions(overview, state.language)}</select>
      <select data-sort aria-label="Sort order">${SORTS.map(
        ([id, label]) => `<option value="${id}" ${id === state.sort ? "selected" : ""}>${label}</option>`,
      ).join("")}</select>
    </div>
    <div class="pagination" data-pagination-top></div>
    <div class="review-list" data-results></div>
    <div class="pagination" data-pagination-bottom></div>`;

  const results = root.querySelector("[data-results]");
  let runToken = 0;

  async function load() {
    context.setParams({ ...state, page: state.page > 1 ? state.page : "" });
    const token = ++runToken;
    results.innerHTML = loading("Loading reviews…");
    const rating = RATING_PRESETS.find((preset) => preset.id === state.rating);
    try {
      const page = await get(`/api/apps/${context.app}/reviews`, {
        sort: state.sort,
        limit: PAGE_SIZE,
        offset: (state.page - 1) * PAGE_SIZE,
        store: state.store,
        language: state.language,
        min_rating: rating?.min,
        max_rating: rating?.max,
      });
      if (token !== runToken || !context.isCurrent()) return;
      results.innerHTML = page.items.length
        ? page.items.map((review) => reviewCard(review)).join("")
        : empty("No reviews match these filters");
      renderPagination(page);
      rememberTranslations(page.items);
      refreshTranslations(results);
    } catch (error) {
      if (token === runToken) results.innerHTML = failure(error);
    }
  }

  function renderPagination(page) {
    const markup = pagination(page, PAGE_SIZE);
    for (const container of root.querySelectorAll("[data-pagination-top], [data-pagination-bottom]")) {
      container.innerHTML = markup;
    }
  }

  root.addEventListener("click", (event) => {
    const button = event.target.closest("[data-page]");
    if (!button || button.disabled) return;
    state.page += button.dataset.page === "next" ? 1 : -1;
    load();
    root.scrollIntoView({ block: "start" });
  });
  const reset = (key) => (value) => {
    state[key] = value;
    state.page = 1;
    load();
  };
  onSelectChange(root, "store", reset("store"));
  onSelectChange(root, "rating", reset("rating"));
  root.querySelector("[data-language]").addEventListener("change", (event) => reset("language")(event.target.value));
  root.querySelector("[data-sort]").addEventListener("change", (event) => reset("sort")(event.target.value));

  await load();
}
