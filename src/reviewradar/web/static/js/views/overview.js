import { get } from "../api.js";
import { rememberTranslations, refreshTranslations } from "../translations.js";
import {
  STORE_LABELS,
  empty,
  escapeHtml,
  formatNumber,
  formatPercent,
  languageName,
  reviewCard,
} from "../ui.js";

export async function render(root, context) {
  const [overview, critical] = await Promise.all([
    context.overview(),
    get(`/api/apps/${context.app}/reviews`, { sort: "newest", max_rating: 2, limit: 5 }),
  ]);
  if (!context.isCurrent()) return;

  const pipeline = overview.pipeline;
  const stores = new Set(overview.markets.map((market) => market.store));
  const nonEnglish = overview.language_mix
    .filter((row) => row.language && row.language !== "en")
    .reduce((sum, row) => sum + row.review_count, 0);

  root.innerHTML = `
    <section class="kpi-grid" aria-label="Key figures">
      ${kpi("Reviews", formatNumber(overview.total_reviews), `${stores.size} stores, ${overview.markets.length} markets`)}
      ${kpi("Average rating", overview.average_rating ? `${overview.average_rating.toFixed(2)} ★` : "–", "across all markets")}
      ${kpi("Languages", formatNumber(overview.languages), "in reviews")}
      ${kpi("Searchable", formatPercent(pipeline.embedded, pipeline.reviews), "of reviews")}
      ${kpi("Translated", formatPercent(pipeline.translated, nonEnglish), "of non-English reviews")}
    </section>


    <div class="grid-2">
      <section class="card">
        <div class="card-head"><div><h2>Ratings by store</h2></div></div>
        ${ratingBreakdown(overview)}
      </section>
      <section class="card">
        <div class="card-head"><div><h2>Languages</h2></div></div>
        ${languageBreakdown(overview)}
      </section>
    </div>

    <section class="card">
      <div class="card-head">
        <div><h2>Latest critical reviews</h2></div>
        <a class="text-link" href="#/reviews?rating=negative">View all</a>
      </div>
      <div class="review-list">${
        critical.items.length ? critical.items.map((review) => reviewCard(review)).join("") : empty("No critical reviews")
      }</div>
    </section>`;

  rememberTranslations(critical.items);
  refreshTranslations(root);
}

function kpi(label, value, detail) {
  return `<div class="kpi"><div class="kpi-label">${escapeHtml(label)}</div><div class="kpi-value">${escapeHtml(value)}</div><div class="kpi-detail">${escapeHtml(detail)}</div></div>`;
}

function ratingBreakdown(overview) {
  const byStore = new Map();
  for (const row of overview.ratings) {
    if (!byStore.has(row.store)) byStore.set(row.store, [0, 0, 0, 0, 0]);
    byStore.get(row.store)[row.rating - 1] = row.review_count;
  }
  const blocks = [...byStore.entries()].map(([store, counts]) => {
    const total = counts.reduce((sum, count) => sum + count, 0);
    const average = counts.reduce((sum, count, index) => sum + count * (index + 1), 0) / total;
    return `<div class="stack">
      <div class="stack-head"><span>${STORE_LABELS[store]}</span><span class="muted">${average.toFixed(2)} ★ · ${formatNumber(total)} reviews</span></div>
      <div class="stack-bar">${counts
        .map((count, index) =>
          count ? `<span class="r${index + 1}" style="flex:${count}" title="${index + 1} ★ · ${formatNumber(count)} (${formatPercent(count, total)})"></span>` : "",
        )
        .join("")}</div>
    </div>`;
  });
  const legend = [1, 2, 3, 4, 5].map((rating) => `<span><i class="dot r${rating}"></i>${rating} ★</span>`).join("");
  return `${blocks.join("")}<div class="legend">${legend}</div>`;
}

function languageBreakdown(overview) {
  const byStore = new Map();
  for (const row of overview.language_mix) {
    if (!byStore.has(row.store)) byStore.set(row.store, []);
    byStore.get(row.store).push(row);
  }
  return [...byStore.entries()]
    .map(([store, rows]) => {
      const total = rows.reduce((sum, row) => sum + row.review_count, 0);
      const top = rows.slice(0, 6);
      const rest = total - top.reduce((sum, row) => sum + row.review_count, 0);
      const segments = top.map((row, index) => ({ label: languageName(row.language), count: row.review_count, color: index }));
      if (rest > 0) segments.push({ label: "Other", count: rest, color: "other" });
      return `<div class="stack">
        <div class="stack-head"><span>${STORE_LABELS[store]}</span><span class="muted">${formatNumber(total)} reviews</span></div>
        <div class="stack-bar">${segments
          .map((segment) => `<span class="c-${segment.color}" style="flex:${segment.count}" title="${escapeHtml(segment.label)} · ${formatNumber(segment.count)}"></span>`)
          .join("")}</div>
        <div class="legend">${segments
          .map((segment) => `<span><i class="dot c-${segment.color}"></i>${escapeHtml(segment.label)} ${formatPercent(segment.count, total)}</span>`)
          .join("")}</div>
      </div>`;
    })
    .join("");
}
