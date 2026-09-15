import { get, post } from "../api.js";
import { rememberTranslations, refreshTranslations } from "../translations.js";
import {
  RATING_PRESETS,
  empty,
  escapeHtml,
  failure,
  formatNumber,
  formatPercent,
  loading,
  onSegmentChange,
  pagination,
  reviewCard,
  segmented,
} from "../ui.js";

const PAGE_SIZE = 6;
const REPLY_LIMIT = 350;
const GENERATOR_LABELS = {
  "fine-tuned": "Fine-tuned (LoRA)",
  retrieval: "Retrieval few-shot",
  "zero-shot": "Zero-shot",
  reference: "Reference replies",
};
const GENERATOR_HINTS = {
  "fine-tuned": "Qwen3-1.7B with a LoRA adapter trained on published and written replies.",
  retrieval: "The base model with the guidelines and the three most similar answered reviews.",
  "zero-shot": "The base model following the reply guidelines only.",
};
const CHECK_LABELS = {
  non_empty: "Substantive",
  right_language: "Review's language",
  within_limit: "≤ 350 characters",
  no_foreign_brand_or_contact: "No other brand or contact",
  valid_placeholders: "Valid placeholders",
  not_copied: "Not copied from review",
  complete: "Complete sentence",
};
const RATINGS = RATING_PRESETS.filter((preset) => preset.id !== "positive");

export async function render(root, context) {
  const state = {
    rating: context.params.get("rating") ?? "negative",
    page: Math.max(1, Number(context.params.get("page")) || 1),
  };
  const info = await get("/api/replies/generators");
  if (!context.isCurrent()) return;
  // Put the fine-tuned generator first: it is the one being evaluated.
  const generators = [...info.generators].sort(
    (a, b) => Object.keys(GENERATOR_HINTS).indexOf(a) - Object.keys(GENERATOR_HINTS).indexOf(b),
  );
  const status = { loaded: info.loaded };

  root.innerHTML = `
    <section class="card">
      <div class="card-head">
        <div>
          <h2>Offline benchmark</h2>
          <p>${benchmarkCaption(info.benchmark)}</p>
        </div>
      </div>
      ${benchmarkTable(info.benchmark)}
    </section>
    <section class="card">
      <div class="card-head">
        <div>
          <h2>Draft replies</h2>
          <p>Every draft follows the reply guidelines, is stored, and is checked automatically. Drafts are suggestions for a person to review; nothing is posted.</p>
        </div>
      </div>
      <div class="controls">
        ${segmented("rating", RATINGS, state.rating)}
        <span class="muted small">${generators.map((name) => escapeHtml(GENERATOR_LABELS[name] ?? name)).join(" · ")}</span>
      </div>
    </section>
    <div class="pagination" data-pagination-top></div>
    <div class="reply-list" data-results></div>
    <div class="pagination" data-pagination-bottom></div>`;

  const results = root.querySelector("[data-results]");
  let runToken = 0;

  async function load() {
    context.setParams({ rating: state.rating, page: state.page > 1 ? state.page : "" });
    const token = ++runToken;
    results.innerHTML = loading("Loading reviews…");
    const rating = RATINGS.find((preset) => preset.id === state.rating);
    try {
      const page = await get(`/api/apps/${context.app}/reviews`, {
        sort: "newest",
        limit: PAGE_SIZE,
        offset: (state.page - 1) * PAGE_SIZE,
        min_rating: rating?.min,
        max_rating: rating?.max,
      });
      if (token !== runToken || !context.isCurrent()) return;
      results.innerHTML = page.items.length
        ? page.items.map((review) => replyItem(review)).join("")
        : empty("No reviews match this filter");
      for (const container of root.querySelectorAll("[data-pagination-top], [data-pagination-bottom]")) {
        container.innerHTML = pagination(page, PAGE_SIZE);
      }
      rememberTranslations(page.items);
      refreshTranslations(results);
    } catch (error) {
      if (token === runToken) results.innerHTML = failure(error);
    }
  }

  async function draft(button) {
    const reviewId = Number(button.dataset.draft);
    const grid = results.querySelector(`[data-drafts-for="${reviewId}"]`);
    button.disabled = true;
    grid.innerHTML = generators.map((name) => pendingCard(name, status.loaded)).join("");
    for (const name of generators) {
      const slot = grid.querySelector(`[data-generator="${name}"]`);
      try {
        const response = await post("/api/replies/drafts", { review_ids: [reviewId], generator: name });
        if (!context.isCurrent()) return;
        status.loaded = true;
        slot.outerHTML = draftCard(response.items[0], response.took_ms);
      } catch (error) {
        slot.outerHTML = `<article class="reply-card" data-generator="${escapeHtml(name)}">${failure(error)}</article>`;
      }
    }
    button.disabled = false;
    button.textContent = "Draft again";
  }

  root.addEventListener("click", (event) => {
    const draftButton = event.target.closest("[data-draft]");
    if (draftButton && !draftButton.disabled) {
      draft(draftButton);
      return;
    }
    const pageButton = event.target.closest("[data-page]");
    if (pageButton && !pageButton.disabled) {
      state.page += pageButton.dataset.page === "next" ? 1 : -1;
      load();
      root.scrollIntoView({ block: "start" });
    }
  });
  onSegmentChange(root, "rating", (value) => {
    state.rating = value;
    state.page = 1;
    load();
  });

  await load();
}

function replyItem(review) {
  return `<div class="reply-item">
    ${reviewCard(review)}
    <div class="reply-actions">
      <button type="button" class="button primary" data-draft="${review.review_id}">Draft replies</button>
    </div>
    <div class="reply-grid" data-drafts-for="${review.review_id}"></div>
  </div>`;
}

function pendingCard(name, loaded) {
  const message = loaded ? "Drafting…" : "Loading the reply model (first request, ~20 s)…";
  return `<article class="reply-card" data-generator="${escapeHtml(name)}">
    <header class="reply-card-head"><strong>${escapeHtml(GENERATOR_LABELS[name] ?? name)}</strong></header>
    ${loading(message)}
  </article>`;
}

function draftCard(item, tookMs) {
  const label = GENERATOR_LABELS[item.generator] ?? item.generator;
  if (item.status === "unavailable" || !item.text) {
    return `<article class="reply-card" data-generator="${escapeHtml(item.generator)}">
      <header class="reply-card-head"><strong>${escapeHtml(label)}</strong><span class="badge">unavailable</span></header>
      <p class="muted small">This review has no enriched text${item.generator === "retrieval" ? " or embedding" : ""} yet.</p>
    </article>`;
  }
  const length = item.text.length;
  const checks = Object.entries(CHECK_LABELS)
    .map(([key, text]) => `<span class="check-chip${item.checks?.[key] ? "" : " fail"}" title="${escapeHtml(text)}">${item.checks?.[key] ? "✓" : "✕"} ${escapeHtml(text)}</span>`)
    .join("");
  const timing = item.status === "cached" ? "stored draft" : `${formatNumber(Math.round(tookMs / 100) / 10)} s`;
  return `<article class="reply-card" data-generator="${escapeHtml(item.generator)}">
    <header class="reply-card-head">
      <strong title="${escapeHtml(GENERATOR_HINTS[item.generator] ?? "")}">${escapeHtml(label)}</strong>
      <span class="badge ${item.checks?.passed ? "success" : "warning"}">${item.checks?.passed ? "checks passed" : "needs review"}</span>
    </header>
    <p class="reply-text">${escapeHtml(item.text)}</p>
    <div class="reply-meta">
      <span class="${length > REPLY_LIMIT ? "status-error" : "muted"} small">${length}/${REPLY_LIMIT} characters · ${escapeHtml(timing)}</span>
      <button type="button" class="button ghost" data-copy="${escapeHtml(item.text)}">Copy</button>
    </div>
    <div class="check-list">${checks}</div>
  </article>`;
}

function benchmarkCaption(benchmark) {
  if (!benchmark) {
    return "No benchmark yet: run <code>reviewradar replies evaluate</code> to score every generator on held-out reviews.";
  }
  const source = benchmark.source === "written" ? "held-out Duolingo reviews with written reference replies" : `${escapeHtml(benchmark.source)} held-out replies`;
  return `${formatNumber(benchmark.examples)} ${source}, scored with automatic checks · ${new Date(benchmark.created_at).toLocaleString("en-GB", { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" })}`;
}

function benchmarkTable(benchmark) {
  if (!benchmark) return "";
  const rate = (value) => (value === undefined ? "–" : formatPercent(value, 1));
  const rows = benchmark.generators
    .map(
      (row) => `<tr>
        <td><strong>${escapeHtml(GENERATOR_LABELS[row.name] ?? row.name)}</strong></td>
        <td class="num">${rate(row.pass_rate)}</td>
        <td class="num">${rate(row.check_rates.right_language)}</td>
        <td class="num">${rate(row.check_rates.within_limit)}</td>
        <td class="num">${rate(row.check_rates.no_foreign_brand_or_contact)}</td>
        <td class="num">${rate(row.opening_diversity)}</td>
        <td class="num">${formatNumber(Math.round(row.median_chars))}</td>
        <td class="num">${row.name === "reference" ? "–" : `${row.seconds_per_reply.toFixed(2)} s`}</td>
      </tr>`,
    )
    .join("");
  return `<div class="table-wrap"><table>
    <thead><tr>
      <th>Generator</th><th class="num">All checks</th><th class="num">Language</th><th class="num">≤ 350 chars</th>
      <th class="num">No other brand</th><th class="num" title="Distinct 5-word openings per reply">Distinct openings</th>
      <th class="num">Median chars</th><th class="num">Time per reply</th>
    </tr></thead>
    <tbody>${rows}</tbody>
  </table></div>`;
}
