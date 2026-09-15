// Formatting helpers and small HTML components shared by every view.

export const STORE_LABELS = { google_play: "Google Play", app_store: "App Store" };

export const MODES = {
  lexical: {
    label: "Keyword",
    short: "K",
    scoreLabel: "Full-text rank (ts_rank_cd)",
    explain:
      "<strong>Keyword</strong> uses PostgreSQL full-text search. Every word must appear, so it is precise for exact terms but misses paraphrases and other languages.",
  },
  semantic: {
    label: "Semantic",
    short: "S",
    scoreLabel: "Cosine similarity of multilingual embeddings",
    explain:
      "<strong>Semantic</strong> compares meaning with multilingual embeddings in pgvector, so it finds paraphrases and reviews written in other languages.",
  },
  hybrid: {
    label: "Hybrid",
    short: "H",
    scoreLabel: "Reciprocal rank fusion score",
    explain:
      "<strong>Hybrid</strong> merges the keyword and semantic rankings with reciprocal rank fusion: reviews both retrievers rank highly come first.",
  },
};

export const RATING_PRESETS = [
  { id: "", label: "Any rating" },
  { id: "negative", label: "★ 1–2", min: 1, max: 2 },
  { id: "neutral", label: "★ 3", min: 3, max: 3 },
  { id: "positive", label: "★ 4–5", min: 4, max: 5 },
];

export const EXAMPLE_QUERIES = [
  "the energy system limits how many lessons I can do",
  "too many ads",
  "a energia acaba muito rápido",
  "zu viel Werbung",
  "microphone does not recognize my voice",
  "毎日楽しく勉強できる",
];

const languageNames = (() => {
  try {
    return new Intl.DisplayNames(["en"], { type: "language" });
  } catch {
    return null;
  }
})();

export function languageName(code) {
  if (!code) return "Unknown language";
  if (code === "zxx") return "No linguistic content";
  try {
    return languageNames?.of(code) ?? code;
  } catch {
    return code;
  }
}

export function escapeHtml(value) {
  return String(value ?? "").replace(
    /[&<>"']/g,
    (character) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[character],
  );
}

export const $ = (id) => document.getElementById(id);

export const formatNumber = (value) => new Intl.NumberFormat("en-US").format(value);

export function formatPercent(part, total) {
  if (!total) return "–";
  const percent = (100 * part) / total;
  return `${percent >= 99.5 || percent < 1 ? percent.toFixed(percent < 1 && percent > 0 ? 1 : 0) : Math.round(percent)}%`;
}

export const formatDate = (iso) =>
  new Date(iso).toLocaleDateString("en-GB", { day: "numeric", month: "short" });

export const formatDateTime = (iso) =>
  new Date(iso).toLocaleString("en-GB", {
    day: "numeric",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
  });

export function stars(rating) {
  return `<span class="stars" aria-label="${rating} out of 5 stars">${"★".repeat(rating)}<span class="stars-off">${"★".repeat(5 - rating)}</span></span>`;
}

export function loading(message) {
  return `<div class="state"><span class="spinner" aria-hidden="true"></span>${escapeHtml(message)}</div>`;
}

/** An empty state; ``detailHtml`` must already be escaped. */
export function empty(title, detailHtml = "") {
  return `<div class="state"><div><strong>${escapeHtml(title)}</strong>${detailHtml ? `<p>${detailHtml}</p>` : ""}</div></div>`;
}

export function failure(error) {
  return `<div class="state error"><div><strong>Request failed</strong><p>${escapeHtml(error.message)}</p></div></div>`;
}

/** A segmented control; wire it with ``onSegmentChange``. */
export function segmented(name, options, current) {
  return `<div class="segmented" role="radiogroup" data-segment="${name}">${options
    .map(
      (option) =>
        `<button type="button" role="radio" aria-checked="${option.id === current}" data-value="${escapeHtml(option.id)}">${escapeHtml(option.label)}</button>`,
    )
    .join("")}</div>`;
}

export function onSegmentChange(root, name, handler) {
  const control = root.querySelector(`[data-segment="${name}"]`);
  control?.addEventListener("click", (event) => {
    const button = event.target.closest("button[data-value]");
    if (!button) return;
    control.querySelectorAll("button").forEach((option) => {
      option.setAttribute("aria-checked", String(option === button));
    });
    handler(button.dataset.value);
  });
}

export function languageOptions(overview, current) {
  const totals = new Map();
  for (const row of overview.language_mix) {
    if (row.language) totals.set(row.language, (totals.get(row.language) ?? 0) + row.review_count);
  }
  const options = [...totals.entries()]
    .sort((a, b) => b[1] - a[1])
    .map(
      ([code, count]) =>
        `<option value="${escapeHtml(code)}" ${code === current ? "selected" : ""}>${escapeHtml(languageName(code))} · ${formatNumber(count)}</option>`,
    )
    .join("");
  return `<option value="">All languages</option>${options}`;
}

export function reviewCard(review, { score = null, scoreLabel = "", extraBadges = "", compact = false } = {}) {
  const meta = [
    STORE_LABELS[review.store] ?? review.store,
    languageName(review.language),
    formatDate(review.reviewed_at),
    review.app_version ? `v${review.app_version}` : null,
    review.country ? review.country.toUpperCase() : null,
  ]
    .filter(Boolean)
    .map(escapeHtml)
    .join(" · ");
  return `<article class="review${compact ? " compact" : ""}" data-review-id="${review.review_id}" data-language="${escapeHtml(review.language ?? "")}">
    <header class="review-head">
      ${stars(review.rating)}
      <span class="muted">${meta}</span>
      ${extraBadges}
      ${score === null ? "" : `<span class="score" title="${escapeHtml(scoreLabel)}">${score.toFixed(3)}</span>`}
    </header>
    <p class="review-text" data-role="original" title="Click to expand">${escapeHtml(review.text || "(no text)")}</p>
    <p class="review-text translated" data-role="translation" hidden title="Click to expand"></p>
    <footer class="review-foot" data-role="translation-status" hidden></footer>
  </article>`;
}

/** Result range and previous/next buttons for a paged response; buttons carry ``data-page``. */
export function pagination(page, pageSize, noun = "reviews") {
  const current = Math.floor(page.offset / pageSize) + 1;
  const first = page.total ? page.offset + 1 : 0;
  const last = page.offset + page.items.length;
  const pages = Math.max(1, Math.ceil(page.total / pageSize));
  return `
    <span>${formatNumber(first)}–${formatNumber(last)} of ${formatNumber(page.total)} ${escapeHtml(noun)}</span>
    <span class="controls">
      <button type="button" class="button ghost" data-page="prev" ${current <= 1 ? "disabled" : ""}>← Previous</button>
      <span>Page ${current} of ${formatNumber(pages)}</span>
      <button type="button" class="button ghost" data-page="next" ${current >= pages ? "disabled" : ""}>Next →</button>
    </span>`;
}

export function codeBlock(content, { html = false, copyText = null } = {}) {
  const text = copyText ?? (html ? null : content);
  return `<div class="code-block">${
    text === null ? "" : `<button type="button" class="copy" data-copy="${escapeHtml(text)}">Copy</button>`
  }<pre>${html ? content : escapeHtml(content)}</pre></div>`;
}

export function highlightJson(value) {
  const json = escapeHtml(JSON.stringify(value, null, 2) ?? "null");
  return json.replace(
    /(&quot;(?:\\.|[^&\\]|&(?!quot;))*?&quot;)(\s*:)?|\b(true|false|null)\b|-?\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b/g,
    (match, string, colon, literal) => {
      if (string) {
        return colon ? `<span class="j-key">${string}</span>${colon}` : `<span class="j-str">${string}</span>`;
      }
      if (literal) return `<span class="j-lit">${match}</span>`;
      return `<span class="j-num">${match}</span>`;
    },
  );
}

let toastTimer = null;

export function toast(message, { timeout = 4500 } = {}) {
  const element = $("toast");
  element.textContent = message;
  element.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    element.hidden = true;
  }, timeout);
}

/** A labelled select; wire it with ``onSelectChange``. */
export function selectControl(name, options, current, label) {
  return `<select data-select="${name}" aria-label="${escapeHtml(label)}">${options
    .map(
      (option) =>
        `<option value="${escapeHtml(option.id)}" ${option.id === current ? "selected" : ""}>${escapeHtml(option.label)}</option>`,
    )
    .join("")}</select>`;
}

export function onSelectChange(root, name, handler) {
  root.querySelector(`[data-select="${name}"]`)?.addEventListener("change", (event) => handler(event.target.value));
}
