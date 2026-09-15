import { get } from "../api.js";
import {
  STORE_LABELS,
  codeBlock,
  empty,
  escapeHtml,
  formatDateTime,
  formatNumber,
  formatPercent,
} from "../ui.js";

const SOURCE_LABELS = {
  detected: ["Detected", "Confident detection, or a weaker one confirmed by store or storefront evidence."],
  store_hint: ["Store hint", "Google Play served the review for a requested language."],
  market_default: ["Market default", "Short text; the storefront's main language from the catalog."],
  unknown: ["Unknown", "No reliable evidence."],
};

export async function render(root, context) {
  const [overview, ingestion, system] = await Promise.all([
    context.overview(),
    get(`/api/apps/${context.app}/ingestion`, { limit: 30 }),
    get("/api/system"),
  ]);
  if (!context.isCurrent()) return;

  const pipeline = overview.pipeline;
  const nonEnglish = overview.language_mix
    .filter((row) => row.language && row.language !== "en")
    .reduce((sum, row) => sum + row.review_count, 0);

  root.innerHTML = `
    <div class="grid-2">
      <section class="card">
        <div class="card-head"><div><h2>Coverage</h2></div></div>
        ${coverage("Enriched", pipeline.enriched, pipeline.reviews, "Redacted text and resolved language")}
        ${coverage("Embedded", pipeline.embedded, pipeline.reviews, `Vectors from ${system.embedding_model}`)}
        ${coverage("Translated", pipeline.translated, nonEnglish, `Non-English reviews in English via ${system.translation.model}`)}
      </section>
      <section class="card">
        <div class="card-head"><div><h2>How languages were resolved</h2></div></div>
        ${languageSources(pipeline)}
      </section>
    </div>

    <section class="card">
      <div class="card-head"><div><h2>Incremental cursors</h2></div></div>
      ${cursorsTable(ingestion.cursors)}
    </section>

    <section class="card">
      <div class="card-head"><div><h2>Recent ingestion runs</h2></div></div>
      ${runsTable(ingestion.runs)}
    </section>`;
}

function coverage(label, done, total, detail) {
  const percent = total ? Math.min(100, (100 * done) / total) : 0;
  return `<div class="stack">
    <div class="stack-head"><span><strong>${label}</strong> <span class="muted small">${escapeHtml(detail)}</span></span><span class="muted">${formatNumber(done)} / ${formatNumber(total)} · ${formatPercent(done, total)}</span></div>
    <div class="progress"><span style="width:${percent.toFixed(1)}%"></span></div>
  </div>`;
}

function languageSources(pipeline) {
  const total = Object.values(pipeline.language_sources).reduce((sum, count) => sum + count, 0);
  if (!total) return empty("No enriched reviews yet");
  const order = ["detected", "store_hint", "market_default", "unknown"];
  const bar = order
    .map((source, index) => {
      const count = pipeline.language_sources[source] ?? 0;
      return count ? `<span class="c-${index}" style="flex:${count}" title="${SOURCE_LABELS[source][0]} · ${formatNumber(count)}"></span>` : "";
    })
    .join("");
  const rows = order
    .map((source, index) => {
      const count = pipeline.language_sources[source] ?? 0;
      return `<tr><td><i class="dot c-${index}"></i>${SOURCE_LABELS[source][0]}</td><td class="num">${formatNumber(count)}</td><td class="num">${formatPercent(count, total)}</td><td class="wrap muted">${SOURCE_LABELS[source][1]}</td></tr>`;
    })
    .join("");
  return `<div class="stack-bar">${bar}</div><div class="table-wrap" style="margin-top:10px"><table><tbody>${rows}</tbody></table></div>`;
}

function cursorsTable(cursors) {
  if (!cursors.length) return empty("No cursors yet", "Run <code>reviewradar ingest</code> first.");
  return `<div class="table-wrap"><table>
    <thead><tr><th>Store</th><th>Partition</th><th>Newest review ingested</th><th>Updated</th></tr></thead>
    <tbody>${cursors
      .map(
        (cursor) => `<tr>
          <td><span class="badge store ${cursor.store}">${STORE_LABELS[cursor.store]}</span></td>
          <td class="mono">${escapeHtml(cursor.partition_key)}</td>
          <td>${formatDateTime(cursor.newest_reviewed_at)}</td>
          <td class="muted">${formatDateTime(cursor.updated_at)}</td>
        </tr>`,
      )
      .join("")}</tbody></table></div>`;
}

function runsTable(runs) {
  if (!runs.length) return empty("No ingestion runs recorded");
  return `<div class="table-wrap"><table>
    <thead><tr><th>Started</th><th>Store</th><th>Partition</th><th>Status</th><th class="num">Fetched</th><th class="num">New</th><th class="num">Updated</th><th class="num">Unchanged</th><th>Coverage</th><th>Error</th></tr></thead>
    <tbody>${runs
      .map(
        (run) => `<tr>
          <td>${formatDateTime(run.started_at)}</td>
          <td><span class="badge store ${run.store}">${STORE_LABELS[run.store]}</span></td>
          <td class="mono">${escapeHtml(run.partition_key)}</td>
          <td>${statusBadge(run.status)}</td>
          <td class="num">${formatNumber(run.fetched)}</td>
          <td class="num">${formatNumber(run.inserted)}</td>
          <td class="num">${formatNumber(run.updated)}</td>
          <td class="num">${formatNumber(run.unchanged)}</td>
          <td>${coverageBadge(run.coverage_complete)}</td>
          <td class="wrap">${run.error ? `<span class="status-error">${escapeHtml(run.error)}</span>` : ""}</td>
        </tr>`,
      )
      .join("")}</tbody></table></div>`;
}

function statusBadge(status) {
  const tone = { succeeded: "success", failed: "danger", running: "warning" }[status] ?? "";
  return `<span class="badge ${tone}">${escapeHtml(status)}</span>`;
}

function coverageBadge(complete) {
  if (complete === null) return `<span class="muted">—</span>`;
  return complete
    ? `<span class="badge success">complete</span>`
    : `<span class="badge warning" title="The store stopped serving older reviews before the window's start">partial</span>`;
}
