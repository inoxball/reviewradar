import { get, post } from "../api.js";
import { rememberTranslations, refreshTranslations } from "../translations.js";
import { monitorSection, spikeSection } from "./spikes.js";
import {
  empty,
  escapeHtml,
  failure,
  formatDateTime,
  formatNumber,
  formatPercent,
  languageName,
  loading,
  onSegmentChange,
  pagination,
  reviewCard,
  segmented,
  toast,
} from "../ui.js";

const PAGE_SIZE = 20;
const SAMPLES_PER_TOPIC = 2;
const RECENT_DAYS = 7;
const RISING_THRESHOLD = 1.5;
const MIN_RISING_SIZE = 10;
const TOPIC_COUNTS = [10, 20, 30, 40];
const SORTS = [
  { id: "size", label: "Largest" },
  { id: "negative", label: "Most negative" },
  { id: "rising", label: "Rising" },
];
const SCOPES = [
  { id: "critical", label: "★ 1–3", request: { max_rating: 3 } },
  { id: "all", label: "All ratings", request: { all_ratings: true } },
];

export async function render(root, context) {
  const topicId = Number(context.params.get("topic"));
  if (topicId) await renderTopic(root, context, topicId);
  else await renderTopics(root, context);
}

/* Topic overview */

async function renderTopics(root, context) {
  const state = { sort: context.params.get("sort") ?? "size" };
  const [body, anomalies, monitor] = await Promise.all([
    get(`/api/apps/${context.app}/topics`, { samples: SAMPLES_PER_TOPIC }),
    // Spikes are optional: the page still works when detection is unavailable.
    get(`/api/apps/${context.app}/anomalies`, { samples: 2 }).catch(() => null),
    get(`/api/apps/${context.app}/anomalies/monitor`, { samples: 2 }).catch(() => null),
  ]);
  if (!context.isCurrent()) return;

  root.innerHTML = `
    <section class="card">
      <div class="card-head">
        <div><h2>Discovery</h2><p>${runSummary(body.run)}</p></div>
      </div>
      ${runForm(body.run)}
    </section>
    ${anomalies?.run ? spikeSection(anomalies) : ""}
    ${monitor?.run ? monitorSection(monitor) : ""}
    ${
      body.run
        ? `<div class="controls">${segmented("sort", SORTS, state.sort)}<span class="muted small" data-sort-hint></span></div>
           <div class="topic-grid" data-topics></div>`
        : empty(
            "No topics yet",
            `Run discovery above, or <code>reviewradar topics ${escapeHtml(context.app)}</code> from the command line.`,
          )
    }`;
  wireRunForm(root, context, body.run);
  const alertSamples = [
    ...(anomalies?.spikes ?? []).flatMap((spike) => spike.samples),
    ...(monitor?.incidents ?? []).flatMap((incident) => incident.samples),
  ];
  if (alertSamples.length) {
    rememberTranslations(alertSamples);
    refreshTranslations(root);
  }
  if (!body.run) return;

  const momentum = risingFactors(body.topics);
  const container = root.querySelector("[data-topics]");
  const draw = () => {
    context.setParams({ sort: state.sort === "size" ? "" : state.sort });
    root.querySelector("[data-sort-hint]").textContent = SORT_HINTS[state.sort];
    const topics = sortTopics(body.topics, state.sort, momentum);
    container.innerHTML = topics.map((topic) => topicCard(topic, body.run, momentum.get(topic.id))).join("");
    rememberTranslations(topics.flatMap((topic) => topic.samples));
    refreshTranslations(container);
  };
  onSegmentChange(root, "sort", (value) => {
    state.sort = value;
    draw();
  });
  draw();
}

const SORT_HINTS = {
  size: "Topics with the most reviews first.",
  negative: "Topics with the largest share of 1–2 star reviews first.",
  rising: `Topics whose share of reviews grew most in the last ${RECENT_DAYS} days.`,
};

function runSummary(run) {
  if (!run) return "Topics have not been discovered for this app yet.";
  const scope = run.parameters.max_rating ? `rated ★ ≤ ${run.parameters.max_rating}` : "of every rating";
  return `${formatNumber(run.topic_count)} topics from ${formatNumber(run.review_count)} reviews ${scope} with ${run.parameters.min_words}+ words ·
    ${escapeHtml(run.algorithm)} on ${escapeHtml(run.embedding_model.split("/").pop())} embeddings · ${formatDateTime(run.created_at)}`;
}

function runForm(run) {
  const scope = run && run.parameters.max_rating === null ? "all" : "critical";
  const count = run?.parameters.n_clusters ?? 20;
  return `<form class="controls" data-run-form>
    ${segmented("scope", SCOPES, scope)}
    <label class="inline-field">Topics
      <select name="topic_count">${TOPIC_COUNTS.map(
        (value) => `<option value="${value}" ${value === count ? "selected" : ""}>${value}</option>`,
      ).join("")}</select>
    </label>
    <button type="submit" class="button primary">${run ? "Run again" : "Discover topics"}</button>
    <span class="muted small" data-run-status>Clusters stored embeddings with k-means; labels come from English translations.</span>
  </form>`;
}

function wireRunForm(root, context, run) {
  const form = root.querySelector("[data-run-form]");
  let scope = run && run.parameters.max_rating === null ? "all" : "critical";
  onSegmentChange(form, "scope", (value) => {
    scope = value;
  });
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const button = form.querySelector("button[type=submit]");
    const status = form.querySelector("[data-run-status]");
    button.disabled = true;
    status.className = "muted small";
    status.innerHTML = `<span class="spinner small" aria-hidden="true"></span> Clustering review embeddings…`;
    try {
      const result = await post(`/api/apps/${context.app}/topics/runs`, {
        topic_count: Number(form.elements.topic_count.value),
        ...SCOPES.find((option) => option.id === scope).request,
      });
      if (!context.isCurrent()) return;
      toast(
        `Found ${result.topic_count} topics in ${formatNumber(result.review_count)} reviews (${(result.took_ms / 1000).toFixed(1)} s)`,
      );
      await renderTopics(root, context);
    } catch (error) {
      button.disabled = false;
      status.className = "status-error small";
      status.textContent = error.message;
    }
  });
}

/** How much more of each topic's volume falls in the recent window than across all topics. */
function risingFactors(topics) {
  const recent = (daily) => daily.slice(-RECENT_DAYS).reduce((sum, day) => sum + day.count, 0);
  const total = topics.reduce((sum, topic) => sum + topic.size, 0);
  const recentTotal = topics.reduce((sum, topic) => sum + recent(topic.daily), 0);
  const baseline = total ? recentTotal / total : 0;
  return new Map(
    topics.map((topic) => [topic.id, baseline && topic.size ? recent(topic.daily) / topic.size / baseline : 0]),
  );
}

function sortTopics(topics, sort, momentum) {
  const ordered = [...topics];
  if (sort === "negative") {
    ordered.sort((a, b) => b.negative_share - a.negative_share || a.average_rating - b.average_rating);
  } else if (sort === "rising") {
    ordered.sort((a, b) => momentum.get(b.id) - momentum.get(a.id));
  } else {
    ordered.sort((a, b) => b.size - a.size);
  }
  return ordered;
}

function topicCard(topic, run, rising) {
  const risingBadge =
    rising >= RISING_THRESHOLD && topic.size >= MIN_RISING_SIZE
      ? `<span class="badge" title="Share of this topic's reviews from the last ${RECENT_DAYS} days, relative to all topics">Rising ${rising.toFixed(1)}×</span>`
      : "";
  return `<article class="card topic">
    <header class="topic-head">
      <h3><a href="#/topics?topic=${topic.id}">${escapeHtml(topic.label)}</a></h3>
      ${risingBadge}
      <span class="topic-size" title="Reviews in this topic">${formatNumber(topic.size)}</span>
    </header>
    <p class="topic-keywords">${escapeHtml(topic.keywords.join(", "))}</p>
    ${topicStats(topic)}
  </article>`;
}

function keywordChips(keywords) {
  return `<div class="keywords">${keywords
    .map(
      (keyword) =>
        `<a class="chip small" href="#/search?q=${encodeURIComponent(keyword)}" title="Search reviews for ${escapeHtml(keyword)}">${escapeHtml(keyword)}</a>`,
    )
    .join("")}</div>`;
}

function topicStats(topic) {
  const negative = Math.round(topic.negative_share * 100);
  return `<div class="topic-stats">
    <div><span class="muted small">Average</span><strong>${topic.average_rating.toFixed(2)} ★</strong></div>
    <div>
      <span class="muted small">1–2 ★</span><strong>${negative}%</strong>
      <span class="meter" aria-hidden="true"><span style="width:${negative}%"></span></span>
    </div>
    ${topic.daily.length ? `<div><span class="muted small">Reviews per day</span>${sparkline(topic.daily)}</div>` : ""}
  </div>`;
}

function sparkline(daily, { width = 200, height = 34 } = {}) {
  if (daily.length < 2) return `<span class="muted small">–</span>`;
  const peak = Math.max(1, ...daily.map((day) => day.count));
  const step = width / (daily.length - 1);
  const y = (count) => (height - 2 - (count / peak) * (height - 4)).toFixed(1);
  const points = daily.map((day, index) => `${(index * step).toFixed(1)},${y(day.count)}`).join(" ");
  const label = `${daily[0].day} to ${daily.at(-1).day}, peak ${peak} per day`;
  return `<svg class="sparkline" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none" role="img" aria-label="${label}">
    <title>${label}</title>
    <polygon points="0,${height} ${points} ${width},${height}" />
    <polyline points="${points}" />
  </svg>`;
}

/* One topic */

async function renderTopic(root, context, topicId) {
  let page = Math.max(1, Number(context.params.get("page")) || 1);
  root.innerHTML = `
    <a class="text-link back-link" href="#/topics">← All topics</a>
    <section class="card" data-topic-head>${loading("Loading topic…")}</section>
    <div class="pagination" data-pagination-top></div>
    <div class="review-list" data-results></div>
    <div class="pagination" data-pagination-bottom></div>`;
  const head = root.querySelector("[data-topic-head]");
  const results = root.querySelector("[data-results]");

  async function load() {
    context.setParams({ topic: topicId, page: page > 1 ? page : "" });
    try {
      const body = await get(`/api/apps/${context.app}/topics/${topicId}/reviews`, {
        limit: PAGE_SIZE,
        offset: (page - 1) * PAGE_SIZE,
      });
      if (!context.isCurrent()) return;
      head.innerHTML = topicHeader(body.topic);
      results.innerHTML = body.items.length
        ? body.items.map((review) => reviewCard(review)).join("")
        : empty("No reviews on this page");
      for (const container of root.querySelectorAll("[data-pagination-top], [data-pagination-bottom]")) {
        container.innerHTML = pagination(body, PAGE_SIZE);
      }
      rememberTranslations(body.items);
      refreshTranslations(results);
    } catch (error) {
      if (!context.isCurrent()) return;
      head.innerHTML = failure(error);
      results.innerHTML = "";
    }
  }

  root.addEventListener("click", (event) => {
    const button = event.target.closest("[data-page]");
    if (!button || button.disabled) return;
    page += button.dataset.page === "next" ? 1 : -1;
    load();
    root.scrollIntoView({ block: "start" });
  });
  await load();
}

function topicHeader(topic) {
  const languages = Object.entries(topic.languages)
    .map(([code, count]) => `${escapeHtml(languageName(code))} ${formatNumber(count)}`)
    .join(" · ");
  return `<div class="card-head">
      <div>
        <h2>${escapeHtml(topic.label)}</h2>
        <p>${formatNumber(topic.size)} reviews, most typical first: ordered by closeness to the topic's centre.</p>
      </div>
    </div>
    <div class="topic">
      ${keywordChips(topic.keywords)}
      ${topicStats(topic)}
      <div class="muted small">${languages}</div>
    </div>`;
}
