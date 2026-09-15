// Topic spikes: days when one topic suddenly dominated a source's critical reviews.

import { STORE_LABELS, empty, escapeHtml, formatDate, formatNumber, formatPercent, reviewCard } from "../ui.js";

export function spikeSection(report) {
  const tested = report.coverage.filter((entry) => entry.tested);
  const skipped = report.coverage.filter((entry) => !entry.tested);
  const parts = [
    `${formatNumber(report.tests)} topic-days tested across ${tested.length} sources`,
    `a spike needs p ≤ ${report.threshold ? report.threshold.toExponential(1) : "–"}, 5+ reviews and 3× the usual share`,
  ];
  if (skipped.length) parts.push(`${skipped.length} sources skipped for too little coverage`);
  const spikes = report.spikes.length
    ? report.spikes.map(spikeCard).join("")
    : empty("No spikes", "No topic suddenly dominated a source's critical reviews.");
  return `<section class="card">
    <div class="card-head">
      <div>
        <h2>Spikes</h2>
        <p>${escapeHtml(parts.join(" · "))}. Shares are compared per store and market, so changes in how many reviews a store exposes are not mistaken for incidents.</p>
      </div>
    </div>
    <div class="spike-list">${spikes}</div>
  </section>`;
}

function spikeCard(spike) {
  const [store, market] = spike.source.split(":");
  const place = `${STORE_LABELS[store] ?? store} ${market.toUpperCase()}`;
  const version = spike.top_version
    ? ` · ${formatPercent(spike.top_version_share, 1)} on version ${escapeHtml(spike.top_version)}`
    : "";
  return `<article class="spike">
    <header class="spike-head">
      <span class="badge danger">${escapeHtml(formatDate(spike.day))}</span>
      <span class="badge">${escapeHtml(place)}</span>
      <a class="spike-topic" href="#/topics?topic=${spike.topic.id}">${escapeHtml(spike.topic.label)}</a>
    </header>
    <p class="spike-stats">
      <strong>${formatNumber(spike.count)} of ${formatNumber(spike.total)}</strong> critical reviews
      (${formatPercent(spike.share, 1)}, usually ${formatPercent(spike.expected_share, 1)}) ·
      ${spike.lift.toFixed(1)}× · p = ${spike.p_value.toExponential(1)}${version}
    </p>
    <div class="review-list">${spike.samples.map((review) => reviewCard(review, { compact: true })).join("")}</div>
  </article>`;
}

/** The daily monitor replayed over the ingested history, and how it did on known spikes. */
export function monitorSection(report) {
  const incidents = report.incidents.length
    ? report.incidents.map(incidentCard).join("")
    : empty("No incidents", "No day stood out against the days before it.");
  return `<section class="card">
    <div class="card-head">
      <div>
        <h2>Daily monitor <span class="badge accent">backtest</span></h2>
        <p>An alert job replayed over the ingested history. Each day is judged only against the ${report.baseline_days} days before it, once a store and market has ${report.min_baseline_days} earlier days of coverage, with a false-alarm budget of ${report.daily_alpha} per day. ${formatNumber(report.judged_days)} days judged, ${formatNumber(report.alert_days)} with alerts.</p>
      </div>
    </div>
    ${monitorTimeline(report.days)}
    <div class="spike-list">${incidents}</div>
    ${report.spikes.length ? spikeOutcomes(report.spikes) : ""}
  </section>`;
}

function placeName(source) {
  const [store, market] = source.split(":");
  return `${STORE_LABELS[store] ?? store} ${market.toUpperCase()}`;
}

function monitorTimeline(days) {
  if (!days.length) return "";
  const cells = days
    .map((day) => {
      const state = day.alerts ? "alert" : day.sources ? "quiet" : "unjudged";
      const title = day.sources
        ? `${formatDate(day.day)}: ${day.sources} sources judged, ${day.tests} tests, ${day.alerts} alerts`
        : `${formatDate(day.day)}: too little earlier coverage to judge`;
      return `<span class="monitor-day ${state}" title="${escapeHtml(title)}"></span>`;
    })
    .join("");
  return `<div class="monitor-timeline" role="img" aria-label="Monitored days: red days raised alerts">${cells}</div>
    <div class="monitor-legend muted small">
      <span>${escapeHtml(formatDate(days[0].day))}</span>
      <span><i class="monitor-day alert"></i> alert <i class="monitor-day quiet"></i> quiet <i class="monitor-day unjudged"></i> too little history</span>
      <span>${escapeHtml(formatDate(days.at(-1).day))}</span>
    </div>`;
}

function incidentCard(incident) {
  const range =
    incident.first_day === incident.last_day
      ? formatDate(incident.first_day)
      : `${formatDate(incident.first_day)} – ${formatDate(incident.last_day)}`;
  const release = incident.release
    ? `<span class="badge warning" title="Most of the incident's reviews are on a version first seen shortly before it">release ${escapeHtml(incident.release)}</span>`
    : "";
  const span =
    incident.alert_days > 1
      ? ` · alerts on ${incident.alert_days} days, ${formatNumber(incident.review_count)} reviews`
      : "";
  const version = incident.top_version
    ? ` · ${formatPercent(incident.top_version_share, 1)} on version ${escapeHtml(incident.top_version)}`
    : "";
  return `<article class="spike">
    <header class="spike-head">
      <span class="badge danger">${escapeHtml(range)}</span>
      <span class="badge">${escapeHtml(placeName(incident.source))}</span>
      ${release}
      <a class="spike-topic" href="#/topics?topic=${incident.topic.id}">${escapeHtml(incident.topic.label)}</a>
    </header>
    <p class="spike-stats">
      Peak ${escapeHtml(formatDate(incident.peak_day))}: <strong>${formatNumber(incident.peak_count)} of ${formatNumber(incident.peak_total)}</strong> critical reviews
      (${formatPercent(incident.peak_share, 1)}, before ${formatPercent(incident.expected_share, 1)}) ·
      ${incident.lift.toFixed(1)}× · p = ${incident.p_value.toExponential(1)} · judged on ${incident.baseline_days} earlier days${version}${span}
    </p>
    <div class="review-list">${incident.samples.map((review) => reviewCard(review, { compact: true })).join("")}</div>
  </article>`;
}

function spikeOutcomes(spikes) {
  const rows = spikes
    .map((spike) => {
      let outcome;
      if (!spike.caught) {
        outcome = `<span class="status-error">Missed</span>: ${escapeHtml(spike.reason ?? "")}`;
      } else if (spike.delay_days === 0) {
        outcome = `<span class="status-ok">Alerted the same day</span>`;
      } else if (spike.delay_days > 0) {
        outcome = `<span class="status-ok">Alerted ${spike.delay_days} days later</span>`;
      } else {
        outcome = `<span class="status-ok">Alerted ${-spike.delay_days} days earlier</span>`;
      }
      return `<tr>
        <td>${escapeHtml(formatDate(spike.day))}</td>
        <td>${escapeHtml(placeName(spike.source))}</td>
        <td><a href="#/topics?topic=${spike.topic.id}">${escapeHtml(spike.topic.label)}</a></td>
        <td>${outcome}</td>
      </tr>`;
    })
    .join("");
  return `<h3 class="monitor-subhead">Would it have caught the spikes found in hindsight?</h3>
    <div class="monitor-table-wrap">
      <table class="monitor-table">
        <thead><tr><th>Spike</th><th>Store</th><th>Topic</th><th>Daily monitor</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>`;
}
