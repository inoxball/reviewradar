import { get } from "../api.js";
import { MODES, escapeHtml, formatNumber, loading } from "../ui.js";

export async function render(root, context) {
  root.innerHTML = loading("Scoring every search mode against the graded judgments. This takes a few seconds the first time…");
  const evaluation = await get("/api/evaluation");
  if (!context.isCurrent()) return;

  const k = evaluation.k;
  const best = Math.max(...evaluation.modes.map((mode) => mode.ndcg));

  root.innerHTML = `
    <section class="metric-grid">${evaluation.modes
      .map(
        (mode) => `<div class="metric${mode.ndcg === best ? " best" : ""}">
          <div class="step-name"><span>${MODES[mode.mode].label}</span>${mode.ndcg === best ? `<span class="badge accent">best</span>` : ""}</div>
          <div class="metric-value">${mode.ndcg.toFixed(3)}</div>
          <div class="muted small">nDCG@${k}</div>
          <div class="progress"><span style="width:${(mode.ndcg * 100).toFixed(1)}%"></span></div>
          <dl>
            <div><dt>P@${k}</dt><dd>${mode.precision.toFixed(2)}</dd></div>
            <div><dt>MRR</dt><dd>${mode.mrr.toFixed(2)}</dd></div>
            <div><dt>judged@${k}</dt><dd>${mode.judged_fraction.toFixed(2)}</dd></div>
          </dl>
        </div>`,
      )
      .join("")}</section>


    <section class="card">
      <div class="card-head"><div><h2>Metrics</h2></div></div>
      <dl class="definitions">
        <div><dt>nDCG@${k}</dt><dd>Ranking quality with graded relevance; 1.0 means the best reviews come first.</dd></div>
        <div><dt>P@${k}</dt><dd>Share of the top ${k} results that are at least partly relevant.</dd></div>
        <div><dt>MRR</dt><dd>How early the first relevant result appears, averaged over queries.</dd></div>
        <div><dt>judged@${k}</dt><dd>Share of the top ${k} with a relevance grade. Low values mean unjudged results count as misses.</dd></div>
      </dl>
    </section>

    <section class="card">
      <div class="card-head"><div><h2>nDCG@${k} per query</h2></div></div>
      <div class="table-wrap"><table>
        <thead><tr><th>Query</th>${evaluation.modes.map((mode) => `<th class="num">${MODES[mode.mode].label}</th>`).join("")}</tr></thead>
        <tbody>${evaluation.queries
          .map(
            (query) => `<tr>
              <td class="wrap"><a class="text-link" href="#/compare?q=${encodeURIComponent(query.text)}">${escapeHtml(query.text)}</a></td>
              ${evaluation.modes.map((mode) => heatCell(query.ndcg[mode.mode])).join("")}
            </tr>`,
          )
          .join("")}</tbody>
      </table></div>
    </section>`;
}

function heatCell(value) {
  const score = value ?? 0;
  return `<td class="num heat" style="--v:${score.toFixed(3)}">${score.toFixed(2)}</td>`;
}
