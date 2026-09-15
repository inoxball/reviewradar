// App shell: routing, app selection, API inspector, translation switch and global events.

import { clearRecordedCalls, curlFor, get, onCallsChanged, recordedCalls } from "./api.js";
import {
  $,
  codeBlock,
  escapeHtml,
  failure,
  highlightJson,
  loading,
  toast,
} from "./ui.js";
import {
  handleTranslationAction,
} from "./translations.js";
import * as apiPlayground from "./views/api.js";
import * as compare from "./views/compare.js";
import * as evaluation from "./views/evaluation.js";
import * as overview from "./views/overview.js";
import * as pipeline from "./views/pipeline.js";
import * as replies from "./views/replies.js";
import * as reviews from "./views/reviews.js";
import * as search from "./views/search.js";
import * as topics from "./views/topics.js";

const ROUTES = {
  overview: {
    view: overview,
    title: "Overview",
  },
  search: {
    view: search,
    title: "Search",
  },
  compare: {
    view: compare,
    title: "Search modes",
  },
  topics: {
    view: topics,
    title: "Topics",
  },
  replies: {
    view: replies,
    title: "Reply drafting",
  },
  reviews: {
    view: reviews,
    title: "Reviews",
  },
  pipeline: {
    view: pipeline,
    title: "Pipeline",
  },
  evaluation: {
    view: evaluation,
    title: "Search quality",
  },
  api: {
    view: apiPlayground,
    title: "API playground",
  },
};

const APP_KEY = "reviewradar.app";
const state = { app: null, renderToken: 0 };

function currentRoute() {
  const [name, query = ""] = window.location.hash.replace(/^#\/?/, "").split("?");
  return { name: ROUTES[name] ? name : "overview", params: new URLSearchParams(query) };
}

async function renderRoute() {
  const { name, params } = currentRoute();
  const route = ROUTES[name];
  const token = ++state.renderToken;

  document.querySelectorAll("[data-route]").forEach((link) => {
    if (link.dataset.route === name) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  });
  $("page-title").textContent = route.title;
  document.title = `${route.title} · ReviewRadar`;
  clearRecordedCalls();

  const root = $("view");
  root.innerHTML = loading("Loading…");
  let overviewRequest = null;
  const context = {
    app: state.app,
    params,
    isCurrent: () => token === state.renderToken,
    overview: () => {
      overviewRequest ??= get(`/api/apps/${state.app}/overview`);
      return overviewRequest;
    },
    setParams: (values) => {
      const query = new URLSearchParams(
        Object.entries(values).filter(([, value]) => value !== "" && value !== null && value !== undefined),
      ).toString();
      history.replaceState(null, "", `#/${name}${query ? `?${query}` : ""}`);
    },
  };

  try {
    await route.view.render(root, context);
  } catch (error) {
    if (context.isCurrent()) root.innerHTML = failure(error);
  }
}

/* API inspector */

function renderCalls(calls) {
  $("api-count").textContent = String(calls.length);
  const container = $("api-calls");
  const open = new Set(
    [...container.querySelectorAll("details[open]")].map((element) => element.dataset.callId),
  );
  if (!calls.length) {
    container.innerHTML = `<p class="muted small">No requests yet. Interact with the page, or try the API playground.</p>`;
    return;
  }
  container.innerHTML = calls
    .map((call) => {
      const status = call.pending
        ? `<span class="spinner small" aria-hidden="true"></span>`
        : `<span class="${call.error ? "status-error" : "status-ok"}">${call.status ?? "ERR"}</span> · ${call.ms} ms`;
      const isOpen = open.has(String(call.id));
      return `<details class="call" data-call-id="${call.id}" ${isOpen ? "open" : ""}>
        <summary>
          <span class="method ${call.method.toLowerCase()}">${call.method}</span>
          <span class="call-url" title="${escapeHtml(call.url)}">${escapeHtml(call.url)}</span>
          <span class="call-meta">${status}</span>
        </summary>
        <div class="call-body">${isOpen ? callDetails(call) : ""}</div>
      </details>`;
    })
    .join("");
}

function callDetails(call) {
  const parts = [codeBlock(curlFor(call))];
  if (call.error) parts.push(`<p class="status-error small">${escapeHtml(call.error)}</p>`);
  if (call.response !== null) {
    parts.push(codeBlock(highlightJson(call.response), { html: true, copyText: JSON.stringify(call.response, null, 2) }));
  }
  return parts.join("");
}

function initInspector() {
  const drawer = $("api-drawer");
  const toggle = $("api-toggle");
  const setOpen = (open) => {
    drawer.hidden = !open;
    toggle.setAttribute("aria-expanded", String(open));
  };
  toggle.addEventListener("click", () => setOpen(drawer.hidden));
  $("api-close").addEventListener("click", () => setOpen(false));
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !drawer.hidden) setOpen(false);
  });
  $("api-calls").addEventListener(
    "toggle",
    (event) => {
      const details = event.target;
      if (!(details instanceof HTMLDetailsElement) || !details.open) return;
      const call = recordedCalls().find((item) => String(item.id) === details.dataset.callId);
      if (call) details.querySelector(".call-body").innerHTML = callDetails(call);
    },
    true,
  );
  onCallsChanged(renderCalls);
  renderCalls(recordedCalls());
}

/* Sidebar */

async function initApps() {
  const apps = await get("/api/apps", {}, { record: false });
  const select = $("app-select");
  select.innerHTML = apps
    .map((app) => `<option value="${escapeHtml(app.slug)}">${escapeHtml(app.name)}</option>`)
    .join("");
  const saved = safeStorage(() => localStorage.getItem(APP_KEY));
  state.app = apps.some((app) => app.slug === saved) ? saved : apps[0]?.slug ?? null;
  select.value = state.app ?? "";
  select.addEventListener("change", () => {
    state.app = select.value;
    safeStorage(() => localStorage.setItem(APP_KEY, state.app));
    renderRoute();
  });
}

/* Global interactions */

function initGlobalEvents() {
  document.addEventListener("click", async (event) => {
    const action = event.target.closest("[data-action]");
    if (action?.closest(".review")) {
      handleTranslationAction(action);
      return;
    }
    const copy = event.target.closest("[data-copy]");
    if (copy) {
      try {
        await navigator.clipboard.writeText(copy.dataset.copy);
        toast("Copied to the clipboard");
      } catch {
        toast("Copying is not available in this browser");
      }
      return;
    }
    const text = event.target.closest(".review-text");
    if (text && !window.getSelection()?.toString()) text.classList.toggle("expanded");
  });

  document.addEventListener("keydown", (event) => {
    const typing = ["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement?.tagName);
    if (event.key === "/" && !typing) {
      const input = document.querySelector("[data-primary-input]");
      if (input) {
        event.preventDefault();
        input.focus();
        input.select();
      }
    }
  });
}

function safeStorage(operation) {
  try {
    return operation();
  } catch {
    return null;
  }
}

async function start() {
  initInspector();
  initGlobalEvents();
  try {
    await initApps();
  } catch (error) {
    $("view").innerHTML = failure(error);
    return;
  }
  window.addEventListener("hashchange", renderRoute);
  await renderRoute();
}

start();
