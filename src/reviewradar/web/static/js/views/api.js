// API playground generated from the OpenAPI specification that FastAPI publishes.

import { get, send } from "../api.js";
import { codeBlock, empty, escapeHtml, highlightJson } from "../ui.js";
import { curlFor } from "../api.js";

const DEFAULT_QUERY = "too many ads";

export async function render(root, context) {
  const spec = await get("/openapi.json", {}, { record: false });
  if (!context.isCurrent()) return;

  const operations = Object.entries(spec.paths).flatMap(([path, methods]) =>
    Object.entries(methods).map(([method, operation]) => ({
      key: `${method.toUpperCase()} ${path}`,
      method: method.toUpperCase(),
      path,
      operation,
    })),
  );
  const groups = new Map();
  for (const item of operations) {
    const tag = item.operation.tags?.[0] ?? "other";
    if (!groups.has(tag)) groups.set(tag, []);
    groups.get(tag).push(item);
  }

  const requested = context.params.get("endpoint");
  let selected = operations.find((item) => item.key === requested) ?? operations.find((item) => item.path.endsWith("/search")) ?? operations[0];

  root.innerHTML = `
    <div class="playground">
      <nav class="endpoint-list card" aria-label="Endpoints">${[...groups.entries()]
        .map(
          ([tag, items]) => `<div class="endpoint-group"><h3>${escapeHtml(tag)}</h3>${items
            .map(
              (item) => `<button type="button" class="endpoint" data-endpoint="${escapeHtml(item.key)}">
                <span class="method ${item.method.toLowerCase()}">${item.method}</span>
                <span class="endpoint-path">${escapeHtml(item.path)}</span>
                <span class="endpoint-summary">${escapeHtml(item.operation.summary ?? "")}</span>
              </button>`,
            )
            .join("")}</div>`,
        )
        .join("")}
        <p class="muted small" style="margin:0">Also available as <a href="/docs" target="_blank" rel="noopener">Swagger UI</a> and <a href="/openapi.json" target="_blank" rel="noopener">openapi.json</a>.</p>
      </nav>
      <div data-detail></div>
    </div>`;

  const detail = root.querySelector("[data-detail]");

  function select(item) {
    selected = item;
    context.setParams({ endpoint: item.key });
    root.querySelectorAll("[data-endpoint]").forEach((button) => {
      button.setAttribute("aria-current", String(button.dataset.endpoint === item.key));
    });
    renderDetail(detail, spec, item, context);
  }

  root.querySelectorAll("[data-endpoint]").forEach((button) => {
    button.addEventListener("click", () => select(operations.find((item) => item.key === button.dataset.endpoint)));
  });
  select(selected);
}

async function renderDetail(container, spec, item, context) {
  const parameters = item.operation.parameters ?? [];
  const bodySchema = item.operation.requestBody?.content?.["application/json"]?.schema;

  container.innerHTML = `
    <section class="card">
      <div class="card-head">
        <div>
          <h2><span class="method ${item.method.toLowerCase()}">${item.method}</span> <span class="mono">${escapeHtml(item.path)}</span></h2>
          <p>${escapeHtml(item.operation.summary ?? "")}</p>
        </div>
      </div>
      ${item.operation.description ? `<p class="muted" style="margin-top:0;white-space:pre-line">${escapeHtml(item.operation.description)}</p>` : ""}
      <form data-request>
        ${parameters.length ? `<div class="form-grid">${parameters.map((parameter) => field(spec, parameter, context)).join("")}</div>` : `<p class="muted small">No parameters.</p>`}
        ${bodySchema ? `<label style="display:grid;gap:4px;margin-top:12px"><span class="param-name mono">JSON body</span><textarea data-body spellcheck="false">${escapeHtml(JSON.stringify(await exampleBody(spec, bodySchema, context), null, 2))}</textarea></label>` : ""}
        <div class="controls" style="margin-top:14px">
          <button type="submit" class="button primary">Send request</button>
          <span class="muted small">The call also appears in the API inspector.</span>
        </div>
      </form>
    </section>
    <section class="card" data-response>${empty("No response yet", "Fill in the parameters and send the request.")}</section>`;

  container.querySelector("[data-request]").addEventListener("submit", async (event) => {
    event.preventDefault();
    const output = container.querySelector("[data-response]");
    let path = item.path;
    const params = {};
    for (const parameter of parameters) {
      const input = container.querySelector(`[data-param="${CSS.escape(parameter.name)}"]`);
      const value = readValue(spec, parameter, input);
      if (parameter.in === "path") path = path.replace(`{${parameter.name}}`, encodeURIComponent(value ?? ""));
      else if (value !== null) params[parameter.name] = value;
    }
    let body;
    const bodyInput = container.querySelector("[data-body]");
    if (bodyInput) {
      try {
        body = JSON.parse(bodyInput.value);
      } catch (error) {
        output.innerHTML = `<div class="state error"><div><strong>Invalid JSON body</strong><p>${escapeHtml(error.message)}</p></div></div>`;
        return;
      }
    }
    output.innerHTML = `<div class="state"><span class="spinner" aria-hidden="true"></span>Sending ${item.method} ${escapeHtml(path)}…</div>`;
    const call = await send(item.method, path, { params, body });
    output.innerHTML = `
      <div class="card-head"><div><h2>Response</h2><p><span class="${call.error ? "status-error" : "status-ok"}">${call.status ?? "network error"}</span> · ${call.ms} ms</p></div></div>
      <div class="request-line"><span class="method ${call.method.toLowerCase()}">${call.method}</span>${escapeHtml(call.url)}</div>
      <div class="review-list" style="margin-top:10px">
        ${codeBlock(curlFor(call))}
        ${call.error ? `<p class="status-error">${escapeHtml(call.error)}</p>` : ""}
        ${call.response === null ? "" : codeBlock(highlightJson(call.response), { html: true, copyText: JSON.stringify(call.response, null, 2) })}
      </div>`;
  });
}

function resolve(spec, schema) {
  let current = schema ?? {};
  for (let depth = 0; depth < 5; depth += 1) {
    if (current.$ref) {
      current = current.$ref.split("/").slice(1).reduce((node, key) => node?.[key], spec) ?? {};
    } else if (current.anyOf) {
      current = current.anyOf.find((option) => option.type !== "null") ?? {};
    } else {
      break;
    }
  }
  return current;
}

function field(spec, parameter, context) {
  const schema = resolve(spec, parameter.schema);
  const isArray = schema.type === "array";
  const itemSchema = isArray ? resolve(spec, schema.items) : schema;
  const name = escapeHtml(parameter.name);
  const hint = [
    parameter.in === "path" ? "path" : "query",
    parameter.required ? "required" : "optional",
    isArray ? "comma-separated" : itemSchema.type,
  ]
    .filter(Boolean)
    .join(" · ");
  const defaultValue = defaultFor(parameter, schema, context);

  let control;
  if (itemSchema.enum) {
    const options = itemSchema.enum
      .map((value) => `<option value="${escapeHtml(value)}" ${value === defaultValue ? "selected" : ""}>${escapeHtml(value)}</option>`)
      .join("");
    control = `<select data-param="${name}">${parameter.required ? "" : `<option value="">(any)</option>`}${options}</select>`;
  } else {
    const type = !isArray && itemSchema.type === "integer" ? "number" : "text";
    const bounds = [
      itemSchema.minimum !== undefined ? `min="${itemSchema.minimum}"` : "",
      itemSchema.maximum !== undefined ? `max="${itemSchema.maximum}"` : "",
    ].join(" ");
    control = `<input type="${type}" data-param="${name}" value="${escapeHtml(defaultValue ?? "")}" ${bounds} ${parameter.required ? "required" : ""} />`;
  }
  return `<label><span class="param-name">${name}</span>${control}<span class="param-hint">${escapeHtml(hint)}${parameter.description ? ` · ${escapeHtml(parameter.description)}` : ""}</span></label>`;
}

function defaultFor(parameter, schema, context) {
  if (parameter.name === "slug") return context.app;
  if (parameter.name === "q") return DEFAULT_QUERY;
  // Defaults live next to a $ref on the parameter schema, not inside the referenced enum.
  const value = parameter.schema?.default ?? schema.default;
  return value === undefined || value === null ? "" : String(value);
}

function readValue(spec, parameter, input) {
  const raw = input?.value?.trim() ?? "";
  if (!raw) return null;
  const schema = resolve(spec, parameter.schema);
  if (schema.type === "array") return raw.split(",").map((value) => value.trim()).filter(Boolean);
  return raw;
}

async function exampleBody(spec, schema, context) {
  const resolved = resolve(spec, schema);
  if (resolved.properties?.review_ids) {
    try {
      const page = await get(`/api/apps/${context.app}/reviews`, { language: "pt", limit: 3 }, { record: false });
      return { review_ids: page.items.map((review) => review.review_id) };
    } catch {
      return { review_ids: [1] };
    }
  }
  return {};
}
