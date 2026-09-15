// HTTP client for the ReviewRadar API. Every call is recorded for the API inspector.

const MAX_RECORDED_CALLS = 40;
const calls = [];
const listeners = new Set();
let sequence = 0;

export class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.status = status;
  }
}

export function onCallsChanged(listener) {
  listeners.add(listener);
}

export function recordedCalls() {
  return calls;
}

export function clearRecordedCalls() {
  calls.length = 0;
  notify();
}

function notify() {
  for (const listener of listeners) listener(calls);
}

export function buildUrl(path, params = {}) {
  const url = new URL(path, window.location.origin);
  for (const [key, value] of Object.entries(params)) {
    if (value === undefined || value === null || value === "") continue;
    for (const item of Array.isArray(value) ? value : [value]) url.searchParams.append(key, item);
  }
  return url;
}

/** Perform a request and return the recorded call, whatever the outcome. */
export async function send(method, path, { params = {}, body, record = true } = {}) {
  const url = buildUrl(path, params);
  const call = {
    id: ++sequence,
    method,
    url: `${url.pathname}${url.search}`,
    body,
    status: null,
    ms: null,
    response: null,
    error: null,
    pending: true,
  };
  if (record) {
    calls.unshift(call);
    calls.length = Math.min(calls.length, MAX_RECORDED_CALLS);
    notify();
  }

  const started = performance.now();
  try {
    const response = await fetch(url, {
      method,
      headers: body === undefined ? undefined : { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    call.status = response.status;
    call.response = await response.json().catch(() => null);
    if (!response.ok) call.error = describeError(call.response, response.statusText);
  } catch (error) {
    call.error = error.message;
  } finally {
    call.ms = Math.round(performance.now() - started);
    call.pending = false;
    if (record) notify();
  }
  return call;
}

async function request(method, path, options) {
  const call = await send(method, path, options);
  if (call.error) throw new ApiError(call.error, call.status);
  return call.response;
}

export const get = (path, params = {}, options = {}) => request("GET", path, { ...options, params });
export const post = (path, body, options = {}) => request("POST", path, { ...options, body });

export function curlFor(call) {
  const target = `${window.location.origin}${call.url}`;
  if (call.method === "GET") return `curl -s "${target}"`;
  return [
    `curl -s -X ${call.method} "${target}"`,
    `  -H "Content-Type: application/json"`,
    `  -d '${JSON.stringify(call.body ?? {})}'`,
  ].join(" \\\n");
}

function describeError(payload, fallback) {
  const detail = payload?.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail.map((item) => `${(item.loc ?? []).slice(1).join(".")}: ${item.msg}`).join("; ");
  }
  return fallback || "Request failed";
}
