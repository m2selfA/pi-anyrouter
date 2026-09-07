import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { ProxyAgent, fetch as undiciFetch } from "undici";
import { DEBUG_DIR, DEBUG_ENABLED, type FetchInit, type Json } from "./types.js";

// ── Headers ─────────────────────────────────────────────────────────────────

export function redactHeaders(headers: Record<string, string>) {
  const redacted = { ...headers };
  if (redacted.authorization) redacted.authorization = "Bearer ***";
  if (redacted["x-api-key"]) redacted["x-api-key"] = "***";
  return redacted;
}

// ── Proxy ───────────────────────────────────────────────────────────────────

const PROXY_AGENTS = new Map<string, ProxyAgent>();

function hostMatchesNoProxy(hostname: string, pattern: string) {
  const item = pattern.trim().toLowerCase();
  if (!item) return false;
  if (item === "*") return true;
  const host = hostname.toLowerCase();
  if (item.startsWith(".")) return host === item.slice(1) || host.endsWith(item);
  return host === item || host.endsWith(`.${item}`);
}

function getProxyUrl(url: string) {
  const parsed = new URL(url);
  const noProxy = process.env.NO_PROXY || process.env.no_proxy || "";
  if (noProxy.split(",").some((item) => hostMatchesNoProxy(parsed.hostname, item))) return undefined;
  if (parsed.protocol === "https:") return process.env.HTTPS_PROXY || process.env.https_proxy || process.env.HTTP_PROXY || process.env.http_proxy;
  return process.env.HTTP_PROXY || process.env.http_proxy;
}

function getProxyAgent(proxyUrl: string) {
  let agent = PROXY_AGENTS.get(proxyUrl);
  if (!agent) {
    agent = new ProxyAgent(proxyUrl);
    PROXY_AGENTS.set(proxyUrl, agent);
  }
  return agent;
}

export function fetchWithProxy(url: string, init: FetchInit) {
  const proxyUrl = getProxyUrl(url);
  if (!proxyUrl) return fetch(url, init);
  return undiciFetch(url, { ...init, dispatcher: getProxyAgent(proxyUrl) } as any) as unknown as Promise<Response>;
}

// ── Debug ───────────────────────────────────────────────────────────────────

export function writeDebugFile(kind: "request" | "response" | "error", modelId: string, requestId: string | undefined, payload: Json) {
  if (!DEBUG_ENABLED) return;
  mkdirSync(DEBUG_DIR, { recursive: true });
  const safeModel = modelId.replace(/[^a-zA-Z0-9._-]+/g, "_");
  const safeRequestId = (requestId || "no-request-id").replace(/[^a-zA-Z0-9._-]+/g, "_");
  const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
  const path = join(DEBUG_DIR, `${timestamp}-${safeModel}-${safeRequestId}-${kind}.json`);
  writeFileSync(path, JSON.stringify(payload, null, 2), "utf8");
}

// ── Retry ───────────────────────────────────────────────────────────────────

export function delay(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export function isRetryableStatus(status: number) {
  return [408, 409, 429, 500, 502, 503, 504, 520, 522, 524].includes(status);
}

export function parseRetryAfterMs(value: string | null) {
  if (!value) return undefined;
  const seconds = Number(value);
  if (Number.isFinite(seconds) && seconds >= 0) return seconds * 1000;
  const at = Date.parse(value);
  if (Number.isFinite(at)) {
    const delta = at - Date.now();
    return delta > 0 ? delta : 0;
  }
  return undefined;
}

export function getRetryDelayMs(attempt: number, retryAfterMs?: number) {
  if (typeof retryAfterMs === "number") return Math.max(0, Math.min(retryAfterMs, 30_000));
  const base = Math.min(1000 * 2 ** attempt, 15_000);
  const jitter = Math.floor(Math.random() * 250);
  return base + jitter;
}

// ── SSE parsing ─────────────────────────────────────────────────────────────

export function parseSseEvent(chunk: string) {
  let event = "message";
  const data: string[] = [];
  for (const line of chunk.split(/\r?\n/)) {
    if (!line || line.startsWith(":")) continue;
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) data.push(line.slice(5).trimStart());
  }
  return { event, data: data.join("\n") };
}

export function nextSseChunk(buffer: string) {
  const unix = buffer.indexOf("\n\n");
  const dos = buffer.indexOf("\r\n\r\n");
  if (unix === -1 && dos === -1) return undefined;
  if (dos !== -1 && (unix === -1 || dos < unix)) {
    return { chunk: buffer.slice(0, dos), rest: buffer.slice(dos + 4) };
  }
  return { chunk: buffer.slice(0, unix), rest: buffer.slice(unix + 2) };
}
