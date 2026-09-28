import { randomBytes, randomUUID } from "node:crypto";
import { homedir } from "node:os";
import { join } from "node:path";

// ── Local type aliases ──────────────────────────────────────────────────────

export type Json = Record<string, any>;
export type StreamMode = "off" | "auto" | "force";
export type FetchInit = Parameters<typeof fetch>[1];

export type ProviderModelConfig = {
  id: string;
  name?: string;
  api?: string;
  reasoning?: boolean;
  input?: ("text" | "image")[];
  cost?: { input?: number; output?: number; cacheRead?: number; cacheWrite?: number };
  contextWindow?: number;
  maxTokens?: number;
};

export type ProviderConfigFile = {
  baseUrl?: string;
  apiKey?: string;
  models?: ProviderModelConfig[];
};

// ── Constants ───────────────────────────────────────────────────────────────

export const DEFAULT_CONFIG_PATH = join(homedir(), ".pi", "agent", "anyrouter.json");
export const CONFIG_PATH = process.env.PI_ANYROUTER_CC_CONFIG || DEFAULT_CONFIG_PATH;
export const PROVIDER_NAME = "anyrouter";
// Keep this API id unique so pi uses this extension's streamSimple handler
// without touching the built-in anthropic-messages implementation.
export const API_ID = "anyrouter-messages" as import("@earendil-works/pi-ai").Api;
export const DEBUG_ENABLED = process.env.PI_ANYROUTER_CC_DEBUG === "1";
export const DEBUG_DIR = process.env.PI_ANYROUTER_CC_DEBUG_DIR || join(process.cwd(), ".pi", "anyrouter-cc-debug");
// Captured from the locally installed Claude Code on 2026-09-28.
export const CLAUDE_CODE_VERSION = "2.1.281";
export const CLAUDE_CODE_VERSION_BUILD = "2.1.281";
export const STAINLESS_PACKAGE_VERSION = "0.112.1";
export const STAINLESS_OS = "Windows";
export const STAINLESS_ARCH = "x64";
export const STAINLESS_RUNTIME = "node";
export const STAINLESS_RUNTIME_VERSION = "v26.3.0";
export const ANTHROPIC_BETA =
  "claude-code-20250219,context-1m-2025-08-07,interleaved-thinking-2025-05-14,thinking-token-count-2026-05-13,context-management-2025-06-27,prompt-caching-scope-2026-01-05,mid-conversation-system-2026-04-07,per-turn-control-2026-07-01,mid-conversation-tool-changes-2026-07-01,advisor-tool-2026-03-01,advanced-tool-use-2025-11-20,effort-2025-11-24,fallback-credit-2026-06-01";
export const CLAUDE_DEVICE_ID = randomBytes(32).toString("hex");
export const CODEX_VERSION = "0.153.4";
export const CODEX_INSTALLATION_ID = randomUUID();

export const NAME_MAP: Record<string, string> = {
  read: "Read",
  write: "Write",
  edit: "Edit",
  bash: "Bash",
  powershell: "PowerShell",
  grep: "Grep",
  find: "Glob",
  glob: "Glob",
  ls: "LS",
  todowrite: "TodoWrite",
  webfetch: "WebFetch",
  websearch: "WebSearch",
  google_search: "Google_Search",
};
