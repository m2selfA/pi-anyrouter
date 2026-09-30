import { type Api, type AssistantMessage, calculateCost, type Model, type SimpleStreamOptions, type StopReason } from "@earendil-works/pi-ai";
import { NAME_MAP, type StreamMode } from "./types.js";

export function toClaudeCodeName(name?: string | null) {
  if (!name || typeof name !== "string") return name;
  return NAME_MAP[name.toLowerCase()] ?? name.charAt(0).toUpperCase() + name.slice(1);
}

export function fromClaudeCodeName(name?: string | null): string {
  if (!name || typeof name !== "string") return name ?? "";
  const lower = name.toLowerCase();
  for (const [from, to] of Object.entries(NAME_MAP)) {
    if (to.toLowerCase() === lower) return from;
  }
  return name.charAt(0).toLowerCase() + name.slice(1);
}

export function sanitizeText(text: string) {
  return text.replace(/[\uD800-\uDFFF]/g, "\uFFFD");
}

export function mapReasoningEffort(level?: SimpleStreamOptions["reasoning"]) {
  switch (level) {
    case "minimal":
    case "low":
      return "low";
    case "medium":
      return "medium";
    case "high":
      return "high";
    case "xhigh":
      return "xhigh";
    case "max":
      return "max";
    default:
      return "medium";
  }
}

export function mapStopReason(reason: string): StopReason {
  switch (reason) {
    case "end_turn":
    case "pause_turn":
    case "stop_sequence":
      return "stop";
    case "max_tokens":
      return "length";
    case "tool_use":
      return "toolUse";
    default:
      return "error";
  }
}

export function createEmptyUsage() {
  return {
    input: 0,
    output: 0,
    cacheRead: 0,
    cacheWrite: 0,
    reasoning: undefined as number | undefined,
    totalTokens: 0,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
  };
}

export function tryParseJson(text: string) {
  try {
    return text ? JSON.parse(text) : {};
  } catch {
    return undefined;
  }
}

export function extractRequestId(parsed: any, headers: Headers) {
  return parsed?.error?.message?.match(/request id:\s*([^)]+)/i)?.[1] || headers.get("x-oneapi-request-id") || undefined;
}

export function updateUsageFromAnthropic(output: AssistantMessage, usage: any, model: Model<Api>) {
  if (usage?.input_tokens != null) output.usage.input = usage.input_tokens;
  if (usage?.output_tokens != null) output.usage.output = usage.output_tokens;
  if (usage?.cache_read_input_tokens != null) output.usage.cacheRead = usage.cache_read_input_tokens;
  if (usage?.cache_creation_input_tokens != null) output.usage.cacheWrite = usage.cache_creation_input_tokens;
  // Anthropic reports thinking tokens under cache_read_input_tokens in some responses,
  // but the explicit thinking_tokens field (when present) is the authoritative source.
  if (usage?.thinking_tokens != null) output.usage.reasoning = usage.thinking_tokens;
  output.usage.totalTokens = output.usage.input + output.usage.output + output.usage.cacheRead + output.usage.cacheWrite;
  calculateCost(model, output.usage);
}

export function resetOutputState(output: AssistantMessage) {
  output.content = [];
  output.usage = createEmptyUsage();
  output.stopReason = "pending";
  output.timestamp = Date.now();
  output.errorMessage = undefined;
  output.responseId = undefined;
}

// AnyRouter's `*-cc-format` routes speak the Anthropic/Claude Code
// messages protocol even when the model id starts with `gpt`.
const CLAUDE_CODE_FORMAT_MODEL_RE = /(?:^|[-_.])cc[-_.]format(?:[-_.]|$)/i;

export function isCodexModel(modelId: string, configuredApi?: string) {
  if (configuredApi) return configuredApi === "openai-codex-responses";
  if (CLAUDE_CODE_FORMAT_MODEL_RE.test(modelId)) return false;
  return /(?:^|[-_.])(gpt|codex)(?:[-_.]|$)/i.test(modelId) || /^o\d(?:[-_.]|$)/i.test(modelId);
}

export function getStreamMode(): StreamMode {
  // AnyRouter's Claude Code subscription route is SSE-first. Keep the exact
  // transport by default instead of falling back to a generic JSON request.
  const value = String(process.env.PI_ANYROUTER_CC_STREAM_MODE || "force")
    .trim()
    .toLowerCase();
  if (["1", "true", "on", "auto"].includes(value)) return "auto";
  if (["force", "only"].includes(value)) return "force";
  return "off";
}
