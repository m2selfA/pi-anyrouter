import {
  type Api,
  type AssistantMessage,
  type AssistantMessageEventStream,
  type Context,
  calculateCost,
  type ImageContent,
  type Model,
  type SimpleStreamOptions,
  type TextContent,
  type Tool,
  type ToolResultMessage,
} from "@earendil-works/pi-ai";
import {
  delay,
  fetchWithProxy,
  getRetryDelayMs,
  isRetryableErrorType,
  isRetryableStatus,
  nextSseChunk,
  parseRetryAfterMs,
  parseSseEvent,
  RetryableStreamError,
  redactHeaders,
  writeDebugFile,
} from "./http.js";
import { CODEX_INSTALLATION_ID, CODEX_VERSION, type Json } from "./types.js";
import { extractRequestId, mapReasoningEffort, sanitizeText, tryParseJson } from "./utils.js";

// ── URL ─────────────────────────────────────────────────────────────────────

export function getCodexResponsesUrl(baseUrl: string) {
  const normalized = baseUrl.replace(/\/+$/, "");
  if (normalized.endsWith("/responses")) return normalized;
  if (normalized.endsWith("/v1")) return `${normalized}/responses`;
  return `${normalized}/v1/responses`;
}

// ── Message / tool conversion ───────────────────────────────────────────────

export function convertCodexMessages(context: Context) {
  const input: any[] = [];
  if (context.systemPrompt) {
    input.push({
      type: "message",
      role: "developer",
      content: [{ type: "input_text", text: sanitizeText(context.systemPrompt) }],
    });
  }

  for (const msg of context.messages) {
    if (msg.role === "user") {
      if (typeof msg.content === "string") {
        if (msg.content.trim()) {
          input.push({ type: "message", role: "user", content: [{ type: "input_text", text: sanitizeText(msg.content) }] });
        }
      } else {
        const content = msg.content.map((item) =>
          item.type === "text"
            ? { type: "input_text", text: sanitizeText(item.text) }
            : { type: "input_image", detail: "auto", image_url: `data:${item.mimeType};base64,${item.data}` },
        );
        if (content.length) input.push({ type: "message", role: "user", content });
      }
      continue;
    }

    if (msg.role === "assistant") {
      for (const block of msg.content) {
        if (block.type === "thinking" && block.thinkingSignature) {
          const reasoning = tryParseJson(block.thinkingSignature);
          if (reasoning) input.push(reasoning);
        } else if (block.type === "text" && block.text.trim()) {
          input.push({
            type: "message",
            role: "assistant",
            status: "completed",
            content: [{ type: "output_text", text: sanitizeText(block.text), annotations: [] }],
          });
        } else if (block.type === "toolCall") {
          const [callId, itemId] = block.id.split("|");
          input.push({
            type: "function_call",
            ...(itemId ? { id: itemId } : {}),
            call_id: callId,
            name: block.name,
            arguments: JSON.stringify(block.arguments),
          });
        }
      }
      continue;
    }

    if (msg.role === "toolResult") {
      const toolMsg = msg as ToolResultMessage;
      const text = toolMsg.content
        .filter((item) => item.type === "text")
        .map((item) => (item as TextContent).text)
        .join("\n");
      const images = toolMsg.content.filter((item) => item.type === "image") as ImageContent[];
      const output = images.length
        ? [
            ...(text ? [{ type: "input_text", text: sanitizeText(text) }] : []),
            ...images.map((image) => ({ type: "input_image", detail: "auto", image_url: `data:${image.mimeType};base64,${image.data}` })),
          ]
        : sanitizeText(text || (images.length ? "(see attached image)" : "(no tool output)"));
      input.push({ type: "function_call_output", call_id: toolMsg.toolCallId.split("|")[0], output });
    }
  }
  return input;
}

function convertCodexTools(tools: Tool[]) {
  return tools.map((tool) => ({
    type: "function",
    name: tool.name,
    description: tool.description,
    parameters: tool.parameters,
    strict: false,
  }));
}

// ── Headers / metadata ──────────────────────────────────────────────────────

export function createCodexMetadata(sessionId: string, turnId: string) {
  const windowId = `${sessionId}:0`;
  const turnMetadata = JSON.stringify({
    installation_id: CODEX_INSTALLATION_ID,
    session_id: sessionId,
    thread_id: sessionId,
    turn_id: turnId,
    window_id: windowId,
    request_kind: "turn",
    thread_source: "user",
    turn_started_at_unix_ms: Date.now(),
  });
  return {
    windowId,
    turnMetadata,
    clientMetadata: {
      session_id: sessionId,
      thread_id: sessionId,
      turn_id: turnId,
      "x-codex-installation-id": CODEX_INSTALLATION_ID,
      "x-codex-window-id": windowId,
      "x-codex-turn-metadata": turnMetadata,
    },
  };
}

function createCodexHeaders(apiKey: string, sessionId: string, metadata: ReturnType<typeof createCodexMetadata>) {
  return {
    authorization: `Bearer ${apiKey}`,
    accept: "text/event-stream",
    "content-type": "application/json",
    originator: "codex_exec",
    "user-agent": `codex_exec/${CODEX_VERSION} (Linux; x86_64) (codex_exec; ${CODEX_VERSION})`,
    "x-openai-internal-codex-responses-lite": "true",
    "x-codex-beta-features": "remote_compaction_v2",
    "x-codex-window-id": metadata.windowId,
    "x-codex-turn-metadata": metadata.turnMetadata,
    "x-client-request-id": sessionId,
    "session-id": sessionId,
    "thread-id": sessionId,
  };
}

// ── Request body builder ────────────────────────────────────────────────────

export function buildCodexRequestBody(
  model: Model<Api>,
  context: Context,
  options: SimpleStreamOptions | undefined,
  sessionId: string,
  metadata: ReturnType<typeof createCodexMetadata>,
) {
  const body: Json = {
    model: model.id,
    input: convertCodexMessages(context),
    tool_choice: "auto",
    parallel_tool_calls: false,
    reasoning: {
      effort: mapReasoningEffort(options?.reasoning),
      context: "all_turns",
    },
    store: false,
    stream: true,
    text: { verbosity: "low" },
    max_output_tokens: options?.maxTokens || model.maxTokens,
    include: ["reasoning.encrypted_content"],
    prompt_cache_key: sessionId,
    client_metadata: metadata.clientMetadata,
  };
  if (context.tools?.length) body.tools = convertCodexTools(context.tools);
  return body;
}

// ── Usage ───────────────────────────────────────────────────────────────────

function applyCodexUsage(output: AssistantMessage, response: any, model: Model<Api>) {
  const usage = response?.usage;
  if (!usage) return;
  const cached = usage.input_tokens_details?.cached_tokens || 0;
  const cacheWrite = usage.input_tokens_details?.cache_write_tokens || 0;
  output.usage.input = Math.max(0, (usage.input_tokens || 0) - cached - cacheWrite);
  output.usage.output = usage.output_tokens || 0;
  output.usage.cacheRead = cached;
  output.usage.cacheWrite = cacheWrite;
  if (usage.output_tokens_details?.reasoning_tokens != null) output.usage.reasoning = usage.output_tokens_details.reasoning_tokens;
  output.usage.totalTokens = usage.total_tokens || output.usage.input + output.usage.output + cached + cacheWrite;
  calculateCost(model, output.usage);
}

// ── SSE payload processing ──────────────────────────────────────────────────

function applyCodexSsePayload(payload: any, output: AssistantMessage, stream: AssistantMessageEventStream, model: Model<Api>, slots: Map<number, any>) {
  const type = payload?.type;
  if (!type || type === "response.in_progress" || type === "response.metadata") return;
  if (type === "error") {
    const errorType = payload.error?.type || payload.error?.code;
    const message = payload.error?.message || payload.message || JSON.stringify(payload);
    if (isRetryableErrorType(errorType)) throw new RetryableStreamError(message);
    throw new Error(message);
  }
  if (type === "response.failed") {
    const err = payload.response?.error;
    const message = err?.message || "Codex response failed";
    const errorType = err?.type || err?.code;
    if (isRetryableErrorType(errorType)) throw new RetryableStreamError(message);
    throw new Error(message);
  }

  if (type === "response.created") {
    output.responseId = payload.response?.id || output.responseId;
    return;
  }

  if (type === "response.output_item.added") {
    const item = payload.item;
    if (item?.type === "message") {
      const block = { type: "text", text: "" };
      output.content.push(block as any);
      const contentIndex = output.content.length - 1;
      slots.set(payload.output_index, { type: "text", block, contentIndex });
      stream.push({ type: "text_start", contentIndex, partial: output });
    } else if (item?.type === "reasoning") {
      const block = { type: "thinking", thinking: "", thinkingSignature: "" };
      output.content.push(block as any);
      const contentIndex = output.content.length - 1;
      slots.set(payload.output_index, { type: "thinking", block, contentIndex });
      stream.push({ type: "thinking_start", contentIndex, partial: output });
    } else if (item?.type === "function_call") {
      const block = { type: "toolCall", id: `${item.call_id}|${item.id}`, name: item.name, arguments: {}, partialJson: item.arguments || "" };
      output.content.push(block as any);
      const contentIndex = output.content.length - 1;
      slots.set(payload.output_index, { type: "toolCall", block, contentIndex });
      stream.push({ type: "toolcall_start", contentIndex, partial: output });
    }
    return;
  }

  const slot = slots.get(payload.output_index);
  if (type === "response.output_text.delta" && slot?.type === "text") {
    slot.block.text += String(payload.delta || "");
    stream.push({ type: "text_delta", contentIndex: slot.contentIndex, delta: String(payload.delta || ""), partial: output });
  } else if ((type === "response.reasoning_summary_text.delta" || type === "response.reasoning_text.delta") && slot?.type === "thinking") {
    slot.block.thinking += String(payload.delta || "");
    stream.push({ type: "thinking_delta", contentIndex: slot.contentIndex, delta: String(payload.delta || ""), partial: output });
  } else if (type === "response.function_call_arguments.delta" && slot?.type === "toolCall") {
    slot.block.partialJson += String(payload.delta || "");
    const parsed = tryParseJson(slot.block.partialJson);
    if (parsed !== undefined) slot.block.arguments = parsed;
    stream.push({ type: "toolcall_delta", contentIndex: slot.contentIndex, delta: String(payload.delta || ""), partial: output });
  } else if (type === "response.function_call_arguments.done" && slot?.type === "toolCall") {
    slot.block.partialJson = String(payload.arguments || slot.block.partialJson);
    slot.block.arguments = tryParseJson(slot.block.partialJson) || {};
  } else if (type === "response.output_item.done") {
    const item = payload.item;
    if (slot?.type === "text" && item?.type === "message") {
      slot.block.text = item.content?.map((part: any) => part.text || part.refusal || "").join("") || slot.block.text;
      stream.push({ type: "text_end", contentIndex: slot.contentIndex, content: slot.block.text, partial: output });
    } else if (slot?.type === "thinking" && item?.type === "reasoning") {
      slot.block.thinking =
        item.summary?.map((part: any) => part.text).join("\n\n") || item.content?.map((part: any) => part.text).join("\n\n") || slot.block.thinking;
      slot.block.thinkingSignature = JSON.stringify(item);
      stream.push({ type: "thinking_end", contentIndex: slot.contentIndex, content: slot.block.thinking, partial: output });
    } else if (slot?.type === "toolCall" && item?.type === "function_call") {
      slot.block.arguments = tryParseJson(item.arguments || slot.block.partialJson) || {};
      delete slot.block.partialJson;
      stream.push({ type: "toolcall_end", contentIndex: slot.contentIndex, toolCall: slot.block, partial: output });
    }
    slots.delete(payload.output_index);
  } else if (type === "response.completed" || type === "response.incomplete") {
    output.responseId = payload.response?.id || output.responseId;
    applyCodexUsage(output, payload.response, model);
    output.stopReason = type === "response.incomplete" ? "length" : output.content.some((block) => block.type === "toolCall") ? "toolUse" : "stop";
  }
}

// ── Stream consumption ──────────────────────────────────────────────────────

async function consumeCodexStream(response: Response, output: AssistantMessage, stream: AssistantMessageEventStream, model: Model<Api>) {
  if (!response.body) throw new Error("Codex stream response body missing");
  const slots = new Map<number, any>();
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let terminal = false;
  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    let parsedChunk = nextSseChunk(buffer);
    while (parsedChunk) {
      buffer = parsedChunk.rest;
      const event = parseSseEvent(parsedChunk.chunk);
      if (event.data && event.data !== "[DONE]") {
        const payload = tryParseJson(event.data);
        if (!payload) throw new Error(`invalid Codex SSE payload: ${event.data.slice(0, 200)}`);
        applyCodexSsePayload(payload, output, stream, model, slots);
        if (payload.type === "response.completed" || payload.type === "response.incomplete") terminal = true;
      }
      parsedChunk = nextSseChunk(buffer);
    }
    if (done) break;
  }
  const tail = buffer.trim();
  if (tail) {
    const event = parseSseEvent(tail);
    if (event.data && event.data !== "[DONE]") {
      const payload = tryParseJson(event.data);
      if (!payload) throw new Error(`invalid Codex SSE payload: ${event.data.slice(0, 200)}`);
      applyCodexSsePayload(payload, output, stream, model, slots);
      if (payload.type === "response.completed" || payload.type === "response.incomplete") terminal = true;
    }
  }
  if (!terminal) throw new Error("Codex stream ended before a terminal response event");
  return response;
}

// ── Streaming request ───────────────────────────────────────────────────────

export async function tryStreamAnyRouterCodex(
  url: string,
  body: Json,
  apiKey: string,
  model: Model<Api>,
  output: AssistantMessage,
  stream: AssistantMessageEventStream,
  sessionId: string,
  metadata: ReturnType<typeof createCodexMetadata>,
  options?: SimpleStreamOptions,
) {
  const bodyText = JSON.stringify(body);
  const maxRetries = Math.max(0, Number(process.env.PI_ANYROUTER_CC_MAX_RETRIES || options?.maxRetries || "10") || 0);

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    const headers = createCodexHeaders(apiKey, sessionId, metadata);
    if (attempt === 0) writeDebugFile("request", model.id, undefined, { url, headers: redactHeaders(headers), body, transport: "codex-sse" });

    let response: Response;
    try {
      response = await fetchWithProxy(url, { method: "POST", signal: options?.signal, headers, body: bodyText });
    } catch (error) {
      if (attempt < maxRetries && !options?.signal?.aborted) {
        await delay(getRetryDelayMs(attempt), options?.signal);
        continue;
      }
      throw error;
    }

    // HTTP-level error → retry if retryable status
    if (!response.ok) {
      const raw = await response.text();
      const parsed = tryParseJson(raw) || { raw };
      const requestId = extractRequestId(parsed, response.headers);
      writeDebugFile("error", model.id, requestId, { status: response.status, requestId, body: parsed, raw, transport: "codex-sse", retryAttempt: attempt });
      if (attempt < maxRetries && isRetryableStatus(response.status)) {
        await delay(getRetryDelayMs(attempt, parseRetryAfterMs(response.headers.get("retry-after"))), options?.signal);
        continue;
      }
      throw new Error(raw || `HTTP ${response.status}`);
    }

    if (!(response.headers.get("content-type") || "").includes("text/event-stream")) {
      throw new Error(`stream response was not SSE (content-type=${response.headers.get("content-type") || "<missing>"})`);
    }

    if (options?.onResponse) {
      await options.onResponse({ status: response.status, headers: Object.fromEntries(response.headers.entries()) }, model);
    }

    // Consume the SSE stream — retry on RetryableStreamError if no content was emitted
    const contentLenBefore = output.content.length;
    try {
      const resp = await consumeCodexStream(response, output, stream, model);
      writeDebugFile("response", model.id, resp.headers.get("x-oneapi-request-id") || undefined, {
        status: resp.status,
        responseId: output.responseId,
        stopReason: output.stopReason,
        usage: output.usage,
        transport: "codex-sse",
      });
      return;
    } catch (error) {
      if (error instanceof RetryableStreamError && output.content.length === contentLenBefore && attempt < maxRetries && !options?.signal?.aborted) {
        writeDebugFile("error", model.id, undefined, { phase: "sse-stream-retry", errorMessage: error.message, retryAttempt: attempt, transport: "codex-sse" });
        await delay(getRetryDelayMs(attempt, error.retryAfterMs), options?.signal);
        continue;
      }
      throw error;
    }
  }

  throw new Error("Codex request failed after retries");
}
