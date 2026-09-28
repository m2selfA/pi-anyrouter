import { randomUUID } from "node:crypto";
import {
  type Api,
  type AssistantMessage,
  type AssistantMessageEventStream,
  type Context,
  createAssistantMessageEventStream,
  type Model,
  type SimpleStreamOptions,
} from "@earendil-works/pi-ai";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import {
  applyJsonResponseToOutput,
  convertMessages,
  createClaudeCodeMetadata,
  createClaudeCodeSystem,
  ensureClaudeCodeTools,
  postJson,
  tryStreamAnyRouterCc,
} from "./claude-code.js";
import { buildCodexRequestBody, createCodexMetadata, getCodexResponsesUrl, tryStreamAnyRouterCodex } from "./codex.js";
import { loadSourceProvider } from "./config.js";
import { writeDebugFile } from "./http.js";
import { API_ID, CONFIG_PATH, type Json, PROVIDER_NAME } from "./types.js";
import { createEmptyUsage, getStreamMode, isCodexModel, mapReasoningEffort, resetOutputState } from "./utils.js";

function streamAnyRouterCc(model: Model<Api>, context: Context, options?: SimpleStreamOptions): AssistantMessageEventStream {
  const stream = createAssistantMessageEventStream();
  (async () => {
    const output: AssistantMessage = {
      role: "assistant",
      content: [],
      api: model.api,
      provider: model.provider,
      model: model.id,
      usage: createEmptyUsage(),
      stopReason: "pending",
      timestamp: Date.now(),
    };

    try {
      const source = loadSourceProvider();
      const apiKey = options?.apiKey || source.apiKey;
      const sessionId = options?.sessionId || randomUUID();

      const configuredModel = source.models.find((item) => item.id === model.id);
      if (isCodexModel(model.id, configuredModel?.api)) {
        const turnId = randomUUID();
        const metadata = createCodexMetadata(sessionId, turnId);
        let codexBody: Json = buildCodexRequestBody(model, context, options, sessionId, metadata);
        if (options?.onPayload) {
          const replaced = await options.onPayload(codexBody, model);
          if (replaced !== undefined) codexBody = replaced as Json;
        }
        stream.push({ type: "start", partial: output });
        await tryStreamAnyRouterCodex(getCodexResponsesUrl(source.baseUrl), codexBody, apiKey, model, output, stream, sessionId, metadata, options);
        if (options?.signal?.aborted) throw new Error("Request was aborted");
        stream.push({ type: "done", reason: output.stopReason as "stop" | "length" | "toolUse" | "deferred", message: output });
        stream.end();
        return;
      }

      const url = `${source.baseUrl.replace(/\/$/, "")}/v1/messages?beta=true`;
      let requestBody: Json = {
        model: model.id,
        messages: convertMessages(context.messages),
        max_tokens: options?.maxTokens || model.maxTokens || 32000,
        stream: false,
        metadata: createClaudeCodeMetadata(sessionId),
        system: createClaudeCodeSystem(context.systemPrompt || "You are an expert coding assistant operating inside pi."),
        context_management: {
          edits: [{ type: "clear_thinking_20251015", keep: "all" }],
        },
      };
      requestBody.tools = ensureClaudeCodeTools(context.tools);
      if (options?.reasoning && model.reasoning) {
        requestBody.thinking = { type: "adaptive", display: "omitted" };
        requestBody.output_config = { effort: mapReasoningEffort(options.reasoning) };
      }
      if (options?.onPayload) {
        const replaced = await options.onPayload(requestBody, model);
        if (replaced !== undefined) requestBody = replaced as Json;
      }

      stream.push({ type: "start", partial: output });

      const streamMode = getStreamMode();
      if (streamMode !== "off") {
        try {
          await tryStreamAnyRouterCc(url, requestBody, apiKey, model, output, stream, sessionId, options);
          if (options?.signal?.aborted) throw new Error("Request was aborted");
          stream.push({ type: "done", reason: output.stopReason as "stop" | "length" | "toolUse" | "deferred", message: output });
          stream.end();
          return;
        } catch (streamError) {
          if (streamMode === "force" || output.content.length > 0) {
            const errText = `[anyrouter] ${streamError instanceof Error ? streamError.message : String(streamError)}`;
            const contentIndex = output.content.length;
            output.content.push({ type: "text", text: errText } as any);
            output.stopReason = "stop";
            output.errorMessage = errText;
            stream.push({ type: "text_start", contentIndex, partial: output });
            stream.push({ type: "text_delta", contentIndex, delta: errText, partial: output });
            stream.push({ type: "text_end", contentIndex, content: errText, partial: output });
            stream.push({ type: "done", reason: "stop", message: output });
            stream.end();
            return;
          }
          writeDebugFile("error", model.id, undefined, {
            phase: "stream-fallback",
            errorMessage: streamError instanceof Error ? streamError.message : String(streamError),
          });
          resetOutputState(output);
        }
      }

      const response = await postJson(url, requestBody, apiKey, model.id, sessionId, model, options);
      applyJsonResponseToOutput(response, output, stream, model);
      stream.push({ type: "done", reason: output.stopReason as "stop" | "length" | "toolUse" | "deferred", message: output });
      stream.end();
    } catch (error) {
      output.stopReason = options?.signal?.aborted ? "aborted" : "error";
      output.errorMessage = error instanceof Error ? `[anyrouter] ${error.message}` : String(error);
      writeDebugFile("error", model.id, undefined, {
        stopReason: output.stopReason,
        errorMessage: output.errorMessage,
      });
      stream.push({ type: "error", reason: output.stopReason, error: output });
      stream.end();
    }
  })();
  return stream;
}

export default function (pi: ExtensionAPI) {
  try {
    const source = loadSourceProvider();
    pi.registerProvider(PROVIDER_NAME, {
      baseUrl: source.baseUrl,
      apiKey: source.apiKey,
      api: API_ID,
      models: source.models.map((model) => ({
        id: model.id,
        name: model.name ? `${model.name} (AnyRouter)` : `${model.id} (AnyRouter)`,
        api: API_ID,
        reasoning: model.reasoning ?? true,
        thinkingLevelMap:
          (model.reasoning ?? true) ? { off: "off", minimal: "minimal", low: "low", medium: "medium", high: "high", xhigh: "xhigh", max: "max" } : undefined,
        input: model.input ?? ["text"],
        cost: {
          input: model.cost?.input ?? 0,
          output: model.cost?.output ?? 0,
          cacheRead: model.cost?.cacheRead ?? 0,
          cacheWrite: model.cost?.cacheWrite ?? 0,
        },
        contextWindow: model.contextWindow ?? 200000,
        maxTokens: model.maxTokens ?? 32000,
      })),
      streamSimple: streamAnyRouterCc,
    });
  } catch (error) {
    console.error(`[anyrouter] Failed to register provider: ${error instanceof Error ? error.message : String(error)}`);
    console.error(`[anyrouter] Config path: ${CONFIG_PATH}`);
    console.error(`[anyrouter] You can override with PI_ANYROUTER_CC_CONFIG, PI_ANYROUTER_CC_BASE_URL, PI_ANYROUTER_CC_API_KEY`);
  }
}
