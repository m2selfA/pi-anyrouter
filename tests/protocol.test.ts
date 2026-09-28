import assert from "node:assert/strict";
import test from "node:test";
import { convertMessages, ensureClaudeCodeTools, getClaudeCodeHeaders } from "../src/claude-code.js";
import { nextSseChunk, parseSseEvent } from "../src/http.js";

test("normalizes a plain user message as Claude Code does", () => {
  assert.deepEqual(convertMessages([{ role: "user", content: "hello" }] as any), [{ role: "user", content: "hello" }]);
});

test("pads sparse tool declarations to the AnyRouter route minimum", () => {
  const tools = ensureClaudeCodeTools([] as any);
  assert.equal(tools.length, 4);
  assert.deepEqual(
    tools.map((tool) => tool.name),
    ["Read", "Write", "Edit", "Bash"],
  );
});

test("preserves real tools while adding only the missing compatibility entries", () => {
  const tools = ensureClaudeCodeTools([
    {
      name: "grep",
      description: "Search files",
      parameters: { type: "object", properties: { pattern: { type: "string" } }, required: ["pattern"] },
    },
  ] as any);
  assert.equal(tools.length, 4);
  assert.equal(tools[0].name, "Grep");
  assert.deepEqual(
    tools.slice(1).map((tool) => tool.name),
    ["Read", "Write", "Edit"],
  );
});

test("emits the captured Claude Code authentication and version headers", () => {
  const headers = getClaudeCodeHeaders("secret", 0, "session");
  assert.equal(headers["x-api-key"], "secret");
  assert.equal(headers["user-agent"], "claude-cli/2.1.281 (external, sdk-cli)");
  assert.equal(headers["x-stainless-package-version"], "0.112.1");
  assert.match(headers["anthropic-beta"], /per-turn-control-2026-07-01/);
});

test("parses SSE frames when boundaries split through event data", () => {
  const input =
    'event: content_block_delta\ndata: {"type":"content_block_delta","delta":{"text":"hel\\nlo"}}\n\nevent: message_stop\ndata: {}\n\n';
  let buffer = "";
  const events: ReturnType<typeof parseSseEvent>[] = [];
  for (const chunk of [input.slice(0, 7), input.slice(7, 29), input.slice(29, 61), input.slice(61)]) {
    buffer += chunk;
    let frame = nextSseChunk(buffer);
    while (frame) {
      events.push(parseSseEvent(frame.chunk));
      buffer = frame.rest;
      frame = nextSseChunk(buffer);
    }
  }
  assert.deepEqual(
    events.map((event) => event.event),
    ["content_block_delta", "message_stop"],
  );
  assert.match(events[0].data, /hel/);
  assert.equal(buffer, "");
  assert.equal(parseSseEvent('event: error\ndata: {"type":"error","error":{"message":"boom"}}').event, "error");
});
