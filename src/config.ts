import { readFileSync } from "node:fs";
import { CONFIG_PATH, type ProviderConfigFile } from "./types.js";

export function resolveConfigValue(value?: string) {
  if (!value) return "";
  if (value.startsWith("!")) {
    throw new Error("anyrouter does not support shell-command apiKey values. Use a literal key, env var name, or PI_ANYROUTER_CC_API_KEY.");
  }
  return process.env[value] || value;
}

export function loadSourceProvider() {
  let content = "";
  try {
    content = readFileSync(CONFIG_PATH, "utf8");
  } catch {
    throw new Error(`Config file not found: ${CONFIG_PATH}. Create ~/.pi/agent/anyrouter.json or set PI_ANYROUTER_CC_CONFIG.`);
  }

  let parsed: ProviderConfigFile;
  try {
    parsed = JSON.parse(content) as ProviderConfigFile;
  } catch (error) {
    throw new Error(`Invalid JSON in ${CONFIG_PATH}: ${error instanceof Error ? error.message : String(error)}`);
  }

  const baseUrl = process.env.PI_ANYROUTER_CC_BASE_URL || parsed.baseUrl;
  const apiKey = process.env.PI_ANYROUTER_CC_API_KEY || resolveConfigValue(parsed.apiKey);
  const models = parsed.models || [];

  if (!baseUrl) throw new Error(`Missing baseUrl in ${CONFIG_PATH}. You can also set PI_ANYROUTER_CC_BASE_URL.`);
  if (!apiKey) throw new Error(`Missing apiKey in ${CONFIG_PATH}. You can also set PI_ANYROUTER_CC_API_KEY.`);
  if (!models.length) throw new Error(`No models configured in ${CONFIG_PATH}. Add at least one model entry.`);

  return { baseUrl, apiKey, models };
}
