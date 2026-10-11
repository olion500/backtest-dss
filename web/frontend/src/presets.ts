import type { Preset, StrategySettings } from "./types";

function canonical(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(canonical);
  if (value && typeof value === "object") {
    return Object.fromEntries(Object.entries(value as Record<string, unknown>).sort(([a], [b]) => a.localeCompare(b)).map(([k, v]) => [k, canonical(v)]));
  }
  return value;
}

export function sameStrategy(a: StrategySettings, b: StrategySettings) {
  return JSON.stringify(canonical(a)) === JSON.stringify(canonical(b));
}

/** The preset whose strategy matches exactly, or undefined for hand-edited settings. */
export function matchPreset(presets: Preset[], strategy: StrategySettings) {
  return presets.find((preset) => sameStrategy(preset.strategy, strategy));
}
