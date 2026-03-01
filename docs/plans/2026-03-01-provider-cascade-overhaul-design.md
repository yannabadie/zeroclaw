# Provider Cascade Overhaul

**Date:** 2026-03-01
**Status:** Approved
**Approach:** B (Config + Code)

## Problem

ZeroClaw's provider cascade has three issues:
1. Gemini free tier (5-15 RPM) is rate-limited constantly, triggering cascade on every burst
2. Cross-provider model names cause 400 errors (e.g., "deepseek-chat" sent to Gemini)
3. Stale config: moonshot key invalid (401), 23 orphaned cron jobs, unused model routes

## Decisions

| Decision | Choice |
|----------|--------|
| Primary provider | DeepSeek V3.2 (`deepseek-chat`, $0.28/$0.42 per 1M tokens) |
| Model routing | All hints (code/fast/reasoning) use DeepSeek V3.2 |
| Fallback chain | DeepSeek -> Gemini Flash -> MiniMax M2.5 -> Kimi K2.5 |
| Model mapping | Per-provider remaps via `provider_model_fallbacks` |
| Cleanup | Delete disabled crons, remove moonshot (keep moonshot-intl) |

## Design

### 1. Config Changes (`~/.zeroclaw/config.toml`)

```toml
default_provider = "deepseek"
default_model = "deepseek-chat"

[reliability]
fallback_providers = ["gemini", "minimax", "moonshot-intl"]

[reliability.model_fallbacks]
gemini = ["gemini-2.5-flash"]
minimax = ["MiniMax-M2.5"]
"moonshot-intl" = ["kimi-k2.5"]

[[model_routes]]
hint = "code"
provider = "deepseek"
model = "deepseek-chat"

[[model_routes]]
hint = "fast"
provider = "deepseek"
model = "deepseek-chat"

[[model_routes]]
hint = "reasoning"
provider = "deepseek"
model = "deepseek-chat"

[cost.prices."deepseek/deepseek-chat"]
input = 0.28
output = 0.42
```

Removed:
- `moonshot` from fallback_providers (invalid API key)
- Old cross-provider model_fallbacks entries
- Gemini 3.1 Pro model routes

### 2. Code Fix (`src/providers/reliable.rs`)

**Function:** `provider_model_chain()` (~line 300-330)

**Bug:** Non-primary providers without configured remaps fall back to the original model name (line 326: `chain.push(model)`), sending provider-specific model names to incompatible providers.

**Fix:** Return empty chain for non-primary providers without remaps. The caller's for-loop over `sent_models` naturally skips the provider.

```rust
// Before (line 324-326):
if chain.is_empty() {
    chain.push(model);
}

// After:
if chain.is_empty() && !is_primary_provider {
    tracing::debug!(
        provider = provider_name,
        model = model,
        "Skipping provider: no model remap configured"
    );
} else if chain.is_empty() {
    chain.push(model);
}
```

**Test:** Add test verifying that a fallback provider without remap is skipped (no request sent, no error).

### 3. Cleanup

- `DELETE FROM jobs WHERE enabled = 0` in `~/.zeroclaw/workspace/cron/jobs.db`
- Verify 3 active jobs remain: briefing-matinal, resume-journee, audit-securite-hebdo
- Remove any moonshot references from config (keep moonshot-intl only)

## Cascade Flow After Changes

```
Request with model="deepseek-chat"

1. DeepSeek (primary): try "deepseek-chat" -> OK or fail
2. Gemini (fallback):  remap -> try "gemini-2.5-flash" -> OK or fail
3. MiniMax (fallback):  remap -> try "MiniMax-M2.5" -> OK or fail
4. Kimi (fallback):     remap -> try "kimi-k2.5" -> OK or fail
5. All failed -> error

No cross-provider model name errors. No 400s in logs.
```

## Cost Impact

| Provider | Input/1M | Output/1M | Role |
|----------|----------|-----------|------|
| DeepSeek V3.2 | $0.28 | $0.42 | Primary (all traffic) |
| Gemini 2.5 Flash | Free | Free | Fallback #1 |
| MiniMax M2.5 | $0.30 | $1.20 | Fallback #2 |
| Kimi K2.5 | $0.60 | $2.50 | Fallback #3 |

Previous cost (Gemini primary): $0.15/$0.60 but rate-limited, cascading to $0.30-$2.50 fallbacks constantly. Net cost likely higher than DeepSeek at $0.28/$0.42 with no rate limits.

## Env Requirements

- `DEEPSEEK_API_KEY` in `.env` (confirmed present: `sk-9e4c4d14...`)
- `GEMINI_*` credentials (confirmed working via OAuth)
- `MINIMAX_API_KEY` (confirmed working)
- `KIMI_API_KEY` / `MOONSHOT_API_KEY` (confirmed working on api.moonshot.ai)
