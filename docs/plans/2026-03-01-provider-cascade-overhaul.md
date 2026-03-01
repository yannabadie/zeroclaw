# Provider Cascade Overhaul — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Switch ZeroClaw from Gemini (rate-limited) to DeepSeek V3.2 as primary provider, fix cross-provider model-name cascade bug, and clean up stale config/cron.

**Architecture:** Config-driven provider swap (config.toml edits), plus one targeted Rust code fix in `ReliableProvider::provider_model_chain()` to skip fallback providers that have no model remap instead of sending wrong model names.

**Tech Stack:** Rust, TOML config, SQLite (cron jobs.db)

**Design Doc:** `docs/plans/2026-03-01-provider-cascade-overhaul-design.md`

---

### Task 1: Write failing test for cross-provider skip behavior

**Files:**
- Modify: `src/providers/reliable.rs:1472` (add test after `provider_keyed_model_fallbacks_remap_fallback_provider_models`)

**Step 1: Write the failing test**

Add this test after the existing `provider_keyed_model_fallbacks_remap_fallback_provider_models` test (after line 1472):

```rust
    #[tokio::test]
    async fn fallback_provider_without_remap_is_skipped() {
        // Primary provider fails on its model.
        // Fallback provider has NO remap configured.
        // Expected: fallback is skipped entirely (no request sent), cascade fails.
        let primary = Arc::new(ModelAwareMock {
            calls: Arc::new(AtomicUsize::new(0)),
            models_seen: parking_lot::Mutex::new(Vec::new()),
            fail_models: vec!["deepseek-chat"],
            response: "never",
        });
        let fallback_no_remap = Arc::new(ModelAwareMock {
            calls: Arc::new(AtomicUsize::new(0)),
            models_seen: parking_lot::Mutex::new(Vec::new()),
            fail_models: vec![],
            response: "should not reach",
        });

        // Only configure remap for primary, NOT for the fallback
        let mut fallbacks = HashMap::new();
        fallbacks.insert("deepseek".to_string(), vec![]); // primary: no extra models

        let provider = ReliableProvider::new(
            vec![
                (
                    "deepseek".into(),
                    Box::new(primary.clone()) as Box<dyn Provider>,
                ),
                (
                    "gemini".into(),
                    Box::new(fallback_no_remap.clone()) as Box<dyn Provider>,
                ),
            ],
            0,
            1,
        )
        .with_model_fallbacks(fallbacks);

        let err = provider
            .simple_chat("hello", "deepseek-chat", 0.0)
            .await
            .expect_err("should fail since primary fails and fallback is skipped");

        assert!(err.to_string().contains("All providers/models failed"));

        // Primary should have been called with "deepseek-chat"
        let primary_seen = primary.models_seen.lock();
        assert_eq!(primary_seen.len(), 1);
        assert_eq!(primary_seen[0], "deepseek-chat");

        // Fallback should NOT have been called at all
        let fallback_seen = fallback_no_remap.models_seen.lock();
        assert!(
            fallback_seen.is_empty(),
            "Fallback without remap should be skipped, but saw: {:?}",
            *fallback_seen
        );
    }
```

**Step 2: Run the test to verify it fails**

Run: `cargo test --lib fallback_provider_without_remap_is_skipped -- --nocapture 2>&1 | tail -20`

Expected: FAIL — the fallback provider currently receives `"deepseek-chat"` due to the line-326 fallback behavior.

**Step 3: Commit failing test**

```bash
git add src/providers/reliable.rs
git commit -m "test: add failing test for cross-provider model skip

Verifies that fallback providers without a configured model remap
are skipped entirely instead of receiving the primary's model name."
```

---

### Task 2: Fix provider_model_chain to skip unmapped fallbacks

**Files:**
- Modify: `src/providers/reliable.rs:324-327`

**Step 1: Apply the fix**

Replace lines 324–327 in `provider_model_chain()`:

```rust
        // BEFORE (lines 324-327):
        if chain.is_empty() {
            chain.push(model);
        }
```

With:

```rust
        // Skip non-primary providers that have no configured model remap
        // instead of sending the primary provider's model name (which would
        // cause a 400/404 "model not found" error).
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

**Step 2: Run the new test to verify it passes**

Run: `cargo test --lib fallback_provider_without_remap_is_skipped -- --nocapture 2>&1 | tail -20`

Expected: PASS

**Step 3: Run the full test suite to check for regressions**

Run: `cargo test --lib -- --nocapture 2>&1 | tail -30`

Expected: All existing tests pass — particularly:
- `provider_keyed_model_fallbacks_remap_fallback_provider_models` (remapped fallbacks still work)
- `no_model_fallbacks_behaves_like_before` (primary without fallbacks still works)
- `model_failover_all_models_fail` (model-level fallbacks still work)

**Step 4: Commit the fix**

```bash
git add src/providers/reliable.rs
git commit -m "fix(providers): skip fallback providers without model remap

Previously, non-primary providers with no configured model remap would
receive the primary provider's model name (e.g. 'deepseek-chat' sent to
Gemini), causing 400/404 errors. Now they return an empty model chain
and are silently skipped, letting the cascade proceed to the next
configured provider."
```

---

### Task 3: Update daemon config — provider and routing

**Files:**
- Modify: `~/.zeroclaw/config.toml` (lines 1-3, 156-180, 198-211, 503-563)

**Step 1: Switch default provider to DeepSeek**

In `~/.zeroclaw/config.toml`, change lines 1-2:

```toml
# BEFORE:
default_provider = "gemini"
default_model = "gemini-2.5-flash"

# AFTER:
default_provider = "deepseek"
default_model = "deepseek-chat"
```

**Step 2: Update fallback_providers**

Replace `[reliability]` fallback_providers (lines 159-162):

```toml
# BEFORE:
fallback_providers = [
    "minimax",
    "moonshot-intl",
]

# AFTER:
fallback_providers = [
    "gemini",
    "minimax",
    "moonshot-intl",
]
```

**Step 3: Replace model_fallbacks with per-provider remaps**

Replace `[reliability.model_fallbacks]` section (lines 169-180):

```toml
# BEFORE:
[reliability.model_fallbacks]
"gemini-3.1-pro-preview" = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "MiniMax-M2.5",
    "kimi-k2.5",
]
"gemini-2.5-flash" = [
    "gemini-2.0-flash",
    "MiniMax-M2.5",
    "kimi-k2.5",
]

# AFTER:
[reliability.model_fallbacks]
gemini = ["gemini-2.5-flash"]
minimax = ["MiniMax-M2.5"]
"moonshot-intl" = ["kimi-k2.5"]
```

**Step 4: Update model_routes to DeepSeek**

Replace `[[model_routes]]` sections (lines 198-211):

```toml
# BEFORE:
[[model_routes]]
hint = "code"
provider = "gemini"
model = "gemini-3.1-pro-preview"

[[model_routes]]
hint = "fast"
provider = "gemini"
model = "gemini-2.5-flash"

[[model_routes]]
hint = "reasoning"
provider = "gemini"
model = "gemini-3.1-pro-preview"

# AFTER:
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
```

**Step 5: Update agent providers to DeepSeek**

Replace agent provider/model references (lines 503-563):

```toml
# agents.researcher: change provider/model
[agents.researcher]
provider = "deepseek"
model = "deepseek-chat"
# ... rest unchanged

# agents.analyst: change provider/model
[agents.analyst]
provider = "deepseek"
model = "deepseek-chat"
# ... rest unchanged

# agents.coder: change provider/model
[agents.coder]
provider = "deepseek"
model = "deepseek-chat"
# ... rest unchanged
```

**Step 6: Add DeepSeek pricing**

Add after the existing `[cost.prices]` entries (around line 470):

```toml
[cost.prices."deepseek/deepseek-chat"]
input = 0.28
output = 0.42
```

**Step 7: Verify config syntax**

Run: `cat ~/.zeroclaw/config.toml | python3 -c "import sys, tomllib; tomllib.load(sys.stdin.buffer); print('TOML OK')"` (or use `toml` crate validation)

Expected: `TOML OK`

---

### Task 4: Clean up cron jobs database

**Files:**
- Modify: `~/.zeroclaw/workspace/cron/jobs.db`

**Step 1: Verify current state**

Run: `sqlite3 ~/.zeroclaw/workspace/cron/jobs.db "SELECT count(*) FROM cron_jobs WHERE enabled = 0;"`

Expected: `26`

Run: `sqlite3 ~/.zeroclaw/workspace/cron/jobs.db "SELECT id, name FROM cron_jobs WHERE enabled = 1;"`

Expected: 3 rows (briefing-matinal, resume-journee, audit-securite-hebdo)

**Step 2: Delete disabled cron jobs and their run history**

Run:
```bash
sqlite3 ~/.zeroclaw/workspace/cron/jobs.db "DELETE FROM cron_runs WHERE job_id IN (SELECT id FROM cron_jobs WHERE enabled = 0);"
sqlite3 ~/.zeroclaw/workspace/cron/jobs.db "DELETE FROM cron_jobs WHERE enabled = 0;"
sqlite3 ~/.zeroclaw/workspace/cron/jobs.db "VACUUM;"
```

**Step 3: Verify cleanup**

Run: `sqlite3 ~/.zeroclaw/workspace/cron/jobs.db "SELECT count(*) FROM cron_jobs;"`

Expected: `3`

Run: `sqlite3 ~/.zeroclaw/workspace/cron/jobs.db "SELECT name, enabled FROM cron_jobs;"`

Expected:
```
briefing-matinal|1
resume-journee|1
audit-securite-hebdo|1
```

---

### Task 5: Restart daemon and validate end-to-end

**Step 1: Restart the ZeroClaw daemon**

Run: `systemctl --user restart zeroclaw`

**Step 2: Verify daemon starts with DeepSeek**

Run: `systemctl --user status zeroclaw | head -15`

Expected: `active (running)`

Run: `journalctl --user -u zeroclaw --since "1 min ago" --no-pager | head -30`

Expected: Logs show startup with `deepseek` as default provider. No immediate errors.

**Step 3: Send a test message via Telegram**

Send a simple message to the bot (e.g., "test ping") and verify it responds using DeepSeek.

Check logs: `journalctl --user -u zeroclaw --since "2 min ago" --no-pager | grep -i "deepseek\|provider\|model"`

Expected: Logs show `provider=deepseek model=deepseek-chat` for the request.

**Step 4: Verify fallback chain (optional)**

Temporarily check logs for any cascade behavior. If DeepSeek works, no cascade should trigger.

**Step 5: Commit config changes**

```bash
git add -A  # Only if config is in tracked repo; otherwise skip
git commit -m "feat(config): switch to DeepSeek V3.2 primary, 4-level cascade

- default_provider: gemini -> deepseek (deepseek-chat)
- fallback chain: deepseek -> gemini -> minimax -> moonshot-intl
- model_fallbacks: per-provider remaps (gemini-2.5-flash, MiniMax-M2.5, kimi-k2.5)
- model_routes: all hints use deepseek/deepseek-chat
- agents: all use deepseek/deepseek-chat
- cleanup: removed 26 disabled cron jobs from jobs.db"
```
