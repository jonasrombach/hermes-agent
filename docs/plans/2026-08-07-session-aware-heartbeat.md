# Session-Aware Heartbeat Queue/Wake Implementation Plan

> **For Claude:** Use `${SUPERPOWERS_SKILLS_ROOT}/skills/collaboration/executing-plans/SKILL.md` to implement this plan task-by-task.

**Goal:** Add a durable hourly cron mode that wakes Rocky's existing Telegram session as a real, tool-capable turn, queues silently behind an active turn, coalesces duplicate heartbeats, and emits nothing when Rocky returns exactly `NO_REPLY`.

**Architecture:** A new opt-in `session_wake` cron job shape bypasses the isolated cron agent and dispatches its prompt through `gateway.wake.deliver_wake()` to the job's captured origin session. Push adapters mark the synthetic event with first-class session-wake metadata. `BasePlatformAdapter` keeps session wakes in a queue separate from human pending messages, gives human follow-ups priority, and coalesces only matching wake keys. Cron success means the gateway atomically accepted the event into either immediate processing or the session-owned queue; it does not wait for the eventual model turn, avoiding gateway-shutdown deadlocks and duplicate retries.

**Tech Stack:** Python 3.11, asyncio, Hermes gateway adapters, Hermes cron scheduler and JSON job store, pytest, `uv`, Telegram gateway runtime.

**Approved design:** `/home/jonas/hermes-workspace/docs/superpowers/specs/2026-08-01-rocky-pending-heartbeat-design.md`

**Worktree:** `/home/jonas/.hermes/worktrees/heartbeat-nudge-wake`

**Test command:** `./scripts/run_tests.sh ...`

---

## Invariants

1. Standard cron jobs retain byte-compatible isolated-agent behavior.
2. `session_wake` requires an originating gateway session and never fan-outs.
3. A push wake is `MessageEvent(internal=True)` with explicit `session_wake`, `delivery_id`, `coalesce_key`, and memory/transcript provenance metadata.
4. An active turn is never interrupted or steered by a session wake.
5. Human pending messages are never merged with synthetic wakes and run before ambient wakes.
6. At most one unstarted heartbeat with the same coalesce key exists per session; the newest payload wins.
7. The event starts immediately when idle and cascades automatically after busy work without waiting for another owner message.
8. `NO_REPLY` uses the existing exact gateway silence contract and produces no platform send or technical acknowledgement.
9. The synthetic heartbeat prompt is not auto-retained as a Jonas-authored memory.
10. No production runtime or gateway restart occurs before tests, review, and install verification pass.

---

### Task 1: Add the persisted `session_wake` job contract

**Files:**
- Modify: `cron/jobs.py`
- Modify: `tools/cronjob_tools.py`
- Modify: `tests/cron/test_jobs.py`
- Modify: `tests/tools/test_cronjob_tools.py`

**Step 1: Write the failing job-model tests**

Add tests that call `create_job(..., session_wake=True)` and assert:

```python
assert job["session_wake"] is True
assert job["deliver"] == "origin"
assert job["origin"]["platform"] == "telegram"
```

Add one validation test per invalid shape:

```python
@pytest.mark.parametrize("field,value", [
    ("origin", None),
    ("no_agent", True),
    ("script", "collector.py"),
    ("skills", ["x"]),
    ("context_from", ["job-a"]),
    ("workdir", "/tmp"),
])
def test_session_wake_rejects_isolated_cron_axes(field, value): ...
```

Verify normal jobs omit the new key unless explicitly set.

**Step 2: Run RED**

Run:

```bash
./scripts/run_tests.sh -q \
  tests/cron/test_jobs.py \
  tests/tools/test_cronjob_tools.py
```

Expected: FAIL because `create_job` and `cronjob` do not accept `session_wake`.

**Step 3: Implement the minimal job field**

Add `session_wake: bool = False` to `create_job()` and normalize it before the job dict is written. For `session_wake=True`:

- require a non-empty prompt;
- require a valid origin dict with platform and chat ID;
- force `deliver="origin"`;
- reject `no_agent`, script, skills, context chaining, inference overrides, toolset restrictions, and workdir;
- persist `"session_wake": True` only for this mode;
- preserve omission for all existing jobs.

Expose the same create/update field through `tools/cronjob_tools.py`, include it in shaped list/get results, and document that it executes inside the existing origin session rather than a fresh cron session.

**Step 4: Run GREEN**

Run the Task 1 command. Expected: PASS.

**Step 5: Commit**

```bash
git add cron/jobs.py tools/cronjob_tools.py tests/cron/test_jobs.py tests/tools/test_cronjob_tools.py
git commit -m "feat: add session wake cron job contract"
```

---

### Task 2: Carry explicit session-wake metadata through `deliver_wake`

**Files:**
- Modify: `gateway/wake.py`
- Modify: `tests/gateway/test_wake_delivery.py`

**Step 1: Write the failing metadata test**

Call:

```python
await deliver_wake(
    adapter,
    text="heartbeat",
    source=source,
    metadata={
        "session_wake": True,
        "delivery_id": "heartbeat:2026-08-07T21:47:00Z",
        "coalesce_key": "rocky-heartbeat",
        "display_kind": "hidden",
        "skip_external_memory_sync": True,
    },
)
```

Assert the handled event is internal and carries an independent copy of that metadata. Verify the default call still produces an internal event with empty metadata.

**Step 2: Run RED**

```bash
./scripts/run_tests.sh -q tests/gateway/test_wake_delivery.py
```

Expected: FAIL because `deliver_wake()` has no metadata parameter.

**Step 3: Implement metadata forwarding**

Add an optional metadata mapping to `deliver_wake()`. Copy it into the push `MessageEvent` so callers cannot mutate the queued event after dispatch. Keep the API-server self-POST path unchanged for now because Rocky's target is Telegram and raw API sessions have a different completion contract.

**Step 4: Run GREEN**

Run the Task 2 command. Expected: PASS.

**Step 5: Commit**

```bash
git add gateway/wake.py tests/gateway/test_wake_delivery.py
git commit -m "feat: preserve session wake provenance"
```

---

### Task 3: Queue ambient wakes separately and coalesce duplicates

**Files:**
- Modify: `gateway/platforms/base.py`
- Create: `tests/gateway/test_session_wake_queue.py`
- Modify: `tests/gateway/test_internal_event_never_interrupts_busy_session.py`

**Step 1: Write the failing isolation test**

Use a minimal real `BasePlatformAdapter` test double and block its first handler invocation. While active:

1. enqueue a human follow-up in `_pending_messages`;
2. send a heartbeat event with `metadata.session_wake=True` and `coalesce_key="rocky-heartbeat"`;
3. release the active turn.

Assert the handler observes:

```python
["active user turn", "human follow-up", "heartbeat"]
```

Assert the human event text contains no heartbeat text and retains `internal=False`.

**Step 2: Run RED**

```bash
./scripts/run_tests.sh -q tests/gateway/test_session_wake_queue.py::test_busy_session_keeps_human_and_wake_as_distinct_turns
```

Expected: FAIL because the current one-slot pending merge contaminates or replaces one event.

**Step 3: Implement one separate internal queue**

In `BasePlatformAdapter`:

- add a per-session ordered mapping for pending session wakes;
- recognize only `internal=True` plus `metadata["session_wake"] is True`;
- use `coalesce_key`, falling back to `delivery_id` or message ID;
- replace the value for an existing key without changing its queue position;
- never send a busy acknowledgement;
- after a processing task finishes, drain a human pending event first, then one session wake;
- clear pending wakes on destructive session cancellation/reset;
- preserve them on handoffs that currently preserve `_pending_messages`.

Keep the existing behavior of unrelated internal completion events unchanged.

**Step 4: Run GREEN for isolation**

Run the Step 2 command. Expected: PASS.

**Step 5: Write the failing coalescing test**

Queue two heartbeat events with the same key and different delivery IDs/text. Release the active turn and assert exactly one heartbeat turn runs with the newer payload.

**Step 6: Run RED, implement minimal replacement, then run GREEN**

```bash
./scripts/run_tests.sh -q tests/gateway/test_session_wake_queue.py
```

Expected sequence: first FAIL for duplicate delivery, then PASS after keyed replacement.

**Step 7: Run the existing busy-session regression**

```bash
./scripts/run_tests.sh -q \
  tests/gateway/test_session_wake_queue.py \
  tests/gateway/test_internal_event_never_interrupts_busy_session.py
```

Expected: PASS.

**Step 8: Commit**

```bash
git add gateway/platforms/base.py tests/gateway/test_session_wake_queue.py tests/gateway/test_internal_event_never_interrupts_busy_session.py
git commit -m "feat: queue and coalesce ambient session wakes"
```

---

### Task 4: Dispatch `session_wake` jobs without creating a cron agent

**Files:**
- Modify: `cron/scheduler.py`
- Create: `tests/cron/test_scheduler_session_wake.py`
- Modify: `tests/cron/test_scheduler.py`

**Step 1: Write the failing end-to-end scheduler test**

Build a session-wake job with a Telegram origin, a live fake adapter, and a real asyncio loop. Invoke `run_one_job()` and assert:

```python
cron_scheduler.run_job.assert_not_called()
assert accepted_event.text == job["prompt"]
assert accepted_event.source.platform == Platform.TELEGRAM
assert accepted_event.source.chat_id == job["origin"]["chat_id"]
assert accepted_event.metadata["session_wake"] is True
assert accepted_event.metadata["coalesce_key"] == "rocky-heartbeat"
```

Assert the execution and job status are successful only after `adapter.handle_message()` accepts the event.

**Step 2: Run RED**

```bash
./scripts/run_tests.sh -q tests/cron/test_scheduler_session_wake.py
```

Expected: FAIL because `run_one_job()` starts the isolated cron agent.

**Step 3: Implement the push-dispatch branch**

Before the normal secret-scope and `run_job()` path:

- resolve the exact captured origin with `_resolve_origin()`;
- normalize platform to `gateway.config.Platform`;
- resolve the corresponding live adapter;
- construct `gateway.session.SessionSource` with chat, thread, user, and chat-name fields;
- compute a stable delivery ID from job ID plus the scheduled occurrence;
- set `coalesce_key` from job data, defaulting to the job ID;
- schedule `deliver_wake()` onto the gateway loop with `safe_schedule_threadsafe`;
- wait only for `adapter.handle_message()` acceptance, using the existing bounded loop handoff;
- mark the cron execution as accepted/successful without output delivery, agent teardown, or transcript mirroring.

A missing origin, adapter, loop, unsupported platform, or dispatch exception is a normal failed cron execution with a truthful error.

**Step 4: Run GREEN**

Run the Task 4 command. Expected: PASS.

**Step 5: Add failure and non-regression tests**

Cover missing adapter, malformed origin, stopped event loop, and a normal cron job. Verify the normal job still calls `run_job()` and `_deliver_result()` exactly as before.

**Step 6: Run the scheduler suite**

```bash
./scripts/run_tests.sh -q \
  tests/cron/test_scheduler_session_wake.py \
  tests/cron/test_scheduler.py \
  tests/cron/test_scheduler_cron_session_isolation.py
```

Expected: PASS.

**Step 7: Commit**

```bash
git add cron/scheduler.py tests/cron/test_scheduler_session_wake.py tests/cron/test_scheduler.py
git commit -m "feat: dispatch cron jobs into origin sessions"
```

---

### Task 5: Preserve internal provenance and skip automatic memory retention

**Files:**
- Modify: `gateway/run.py`
- Modify: `gateway/turn_context.py`
- Modify: `run_agent.py`
- Modify: `tests/plugins/memory/test_hindsight_provider.py`
- Create: `tests/gateway/test_session_wake_provenance.py`

**Step 1: Write the failing gateway provenance test**

Pass a session-wake `MessageEvent` through the real gateway turn wiring with the agent call captured. Assert `run_conversation()` receives:

```python
persist_user_display_kind="hidden"
persist_user_display_metadata={
    "event_kind": "session_wake",
    "delivery_id": "...",
    "coalesce_key": "rocky-heartbeat",
    "skip_external_memory_sync": True,
}
```

The API message content must remain the canonical heartbeat prompt so Rocky sees it in context.

**Step 2: Run RED**

```bash
./scripts/run_tests.sh -q tests/gateway/test_session_wake_provenance.py
```

Expected: FAIL because gateway `TurnContext` does not carry display metadata into `run_conversation()`.

**Step 3: Wire the persisted synthetic-row metadata**

Add `persist_user_display_kind` and `persist_user_display_metadata` to `gateway.turn_context.TurnContext`, `_run_agent()`, `_run_agent_inner()`, and `TurnRunner.run_sync()`. Populate them only from trusted `internal` session-wake metadata. Do not allow ordinary platform messages to select a hidden display type through untrusted metadata.

**Step 4: Run GREEN for provenance**

Run the Step 2 command. Expected: PASS.

**Step 5: Write the failing memory-boundary test**

Create a completed turn whose latest user row has:

```python
{"role": "user", "display_metadata": {"skip_external_memory_sync": True}}
```

Call `_sync_external_memory_for_turn()` and assert Hindsight `sync_turn()` and post-turn prefetch are not called. Add a control test showing an ordinary user turn still syncs.

**Step 6: Run RED, implement the narrow guard, then GREEN**

In `AIAgent._sync_external_memory_for_turn()`, inspect only the current latest user row in the supplied messages. Return before both sync and post-turn prefetch when the trusted flag is present.

```bash
./scripts/run_tests.sh -q \
  tests/gateway/test_session_wake_provenance.py \
  tests/plugins/memory/test_hindsight_provider.py
```

Expected: PASS.

**Step 7: Commit**

```bash
git add gateway/run.py gateway/turn_context.py run_agent.py \
  tests/gateway/test_session_wake_provenance.py \
  tests/plugins/memory/test_hindsight_provider.py
git commit -m "feat: mark ambient turns and protect memory provenance"
```

---

### Task 6: Prove `NO_REPLY` and immediate post-busy cascade

**Files:**
- Modify or create the narrowest existing gateway response-suppression test file discovered during implementation
- Modify: `tests/gateway/test_session_wake_queue.py`

**Step 1: Locate the exact silence predicate**

Use source search to identify the gateway's existing exact `NO_REPLY` suppression path. Do not add a second silence parser.

**Step 2: Write the failing integration test**

Run a session-wake through a fake handler that returns exactly `NO_REPLY`. Assert:

```python
adapter._send_with_retry.assert_not_awaited()
```

Also assert no busy/queued acknowledgement was sent before the handler ran.

If the existing behavior already passes, retain the regression test and do not change production code for this slice.

**Step 3: Test cascade without another user message**

Hold an active turn, queue one heartbeat, release the turn, and await the adapter's background task. Assert the heartbeat handler runs without invoking `handle_message()` again and without injecting a synthetic owner message.

**Step 4: Run GREEN**

```bash
./scripts/run_tests.sh -q \
  tests/gateway/test_session_wake_queue.py \
  tests/gateway/test_wake_delivery.py \
  tests/gateway/test_internal_event_never_interrupts_busy_session.py
```

Expected: PASS.

**Step 5: Commit**

```bash
git add tests/gateway
git commit -m "test: lock heartbeat silence and cascade semantics"
```

---

### Task 7: Document the new cron mode and configure Rocky's heartbeat

**Files:**
- Modify: `website/docs/user-guide/features/cron.md` or the repository's current canonical cron documentation
- Modify: `HEARTBEAT.md` in the active Rocky workspace only if its current content does not express the approved behavior
- Runtime state: create one cron job in the active `default` profile after installation

**Step 1: Add focused user-facing documentation**

Document `session_wake` as an advanced opt-in mode with these constraints:

- created from a live origin session;
- continues that session with normal tools and memory recall;
- queues silently at a turn boundary when busy;
- no isolated cron agent and no output fan-out;
- exact `NO_REPLY` remains platform-silent;
- repeated unstarted events with the same key coalesce.

Do not imply that standard cron isolation changed.

**Step 2: Validate docs references and examples**

Search the generated docs for stale claims that all cron jobs always use isolated sessions. Narrowly qualify those statements rather than deleting the default invariant.

**Step 3: Commit docs**

```bash
git add website/docs/user-guide/features/cron.md
git commit -m "docs: describe session wake cron jobs"
```

**Step 4: After runtime installation, create the real job**

Create exactly one recurring job:

```text
name: Rocky hourly heartbeat
schedule: 47 * * * *
session_wake: true
deliver: origin
coalesce_key: rocky-heartbeat
origin: current Telegram DM session
```

Use a self-contained prompt that identifies itself as an internal scheduled heartbeat, asks Rocky to inspect the injected `HEARTBEAT.md` guidance and current conversation context, permits normal tools, and requires exact `NO_REPLY` when no contact is useful.

**Step 5: Verify persisted state**

List the job and read back its persisted job record. Verify schedule, origin, session-wake mode, next run, and absence of isolated-agent axes. Do not expose credentials or unrelated job prompts.

---

### Task 8: Run focused, broad, and independent verification

**Files:**
- No production files unless a test exposes a bug

**Step 1: Run the focused feature suite**

```bash
./scripts/run_tests.sh -q \
  tests/gateway/test_wake_delivery.py \
  tests/gateway/test_session_wake_queue.py \
  tests/gateway/test_session_wake_provenance.py \
  tests/gateway/test_internal_event_never_interrupts_busy_session.py \
  tests/run_agent/test_steer.py \
  tests/cron/test_jobs.py \
  tests/cron/test_scheduler.py \
  tests/cron/test_scheduler_session_wake.py \
  tests/cron/test_scheduler_cron_session_isolation.py \
  tests/tools/test_cronjob_tools.py \
  tests/plugins/memory/test_hindsight_provider.py
```

Expected: all pass with no warnings introduced by the change.

**Step 2: Run broader affected directories**

```bash
./scripts/run_tests.sh -q tests/gateway tests/cron tests/tools/test_cronjob_tools.py tests/plugins/memory
```

Expected: PASS. If unrelated pre-existing failures occur, reproduce them on the untouched base branch before classifying them.

**Step 3: Inspect the diff**

```bash
git diff --check
git status --short
git diff --stat
git diff runtime/v2026.8.3-rocky...HEAD
```

Check for secrets, accidental generated files, broad refactors, stale Nudge terminology, and changes to standard cron behavior.

**Step 4: Dispatch independent reviews**

Ask one reviewer to audit spec compliance and concurrency boundaries, and another to audit regression risk, security, provenance, and test quality. Require file/line evidence and no edits. Convert every valid finding into a failing test before fixing it.

**Step 5: Fix valid findings with RED-GREEN**

For each accepted defect:

1. write a reproducing test;
2. run it and observe the expected failure;
3. implement the narrow fix;
4. rerun focused and affected suites;
5. commit.

---

### Task 9: Install, safely restart, and verify the real path

**Files/state:**
- Installed Hermes runtime for active profile `default`
- Gateway service
- Active Telegram session
- Cron job store

**Step 1: Load and follow the `safe-restart` skill**

Prepare the tested runtime using the repository's supported install path. Verify the installed revision and import path before restart. Jonas normally performs restart, but his explicit assignment to finish the feature permits the safest available automated handoff only if the skill and live service support it without lockout risk. Otherwise report the exact manual restart blocker.

**Step 2: Verify post-restart health**

Check gateway process health, Telegram adapter readiness, logs for import/config errors, cron scheduler startup, and the installed source revision.

**Step 3: Run a controlled idle heartbeat**

Trigger the new job manually while the session is idle. Verify from real logs and transcript state that:

- the existing Telegram session ID was reused;
- the event is marked internal/session-wake;
- the agent had the normal toolset and context;
- exact `NO_REPLY` caused no Telegram send.

Do not claim Telegram silence from a command exit code alone. Check outbound adapter evidence and the Telegram chat state if available.

**Step 4: Run a controlled busy heartbeat**

Start a bounded active turn, trigger one heartbeat during it, and verify:

- no second agent run exists for the session;
- no interrupt/steer/ack occurs;
- the heartbeat runs as its own turn immediately after the active turn;
- it does not wait for another Jonas message.

Trigger a second heartbeat while the first remains queued and verify one coalesced follow-up.

**Step 5: Verify memory provenance**

Query the transcript and Hindsight evidence. Confirm the technical heartbeat prompt was not auto-retained as a Jonas statement. Confirm normal user turns still retain normally.

**Step 6: Roll back on any live failure**

Pause the new job, restore the previous runtime through the supported runtime mechanism, safely restart, and verify gateway health. Preserve logs and failing evidence for another RED-GREEN cycle.

---

### Task 10: Finish and push the branch

**Step 1: Final verification**

Run the focused suite once more on the exact final commit and record the real pass count. Verify the live job remains enabled with the next `:47` occurrence.

**Step 2: Final commit if needed**

```bash
git status --short
git add <only intended files>
git commit -m "feat: add session-aware heartbeat wakes"
```

**Step 3: Push**

```bash
git push -u origin feature/session-aware-heartbeat
```

If no writable remote or authentication exists, report that exact blocker and keep the verified commits in the worktree. Do not describe local commits as delivered upstream.

**Step 4: Final report**

Report only:

- live status and next scheduled run;
- branch/commit and push status;
- exact focused and broader test results;
- real idle/busy/`NO_REPLY` verification results;
- review URL for the approved spec;
- any genuine remaining blocker.
