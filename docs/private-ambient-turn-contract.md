# Private ambient turn contract

Private ambient turns are intentionally a narrow, local extension of the gateway.

## Kept

- A plugin injection with `private=True` is queued in the adapter's private queue while its session is active. It is not merged into the ordinary pending-message slot.
- Private events do not interrupt or steer a running agent. Ordinary plugin injection is also queued by default; `busy_policy="steer"` remains the explicit opt-in for an ordinary plugin event.
- Private execution produces no platform response, typing/status rail, TTS, attachment delivery, or error reply.
- Private turn messages are excluded from normal session persistence, transcript snapshots, lifecycle observers, and external-memory sync/prefetch.

## Deliberately dropped from the pre-merge private implementation

- Per-turn plugin lifecycle callbacks (`started`, `finished`, `cancelled`).
- Dispatch-result callbacks and their exactly-once receipt semantics across scheduler rejection, cancellation, adapter failure, and callback exceptions.
- Revalidation of a session entry/origin immediately before adapter handoff, including changed-origin rejection and cached-session identity-refresh guarantees.
- Compatibility tests for mixed public/private global FIFO reconstruction, reset-specific private-wake cancellation reports, topic-recovery marker repair, and long-running/inactivity/auto-reset notice micro-behavior.

The gateway keeps its existing normal queue, recovery, and notice behavior. Private events fail closed at the output and persistence boundaries rather than carrying a second lifecycle protocol.
