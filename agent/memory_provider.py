"""Abstract base class for pluggable memory providers.

Plugins ship in ``plugins/memory/<name>/``, activated via ``memory.provider`` (ONE external
provider at a time). Lifecycle, driven by MemoryManager: initialize -> system_prompt_block /
prefetch / sync_turn per turn -> tool dispatch -> shutdown, plus optional ``on_*`` hooks.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# v1 = best-effort on_pre_compress() with the raw message list; v2 = opt-in fail-closed
# checkpoint (normalized evidence handoff + strict-mode failure propagation).
PRE_COMPRESS_CHECKPOINT_API_VERSION = 2

# Default glyph for recall indicators; providers may use their own brand mark.
INDICATOR_GLYPH = "🧠"


@dataclass(frozen=True)
class RecallStatus:
    """What the last prefetch injected, for the deterministic recall indicator
    (``MemoryManager.describe_recall``). ``count == 0`` means content without a
    discrete count (e.g. a synthesized reflect answer) and renders generically."""

    provider_label: str
    count: int
    glyph: str = INDICATOR_GLYPH


# Prompts with no semantic signal; single source of truth for the core prefetch gate and
# provider-side classifiers. Anchored and followed only by whitespace/punctuation, so
# "k8s"/"yolo"/"note" do NOT match while "hi!"/"thanks :)"/"done???" do.
TRIVIAL_PROMPT_RE = re.compile(
    r'^(yes|no|ok|okay|sure|thanks|thank you|y|n|yep|nope|yeah|nah|'
    r'ja|genau|mach weiter|weiter|'
    r'hi|hey|hello|yo|sup|'
    r'continue|go ahead|do it|proceed|got it|cool|nice|great|done|next|lgtm|k)'
    r'[\s!?.:;,"' + "'" + r'~\u2018\u2019\u201c\u201d\u2014\u2013\u2026()\[\]{}<>*&^%$#@!+=`\u00a0]*$',
    re.IGNORECASE,
)

_MEMORY_CONTEXT_RE = re.compile(r"<memory-context>.*?</memory-context>", re.DOTALL | re.IGNORECASE)
_CODE_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)
_SYMBOL_ONLY_RE = re.compile(r"^[\W_]+$", re.UNICODE)
_SYNTHETIC_USER_PREFIXES = (
    "[System: Your previous response was truncated",
    "[System: The previous response was cut off",
    "[System: Your previous tool call",
    "[Your active task list was preserved across context compression]",
    "[IMPORTANT: Background process ",
    "[CONTEXT COMPACTION]",
)


def _message_text(message: Dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text") or part.get("content") or "")
            for part in content
            if isinstance(part, dict)
        )
    return ""


def _extract_steer_text(message: Dict[str, Any]) -> str:
    if message.get("role") != "tool":
        return ""
    text = _message_text(message)
    if not text:
        return ""
    try:
        from agent.prompt_builder import STEER_MARKER_CLOSE, STEER_MARKER_OPEN
    except Exception:
        return ""
    start = text.find(STEER_MARKER_OPEN)
    if start == -1:
        return ""
    start += len(STEER_MARKER_OPEN)
    end = text.find(STEER_MARKER_CLOSE, start)
    if end == -1:
        return ""
    return text[start:end].strip()


def _auto_recall_entry(message: Any) -> Optional[tuple[str, str]]:
    if not isinstance(message, dict):
        return None
    if message.get("hermes_private_turn"):
        return None
    if message.get("_compressed_summary"):
        return None
    role = message.get("role")
    if role == "tool":
        steer = _clean_auto_recall_text(_extract_steer_text(message))
        return ("User", steer) if steer else None

    text = _clean_auto_recall_text(_message_text(message))
    if not text:
        return None
    if role == "user":
        if (
            text.startswith("/")
            or text.startswith(_SYNTHETIC_USER_PREFIXES)
            or any(key.endswith("_synthetic") and value for key, value in message.items())
        ):
            return None
        return "User", text
    if role == "assistant" and not message.get("tool_calls"):
        return "Assistant", text
    return None


def _clean_auto_recall_text(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    cleaned = _MEMORY_CONTEXT_RE.sub(" ", text)
    cleaned = _CODE_BLOCK_RE.sub(" ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _head_tail(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    if max_chars <= 1:
        return text[:max_chars]
    head = max_chars // 2
    return text[:head] + "…" + text[-(max_chars - head - 1):]


def build_auto_recall_query(
    current_user_message: str,
    prior_messages: List[Dict[str, Any]],
    *,
    max_chars: int = 2100,
) -> str:
    """Build an automatic recall query from recent semantic conversation text."""
    if max_chars <= 0:
        return ""
    current = _head_tail(_clean_auto_recall_text(current_user_message), min(800, max_chars))
    current_section = f"Current user message:\n{current}"
    if len(current_section) >= max_chars:
        return _head_tail(current_section, max_chars)

    context_prefix = "\n\nRecent conversation context:\n"
    available_total = max_chars - len(current_section) - len(context_prefix)
    remaining_content = 1200
    selected_reversed: List[tuple[str, str]] = []
    rendered_size = 0

    for message in reversed(prior_messages):
        entry = _auto_recall_entry(message)
        if entry is None:
            continue
        role, text = entry
        separator_size = 2 if selected_reversed else 0
        label_size = len(role) + 2  # ``Role:\n``
        available_for_text = available_total - rendered_size - separator_size - label_size
        excerpt_budget = min(remaining_content, available_for_text)
        if excerpt_budget <= 0:
            break
        excerpt = _head_tail(text, excerpt_budget)
        selected_reversed.append((role, excerpt))
        rendered_size += separator_size + label_size + len(excerpt)
        remaining_content -= len(excerpt)
        if len(excerpt) < len(text) or remaining_content <= 0:
            break

    if not selected_reversed:
        return current_section

    chronological = reversed(selected_reversed)
    rendered_context = "\n\n".join(
        f"{role}:\n{text}" for role, text in chronological
    )
    return current_section + context_prefix + rendered_context


def is_trivial_prompt(text: Optional[str]) -> bool:
    """True for empty input, slash commands and bare greetings/acknowledgements (skipping
    recall saves a round-trip and keeps stale context from derailing one-word replies)."""
    stripped = (text or "").strip()
    if not stripped or stripped.startswith("/"):
        return True
    return bool(TRIVIAL_PROMPT_RE.match(stripped)) or bool(_SYMBOL_ONLY_RE.fullmatch(stripped))


class MemoryProvider(ABC):
    """Abstract base class for memory providers."""

    # Providers that durably checkpoint every successful on_pre_compress() set this to
    # PRE_COMPRESS_CHECKPOINT_API_VERSION; 1 = best-effort legacy.
    pre_compress_checkpoint_api_version = 1

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this provider (e.g. 'builtin', 'honcho', 'hindsight')."""

    # -- Core lifecycle (implement these) ------------------------------------

    @abstractmethod
    def is_available(self) -> bool:
        """Configured, credentialed and ready? Gates activation; check config/deps only, no network."""

    @abstractmethod
    def initialize(self, session_id: str, **kwargs) -> None:
        """Initialize once at agent startup (connections, resources, threads).

        kwargs always include ``hermes_home`` (profile-scoped storage; never hardcode
        ``~/.hermes``) and ``platform``; may include ``agent_context`` ("primary" |
        "subagent" | "cron" | "flush" — skip writes for non-primary contexts),
        ``agent_identity``, ``agent_workspace``, ``parent_session_id``, ``user_id``, ``user_id_alt``.
        """

    def unavailable_reason(self) -> str:
        """User-facing hint for the "provider unavailable" warning (``initialize()`` never runs then)."""
        return ""

    def system_prompt_block(self) -> str:
        """STATIC system-prompt text; "" to skip. Recalled context goes through prefetch(), not here."""
        return ""

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Formatted recall context for the upcoming turn ("" if none). Must be fast — recall
        in the background and return cached results; ``session_id`` scopes concurrent sessions."""
        return ""

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """Queue a background recall after each turn; prefetch() consumes it next turn."""

    def recall_status(self) -> Optional[RecallStatus]:
        """What the most recent :meth:`prefetch` injected (``None`` = no indicator). Must reflect
        only the LAST prefetch, never a stale prior count."""
        return None

    def sync_turn(
        self, user_content: str, assistant_content: str, *,
        session_id: str = "", messages: Optional[List[Dict[str, Any]]] = None,
        turn_author: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Persist a completed turn (non-blocking). ``messages`` is the OpenAI-style list so far.
        ``turn_author`` (``{"id", "name", "is_bot"}``) is who wrote the user side; the manager sends it only to signatures that accept it."""

    @abstractmethod
    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """OpenAI function-calling schemas ({"name", "description", "parameters"}); [] if none."""

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        """Handle one of this provider's tools; must return a JSON string."""
        raise NotImplementedError(f"Provider {self.name} does not handle tool {tool_name}")

    def shutdown(self) -> None:
        """Clean shutdown — flush queues, close connections."""

    # -- Optional hooks (override to opt in) ---------------------------------

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        """Per-turn tick. kwargs may include remaining_tokens, model, platform, tool_count, author_id, author_name,
        author_is_bot. The author trio names who wrote THIS turn (None, None, False without one): a shared session
        carries several participants, so a provider keying durable state on identity must read it per turn."""

    def identity_signature(self) -> Dict[str, Any]:
        """Identity-mapping values that must bust a cached gateway agent when they change (writer identity, alias
        tables, session-name prefixing). Provider-namespaced keys, JSON-serializable values. The gateway calls this
        on an uninitialized instance on every inbound message, so keep it cheap and read-only."""
        return {}

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """End-of-session extraction; fires only at real session boundaries, never per-turn."""

    def on_session_switch(
        self, new_session_id: str, *, parent_session_id: str = "", reset: bool = False, rewound: bool = False, **kwargs,
    ) -> None:
        """session_id reassigned mid-process (/resume, /branch, /reset, /new, compression)
        without teardown: rebind per-session state so later writes land in the right record.
        ``reset`` is True only for a genuinely new conversation (flush buffers); ``rewound``:
        same id but the transcript was truncated."""

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        """Extract insights from ``messages`` about to be compressed, fed into the summary prompt."""
        return ""

    def on_delegation(self, task: str, result: str, *, child_session_id: str = "", **kwargs) -> None:
        """PARENT-side observation of a completed delegation (the subagent has no provider session)."""

    def get_config_schema(self) -> List[Dict[str, Any]]:
        """Setup fields for ``hermes memory setup`` ([] if none): ``key``, ``description``,
        optional ``secret`` (goes to .env), ``required``, ``default``, ``choices``, ``type``
        (text | integer | number | boolean), ``minimum``/``maximum``/``step``, ``url``,
        ``env_var`` (explicit secret env var; default auto-generated)."""
        return []

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        """Write non-secret setup ``values`` to the provider's native config. Plugins MUST either
        override this or use only env vars (every schema field carrying ``env_var``)."""

    def on_memory_write(self, action: str, target: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Mirror a built-in memory-tool write (``action``: add | replace | remove; ``target``:
        memory | user; ``metadata``: provenance such as write_origin, session_id, tool_name)."""

    def backup_paths(self) -> List[str]:
        """Absolute paths of provider state OUTSIDE HERMES_HOME for ``hermes backup``/``import``
        (paths outside the home dir are skipped). MUST work without ``initialize()`` or network."""
        return []
