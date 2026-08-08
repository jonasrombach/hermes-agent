"""Plugin-safe delivery of trusted, non-conversational ambient turns.

Plugins receive :class:`AmbientTurnService`, never a gateway runner or adapter.
The gateway publishes its live runtime only after available adapters are
initialized; this module uses that shared wake runtime solely to resolve an explicit,
persisted :class:`~gateway.session.SessionSource` to its owning adapter.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import re
from typing import Any, Mapping

from gateway.session import SessionSource
from gateway.wake import deliver_wake, get_wake_runtime

_MAX_TEXT_BYTES = 16 * 1024
_MAX_ID_LENGTH = 160
_MAX_SOURCE_LABEL_LENGTH = 160
_EVENT_KIND_RE = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,63}$")
def resolve_ambient_adapter(source: SessionSource, adapters: Mapping[Any, Any], *, profile_adapters: Mapping[str, Mapping[Any, Any]] | None = None) -> Any:
    """Resolve an explicit source without a request-derived fallback.

    A source stamped for a secondary multiplex profile is fail-closed: it must
    use that profile's adapter and can never fall back to the primary adapter.
    """
    if not isinstance(source, SessionSource):
        raise ValueError("ambient delivery requires an explicit SessionSource")
    profile = (source.profile or "").strip()
    if profile and profile != "default":
        profile_map = (profile_adapters or {}).get(profile)
        if profile_map is None:
            return None
        return profile_map.get(source.platform)
    return adapters.get(source.platform)


def _parse_time(value: Any, field: str) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        result = value
    elif isinstance(value, str):
        try:
            result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"{field} must be an ISO-8601 timestamp") from exc
    else:
        raise ValueError(f"{field} must be a datetime or ISO-8601 timestamp")
    if result.tzinfo is None:
        raise ValueError(f"{field} must include a timezone")
    return result.astimezone(timezone.utc)


def _bounded_string(value: Any, field: str, *, required: bool = False, limit: int = _MAX_ID_LENGTH) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    result = value.strip()
    if required and not result:
        raise ValueError(f"{field} must be non-empty")
    if len(result) > limit:
        raise ValueError(f"{field} is too long")
    return result


class AmbientTurnService:
    """Host-owned facade plugins use to enqueue hidden ambient turns."""

    def __init__(self, plugin_id: str):
        self._plugin_id = _bounded_string(plugin_id, "plugin_id", required=True)

    async def deliver(
        self,
        *,
        source: Mapping[str, Any] | SessionSource,
        text: str,
        event_kind: str,
        delivery_id: str,
        coalesce_key: str | None = None,
        occurred_at: datetime | str | None = None,
        expires_at: datetime | str | None = None,
        source_label: str | None = None,
    ) -> None:
        """Deliver one event through the shared internal wake path.

        Errors are explicit so plugins can retain and retry their event.
        """
        if not isinstance(text, str) or not text:
            raise ValueError("text must be non-empty")
        if len(text.encode("utf-8")) > _MAX_TEXT_BYTES:
            raise ValueError("text exceeds the 16384-byte limit")
        event_kind = _bounded_string(event_kind, "event_kind", required=True, limit=64)
        if not _EVENT_KIND_RE.fullmatch(event_kind):
            raise ValueError("event_kind has an invalid format")
        delivery_id = _bounded_string(delivery_id, "delivery_id", required=True)
        coalesce_key = _bounded_string(coalesce_key or "", "coalesce_key")
        source_label = _bounded_string(
            source_label or "", "source_label", limit=_MAX_SOURCE_LABEL_LENGTH
        )
        occurred = _parse_time(occurred_at, "occurred_at")
        expires = _parse_time(expires_at, "expires_at")
        if expires is not None and expires <= datetime.now(timezone.utc):
            raise ValueError("ambient event is expired")
        if occurred is not None and expires is not None and occurred > expires:
            raise ValueError("occurred_at must not be after expires_at")
        if isinstance(source, SessionSource):
            session_source = source
        elif isinstance(source, Mapping):
            try:
                session_source = SessionSource.from_dict(dict(source))
            except Exception as exc:
                raise ValueError(f"source is malformed: {exc}") from exc
        else:
            raise ValueError("source must be an explicit SessionSource mapping")

        runtime = get_wake_runtime()
        if runtime is None:
            raise RuntimeError("Gateway ambient runtime is unavailable")
        runner, loop = runtime
        if getattr(runner, "_draining", False) or not getattr(runner, "_running", False):
            raise RuntimeError("Gateway ambient runtime is stopping")
        if getattr(loop, "is_closed", lambda: True)():
            raise RuntimeError("Gateway ambient runtime event loop is closed")
        if not getattr(loop, "is_running", lambda: False)():
            raise RuntimeError("Gateway ambient runtime event loop is not running")
        adapter = resolve_ambient_adapter(
            session_source,
            getattr(runner, "adapters", {}) or {},
            profile_adapters=getattr(runner, "_profile_adapters", {}) or {},
        )
        if adapter is None:
            raise RuntimeError(
                f"Gateway adapter is unavailable for {session_source.platform.value}"
            )
        metadata: dict[str, Any] = {
            "internal_ambient": True,
            "plugin_id": self._plugin_id,
            "delivery_id": delivery_id,
            "event_kind": event_kind,
        }
        if coalesce_key:
            metadata["coalesce_key"] = f"plugin:{self._plugin_id}:{coalesce_key}"
        if source_label:
            metadata["source_label"] = source_label
        if occurred:
            metadata["occurred_at"] = occurred.isoformat()
        if expires:
            metadata["expires_at"] = expires.isoformat()

        coro = deliver_wake(adapter, text=text, source=session_source, metadata=metadata)
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None
        if current_loop is loop:
            await coro
        else:
            future = asyncio.run_coroutine_threadsafe(coro, loop)
            await asyncio.wrap_future(future)
