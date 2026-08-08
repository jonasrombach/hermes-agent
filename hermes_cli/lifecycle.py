"""Hermes lifecycle dispatch for first-party observers and plugins."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine
from typing import Any, List

logger = logging.getLogger(__name__)


class GatewayLifecycleTasks:
    """Small owner for plugin background services tied to one gateway run."""

    def __init__(self) -> None:
        self._tasks: set[asyncio.Task[Any]] = set()
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def create_task(self, awaitable: Coroutine[Any, Any, Any], *, name: str | None = None) -> asyncio.Task[Any]:
        """Start and track one service task on the current gateway loop."""
        if self._closed:
            awaitable.close()
            raise RuntimeError("Gateway lifecycle task owner is closed")
        task = asyncio.create_task(awaitable, name=name)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def cancel_and_wait(self) -> None:
        """Cancel all tracked services and wait until their cleanup completes."""
        self._closed = True
        current = asyncio.current_task()
        while True:
            tasks = [task for task in self._tasks if task is not current and not task.done()]
            if not tasks:
                break
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()


def invoke_hook(hook_name: str, **kwargs: Any) -> List[Any]:
    """Notify first-party observers, then invoke compatibility plugin hooks."""
    try:
        from hermes_cli.observability import observe_lifecycle

        observe_lifecycle(hook_name, **kwargs)
    except Exception:
        logger.warning("Built-in observability hook failed", exc_info=True)

    from hermes_cli import plugins

    return plugins.invoke_hook(hook_name, **kwargs)


async def invoke_hook_async(hook_name: str, **kwargs: Any) -> List[Any]:
    """Notify plugins through a lifecycle hook that may be asynchronous."""
    try:
        from hermes_cli.observability import observe_lifecycle

        observe_lifecycle(hook_name, **kwargs)
    except Exception:
        logger.warning("Built-in observability hook failed", exc_info=True)

    from hermes_cli import plugins

    return await plugins.invoke_hook_async(hook_name, **kwargs)


def has_hook(hook_name: str) -> bool:
    """Return whether a first-party observer or plugin consumes a hook."""
    try:
        from hermes_cli.observability import handles_hook

        if handles_hook(hook_name):
            return True
    except Exception:
        logger.warning("Unable to inspect built-in observability hooks", exc_info=True)

    from hermes_cli import plugins

    return plugins.has_hook(hook_name)


def finalize_session(**kwargs: Any) -> List[Any]:
    """Notify observers and hard-close one core-owned Relay conversation."""
    try:
        from hermes_cli.observability import observe_lifecycle

        observe_lifecycle("on_session_finalize", **kwargs)
    except Exception:
        logger.warning("Built-in observability hook failed", exc_info=True)

    session_id = str(kwargs.get("session_id") or "")
    if session_id:
        try:
            from agent import relay_runtime

            relay_runtime.SESSION_COORDINATOR.finalize_conversation(
                profile_key=relay_runtime.current_profile_key(),
                session_id=session_id,
            )
        except Exception:
            logger.warning("Core Relay session finalization failed", exc_info=True)

    from hermes_cli import plugins

    return plugins.invoke_hook("on_session_finalize", **kwargs)
