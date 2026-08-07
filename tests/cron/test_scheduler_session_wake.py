from __future__ import annotations

import asyncio
import threading
from contextlib import contextmanager
from unittest.mock import AsyncMock, patch

import pytest

from cron import scheduler
from cron.scheduler_provider import InProcessCronScheduler
from gateway.config import Platform


@contextmanager
def running_loop():
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    try:
        yield loop
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=2)
        loop.close()


def session_wake_job(**overrides):
    job = {
        "id": "heartbeat-job",
        "name": "Rocky heartbeat",
        "prompt": "Check in on the current session",
        "session_wake": True,
        "next_run_at": "2026-08-08T10:00:00+00:00",
        "origin": {
            "platform": "telegram",
            "chat_id": "123456",
            "chat_name": "Jonas",
            "chat_type": "dm",
            "user_id": "789",
            "user_name": "jonas",
            "thread_id": "42",
            "profile": "default",
        },
    }
    job.update(overrides)
    return job


def run_with_bookkeeping_patched(job, *, adapters, loop):
    with patch.object(scheduler, "claim_dispatch", return_value=True), \
         patch.object(scheduler, "create_execution", return_value={"id": "exec-1"}), \
         patch.object(scheduler, "mark_execution_running") as mark_running, \
         patch.object(scheduler, "mark_job_run") as mark_run, \
         patch.object(scheduler, "finish_execution") as finish, \
         patch.object(scheduler, "run_job") as isolated_run, \
         patch.object(scheduler, "_deliver_result") as normal_delivery:
        result = scheduler.run_one_job(job, adapters=adapters, loop=loop)
    return result, mark_running, mark_run, finish, isolated_run, normal_delivery


def test_session_wake_prompt_file_is_read_fresh_and_bounded(tmp_path):
    prompt_file = tmp_path / "HEARTBEAT.md"
    job = session_wake_job(prompt_file=str(prompt_file), prompt="wrapper")

    prompt_file.write_text("first stance", encoding="utf-8")
    first_prompt = scheduler._session_wake_prompt(job)
    assert "[HEARTBEAT.md]\nfirst stance\n[END HEARTBEAT.md]" in first_prompt
    prompt_file.write_text("second stance", encoding="utf-8")
    second_prompt = scheduler._session_wake_prompt(job)
    assert "[HEARTBEAT.md]\nsecond stance\n[END HEARTBEAT.md]" in second_prompt

    prompt_file.write_bytes(b"x" * (16 * 1024 + 1))
    with pytest.raises(RuntimeError, match="16384-byte"):
        scheduler._session_wake_prompt(job)


def test_session_wake_prompt_file_fails_closed_when_missing_or_invalid(tmp_path):
    missing = session_wake_job(prompt_file=str(tmp_path / "missing.md"))
    with pytest.raises(RuntimeError, match="missing"):
        scheduler._session_wake_prompt(missing)

    invalid = tmp_path / "invalid.md"
    invalid.write_bytes(b"\xff")
    with pytest.raises(RuntimeError, match="UTF-8"):
        scheduler._session_wake_prompt(
            session_wake_job(prompt_file=str(invalid))
        )


def test_session_wake_dispatches_to_exact_origin_without_isolated_cron_agent():
    adapter = AsyncMock()
    job = session_wake_job()

    with running_loop() as loop:
        result, mark_running, mark_run, finish, isolated_run, normal_delivery = (
            run_with_bookkeeping_patched(
                job,
                adapters={Platform.TELEGRAM: adapter},
                loop=loop,
            )
        )

    assert result is True
    isolated_run.assert_not_called()
    normal_delivery.assert_not_called()
    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.text == job["prompt"]
    assert event.internal is True
    assert event.source.platform is Platform.TELEGRAM
    assert event.source.chat_id == "123456"
    assert event.source.chat_name == "Jonas"
    assert event.source.chat_type == "dm"
    assert event.source.user_id == "789"
    assert event.source.user_name == "jonas"
    assert event.source.thread_id == "42"
    assert event.source.profile == "default"
    assert event.metadata == {
        "session_wake": True,
        "delivery_id": "heartbeat-job:2026-08-08T10:00:00+00:00",
        "coalesce_key": "cron:heartbeat-job",
        "display_kind": "hidden",
        "skip_external_memory_sync": True,
        "event_kind": "cron_session_wake",
    }
    mark_running.assert_called_once_with("exec-1")
    mark_run.assert_called_once_with("heartbeat-job", True, None)
    finish.assert_called_once_with(
        "exec-1",
        success=True,
        error=None,
        delivery_outcome="accepted",
    )


@pytest.mark.parametrize(
    ("job_change", "adapters", "loop_factory", "error_fragment"),
    [
        ({"origin": None}, {}, lambda: None, "origin"),
        ({}, {}, lambda: None, "adapter"),
        ({}, {Platform.TELEGRAM: AsyncMock()}, lambda: None, "loop"),
    ],
)
def test_session_wake_missing_runtime_dependency_is_failed_run(
    job_change, adapters, loop_factory, error_fragment
):
    job = session_wake_job(**job_change)
    loop = loop_factory()

    result, _, mark_run, finish, isolated_run, normal_delivery = (
        run_with_bookkeeping_patched(job, adapters=adapters, loop=loop)
    )

    assert result is False
    isolated_run.assert_not_called()
    normal_delivery.assert_not_called()
    mark_run.assert_called_once()
    assert mark_run.call_args.args[:2] == (job["id"], False)
    assert error_fragment in mark_run.call_args.args[2].lower()
    assert finish.call_args.kwargs["success"] is False
    assert error_fragment in finish.call_args.kwargs["error"].lower()


def test_external_provider_preserves_claimed_scheduled_instant_for_delivery_id():
    before_claim = session_wake_job(next_run_at="2026-08-08T10:00:00+00:00")
    after_claim = session_wake_job(
        next_run_at="2026-08-08T11:00:00+00:00",
        fire_claim={"at": "2026-08-08T10:00:01+00:00", "by": "node-a"},
    )

    with patch("cron.jobs.get_job", side_effect=[before_claim, after_claim]), \
         patch("cron.jobs.claim_job_for_fire", return_value=True), \
         patch("cron.executions.create_execution", return_value={"id": "exec-1"}), \
         patch("cron.scheduler.run_one_job", return_value=True) as run_one:
        result = InProcessCronScheduler().fire_due("heartbeat-job")

    assert result is True
    dispatched_job = run_one.call_args.args[0]
    assert dispatched_job["_claimed_scheduled_at"] == "2026-08-08T10:00:00+00:00"


def test_session_wake_stopped_loop_is_a_clear_failed_run():
    adapter = AsyncMock()
    loop = asyncio.new_event_loop()
    try:
        result, _, mark_run, finish, isolated_run, _ = run_with_bookkeeping_patched(
            session_wake_job(),
            adapters={Platform.TELEGRAM: adapter},
            loop=loop,
        )
    finally:
        loop.close()

    assert result is False
    isolated_run.assert_not_called()
    adapter.handle_message.assert_not_awaited()
    assert "loop is not running" in mark_run.call_args.args[2].lower()
    assert "loop is not running" in finish.call_args.kwargs["error"].lower()


def test_session_wake_rejected_by_adapter_is_not_marked_successful():
    adapter = AsyncMock()
    adapter.handle_message.side_effect = RuntimeError("queue rejected")

    with running_loop() as loop:
        result, _, mark_run, finish, isolated_run, normal_delivery = (
            run_with_bookkeeping_patched(
                session_wake_job(name="Ambient nudge"),
                adapters={Platform.TELEGRAM: adapter},
                loop=loop,
            )
        )

    assert result is False
    isolated_run.assert_not_called()
    normal_delivery.assert_not_called()
    mark_run.assert_called_once()
    assert mark_run.call_args.args[1] is False
    assert "queue rejected" in mark_run.call_args.args[2]
    assert finish.call_args.kwargs["success"] is False


def test_session_wake_event_kind_is_independent_of_job_name():
    adapter = AsyncMock()

    with running_loop() as loop:
        result, *_ = run_with_bookkeeping_patched(
            session_wake_job(name="Ambient nudge"),
            adapters={Platform.TELEGRAM: adapter},
            loop=loop,
        )

    assert result is True
    event = adapter.handle_message.await_args.args[0]
    assert event.metadata["event_kind"] == "cron_session_wake"


def test_normal_cron_job_keeps_isolated_execution_and_result_delivery():
    job = session_wake_job(session_wake=False, deliver="origin")

    with patch.object(scheduler, "claim_dispatch", return_value=True), \
         patch.object(scheduler, "create_execution", return_value={"id": "exec-1"}), \
         patch.object(scheduler, "mark_execution_running"), \
         patch.object(scheduler, "save_job_output", return_value="output.md"), \
         patch.object(scheduler, "mark_job_run") as mark_run, \
         patch.object(scheduler, "finish_execution") as finish, \
         patch.object(
             scheduler,
             "run_job",
             return_value=(True, "full output", "normal result", None),
         ) as isolated_run, \
         patch.object(scheduler, "_deliver_result", return_value=None) as normal_delivery:
        result = scheduler.run_one_job(job, adapters={}, loop=None)

    assert result is True
    isolated_run.assert_called_once()
    normal_delivery.assert_called_once_with(
        job,
        "normal result",
        adapters={},
        loop=None,
    )
    mark_run.assert_called_once_with(job["id"], True, None, delivery_error=None)
    assert finish.call_args.kwargs["success"] is True
