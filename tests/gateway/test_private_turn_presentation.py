"""Private gateway turns preserve provenance for automatic recall exclusion."""


def test_private_turn_marks_only_its_user_assistant_and_tool_messages():
    from run_agent import _mark_private_turn_messages

    history = {"role": "assistant", "content": "ordinary history"}
    private_user = {"role": "user", "content": "private wake"}
    private_assistant = {"role": "assistant", "content": "private result"}
    private_tool = {"role": "tool", "content": "private tool result"}
    messages = [history, private_user, private_assistant, private_tool]

    _mark_private_turn_messages(messages, start_index=1)

    assert "hermes_private_turn" not in history
    assert all(message["hermes_private_turn"] is True for message in messages[1:])