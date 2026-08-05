"""
Tests for the conversational tool-calling loop in src/agent/orchestrator.py.

These tests never call a real LLM. They monkeypatch the OpenAI client's
`chat.completions.create` method with fake, scripted responses shaped
like the real SDK's return objects, so we can verify the loop's control
flow: does it execute the tool the model asked for, feed the result back
correctly, collect the resolution trace onto the `AgentReply`, and stop
when the model gives a final answer (or when it hits the round limit)?

`run_tool` itself is also monkeypatched here (rather than letting it hit
real analytics/data-source code) so these tests stay focused on
orchestration, not on any particular tool's behavior - that's covered in
tests/test_tools.py and tests/test_kpis.py.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.agent.orchestrator import PRUNED_TOOL_PLACEHOLDER, SUMMARY_HEADER, AirportIntelligenceAgent
from src.config import llm_settings


def _fake_tool_call(call_id: str, name: str, arguments: dict) -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


def _fake_response(role: str, content: str | None, tool_calls: list | None = None) -> SimpleNamespace:
    message = SimpleNamespace(role=role, content=content, tool_calls=tool_calls or None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


@pytest.fixture
def agent() -> AirportIntelligenceAgent:
    return AirportIntelligenceAgent(model="test/model")


def test_direct_answer_without_any_tool_call(agent):
    agent.client.chat.completions.create = MagicMock(
        return_value=_fake_response("assistant", "Boston Logan looks strongest.")
    )

    reply = agent.send_message("Which New England airport looks strongest?")

    assert reply.text == "Boston Logan looks strongest."
    assert reply.tool_calls == []
    assert agent.client.chat.completions.create.call_count == 1


def test_executes_requested_tool_and_collects_resolution_trace(agent):
    tool_call = _fake_tool_call("call_1", "get_long_haul_flight_share", {"airport": "ANC"})
    first_response = _fake_response("assistant", None, tool_calls=[tool_call])
    second_response = _fake_response("assistant", "About 5% of ANC departures are long-haul.")
    agent.client.chat.completions.create = MagicMock(side_effect=[first_response, second_response])

    fake_tool_result = {
        "airport": "ANC",
        "long_haul_flight_share_pct": 4.9,
        "resolution_trace": [
            {"tier": "primary", "source_name": "BTS On-Time Performance", "status": "success", "detail": ""}
        ],
    }
    with patch("src.agent.orchestrator.run_tool", return_value=fake_tool_result) as mock_run_tool:
        reply = agent.send_message("What share of ANC flights are long haul?")

    assert reply.text == "About 5% of ANC departures are long-haul."
    assert agent.client.chat.completions.create.call_count == 2
    mock_run_tool.assert_called_once_with("get_long_haul_flight_share", {"airport": "ANC"})

    assert len(reply.tool_calls) == 1
    record = reply.tool_calls[0]
    assert record.tool_name == "get_long_haul_flight_share"
    assert record.arguments == {"airport": "ANC"}
    assert record.resolution_trace == fake_tool_result["resolution_trace"]

    tool_messages = [m for m in agent.messages if m.get("role") == "tool"]
    assert len(tool_messages) == 1
    tool_payload = json.loads(tool_messages[0]["content"])
    assert tool_payload["airport"] == "ANC"


def test_conversation_history_grows_across_turns(agent):
    agent.client.chat.completions.create = MagicMock(return_value=_fake_response("assistant", "Sure."))

    agent.send_message("Hello")
    length_after_first = len(agent.messages)
    agent.send_message("Follow-up question")

    assert len(agent.messages) > length_after_first


def test_reset_clears_history_back_to_system_prompt_only(agent):
    agent.client.chat.completions.create = MagicMock(return_value=_fake_response("assistant", "Sure."))
    agent.send_message("Hello")

    agent.reset()

    assert len(agent.messages) == 1
    assert agent.messages[0]["role"] == "system"


def test_stops_after_max_rounds_if_model_never_finishes(agent):
    tool_call = _fake_tool_call("call_x", "get_airport_weather", {"airport": "SFO"})
    agent.client.chat.completions.create = MagicMock(
        return_value=_fake_response("assistant", None, tool_calls=[tool_call])
    )

    with patch("src.agent.orchestrator.run_tool", return_value={"resolution_trace": []}):
        reply = agent.send_message("Keep looping forever")

    assert "too many tool calls" in reply.text.lower()
    assert agent.client.chat.completions.create.call_count == llm_settings.max_tool_call_rounds
    assert len(reply.tool_calls) == llm_settings.max_tool_call_rounds


def test_unknown_tool_call_produces_error_payload_instead_of_raising(agent):
    tool_call = _fake_tool_call("call_bad", "not_a_real_tool", {})
    first_response = _fake_response("assistant", None, tool_calls=[tool_call])
    second_response = _fake_response("assistant", "I couldn't find that data.")
    agent.client.chat.completions.create = MagicMock(side_effect=[first_response, second_response])

    error_result = {"error": "Unknown tool 'not_a_real_tool'"}
    with patch("src.agent.orchestrator.run_tool", return_value=error_result):
        reply = agent.send_message("Ask something that triggers a bad tool call")

    assert reply.text == "I couldn't find that data."
    assert reply.tool_calls[0].resolution_trace == []  # error results have no trace


def _fake_turn(index: int) -> list[dict]:
    """One complete turn's worth of messages: user -> assistant(tool_calls) -> tool -> assistant(final)."""
    return [
        {"role": "user", "content": f"question {index}"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": f"call{index}", "type": "function", "function": {"name": "x", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": f"call{index}", "content": f"HEAVY_RAW_JSON_{index}" * 20},
        {"role": "assistant", "content": f"answer {index}"},
    ]


def _patch_max_history_turns(monkeypatch, n: int) -> None:
    # llm_settings is a frozen dataclass, so swap the module-level reference
    # orchestrator.py reads from rather than mutating a field in place.
    import dataclasses

    narrow_settings = dataclasses.replace(llm_settings, max_history_turns=n)
    monkeypatch.setattr("src.agent.orchestrator.llm_settings", narrow_settings)


class TestCompactHistory:
    """Unit tests for the three-layer context management in AirportIntelligenceAgent._compact_history()."""

    def test_system_message_content_starts_as_plain_prompt(self, agent):
        assert agent.messages[0]["role"] == "system"
        assert SUMMARY_HEADER not in agent.messages[0]["content"]

    def test_short_history_is_a_no_op(self, agent):
        original = agent.messages
        agent._compact_history()
        assert agent.messages == original

    def test_completed_turns_tool_messages_get_pruned(self, agent):
        agent.messages += _fake_turn(0) + _fake_turn(1)
        agent._compact_history()
        tool_messages = [m for m in agent.messages if m["role"] == "tool"]
        assert len(tool_messages) == 2
        assert all(m["content"] == PRUNED_TOOL_PLACEHOLDER for m in tool_messages)

    def test_assistant_final_answers_survive_pruning(self, agent):
        agent.messages += _fake_turn(0) + _fake_turn(1)
        agent._compact_history()
        final_answers = [m["content"] for m in agent.messages if m["role"] == "assistant" and m["content"]]
        assert final_answers == ["answer 0", "answer 1"]

    def test_within_window_stays_untouched_in_count(self, agent):
        for i in range(3):
            agent.messages += _fake_turn(i)
        before = len(agent.messages)
        agent._compact_history()
        assert len(agent.messages) == before

    def test_tool_calls_field_on_assistant_message_is_preserved(self, agent):
        # Pruning must only touch role="tool" content, never strip the
        # tool_calls list off the assistant message that requested them -
        # the API requires that structural pairing to stay intact.
        agent.messages += _fake_turn(0)
        agent._compact_history()
        assistant_with_calls = next(m for m in agent.messages if m["role"] == "assistant" and m.get("tool_calls"))
        assert assistant_with_calls["tool_calls"][0]["id"] == "call0"

    def test_sliding_window_summarizes_instead_of_deleting(self, agent, monkeypatch):
        _patch_max_history_turns(monkeypatch, 2)
        for i in range(5):
            agent.messages += _fake_turn(i)
        agent.client.chat.completions.create = MagicMock(
            return_value=_fake_response("assistant", "Turns 0-2 covered ANC and BOS questions.")
        )

        agent._compact_history()

        # Old turns are gone from the raw list...
        user_questions = [m["content"] for m in agent.messages if m["role"] == "user"]
        assert user_questions == ["question 3", "question 4"]
        # ...but folded into the running summary, not just discarded.
        assert agent.conversation_summary == "Turns 0-2 covered ANC and BOS questions."
        assert SUMMARY_HEADER in agent.messages[0]["content"]
        assert "Turns 0-2 covered ANC and BOS questions." in agent.messages[0]["content"]

    def test_summarizer_is_given_the_previous_summary_to_extend(self, agent, monkeypatch):
        _patch_max_history_turns(monkeypatch, 1)
        agent.conversation_summary = "Earlier: discussed LAX vs SNA."
        agent.messages[0] = agent._build_system_message()
        agent.messages += _fake_turn(0) + _fake_turn(1)

        create_mock = MagicMock(return_value=_fake_response("assistant", "Updated summary."))
        agent.client.chat.completions.create = create_mock

        agent._compact_history()

        sent_messages = create_mock.call_args.kwargs["messages"]
        user_prompt = next(m["content"] for m in sent_messages if m["role"] == "user")
        assert "Earlier: discussed LAX vs SNA." in user_prompt
        assert agent.conversation_summary == "Updated summary."

    def test_summarization_failure_keeps_previous_summary_and_does_not_raise(self, agent, monkeypatch):
        _patch_max_history_turns(monkeypatch, 1)
        agent.conversation_summary = "Prior summary survives failure."
        agent.messages[0] = agent._build_system_message()
        agent.messages += _fake_turn(0) + _fake_turn(1)
        agent.client.chat.completions.create = MagicMock(side_effect=RuntimeError("network down"))

        agent._compact_history()  # must not raise

        assert agent.conversation_summary == "Prior summary survives failure."
        # The turns still age out of the raw list even though summarization failed -
        # losing the running summary's update is better than growing unboundedly.
        user_questions = [m["content"] for m in agent.messages if m["role"] == "user"]
        assert user_questions == ["question 1"]


class TestContextCompactionIntegration:
    """Integration-level: does send_message() actually prune between calls?"""

    def test_previous_turns_tool_results_are_pruned_on_next_send_message(self, agent):
        tool_call = _fake_tool_call("call_1", "get_long_haul_flight_share", {"airport": "ANC"})
        responses = [
            _fake_response("assistant", None, tool_calls=[tool_call]),
            _fake_response("assistant", "First answer."),
            _fake_response("assistant", "Second answer, no tools needed."),
        ]
        agent.client.chat.completions.create = MagicMock(side_effect=responses)

        big_result = {"airport": "ANC", "data": "X" * 5000, "resolution_trace": []}
        with patch("src.agent.orchestrator.run_tool", return_value=big_result):
            agent.send_message("What share of ANC flights are long haul?")

        # Before the second call, compaction hasn't run yet - the raw tool
        # JSON from turn 1 is still sitting in history.
        tool_messages_before = [m for m in agent.messages if m.get("role") == "tool"]
        assert any(m["content"] != PRUNED_TOOL_PLACEHOLDER for m in tool_messages_before)

        agent.send_message("Thanks, what about a follow-up with no tools?")

        # Compaction ran at the start of that second call: turn 1's tool
        # result is now pruned, even though turn 1's text answer survives.
        tool_messages_after = [m for m in agent.messages if m.get("role") == "tool"]
        assert all(m["content"] == PRUNED_TOOL_PLACEHOLDER for m in tool_messages_after)
        assistant_texts = [m["content"] for m in agent.messages if m.get("role") == "assistant" and m["content"]]
        assert "First answer." in assistant_texts

    def test_conversation_stays_bounded_over_many_turns(self, agent, monkeypatch):
        _patch_max_history_turns(monkeypatch, 3)
        agent.client.chat.completions.create = MagicMock(
            return_value=_fake_response("assistant", "Sure.")
        )

        for i in range(10):
            agent.send_message(f"question {i}")

        # Compaction runs at the START of send_message, before that call's
        # new turn is appended - so right after a call returns, history
        # holds the trimmed max_turns plus the turn just completed: one
        # more than max_turns, not exactly max_turns. It never grows past
        # that bound, which is what actually matters here.
        user_messages = [m for m in agent.messages if m.get("role") == "user"]
        assert len(user_messages) == 4
        assert [m["content"] for m in user_messages] == ["question 6", "question 7", "question 8", "question 9"]

    def test_old_turns_are_summarized_not_silently_lost(self, agent, monkeypatch):
        _patch_max_history_turns(monkeypatch, 2)
        agent.client.chat.completions.create = MagicMock(
            return_value=_fake_response("assistant", "Covered ANC, BOS, and SFO so far.")
        )

        for i in range(5):
            agent.send_message(f"question {i}")

        # Old turns (well beyond the window) no longer appear as raw
        # messages, but their content lives on in the running summary,
        # which is folded into the system message sent on every call.
        user_messages = [m["content"] for m in agent.messages if m.get("role") == "user"]
        assert "question 0" not in user_messages
        assert agent.conversation_summary == "Covered ANC, BOS, and SFO so far."
        assert "Covered ANC, BOS, and SFO so far." in agent.messages[0]["content"]

    def test_summarizer_only_called_once_window_is_exceeded(self, agent, monkeypatch):
        _patch_max_history_turns(monkeypatch, 10)
        create_mock = MagicMock(return_value=_fake_response("assistant", "Sure."))
        agent.client.chat.completions.create = create_mock

        for i in range(3):
            agent.send_message(f"question {i}")

        # 3 turns, well under the window of 10: one API call per turn for
        # the main reply, zero extra calls for summarization.
        assert create_mock.call_count == 3
        assert agent.conversation_summary == ""
