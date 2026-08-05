"""
Streamlit chat UI for the Airport Investment Intelligence Agent.

What this file does: renders a simple chat window, keeps one
`AirportIntelligenceAgent` per browser session, and - for every assistant
reply - shows a "Data source trace" panel listing exactly which tools
were called and, for each one, every data source that was attempted: a
real live API that succeeded, a real live API that failed and was skipped
in favor of the next one, or (only as a last resort) an LLM estimate.

Why it exists: the assignment requires a chat interface, and the project's
data policy (live sources first, LLM only as a last resort, never silent)
needs to be *visible*, not just true in the code. Rendering the trace
directly from the tool-call data - rather than trusting the model to
narrate it faithfully every time - makes that policy verifiable by the
user on every single answer.

How data moves: user types in the chat box -> this file calls
`agent.send_message(text)` -> the orchestrator runs its tool-calling loop
against `src/agent/tools.py` and returns an `AgentReply` (final text plus
every tool call's resolution trace) -> the text is rendered as the chat
message, and the trace is rendered underneath in an expander.

Run with: streamlit run src/ui/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# `streamlit run src/ui/app.py` only puts this file's own directory on
# sys.path, not the project root - so the `src.*` absolute imports below
# would fail without this. Adding the root here keeps `streamlit run
# src/ui/app.py` working from any working directory, with no PYTHONPATH
# setup required (matters for both local runs and the Docker CMD).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from src.agent.orchestrator import AgentReply, AirportIntelligenceAgent, ToolCallRecord
from src.config import llm_settings

st.set_page_config(page_title="Airport Investment Intelligence Agent", page_icon="✈️", layout="centered")

EXAMPLE_QUESTIONS = [
    "Which airports in New England are strong candidates for terminal expansion?",
    "Compare LA and Santa Ana airport congestion levels.",
    "What is the percentage of long haul flights out of Anchorage airport?",
    "What is the unmet flight demand in SFO airport and why?",
]

STATUS_ICON = {"success": "✅", "failed": "❌"}
TIER_LABEL = {"primary": "Primary API", "secondary": "Secondary API", "llm_fallback": "LLM last resort"}


def _get_agent() -> AirportIntelligenceAgent:
    """Get (or create) the one agent instance for this browser session."""
    if "agent" not in st.session_state:
        st.session_state.agent = AirportIntelligenceAgent()
    return st.session_state.agent


def _get_chat_history() -> list[dict]:
    """Get (or create) the list of {role, content, tool_calls} dicts rendered in the chat window."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    return st.session_state.chat_history


def _render_trace(tool_calls: list[ToolCallRecord]) -> None:
    """Render the 'which data source actually answered this' panel for one assistant turn."""
    if not tool_calls:
        return
    with st.expander(f"🔎 Data source trace ({len(tool_calls)} tool call{'s' if len(tool_calls) != 1 else ''})"):
        for call in tool_calls:
            args_str = ", ".join(f"{k}={v}" for k, v in call.arguments.items()) or "(defaults)"
            st.markdown(f"**`{call.tool_name}`**({args_str})")
            if not call.resolution_trace:
                st.caption("No resolution trace returned.")
                continue
            for step in call.resolution_trace:
                icon = STATUS_ICON.get(step["status"], "•")
                tier = TIER_LABEL.get(step["tier"], step["tier"])
                line = f"{icon} *{tier}* — {step['source_name']}"
                if step["status"] == "failed" and step.get("detail"):
                    line += f" — _{step['detail']}_"
                st.markdown(line)
            st.divider()


def main() -> None:
    st.title("✈️ Airport Investment Intelligence Agent")
    st.caption(
        "Ask about terminal-expansion candidates, congestion, long-haul share, or unmet demand. "
        "All numbers come from deterministic scoring tools fed by real live public data - never "
        "the language model - with an LLM estimate used only if every real source fails, and "
        "always shown in the trace below each answer."
    )

    if not llm_settings.api_key:
        st.warning(
            "OPENROUTER_API_KEY is not set. Copy .env.example to .env and add your key before chatting. "
            "The deterministic analytics still work without a key (see tests/test_kpis.py), but the "
            "chat agent needs an LLM to reason and call tools.",
            icon="⚠️",
        )

    with st.sidebar:
        st.subheader("Model")
        st.write(f"`{llm_settings.model_name}` via OpenRouter")
        st.subheader("Try asking")
        for q in EXAMPLE_QUESTIONS:
            st.markdown(f"- {q}")
        if st.button("Reset conversation"):
            st.session_state.pop("agent", None)
            st.session_state.pop("chat_history", None)
            st.rerun()

    history = _get_chat_history()
    for turn in history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            if turn["role"] == "assistant":
                _render_trace(turn.get("tool_calls", []))

    user_text = st.chat_input("Ask about an airport investment opportunity...")
    if not user_text:
        return

    history.append({"role": "user", "content": user_text, "tool_calls": []})
    with st.chat_message("user"):
        st.markdown(user_text)

    agent = _get_agent()
    with st.chat_message("assistant"):
        with st.spinner("Calling real data source APIs and reasoning..."):
            try:
                reply: AgentReply = agent.send_message(user_text)
            except Exception as exc:  # noqa: BLE001 - surface the error in the chat, don't crash the app
                reply = AgentReply(
                    text=(
                        f"Sorry, something went wrong calling the LLM: {exc}\n\n"
                        "Check that OPENROUTER_API_KEY is set and valid in your .env file."
                    )
                )
        st.markdown(reply.text)
        _render_trace(reply.tool_calls)
    history.append({"role": "assistant", "content": reply.text, "tool_calls": reply.tool_calls})


if __name__ == "__main__":
    main()
