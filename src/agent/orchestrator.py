"""
Conversational agent orchestrator (OpenRouter tool-calling loop).

What this file does: owns the chat conversation with the LLM. It sends
the user's question plus the tool schemas from `src/agent/tools.py` to an
OpenRouter-hosted model, executes whichever tools the model asks for,
feeds the results back, and repeats until the model has enough
information to answer in plain English. It also collects, for every tool
call made along the way, the exact `resolution_trace` (which real data
sources were tried, in what order, and whether the LLM had to fill in a
last-resort estimate) so the UI can display it deterministically -
independent of whether the model chooses to mention it in prose.

Why it exists: this is the ONLY file in the project that talks to the
LLM. Keeping it separate from `src/analytics/` (pure math) and
`src/agent/tools.py` (pure plumbing) means the non-deterministic part of
the system - the LLM's reasoning and phrasing - is fully isolated and can
be swapped, mocked, or removed without touching how any score is computed.

How data moves: UI (`src/ui/app.py`) calls `Agent.send_message(text)` ->
this file calls the OpenRouter chat-completions API with the running
message history and `TOOL_SCHEMAS` -> if the model requests tool calls,
`run_tool()` executes them against real analytics/data-source code ->
their JSON results (including each one's resolution_trace) are appended
back into the message history as "tool" messages, and also collected
into the `AgentReply` returned to the caller -> the loop repeats until
the model responds with plain text -> that text, plus the full list of
tool calls and their traces, is returned to the UI.

Context window management: since every call re-sends the full message
history (the chat-completions API is stateless), a long conversation
would otherwise grow without bound. `_compact_history()` runs at the
start of every `send_message()` and applies three layers - see its
docstring: (1) a stable system+tools prefix for provider-side prompt
caching, (2) pruning old tool-call JSON once the turn that needed it is
done, (3) a sliding window beyond `llm_settings.max_history_turns` that
does NOT delete old turns - it folds them into a running LLM-written
summary (`self.conversation_summary`) kept in the system message, so
older context is compressed rather than lost.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from openai import OpenAI

from src.agent.tools import TOOL_SCHEMAS, run_tool
from src.config import llm_settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are an Airport Investment Intelligence Agent for a firm that invests in \
US airport modernization projects. Analysts ask you which airports are strong \
candidates for terminal expansion, and you help them compare congestion, \
long-haul traffic share, and unmet demand across airports.

Ground rules you must always follow:
1. You do not calculate scores, rankings, percentages, or indices yourself. \
Every number in your answer must come from a tool call. If you have not \
called a tool for a number, do not state that number.
2. Always call the most relevant tool(s) before answering a data question. \
Prefer the specific ranking/comparison/index tools over raw data lookups \
when the question maps directly to one of them (expansion-candidate ranking, \
two-airport congestion comparison, long-haul share, unmet demand index). \
These tools work for any US airport(s), not only their documented examples \
(New England expansion, LAX vs SNA, ANC long-haul, SFO unmet demand) - do \
not refuse or hedge just because a question uses different airports than \
those examples.
3. Every tool result includes a `resolution_trace` (or `uses_llm_estimate` \
flags) showing exactly which data source answered each number: a specific \
real government API/dataset (tier 'primary' or 'secondary'), or, only if \
every real source failed, an LLM estimate (tier 'llm_fallback'). ALWAYS \
summarize this in your answer - name the real data source(s) used (e.g. \
"BTS On-Time Performance, May 2026") and explicitly flag any number that \
came from an LLM estimate rather than real data, since that number is far \
less reliable.
4. Explain your reasoning: name the factors that drove a score, not just \
the final number. Investors want to know *why*, not just *what*.
5. Surface each tool result's `assumptions` list in plain English, not as \
jargon - these are real scoping limits (e.g. "based on one month of data," \
"passenger-weighting isn't available for this metric") that affect confidence.
6. Be explicit about uncertainty and scope. This system estimates unmet \
demand and expansion candidacy from public data; it does not replace a \
full engineering or financial feasibility study.
7. Support follow-up questions using the conversation so far - if the user \
asks "what about BDL?" after a New England ranking, you already have that \
context.
8. Keep answers concise and structured (short paragraphs or bullet points). \
Analysts are busy; lead with the answer, then the reasoning, then a short \
"data sources" line.
"""

PRUNED_TOOL_PLACEHOLDER = (
    "[tool result pruned to save context space - the assistant's reply that "
    "followed already summarized what mattered from this result]"
)

SUMMARY_HEADER = "Summary of earlier parts of this conversation (older turns were condensed here to save space):"

SUMMARIZER_SYSTEM_PROMPT = (
    "You maintain a running summary of an ongoing conversation between an airport-investment "
    "analyst and an AI agent, so older turns can be dropped from the raw message history without "
    "losing what was discussed. Be concise - a few sentences, not a retelling. Preserve specific "
    "facts that a follow-up question might reference: airport codes, scores/percentages/indices "
    "given, and conclusions reached. Do not add commentary or preamble; output only the summary text."
)


def _messages_to_plain_text(messages: list[dict]) -> str:
    """
    Render a list of chat messages as plain 'User: ...' / 'Assistant: ...' text for summarization.

    Tool-call plumbing (the assistant's tool_calls request, the tool's raw
    JSON result) is skipped - by the time a turn is old enough to be
    summarized, its tool messages have already been pruned to a
    placeholder (see `_compact_history`), and the assistant's own final
    reply already turned whatever mattered from them into plain English.
    """
    lines = []
    for m in messages:
        role, content = m.get("role"), m.get("content")
        if role == "user" and content:
            lines.append(f"User: {content}")
        elif role == "assistant" and content:
            lines.append(f"Assistant: {content}")
    return "\n".join(lines)


@dataclass
class ToolCallRecord:
    """One tool call made during a single `send_message()` turn, with its full trace."""

    tool_name: str
    arguments: dict
    resolution_trace: list[dict] = field(default_factory=list)


@dataclass
class AgentReply:
    """What `send_message()` returns: the model's final text plus every tool call it made."""

    text: str
    tool_calls: list[ToolCallRecord] = field(default_factory=list)


class AirportIntelligenceAgent:
    """
    Stateful wrapper around one chat conversation with the LLM.

    Create one instance per conversation (e.g. per Streamlit session). It
    keeps the full message history, including tool calls and their
    results, so follow-up questions have full context. Once the
    conversation passes `llm_settings.max_history_turns`, the oldest turns
    are folded into `self.conversation_summary` (see `_compact_history()`)
    rather than deleted, so older context is compressed, not lost.
    """

    def __init__(self, model: str | None = None) -> None:
        if not llm_settings.api_key:
            logger.warning(
                "OPENROUTER_API_KEY is not set. The agent will fail on send_message() "
                "until a key is provided in the environment or .env file."
            )
        self.client = OpenAI(base_url=llm_settings.base_url, api_key=llm_settings.api_key)
        self.model = model or llm_settings.model_name
        self.conversation_summary: str = ""
        self.messages: list[dict] = [self._build_system_message()]

    def reset(self) -> None:
        """Start a brand new conversation, discarding all history and the running summary."""
        self.conversation_summary = ""
        self.messages = [self._build_system_message()]

    def _build_system_message(self) -> dict:
        """The system message content, with the running summary (if any) appended."""
        content = SYSTEM_PROMPT
        if self.conversation_summary:
            content = f"{content}\n\n{SUMMARY_HEADER}\n{self.conversation_summary}"
        return {"role": "system", "content": content}

    def _compact_history(self) -> None:
        """
        Keep the context window bounded across a long conversation, in three layers.

        A "turn" here means one user message plus everything the model did
        in response to it, up to the next user message. Called once at the
        start of every `send_message()`, before that call's new user
        message is appended - so it only ever touches already-completed
        turns, never the one about to be built, which still needs its own
        tool results intact while the model reasons over them.

        1. Cache-friendly structure: the system message and the `tools`
           schema (see `TOOL_SCHEMAS`) are unchanged between calls unless
           the running summary itself just updated - which only happens
           when layer 3 fires, not on every turn - so most calls still see
           a byte-identical prefix a provider's prompt cache can reuse.
        2. Tool message pruning: a tool call's raw JSON result is only
           useful for the model to reason over *within the turn that
           fetched it*. Once a turn is complete, its tool messages are
           replaced with a small placeholder - the assistant's own reply
           already extracted whatever mattered, so a payload that can be
           large (e.g. a 6-airport ranking with a full resolution trace
           per airport) doesn't round-trip on every later call.
        3. Sliding window: once there are more than `max_history_turns`
           completed turns, the oldest ones are folded into
           `self.conversation_summary` via one LLM call (not deleted) and
           removed from the raw message list - the summary is kept in the
           system message instead, so older context is compressed, not lost.
        """
        if len(self.messages) <= 1:
            return

        rest = self.messages[1:]

        for m in rest:
            if m.get("role") == "tool" and m.get("content") != PRUNED_TOOL_PLACEHOLDER:
                m["content"] = PRUNED_TOOL_PLACEHOLDER

        turn_starts = [i for i, m in enumerate(rest) if m.get("role") == "user"]
        max_turns = llm_settings.max_history_turns
        if len(turn_starts) > max_turns:
            cutoff = turn_starts[-max_turns]
            dropped, rest = rest[:cutoff], rest[cutoff:]
            self.conversation_summary = self._summarize_dropped_turns(dropped)

        self.messages = [self._build_system_message()] + rest

    def _summarize_dropped_turns(self, dropped_messages: list[dict]) -> str:
        """Ask the LLM to fold turns aging out of the window into the running summary."""
        turns_text = _messages_to_plain_text(dropped_messages)
        if not turns_text.strip():
            return self.conversation_summary

        user_prompt = (
            f"Existing summary so far (may be empty):\n{self.conversation_summary or '(none yet)'}\n\n"
            f"Conversation turns to fold into it:\n{turns_text}\n\n"
            "Write one updated, concise summary covering the existing summary plus these new turns."
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SUMMARIZER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                timeout=llm_settings.request_timeout_seconds,
                extra_headers={
                    "HTTP-Referer": llm_settings.site_url,
                    "X-Title": llm_settings.site_name,
                },
            )
            summary = response.choices[0].message.content
            return summary.strip() if summary else self.conversation_summary
        except Exception as exc:  # noqa: BLE001 - a failed summary must not break the conversation
            logger.warning("Conversation summarization failed (%s); keeping the previous summary", exc)
            return self.conversation_summary

    def send_message(self, user_text: str) -> AgentReply:
        """
        Send one user message, run the tool-calling loop, and return the final reply.

        Internally this may call the LLM more than once (once per round of
        tool calls), capped by `llm_settings.max_tool_call_rounds` so a
        confused model can never loop forever. Every tool call made along
        the way - and its full resolution trace - is collected onto the
        returned `AgentReply`, independent of whether the model's own
        prose mentions it.
        """
        self._compact_history()
        self.messages.append({"role": "user", "content": user_text})
        tool_call_records: list[ToolCallRecord] = []

        for _ in range(llm_settings.max_tool_call_rounds):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
                timeout=llm_settings.request_timeout_seconds,
                extra_headers={
                    "HTTP-Referer": llm_settings.site_url,
                    "X-Title": llm_settings.site_name,
                },
            )
            message = response.choices[0].message
            self.messages.append(_message_to_dict(message))

            if not message.tool_calls:
                return AgentReply(text=message.content or "", tool_calls=tool_call_records)

            for tool_call in message.tool_calls:
                arguments, result = self._execute_tool_call(tool_call)
                tool_call_records.append(
                    ToolCallRecord(
                        tool_name=tool_call.function.name,
                        arguments=arguments,
                        resolution_trace=result.get("resolution_trace", []),
                    )
                )
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result, default=str),
                    }
                )

        return AgentReply(
            text=(
                "I made too many tool calls without reaching a final answer. "
                "Please rephrase your question or ask something more specific."
            ),
            tool_calls=tool_call_records,
        )

    @staticmethod
    def _execute_tool_call(tool_call) -> tuple[dict, dict]:
        """Parse one tool call's arguments and run it, never raising back into the loop."""
        try:
            arguments = json.loads(tool_call.function.arguments or "{}")
        except json.JSONDecodeError as exc:
            return {}, {"error": f"Could not parse arguments for '{tool_call.function.name}': {exc}"}
        return arguments, run_tool(tool_call.function.name, arguments)


def _message_to_dict(message) -> dict:
    """Convert an OpenAI SDK message object into the plain dict shape the API expects back."""
    entry: dict = {"role": message.role, "content": message.content or ""}
    if message.tool_calls:
        entry["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in message.tool_calls
        ]
    return entry
