# Interview Explainer

This document is written for you, the candidate, to prepare for the technical
interview about this project. It is not user-facing documentation - it is a
study guide. Read `README.md` first for the factual architecture description;
read this for *why* things are built this way and how to talk about it.

---

## 1. The 60-second pitch

"I built a conversational agent that answers airport-investment questions by
having an LLM call deterministic Python tools, never do math itself. The four
scoring formulas from the assignment - New England expansion ranking, LAX/SNA
congestion, ANC long-haul share, SFO unmet demand - live in a `src/analytics/`
layer with zero LLM dependency, unit-tested directly. Every input to those
formulas comes from a real, live public API - verified by directly querying
each one during development, not assumed from documentation - through a
waterfall: try the best real source, try a fallback real source if there is
one, and only if every real source fails, ask the LLM for a structured,
schema-validated estimate as an honest last resort. Every answer shows exactly
which tier answered each number, in the chat UI, independent of whether the
model mentions it in prose."

---

## 2. End-to-end data flow (walk through this out loud)

```
User Query
   -> src/ui/app.py (Streamlit chat_input)
   -> src/agent/orchestrator.py: AirportIntelligenceAgent.send_message()
        - appends user message to conversation history
        - calls OpenRouter chat.completions.create(messages, tools=TOOL_SCHEMAS)
   -> LLM Tool Call
        - model returns a `tool_calls` array, e.g.
          get_unmet_demand_index(airport="SFO")
   -> src/agent/tools.py: run_tool(name, arguments)
        - looks up the Python function for that tool name, calls it
   -> Python Analytics Engine
        - src/analytics/unmet_demand_calculator.py: calculate_unmet_demand()
        - pure math: normalizes inputs, applies fixed weights, no LLM involved
   -> Data Source Waterfall
        - each raw input comes from src/data_sources/*.py via resolve()
        - tries a real live API (e.g. BTS T-100 on data.bts.gov) first
        - tries a second real source if one exists
        - only if every real source fails, asks the LLM for a schema-
          constrained estimate (src/data_sources/llm_fallback_client.py),
          tagged tier="llm_fallback" so it's never confused with real data
   -> LLM Reasoning
        - the structured result (scores + weights + assumptions +
          resolution_trace) is JSON-serialized and appended to the
          conversation as a "tool" role message
        - the orchestrator calls the LLM again; it now reasons over real
          numbers and produces a plain-English answer, instructed to name
          its real sources and flag any LLM-estimated number
   -> UI Response
        - orchestrator returns an AgentReply(text, tool_calls) - not just a
          string - so the UI can render the resolution trace directly from
          structured data, not by parsing the model's prose
        - Streamlit renders the text AND a "Data source trace" panel built
          straight from tool_calls, independent of what the model said
```

The loop can repeat (capped at 5 rounds): if the model needs a second tool, it
calls it, gets another result appended, and only stops once it has enough to
answer in prose.

---

## 3. Why deterministic logic is isolated from the LLM

**The rule:** `src/analytics/*.py` never imports anything from `src/agent/` or
any LLM SDK. `src/agent/orchestrator.py` never contains a formula or an
arithmetic operation beyond parsing JSON. The one narrow, deliberate exception
is `llm_fallback_client.py` - and it lives in `src/data_sources/`, not
`src/agent/`, on purpose: from the analytics layer's point of view, an LLM
estimate is just another data source (a low-reliability one), not a reasoning
step. It's called only when every real API for one specific raw fact has
failed, it's constrained to a Pydantic schema, and it's always tagged
`is_llm_estimate=True` - it never computes a score itself.

**Why this matters, concretely:**

- **Reproducibility / trust.** Given the same live data, the same formula
  always produces the same score.
- **Auditability.** Anyone can open `congestion_comparator.py`, read the
  formula, and verify it - no need to interrogate a model's hidden reasoning.
- **Testability.** `pytest` runs all 59 tests in well under a second, no
  network, no API key, by monkeypatching each client's *methods* to return
  canned `ResolvedData` built from real-shaped fixtures. The LLM layer is
  tested separately, only for control flow.
- **Assignment requirement**, and now also **the user's explicit requirement**
  that nothing in production comes from anywhere but live data or a clearly
  labeled last-resort LLM estimate.

If asked "couldn't the LLM just compute this from raw data you hand it in the
prompt?" - yes, technically, but then identical questions could get different
scores depending on model sampling, with no formula to point to when someone
asks "why is BOS ranked above BDL."

---

## 4. Key engineering trade-offs

Be ready to name these unprompted.

1. **Real government bulk files instead of giving up on missing APIs.** Two
   inputs (On-Time Performance delay/cancellation data, and per-route T-100
   segment data for ANC) turned out to have no queryable REST API at all -
   confirmed by direct testing, not assumed. Rather than fall back to an LLM
   guess for something this central, I found the real predictable bulk-file
   URL pattern BTS actually publishes to (`transtats.bts.gov/PREZIP/...`),
   and built a small pipeline: download once per month, parse the ~276MB
   national CSV in one pass, cache a compact per-airport summary. First
   question in a new month costs ~45 seconds; every question after that,
   for any airport, is sub-second. That's a real trade-off I'd defend: heavier
   up front, but real data beats a guess for the two questions that matter most.

2. **The LLM fallback is real but narrow.** It's schema-constrained (Pydantic
   `response_format`), always tagged, and only used per-field when a live
   source genuinely fails - not a general "make something up" escape hatch.
   I'd point to the Connecticut example below as proof it actually triggers
   correctly rather than being dead code.

3. **A real, live data pipeline surfaces real geography problems - and that's
   a feature of the design, not a bug I hid.** During development, querying
   2024 Census data for Hartford County, CT (BDL's home county) returned
   empty. Why: Connecticut retired its traditional counties for new "planning
   regions" around 2022, and the Census Bureau's newer ACS vintage uses the
   new geography, so the old FIPS code doesn't resolve. My waterfall didn't
   crash or silently return zero - it recorded the failure, fell through to
   the LLM estimate, and tagged the result so the trace shows exactly what
   happened. I'd use this as the concrete answer to "how do you know your
   error handling actually works" - I didn't have to write a synthetic test
   for this case; a real one showed up during normal development.

4. **Streamlit over a custom frontend**, for the same 1-day-scope reason as
   before: the interesting engineering is in the agent/data layers.

5. **Relative (min-max) normalization** for the expansion and congestion
   scores, scoped explicitly to the requested airport set - documented in
   each result's `assumptions` list rather than left as a silent gotcha.

6. **Passenger-weighted long-haul share was cut, not faked.** The original
   design wanted both a flight-count and a passenger-weighted long-haul
   percentage. Once I confirmed no free live API publishes real per-route
   passenger counts, I removed that half of the metric rather than
   backfilling it with an invented number - the result says so explicitly in
   its `assumptions` list. Given the choice between "quietly wrong" and
   "openly incomplete," I chose incomplete.

---

## 5. Context window management

Worth raising proactively if it doesn't come up: the chat-completions API is
stateless, so every call re-sends the whole conversation. Left alone, that
grows without bound - a real gap I only closed after being asked about it
directly, which is itself worth being upfront about if asked "did you think of
this yourself." `_compact_history()` in `orchestrator.py` runs at the start of
every `send_message()`, before that turn's new message is appended, in three
layers:

1. **Stable prefix for prompt caching.** The system message and the tool
   schemas stay unchanged between calls except when layer 3 just updated the
   running summary - compaction only ever touches content after that fixed
   prefix, so most calls still let a caching-capable provider reuse it.
2. **Tool-result pruning.** A tool call's raw JSON is only needed *within the
   turn that fetched it*, for the model to reason over before writing that
   turn's answer. Once the turn is done, its tool messages collapse to a
   placeholder - the assistant's own reply already extracted what mattered.
3. **Sliding window that summarizes, not deletes.** Beyond `MAX_HISTORY_TURNS`
   (default 10) completed turns, the oldest are removed from the raw message
   list - but only after being folded into a running summary
   (`self.conversation_summary`) via one extra LLM call, kept in the system
   message. I changed this from a straight drop after getting feedback that
   silently discarding old turns loses real information a much later
   follow-up might still need - a summary keeps the gist without the token cost.

**Why pruning and the sliding window are two separate layers, not one:** a
tool result is dead weight almost immediately (as soon as that turn's answer
is written), but the *conversation text* (what the user asked, what the
assistant answered) stays useful for follow-ups much longer. Collapsing both
at the same cutoff would either keep raw JSON around far longer than it's
useful, or throw away conversational context far sooner than necessary.

**Why summarization is a legitimate LLM task here, when the project is so
strict elsewhere about the LLM never computing things itself:** the
deterministic-only rule is about the four scoring formulas - reproducible
business numbers. Summarizing prior conversation turns isn't a business
calculation, it's exactly the kind of natural-language task an LLM is for,
the same category as writing the final answer in the first place. If the
summarization call itself fails, `_summarize_dropped_turns()` catches the
exception, logs it, and keeps the previous summary unchanged rather than
raising - the turns still age out either way, so a failed summary means one
missed update, not a crashed conversation.

**The one nuance to know if asked "why isn't the history exactly
`MAX_HISTORY_TURNS` turns long":** compaction runs *before* the new turn is
added, so right after a call returns, history holds the trimmed window plus
the turn just completed - a steady-state bound of `MAX_HISTORY_TURNS + 1`, not
exactly `MAX_HISTORY_TURNS`. That's intentional, not an off-by-one bug: the
turn that was just answered should stay fully available for an immediate
follow-up ("what about BDL?") without waiting for the *next* call to decide
whether to keep it.

---

## 6. Five likely interview questions, with the answers to give

**Q1: "Why not just let the LLM read the raw data and compute the score in
its own reasoning?"**
> Same answer as before, plus: I have a stronger reason now than "in
> principle" - the LLM fallback in this project genuinely fires sometimes
> (see the Connecticut example), and if the LLM were also doing the *scoring*
> math, I'd have no way to distinguish "the score is uncertain because one
> input was estimated" from "the score is uncertain because the model did
> arithmetic differently this run." Keeping math and estimation cleanly
> separated is what makes the `resolution_trace` meaningful at all.

**Q2: "What happens if a live government API changes its schema or goes
down during the demo?"**
> It's handled at two levels. Each client's waterfall catches any failure -
> wrong field names, timeout, empty response - and either tries the next real
> source or falls through to the LLM estimate, always recorded in the trace.
> I hit this for real during development: my first guess at the T-100 API
> shape was wrong (real Socrata dataset IDs redirect to a different domain,
> `data.bts.gov`, with different field names than I'd assumed), and I caught
> it by actually querying the live endpoint rather than trusting the
> assignment's example URLs at face value - that's also just good practice
> for working with any third-party API you haven't personally verified.

**Q3: "Walk me through the BTS On-Time Performance pipeline - why isn't it
just another REST client like the others?"**
> BTS doesn't publish delay/cancellation data as a queryable API - I
> confirmed this by searching their Socrata catalog under multiple terms and
> across both `data.transportation.gov` and `data.bts.gov` before concluding
> it doesn't exist. What they do publish is a real, free, no-key monthly bulk
> CSV at a predictable URL. So that client does something the others don't:
> downloads and unzips the file, does a single streaming pass building a
> per-airport summary for every origin airport at once (not just the one
> requested), and writes that summary to a local cache keyed by year-month.
> That one pass also solves a second problem for free - the same file has
> real per-flight destination and distance, which is what actually answers
> the ANC long-haul question, since there's no separate queryable route-level
> API either.

**Q4: "How would you extend this to more airports or handle a metric with no
data at all?"**
> The airport list is already a parameter, not hardcoded. Scaling mainly
> means confirming county-FIPS/ICAO-code reference tables cover the new
> airports (small, static lookups - not data). For a metric with genuinely no
> live source (like SFO market leakage, which I approximate from a real but
> imperfect proxy - ticketed passenger splits in the BTS airfare survey), the
> honest move is what I already did: use the closest real proxy available,
> document exactly why it's a proxy in `assumptions`, and let the LLM
> fallback catch total data unavailability rather than silently returning zero.

**Q5: "Your unmet demand index has five weighted components - how did you
pick those weights, and how would you validate them?"**
> They're a reasoned starting point, not a fitted model, and I say that
> explicitly in the README rather than implying they're empirically derived.
> To validate them for real, I'd want historical cases where an airport did
> expand capacity, and check which of these five signals moved *before* that
> decision, then fit weights via regression against realized outcomes -
> that's future work, out of scope for a 1-day build.

---

## 7. If they ask you to code live from this project

- **Add a 5th business question**: new file in `src/analytics/`, a
  `kpi_models.py` result schema, a tool wrapper in `src/agent/tools.py`, a
  `TOOL_SCHEMAS` entry. No orchestrator changes needed.
- **Add a new data source**: subclass `BaseDataClient`, implement
  `_live_tiers()` (ordered list of real API attempts) and optionally
  `_llm_fallback()`. `resolve()` handles the looping/tracing automatically.
- **Change a scoring weight**: a single named constant in
  `src/config.py::ScoringConstants`.
- **Add a fixture-backed test for a new client**: capture one real response
  from the live endpoint during development, save it under `tests/fixtures/`
  with a `_comment` noting it's real-shaped, and monkeypatch `requests.get`
  to return it - see `tests/test_data_sources.py` for the pattern used
  throughout.
