# Airport Investment Intelligence Agent — Design & Architecture

A conversational agent that helps airport-investment analysts identify strong
terminal-expansion opportunities. Every number it reports comes from a
deterministic formula fed by real, live public data; the LLM is used only for
reasoning, tool selection, and narration — never for computing a score.

## Architecture overview

```
 User question
      │
      ▼
 src/ui/app.py  (Streamlit chat)
      │
      ▼
 src/agent/orchestrator.py  ──calls──▶  OpenRouter (LLM)
      │  tool_call(name, args)              │ decides which tool to call
      ▼                                     │
 src/agent/tools.py  (JSON-schema ↔ Python bridge, no math)
      │
      ▼
 src/analytics/*.py  (pure deterministic scoring — no LLM calls, ever)
      │  resolve(airport=..., mode=...)
      ▼
 src/data_sources/*.py  (waterfall: primary live API → secondary live API →
                          LLM last resort, only if every real source fails)
```

Each tool result carries a `resolution_trace` (which source actually answered
each number) and an `assumptions` list back up through this chain, so the
final answer — and the UI's "data source trace" panel — can show exactly
where every number came from.

## Scoring methodology

All formulas and their weights/thresholds live in `src/config.py`
(`ScoringConstants`) as named constants, not buried magic numbers.

| Question | Module | Formula (weights sum to 1.0) |
|---|---|---|
| Airport terminal-expansion ranking | `expansion_scorer.py` | `0.35·capacity_pressure + 0.25·load_factor + 0.20·runway_constraint + 0.20·census_growth`, each min-max normalized across the requested airports, scaled to 0–100 |
| Multi-airport congestion comparison | `congestion_comparator.py` | `0.35·taxi_out(norm) + 0.30·dep_delay_≥15min_rate + 0.20·cancellation_rate + 0.15·nas_delay(norm)`, scaled to 0–100, ranked across any 2+ airports |
| Long-haul flight share | `longhaul_calculator.py` | `long_haul = distance_miles ≥ 3000`; real per-flight distance from BTS On-Time Performance data, flight-count share (not passenger-weighted — see trade-offs) |
| SFO unmet demand index | `unmet_demand_calculator.py` | `0.30·load_factor_pressure + 0.25·(pax_growth − seat_growth) + 0.20·market_leakage(OAK/SJC) + 0.15·fare_premium + 0.10·cancellation_gap`, each normalized, scaled to 0–100 |

Each factor is computed from real public data (BTS T-100, BTS On-Time
Performance, FAA facilities/runways, Census, BEA, BTS airfare survey — see
each analytics module's docstring for the exact source), never from the LLM.

## Key trade-offs

- **Live data first, LLM estimate only as a documented last resort.** Every
  data source tries a real API (sometimes a second real API) before ever
  asking the LLM to fill in a raw fact — and when it does, the result is
  tagged `tier="llm_fallback"` so it's never confused with measured data.
- **Two required inputs have no queryable public API at all**, confirmed by
  direct testing, not assumed: BTS On-Time Performance (delay/cancellation
  data) and route-level distance data. Rather than fall back to a guess, the
  real monthly bulk-download file BTS actually publishes
  (`transtats.bts.gov/PREZIP/...`) is downloaded and parsed directly — one
  dataset that happens to answer both the congestion and long-haul questions.
  First question in a new month costs ~45s (download + parse); every question
  after that reads a local cache in well under a second.
- **Long-haul share is flight-count only, not passenger-weighted.** Real
  per-route passenger counts aren't published via any free live API (only
  BTS's internal, non-queryable system has that). Rather than fabricate a
  weighting, the metric says so explicitly in its `assumptions`.
- **Congestion and long-haul share are single-month snapshots**, not
  full-year averages — 12 bulk-file downloads per question isn't a
  reasonable chat-response time. Every result states exactly which month.
- **SFO's fare-premium/market-leakage metric is SFO-specific**, not
  general-purpose. The one real dataset that supports this kind of
  comparison (BTS's Consumer Airfare Report) only works for multi-airport
  metro markets like SFO/OAK/SJC.
- **A live data pipeline surfaces real-world geography change.** During
  development, a 2024 Census query for BDL's home county returned empty
  because Connecticut retired its counties for new "planning regions" in
  2022 — the waterfall caught the failure and fell through to an LLM
  estimate automatically, exactly as designed, rather than crashing or
  silently returning zero.
- **Context window is actively managed, not left to grow unbounded**, since
  the chat API is stateless and re-sends full history every call: tool
  results are pruned once their turn is answered, and turns beyond a sliding
  window are folded into a running LLM-written summary rather than deleted
  or left to grow forever (see "Where AI is used" below).
- **Mocks are test-only.** `tests/fixtures/` holds real-shaped API responses
  captured during development; the running application never reads them —
  every production code path calls a real live endpoint.

## Where/how AI is used

The LLM (via OpenRouter, `src/agent/orchestrator.py`) is used for exactly
four things, and nothing else:

1. **Tool selection.** Given the user's question and the tool schemas in
   `src/agent/tools.py`, the model decides which deterministic tool(s) to
   call and with what arguments. It never computes the answer itself.
2. **Narration.** Once a tool returns a structured result (score, weights,
   `resolution_trace`, `assumptions`), the model turns it into a clear,
   sourced, plain-English answer — instructed to name the real data
   source(s) used and explicitly flag any number that came from an LLM
   estimate rather than live data.
3. **Last-resort structured estimation** (`src/data_sources/llm_fallback_client.py`).
   Only when every real API for one specific raw input has failed, the model
   is asked for a value constrained to a Pydantic schema (not free text),
   tagged `is_llm_estimate=True`. It fills in one missing fact, never a
   score or ranking.
4. **Conversation summarization** (`orchestrator.py::_summarize_dropped_turns`).
   When the sliding context window fills, the model is asked to fold aging-out
   turns into a running summary (preserving airport codes, numbers, and
   conclusions), kept in the system message — a legitimate language task,
   distinct from the scoring math the LLM is never allowed to touch.

A fifth, test-only use: `tests/test_evals.py` uses a real LLM as an independent
judge to grade the agent's actual answers for a specific failure mode -
hallucinating confident statistics for airports outside this system's real
data coverage (non-US airports) instead of admitting the gap. It's opt-in
(`RUN_LLM_EVALS=1`), since it costs real tokens and isn't part of the
otherwise fully offline test suite.

**Where AI is deliberately NOT used:** any arithmetic behind a score,
ranking, percentage, or index. `src/analytics/*.py` has zero LLM
dependency — the same inputs always produce the same output, and every
formula is auditable by reading the code, not by trusting a model's
reasoning.
