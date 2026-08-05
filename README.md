# Airport Investment Intelligence Agent

A conversational AI agent that helps airport-investment analysts identify strong
terminal-expansion opportunities, using **deterministic scoring logic** for every
number and an **LLM** only for reasoning, tool selection, and plain-English
explanation.

Built to answer four seed questions out of the box (and follow-up questions about
them):

- Which airports in New England are strong candidates for terminal expansion?
- Compare LA (LAX) and Santa Ana (SNA) airport congestion levels.
- What is the percentage of long-haul flights out of Anchorage (ANC) airport?
- What is the unmet flight demand in SFO airport, and why?

**Data policy: live data only in production.** Every number the agent uses comes
from a real, live public API - see [Data sources](#3-data-sources-and-the-waterfall)
below. There is no "load from a local mock file" fallback in the running app. Mock
JSON fixtures exist only under `tests/`, used exclusively by the test suite.

---

## 1. Quick start

### Option A: pip + venv

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env - see "API keys" below for what's required vs. optional

streamlit run src/ui/app.py
```

### Option B: Docker

```bash
cp .env.example .env
# edit .env

docker compose up --build
# open http://localhost:8501
```

### API keys

| Key | Required? | Get one | What breaks without it |
|---|---|---|---|
| `OPENROUTER_API_KEY` | **Required** | https://openrouter.ai/keys | The chat agent can't run at all (deterministic analytics/tests still work) |
| `CENSUS_API_KEY` | Optional | https://api.census.gov/data/key_signup.html (instant) | Population growth in the New England expansion score falls back to an LLM estimate |
| `BEA_API_KEY` | Optional | https://apps.bea.gov/api/signup/ (instant) | Income growth in the New England expansion score falls back to an LLM estimate |

### Run the tests (fully offline, no API key needed)

```bash
pytest
```

All 59 tests run against real-shaped fixture data (`tests/fixtures/`) with the
network fully mocked out - see [Testing strategy](#8-testing-strategy).

---

## 2. Architecture overview

```
                         ┌─────────────────────────┐
                         │   src/ui/app.py          │   Streamlit chat window +
                         │   (chat interface)       │   "data source trace" panel
                         └────────────┬─────────────┘
                                      │ user text
                                      ▼
                         ┌─────────────────────────┐
                         │ src/agent/orchestrator.py│   Talks to the LLM via
                         │ (OpenRouter + tool loop) │   OpenRouter (OpenAI SDK)
                         └────────────┬─────────────┘
                                      │ tool_call(name, args)
                                      ▼
                         ┌─────────────────────────┐
                         │  src/agent/tools.py       │   JSON-schema <-> Python
                         │  (LLM <-> Python bridge)  │   function bridge. No math.
                         └────────────┬─────────────┘
                                      │ plain function call
                                      ▼
                         ┌─────────────────────────┐
                         │  src/analytics/*.py       │   PURE deterministic math.
                         │  (scoring engines)        │   No LLM calls, ever.
                         └────────────┬─────────────┘
                                      │ resolve(airport=..., mode=...)
                                      ▼
                         ┌─────────────────────────┐
                         │ src/data_sources/*.py     │   Waterfall: try each real
                         │ (waterfall resolution)    │   live source in order.
                         └────────────┬─────────────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              ▼                       ▼                       ▼
      Primary live API      Secondary live API      LLM last resort (only if
      (real gov't data)     (real gov't data)        every real source fails)
```

**Data flow for one question**, e.g. *"What is the unmet flight demand in SFO?"*:

1. User types the question in the Streamlit chat box (`src/ui/app.py`).
2. `AirportIntelligenceAgent.send_message()` sends it to the LLM (via OpenRouter)
   together with the tool schemas from `src/agent/tools.py`.
3. The LLM calls `get_unmet_demand_index(airport="SFO")`.
4. `run_tool()` calls `src/analytics/unmet_demand_calculator.py`, which pulls real
   BTS T-100 traffic, real BTS airfare/leakage data, and real BTS On-Time
   Performance cancellation rates - each via its own waterfall - and computes the
   index with a **fixed formula**. No LLM involved in the math.
5. Every data point's `resolution_trace` (which real source answered, or that the
   LLM had to estimate it) travels with the result back to the orchestrator.
6. The LLM turns the structured result into a plain-English answer and **must**
   summarize the trace (see the system prompt in `orchestrator.py`).
7. The UI renders the reply AND, independent of what the LLM said, a **Data
   source trace** panel built directly from the same trace data - so the
   provenance is visible even if the model's prose doesn't mention it.

---

## 3. Data sources and the waterfall

Every client in `src/data_sources/` follows the same pattern (`base_client.py`):
try a primary real API, try a secondary real API if there is one, and only if
every real source fails, ask the LLM for a structured, schema-validated estimate
(`llm_fallback_client.py`) - clearly tagged `tier="llm_fallback"` in the trace.

| Client | Real source(s) | Verified during development |
|---|---|---|
| `bts_t100_client.py` | BTS T-100 by-origin-airport summary, `data.bts.gov/resource/r495-tyji.json` (Socrata SODA API) | Queried live; real passengers/seats/departures/load-factor per airport-year |
| `bts_ontime_client.py` | BTS On-Time Performance monthly bulk file, `transtats.bts.gov/PREZIP/On_Time_Reporting_..._{year}_{month}.zip` | Downloaded and parsed live; real per-flight Origin/Dest/Distance/TaxiOut/DepDel15/Cancelled/NASDelay |
| `faa_arcgis_client.py` | FAA NTAD Aviation Facilities + Runways, ArcGIS FeatureServer | Queried live; BOS's real 6 runways confirmed |
| `faa_nas_client.py` | FAA live NAS status, `nasstatus.faa.gov/api/airport-status-information` (XML) | Queried live; showed a real active closure notice during testing |
| `nws_weather_client.py` | NWS station observations + alerts, `api.weather.gov` | Queried live; real current visibility/wind/alerts |
| `macro_data_client.py` | US Census ACS5 (population), BEA CAINC1 (income), BTS Consumer Airfare Report `tfrh-tu9e` (fares + real SFO/OAK/SJC passenger split) | Queried live; all three confirmed working with real keys |

**No queryable public API exists for two things this project needs** (confirmed by
direct testing during development, not assumed):

- **BTS On-Time Performance** (delay/taxi/cancellation data) has no REST API -
  only a monthly bulk-download file. `bts_ontime_client.py` downloads and parses
  it directly (see below).
- **Route-level T-100 segment data** (origin-destination with distance) also has
  no REST API. It turns out the On-Time Performance bulk file already carries
  real per-flight `Dest`/`Distance`, so `bts_ontime_client.py`'s route method
  covers the ANC long-haul question too - one real dataset, two questions.

**The bulk-file client is cached.** The first request in a given calendar month
downloads the ~30MB file, decompresses the ~276MB national CSV, and does one
pass building a compact per-airport summary (all origins at once) - about 40-50
seconds. That summary is cached to `.cache/bts_ontime/{year}_{month}_summary.json`
(gitignored), so every subsequent question that month, for any airport, reads the
small cache instead - well under a second.

---

## 4. Why deterministic logic is isolated from the LLM

Every number an analyst might act on - a score, a percentage, an index - is
computed in `src/analytics/`, which is **pure Python with no LLM calls**. The LLM
in `src/agent/orchestrator.py` only ever does two things: decide *which* tool to
call, and *narrate* the tool's already-computed result. The one narrow exception
is the LLM-fallback data source (`llm_fallback_client.py`) - used only when every
real API for one specific raw input has failed, constrained to a Pydantic schema,
and always tagged `is_llm_estimate=True` in the trace. Even then, the LLM fills in
a raw fact (e.g. "roughly what is BOS's population growth"), never a score.

This split exists because:

- **Reproducibility.** The same inputs always produce the same score.
- **Auditability.** An investor can open `src/analytics/congestion_comparator.py`
  and see the exact formula and weights - no need to trust a language model's
  internal reasoning.
- **Testability.** `tests/test_kpis.py` tests the scoring formulas directly with
  zero network calls. LLM behavior is tested separately, and only for control
  flow, never for numeric correctness.
- **Assignment requirement.** The brief explicitly requires "deterministic
  scoring or ranking logic (not only LLM output)."

## 5. Context window management

The chat-completions API is stateless - every call re-sends the entire message
history - so a long conversation needs active management or it grows without
bound. `AirportIntelligenceAgent._compact_history()` (in `orchestrator.py`) runs
at the start of every `send_message()`, before that turn's new message is added,
in three layers:

1. **Cache-friendly structure.** The system prompt (`messages[0]`) and the tool
   schemas (`TOOL_SCHEMAS`, passed as the separate `tools` parameter) stay
   byte-for-byte identical on every call *except* the ones where layer 3 below
   just updated the running summary - compaction only ever touches content
   after that prefix, so most calls still let a provider's prompt cache reuse it.
2. **Tool message pruning.** A tool call's raw JSON result (which can be large -
   a 6-airport ranking with a full resolution trace per airport, for example)
   is only useful for the model to reason over *within the turn that fetched
   it*. Once that turn's final answer has been written, its tool messages are
   replaced with a short placeholder; the assistant's own reply already
   captured what mattered.
3. **Sliding window with summarization, not deletion.** Once there are more
   than `MAX_HISTORY_TURNS` (default 10, configurable in `.env`) completed
   turns, the oldest ones are removed from the raw message list - but first
   folded into `self.conversation_summary` via one LLM call (asked to extend
   the existing summary with the new turns, preserving specific facts like
   airport codes and numbers), which is kept in the system message. Older
   context is compressed, never silently lost. If that summarization call
   fails, the previous summary is kept as-is and the turns still age out -
   losing one update is better than the conversation growing unbounded.

Tests for this live in `tests/test_orchestrator.py::TestCompactHistory` and
`TestContextCompactionIntegration`.

## 6. Scoring methodology

All formulas and their weights/thresholds live in `src/config.py`
(`ScoringConstants`).

| Question | Module | Formula (weights sum to 1.0) |
|---|---|---|
| New England terminal expansion ranking | `expansion_scorer.py` | `0.35 * capacity_pressure + 0.25 * load_factor + 0.20 * runway_constraint + 0.20 * census_growth`, each min-max normalized across the requested airports, scaled to 0-100 |
| LAX vs SNA congestion | `congestion_comparator.py` | `0.35 * taxi_out(norm) + 0.30 * dep_delay_ge15_rate + 0.20 * cancellation_rate + 0.15 * nas_delay(norm)`, scaled to 0-100 |
| ANC long-haul share | `longhaul_calculator.py` | `long_haul = distance_miles >= 3000`; real per-flight distance from BTS On-Time Performance, flight-count share only (see limitation below) |
| SFO unmet demand index | `unmet_demand_calculator.py` | `0.30 * load_factor_pressure + 0.25 * (pax_growth - seat_growth) + 0.20 * market_leakage(OAK/SJC) + 0.15 * fare_premium + 0.10 * cancellation_gap`, each normalized, scaled to 0-100 |

Each result also returns an `assumptions` list and a `resolution_trace`, which the
agent is instructed to surface rather than hide.

## 7. Scoping assumptions and limitations

- **Long-haul share is flight-count only, not passenger-weighted.** The original
  design wanted both. Real per-route passenger counts (as opposed to per-route
  flight counts) are not published via any free live API - only BTS's internal
  TranStats system has that at the route level, and it's not queryable. Rather
  than fabricate a passenger weighting, this metric says so explicitly.
- **Congestion and long-haul share are single-month snapshots**, not full-year
  averages - the bulk file has to be downloaded per month, and 12 downloads per
  question isn't a reasonable chat-response time. `source_month` on every result
  says exactly which month.
- **Covers passenger-carrier flights only.** BTS On-Time Performance doesn't
  include cargo-only carriers (e.g. FedEx/UPS at ANC), which is a real share of
  some airports' traffic.
- **SFO's fare-premium and market-leakage metrics are SFO-specific.** The real
  BTS airfare dataset (`tfrh-tu9e`) only supports this kind of comparison for
  multi-airport metro markets (like SFO/OAK/SJC); it isn't a general-purpose
  "any airport's fare vs. national average" source.
- **Census/BEA lookups are county-level**, using each airport's home county as a
  proxy for its catchment area, and require the (free, instant) API keys above.
- **Geography can genuinely shift under a live pipeline.** During development,
  Connecticut's 2024 Census data returned empty for BDL's old county FIPS code
  (Connecticut retired counties for new "planning regions" around 2022) - the
  waterfall caught it, logged the failure, and fell through to the LLM estimate
  automatically, exactly as designed. That's a real illustration of why the
  fallback chain and its trace exist, not a bug.
- **A free OpenRouter model is the default** (`.env.example`); swap `MODEL_NAME`
  for a paid model for materially faster, more reliable tool-calling.

## 8. Testing strategy

```
tests/test_data_sources.py  # Real-schema parsing + waterfall fallback behavior
tests/test_kpis.py          # Deterministic formulas, clients monkeypatched
tests/test_tools.py         # LLM<->Python tool bridge, clients monkeypatched
tests/test_orchestrator.py  # Tool-calling loop + context compaction, OpenAI client mocked
tests/fixtures/*            # Real-shaped API response fixtures - TEST-ONLY
```

Fixtures in `tests/fixtures/` are built from actual API responses captured while
building each client (see the `_comment` field in each fixture file) - not
guessed shapes. The running application never reads these files; only `pytest`
does.

## 9. Project layout

```
├── .env.example
├── README.md
├── INTERVIEW_EXPLAINER.md
├── docker-compose.yml
├── Dockerfile
├── pytest.ini
├── requirements.txt
├── src/
│   ├── config.py                    # Env config + all scoring weights/thresholds
│   ├── data_sources/                # One client per public data source, each a waterfall
│   │   ├── base_client.py           # Shared resolve()/ResolvedData/ResolutionStep
│   │   ├── llm_fallback_client.py   # Last-resort structured LLM estimate
│   │   ├── bts_t100_client.py
│   │   ├── bts_ontime_client.py
│   │   ├── faa_arcgis_client.py
│   │   ├── faa_nas_client.py
│   │   ├── nws_weather_client.py
│   │   └── macro_data_client.py
│   ├── analytics/                   # Pure deterministic scoring, no LLM
│   │   ├── kpi_models.py
│   │   ├── expansion_scorer.py
│   │   ├── congestion_comparator.py
│   │   ├── longhaul_calculator.py
│   │   └── unmet_demand_calculator.py
│   ├── agent/
│   │   ├── tools.py                 # LLM tool schemas + dispatcher
│   │   └── orchestrator.py          # OpenRouter chat + tool-calling loop
│   └── ui/
│       └── app.py                   # Streamlit chat UI + trace panel
└── tests/
    └── fixtures/                    # Real-shaped mock data - test-only
```
