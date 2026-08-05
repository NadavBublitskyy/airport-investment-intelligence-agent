"""
Agent tools: the bridge between the LLM and the deterministic engines.

What this file does: wraps every deterministic analytics function (and a
few raw data-source lookups) as an OpenAI-style "function calling" tool -
a JSON schema the LLM can read, plus a Python callable that actually runs
it and returns plain JSON-serializable data, including a `resolution_trace`
showing exactly which real data source answered (or, as a last resort,
that the LLM had to estimate it).

Why it exists: this is the file that enforces the project's most
important design rule - the LLM never computes a score or a ranking
itself. It only chooses WHICH tool to call and WHEN, based on the user's
question. All arithmetic happens in `src/analytics/`, safely outside the
LLM's reach. This file is intentionally "dumb": it does not reason about
the user's intent (that is the orchestrator/LLM's job) and it does not
compute anything (that is the analytics layer's job) - it only adapts
between the two, and makes sure every tool result carries its own data
provenance.

How data moves: `src/agent/orchestrator.py` sends the LLM this module's
`TOOL_SCHEMAS`. When the LLM responds with a tool call, the orchestrator
looks up the matching Python function in `TOOL_REGISTRY` and calls
`run_tool(name, arguments)`, which returns a dict ready to be JSON-encoded
back to the LLM as the tool's result. The orchestrator also reads each
result's `resolution_trace` to show the user which sources were tried.
"""

from __future__ import annotations

from typing import Any, Callable

from src.analytics.congestion_comparator import compare_congestion
from src.analytics.expansion_scorer import score_new_england_expansion
from src.analytics.longhaul_calculator import calculate_long_haul_share
from src.analytics.unmet_demand_calculator import calculate_unmet_demand
from src.data_sources.base_client import ResolvedData
from src.data_sources.faa_arcgis_client import FaaArcgisClient
from src.data_sources.faa_nas_client import FaaNasClient
from src.data_sources.nws_weather_client import NwsWeatherClient


def _trace_dicts(resolved: ResolvedData) -> list[dict]:
    return [
        {"tier": s.tier, "source_name": s.source_name, "status": s.status, "detail": s.detail}
        for s in resolved.trace
    ]


def _rank_airport_expansion_candidates(airports: list[str] | None = None) -> dict:
    """Tool implementation: terminal-expansion ranking for any airport set (defaults to New England)."""
    result = score_new_england_expansion(airports=airports)
    return result.model_dump()


def _compare_airport_congestion(airports: list[str] | None = None) -> dict:
    """Tool implementation: congestion comparison across any set of airports (defaults to LAX vs SNA)."""
    result = compare_congestion(airports=airports)
    return result.model_dump()


def _get_long_haul_flight_share(airport: str = "ANC") -> dict:
    """Tool implementation: long-haul (>=3000 mi) flight share."""
    result = calculate_long_haul_share(airport=airport)
    return result.model_dump()


def _get_unmet_demand_index(airport: str = "SFO") -> dict:
    """Tool implementation: Estimated Unmet Demand Index and root-cause breakdown."""
    result = calculate_unmet_demand(airport=airport)
    return result.model_dump()


def _get_airport_facility_info(airport: str) -> dict:
    """Tool implementation: raw FAA facility/runway metadata for one airport."""
    result = FaaArcgisClient().get_facility(airport)
    return {"airport": airport, "facility": result.data, "resolution_trace": _trace_dicts(result)}


def _get_airport_live_status(airport: str) -> dict:
    """Tool implementation: live FAA ground stop / ground delay / closure status."""
    result = FaaNasClient().get_status(airport)
    return {"airport": airport, "status": result.data, "resolution_trace": _trace_dicts(result)}


def _get_airport_weather(airport: str) -> dict:
    """Tool implementation: current NWS weather conditions and alerts."""
    result = NwsWeatherClient().get_weather(airport)
    return {"airport": airport, "weather": result.data, "resolution_trace": _trace_dicts(result)}


# --- OpenAI/OpenRouter tool-calling JSON schemas ---
# Field names and descriptions here are what the LLM actually reads to
# decide which tool to call - keep them precise and unambiguous.

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "rank_airport_expansion_candidates",
            "description": (
                "Score and rank any set of US airports on how strong a candidate each is for terminal "
                "expansion investment. Returns a deterministic 0-100 score per airport built from real "
                "BTS passenger/seat growth, real FAA runway data, and real Census population growth for "
                "each airport's home county. Pass any list of IATA codes; if omitted, defaults to a "
                "New England set (BOS, BDL, PVD, PWM, MHT, BTV). "
                'Example prompt this answers: "Which airports in New England are strong candidates for '
                'terminal expansion?"'
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "airports": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Optional list of IATA airport codes to score, e.g. ['BOS','BDL'] or "
                            "['ORD','MDW']. Omit to score the default New England set."
                        ),
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_airport_congestion",
            "description": (
                "Compare operational congestion across two or more US airports using a deterministic "
                "Congestion Score built from real BTS On-Time Performance data (taxi-out time, "
                "departure delay rate, cancellation rate, NAS delay minutes) plus real FAA runway "
                "counts. Returns every airport ranked from most to least congested. Pass any list of "
                "IATA codes (two or more); defaults to LAX vs SNA if omitted. "
                'Example prompt this answers: "Compare LA and Santa Ana airport congestion levels."'
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "airports": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "List of two or more IATA airport codes to compare, e.g. ['LAX','SNA'] or "
                            "['ORD','MDW','JFK']. Omit to compare the default LAX vs SNA."
                        ),
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_long_haul_flight_share",
            "description": (
                "Compute what share of any US airport's real departures are long-haul (nonstop distance "
                ">= 3000 statute miles), from real per-flight BTS route data. Works for any IATA code; "
                "defaults to ANC (Anchorage) if none is given. "
                'Example prompt this answers: "What is the percentage of long haul flights out of '
                'Anchorage airport?"'
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "airport": {"type": "string", "description": "IATA airport code, e.g. 'ANC' or 'JFK'."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_unmet_demand_index",
            "description": (
                "Compute an Estimated Unmet Demand Index (0-100) combining load-factor pressure, "
                "passenger-vs-seat growth gap, market leakage to nearby secondary airports, fare "
                "premium, and cancellation-rate gap, plus a root-cause breakdown (gate/terminal limits "
                "vs runway bottleneck vs weather vs regional pricing). NOTE: unlike the other tools "
                "here, this one is not airport-agnostic - the market-leakage and fare-premium "
                "components are only wired up to real live data for SFO compared against its Bay Area "
                "alternates OAK/SJC, since that's the one real public dataset (BTS airfare survey) that "
                "supports this kind of multi-airport-market comparison. Call it for SFO. "
                'Example prompt this answers: "What is the unmet flight demand in SFO airport and why?"'
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "airport": {"type": "string", "description": "IATA airport code - written for 'SFO'."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_airport_facility_info",
            "description": (
                "Get real raw physical facility data for any US airport from the live FAA database: "
                "number of runways, each runway's real length/width/surface type, and site acreage. "
                "Use this when the user asks about physical infrastructure directly, or needs "
                "supporting detail behind a score. "
                'Example prompt this answers: "How many runways does BOS have, and how long is the '
                'longest one?"'
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "airport": {"type": "string", "description": "IATA airport code, e.g. 'BOS'."},
                },
                "required": ["airport"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_airport_live_status",
            "description": (
                "Get the current real-time live FAA operational status for any US airport: active "
                "ground stops, ground delay programs, closures, and general arrival/departure delay "
                "advisories happening right now. Use this when the user asks about current/live "
                "conditions, not historical averages. "
                'Example prompt this answers: "Is there a ground stop at SFO right now?"'
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "airport": {"type": "string", "description": "IATA airport code, e.g. 'SFO'."},
                },
                "required": ["airport"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_airport_weather",
            "description": (
                "Get real current NWS weather conditions for any US airport: visibility, wind "
                "speed/gusts, conditions, and active weather alerts. Use this to help explain "
                "weather-driven congestion or delays. Active-alert lookups are verified for a reference "
                "set of major airports; for others, current conditions still resolve but alerts may "
                "come back empty rather than falling back to an estimate. "
                'Example prompt this answers: "What\'s the weather like at SFO right now, and could it '
                'be causing delays?"'
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "airport": {"type": "string", "description": "IATA airport code, e.g. 'SFO'."},
                },
                "required": ["airport"],
            },
        },
    },
]

TOOL_REGISTRY: dict[str, Callable[..., dict]] = {
    "rank_airport_expansion_candidates": _rank_airport_expansion_candidates,
    "compare_airport_congestion": _compare_airport_congestion,
    "get_long_haul_flight_share": _get_long_haul_flight_share,
    "get_unmet_demand_index": _get_unmet_demand_index,
    "get_airport_facility_info": _get_airport_facility_info,
    "get_airport_live_status": _get_airport_live_status,
    "get_airport_weather": _get_airport_weather,
}


def run_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """
    Execute one tool by name with the arguments the LLM supplied.

    This is the single choke point every tool call passes through, so
    unknown tool names or bad arguments fail predictably instead of
    crashing the whole conversation. Every successful result carries a
    `resolution_trace` key; the orchestrator reads it to show the user
    exactly which data sources were tried for this answer.
    """
    func = TOOL_REGISTRY.get(name)
    if func is None:
        return {"error": f"Unknown tool '{name}'. Available tools: {sorted(TOOL_REGISTRY)}"}
    try:
        return func(**arguments)
    except TypeError as exc:
        return {"error": f"Bad arguments for tool '{name}': {exc}"}
    except Exception as exc:  # noqa: BLE001 - report to the LLM rather than crash the chat
        return {"error": f"Tool '{name}' failed: {exc}"}
