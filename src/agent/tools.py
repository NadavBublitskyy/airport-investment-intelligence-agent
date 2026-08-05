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


def _rank_new_england_expansion_candidates(airports: list[str] | None = None) -> dict:
    """Tool implementation: New England terminal-expansion ranking."""
    result = score_new_england_expansion(airports=airports)
    return result.model_dump()


def _compare_airport_congestion(airport_a: str = "LAX", airport_b: str = "SNA") -> dict:
    """Tool implementation: two-airport congestion comparison."""
    result = compare_congestion(airport_a=airport_a, airport_b=airport_b)
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
            "name": "rank_new_england_expansion_candidates",
            "description": (
                "Rank New England airports (default: BOS, BDL, PVD, PWM, MHT, BTV) by how strong a "
                "candidate each is for terminal expansion investment. Returns a deterministic 0-100 "
                "score per airport built from real BTS passenger/seat growth, real FAA runway data, "
                "and real Census population growth. Use this for any question about which New England "
                "airport(s) should get terminal expansion investment."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "airports": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Optional list of IATA airport codes to score, e.g. ['BOS','BDL']. "
                            "Omit to score the full default New England set."
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
                "Compare two airports' operational congestion using a deterministic Congestion Score "
                "built from real BTS On-Time Performance data (taxi-out time, departure delay rate, "
                "cancellation rate, NAS delay minutes) plus real FAA runway counts. Defaults to LAX vs "
                "SNA. Use this for any question comparing congestion, delays, or reliability between "
                "two specific airports."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "airport_a": {"type": "string", "description": "First IATA airport code, e.g. 'LAX'."},
                    "airport_b": {"type": "string", "description": "Second IATA airport code, e.g. 'SNA'."},
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
                "Compute the percentage of an airport's real departures that are long-haul (nonstop "
                "distance >= 3000 statute miles), from real per-flight BTS route data. Defaults to ANC "
                "(Anchorage). Use this for any question about long-haul flight share."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "airport": {"type": "string", "description": "IATA airport code, e.g. 'ANC'."},
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
                "Compute the Estimated Unmet Demand Index (0-100) for SFO, combining real load-factor "
                "pressure, real passenger-vs-seat growth gap, real market leakage to OAK/SJC (from real "
                "ticketed passenger volumes), real fare premium, and real cancellation-rate gap. Also "
                "returns a root-cause breakdown (gate/terminal limits vs runway bottleneck vs weather vs "
                "regional pricing). Written specifically for SFO vs. OAK/SJC. Use this for any question "
                "about SFO's unmet demand, capacity shortfall, or 'why can't SFO handle more flights.'"
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
                "Get real raw physical facility data for one airport from the live FAA database: "
                "number of runways, each runway's real length/width/surface type, and site acreage. "
                "Use this when the user asks about physical infrastructure directly, or needs "
                "supporting detail behind a score."
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
                "Get the current real-time live FAA operational status for one airport: active ground "
                "stops, ground delay programs, closures, and general arrival/departure delay advisories "
                "happening right now. Use this when the user asks about current/live conditions, not "
                "historical averages."
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
                "Get real current NWS weather conditions for one airport: visibility, wind speed/gusts, "
                "conditions, and active weather alerts. Use this to help explain weather-driven "
                "congestion or delays."
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
    "rank_new_england_expansion_candidates": _rank_new_england_expansion_candidates,
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
