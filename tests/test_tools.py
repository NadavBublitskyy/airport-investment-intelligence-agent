"""
Tests for the agent tool wrappers in src/agent/tools.py.

These never hit the network: the analytics functions and raw client
lookups each tool wraps are monkeypatched to return canned data (built
the same way as in tests/test_kpis.py). What's under test here is the
*wrapping* - does each tool return JSON-serializable data with a
`resolution_trace`, does the schema list match the registry, and does
`run_tool` handle unknown tools / bad arguments without raising - not
the formulas themselves (covered in test_kpis.py) or live parsing
(covered in test_data_sources.py).
"""

from __future__ import annotations

import json
from unittest.mock import patch

from src.agent.tools import TOOL_REGISTRY, TOOL_SCHEMAS, run_tool
from src.data_sources.base_client import ResolutionStep, ResolvedData


def test_every_schema_has_a_matching_registered_function():
    schema_names = {schema["function"]["name"] for schema in TOOL_SCHEMAS}
    assert schema_names == set(TOOL_REGISTRY)


def test_unknown_tool_returns_error_dict_instead_of_raising():
    result = run_tool("this_tool_does_not_exist", {})
    assert "error" in result


def test_bad_arguments_return_error_dict_instead_of_raising():
    result = run_tool("get_airport_facility_info", {"not_a_real_argument": "x"})
    assert "error" in result


def test_get_airport_facility_info_wraps_client_result_with_trace():
    fake_resolved = ResolvedData(
        data={"num_runways": 4, "longest_runway_ft": 12000, "is_llm_estimate": False},
        trace=[ResolutionStep("primary", "FAA ArcGIS", "success")],
    )
    with patch("src.agent.tools.FaaArcgisClient.get_facility", return_value=fake_resolved):
        result = run_tool("get_airport_facility_info", {"airport": "LAX"})

    assert result["airport"] == "LAX"
    assert result["facility"]["num_runways"] == 4
    assert result["resolution_trace"] == [
        {"tier": "primary", "source_name": "FAA ArcGIS", "status": "success", "detail": ""}
    ]
    json.dumps(result)  # must be JSON-serializable


def test_get_airport_live_status_wraps_client_result_with_trace():
    fake_resolved = ResolvedData(
        data={"airport": "SFO", "events": [], "has_ground_stop": False},
        trace=[ResolutionStep("primary", "FAA NAS", "success")],
    )
    with patch("src.agent.tools.FaaNasClient.get_status", return_value=fake_resolved):
        result = run_tool("get_airport_live_status", {"airport": "SFO"})

    assert result["status"]["has_ground_stop"] is False
    json.dumps(result)


def test_get_airport_weather_wraps_client_result_with_trace():
    fake_resolved = ResolvedData(
        data={"airport": "SFO", "conditions": "Foggy", "active_alerts": []},
        trace=[ResolutionStep("primary", "NWS", "success")],
    )
    with patch("src.agent.tools.NwsWeatherClient.get_weather", return_value=fake_resolved):
        result = run_tool("get_airport_weather", {"airport": "SFO"})

    assert result["weather"]["conditions"] == "Foggy"
    json.dumps(result)


def test_analytics_tools_propagate_exceptions_as_error_dicts():
    with patch("src.agent.tools.score_new_england_expansion", side_effect=ValueError("boom")):
        result = run_tool("rank_new_england_expansion_candidates", {})
    assert "error" in result
    assert "boom" in result["error"]
