"""
Tests for src/data_sources/*.py: real-schema parsing and waterfall behavior.

These tests never hit the network. They monkeypatch `requests.get` (or
feed fixture data directly into pure parsing/aggregation functions) using
the real-shaped fixtures under `tests/fixtures/` - built from actual API
responses captured during development - so the parsing code is exercised
against the real field names, not guessed ones. They also verify the
waterfall itself: when the primary source fails, does resolution move to
the next tier and record why?
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.data_sources.bts_ontime_client import BtsOnTimeClient, _aggregate_rows, _routes_from_summary, _stats_from_summary
from src.data_sources.bts_t100_client import BtsT100Client
from src.data_sources.faa_arcgis_client import FaaArcgisClient
from src.data_sources.faa_nas_client import _parse_events
from src.data_sources.macro_data_client import _bea_income_growth, _census_population
from src.data_sources.nws_weather_client import NwsWeatherClient, _unit_value


def _fake_response(json_data=None, text_data=None, status_code=200):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.text = text_data
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = Exception(f"HTTP {status_code}")
    return resp


class TestBtsT100Client:
    def test_parses_real_shaped_soda_response(self, t100_fixture):
        with patch("src.data_sources.bts_t100_client.requests.get") as mock_get:
            mock_get.return_value = _fake_response(json_data=t100_fixture["BOS_2019"])
            result = BtsT100Client().get_annual_totals("BOS", [2019])[2019]

        assert result.data["passengers"] == 20742138.0
        assert result.data["available_seats"] == 25332809.0
        assert round(result.data["load_factor"], 4) == round(20742138.0 / 25332809.0, 4)
        assert result.trace[0].status == "success"

    def test_empty_response_falls_back_to_llm(self, t100_fixture):
        with patch("src.data_sources.bts_t100_client.requests.get") as mock_get, patch(
            "src.data_sources.bts_t100_client.estimate_structured"
        ) as mock_estimate:
            mock_get.return_value = _fake_response(json_data=t100_fixture["empty"])
            mock_estimate.return_value = {
                "passengers": 1000.0,
                "available_seats": 1200.0,
                "scheduled_departures": 100.0,
                "departures_performed": 99.0,
                "avg_distance_miles": 500.0,
                "confidence": "low",
                "caveat": "test caveat",
            }
            result = BtsT100Client().get_annual_totals("ZZZ", [2019])[2019]

        assert result.data["is_llm_estimate"] is True
        assert result.trace[0].status == "failed"
        assert result.trace[-1].tier == "llm_fallback"
        assert result.trace[-1].status == "success"


class TestBtsOnTimeAggregation:
    def test_aggregates_real_shaped_rows(self, ontime_fixture):
        summary = _aggregate_rows(ontime_fixture["lax_may_2026"], "2026-05")
        agg = summary["by_origin"]["LAX"]
        assert agg["departures"] == 5
        assert agg["cancelled"] == 1
        # dep_del15 only counts non-cancelled flights: 2 of the 4 completed flights are DepDel15=1
        assert agg["dep_del15_count"] == 2

    def test_stats_computation_matches_expected_rates(self, ontime_fixture):
        summary = _aggregate_rows(ontime_fixture["lax_may_2026"], "2026-05")
        stats = _stats_from_summary("LAX", summary["by_origin"]["LAX"], "2026-05")
        assert stats["cancellation_rate"] == round(1 / 5, 4)
        assert stats["departure_delay_ge15_rate"] == round(2 / 4, 4)
        assert 0.0 <= stats["congestion_score"] <= 100.0

    def test_routes_include_real_distance_and_dest(self, ontime_fixture):
        summary = _aggregate_rows(ontime_fixture["anc_may_2026"], "2026-05")
        routes = _routes_from_summary("ANC", summary["by_origin"]["ANC"], "2026-05")
        by_dest = {r["dest"]: r for r in routes["routes"]}
        assert by_dest["ATL"]["distance_miles"] == 3417.0
        assert by_dest["SEA"]["distance_miles"] == 1449.0

    def test_client_uses_cache_after_first_month_summary(self, tmp_path, ontime_fixture, monkeypatch):
        import src.data_sources.bts_ontime_client as mod

        monkeypatch.setattr(mod, "CACHE_DIR", tmp_path)
        client = BtsOnTimeClient()
        summary = _aggregate_rows(ontime_fixture["sna_may_2026"], "2026-05")
        cache_file = tmp_path / "2026_05_summary.json"
        tmp_path.mkdir(parents=True, exist_ok=True)
        import json

        cache_file.write_text(json.dumps(summary))

        with patch.object(mod, "_recent_months", return_value=[(2026, 5)]), patch(
            "src.data_sources.bts_ontime_client.requests.get"
        ) as mock_get:
            result = client.get_airport_stats("SNA")

        mock_get.assert_not_called()  # cache hit, no download needed
        assert result.data["airport"] == "SNA"


class TestFaaArcgisClient:
    def test_parses_real_shaped_facility_and_runway_response(self, arcgis_fixture):
        def fake_get(url, params=None, timeout=None):
            if "NTAD_Aviation_Facilities" in url:
                return _fake_response(json_data=arcgis_fixture["facilities_BOS"])
            return _fake_response(json_data=arcgis_fixture["runways_BOS"])

        with patch("src.data_sources.faa_arcgis_client.requests.get", side_effect=fake_get):
            result = FaaArcgisClient().get_facility("BOS")

        assert result.data["num_runways"] == 6
        assert result.data["longest_runway_ft"] == 10083
        assert result.data["surface_types"] == ["ASPH"]

    def test_no_facility_features_falls_back_to_llm(self, arcgis_fixture):
        def fake_get(url, params=None, timeout=None):
            return _fake_response(json_data=arcgis_fixture["facilities_empty"])

        with patch("src.data_sources.faa_arcgis_client.requests.get", side_effect=fake_get), patch(
            "src.data_sources.faa_arcgis_client.estimate_structured"
        ) as mock_estimate:
            mock_estimate.return_value = {
                "num_runways": 2,
                "longest_runway_ft": 8000,
                "confidence": "low",
                "caveat": "test",
            }
            result = FaaArcgisClient().get_facility("ZZZ")

        assert result.data["is_llm_estimate"] is True


class TestFaaNasXmlParsing:
    def test_parses_multiple_delay_type_categories(self, nas_xml_fixture):
        events = _parse_events(nas_xml_fixture)
        airports = {e["airport"] for e in events}
        assert "LGA" in airports
        assert "SNA" in airports

    def test_arrival_departure_delay_detail_extracted(self, nas_xml_fixture):
        events = _parse_events(nas_xml_fixture)
        lga_event = next(e for e in events if e["airport"] == "LGA")
        assert lga_event["category"] == "General Arrival/Departure Delay Info"
        assert lga_event["delay_direction"] == "Departure"
        assert lga_event["min_delay"] == "31 minutes"

    def test_closure_detail_extracted(self, nas_xml_fixture):
        events = _parse_events(nas_xml_fixture)
        sna_event = next(e for e in events if e["airport"] == "SNA")
        assert sna_event["category"] == "Airport Closures"
        assert sna_event["reopen"] is not None


class TestNwsWeatherClient:
    def test_unit_value_extracts_and_handles_none(self):
        assert _unit_value({"value": 16093.44, "unitCode": "wmoUnit:m"}) == 16093.44
        assert _unit_value({"value": None, "unitCode": "wmoUnit:m"}) is None
        assert _unit_value(None) is None

    def test_direct_station_fetch_converts_units(self, nws_fixture):
        with patch("src.data_sources.nws_weather_client.requests.get") as mock_get:
            mock_get.side_effect = [
                _fake_response(json_data=nws_fixture["observation_sfo"]),
                _fake_response(json_data=nws_fixture["alerts_none"]),
            ]
            result = NwsWeatherClient().get_weather("SFO")

        assert result.data["visibility_miles"] == round(16093.44 / 1609.34, 1)
        assert result.data["conditions"] == "Mostly Cloudy"

    def test_duplicate_alerts_are_deduplicated(self, nws_fixture):
        with patch("src.data_sources.nws_weather_client.requests.get") as mock_get:
            mock_get.side_effect = [
                _fake_response(json_data=nws_fixture["observation_sfo"]),
                _fake_response(json_data=nws_fixture["alerts_ca_deduped_example"]),
            ]
            result = NwsWeatherClient().get_weather("SFO")

        assert result.data["active_alerts"] == ["Heat Advisory", "Extreme Heat Warning", "Red Flag Warning"]


class TestCensusAndBea:
    def test_census_population_parses_real_shaped_response(self, census_bea_fixture):
        with patch("src.data_sources.macro_data_client.requests.get") as mock_get:
            mock_get.return_value = _fake_response(json_data=census_bea_fixture["census_population_2024"])
            pop = _census_population("25", "025", 2024, 8.0)
        assert pop == 785121.0

    def test_census_no_data_row_returns_none(self, census_bea_fixture):
        with patch("src.data_sources.macro_data_client.requests.get") as mock_get:
            mock_get.return_value = _fake_response(json_data=census_bea_fixture["census_no_data"])
            assert _census_population("25", "025", 2024, 8.0) is None

    def test_bea_income_growth_computes_real_percentage(self, census_bea_fixture):
        with patch("src.data_sources.macro_data_client.requests.get") as mock_get:
            mock_get.return_value = _fake_response(json_data=census_bea_fixture["bea_income_growth"])
            growth = _bea_income_growth("25", "025", 8.0)
        expected = (105596 - 81383) / 81383 * 100
        assert round(growth, 4) == round(expected, 4)

    def test_bea_error_response_returns_none(self, census_bea_fixture):
        with patch("src.data_sources.macro_data_client.requests.get") as mock_get:
            mock_get.return_value = _fake_response(json_data=census_bea_fixture["bea_no_data"])
            assert _bea_income_growth("25", "025", 8.0) is None


class TestBtsAirfareLeakage:
    def test_fare_premium_and_leakage_computed_from_real_shaped_rows(self, airfare_fixture):
        from src.data_sources.macro_data_client import MacroDataClient

        with patch("src.data_sources.macro_data_client.requests.get") as mock_get:
            mock_get.return_value = _fake_response(json_data=airfare_fixture["bay_area_sample"])
            result = MacroDataClient().get_sfo_fare_and_leakage()

        assert result.data["avg_sfo_fare_usd"] > result.data["avg_oak_sjc_fare_usd"]
        assert result.data["fare_premium_pct"] > 0
        assert 0 <= result.data["total_leakage_pct"] <= 100
