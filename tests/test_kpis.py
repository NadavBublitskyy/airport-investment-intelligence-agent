"""
Tests for the deterministic scoring formulas in src/analytics/.

These tests never hit the network. They monkeypatch each analytics
module's data-source client *methods* (e.g. `BtsT100Client.get_annual_totals`)
to return canned `ResolvedData` objects built from the real-shaped
fixtures under `tests/fixtures/`, so the formulas run against realistic
numbers without any live API calls. This checks formula properties
(score ranges, ordering, weight sums) and specific arithmetic, not
network behavior - that's covered in `tests/test_data_sources.py`.
"""

from __future__ import annotations

from unittest.mock import patch

from src.analytics.congestion_comparator import compare_congestion
from src.analytics.expansion_scorer import _min_max_normalize, score_new_england_expansion
from src.analytics.longhaul_calculator import calculate_long_haul_share
from src.analytics.unmet_demand_calculator import calculate_unmet_demand
from src.config import scoring_constants
from src.data_sources.base_client import ResolutionStep, ResolvedData


def _rd(data, source_name="test-source", tier="primary"):
    return ResolvedData(data=data, trace=[ResolutionStep(tier, source_name, "success")])


def _t100_row(airport, year, passengers, seats, departures=1000.0, distance=1000.0):
    return {
        "airport": airport,
        "year": year,
        "passengers": passengers,
        "available_seats": seats,
        "scheduled_departures": departures,
        "departures_performed": departures,
        "avg_distance_miles": distance,
        "load_factor": round(passengers / seats, 4),
        "is_llm_estimate": False,
    }


def _facility(num_runways, longest_ft):
    return {"num_runways": num_runways, "longest_runway_ft": longest_ft, "is_llm_estimate": False}


def _growth(pop_growth_pct):
    return {"population_growth_pct_5yr": pop_growth_pct, "income_growth_pct_5yr": 10.0, "is_llm_estimate": False}


class TestMinMaxNormalize:
    def test_spreads_values_into_zero_one_range(self):
        assert _min_max_normalize([10, 20, 30]) == [0.0, 0.5, 1.0]

    def test_flat_input_maps_to_midpoint(self):
        assert _min_max_normalize([5, 5, 5]) == [0.5, 0.5, 0.5]


class TestExpansionScorer:
    AIRPORTS = ["BOS", "BDL"]

    def _patch_all(self, t100_by_airport, facility_by_airport, growth_by_airport):
        def fake_t100(self, airport, years):
            return {y: _rd(_t100_row(airport, y, *t100_by_airport[airport][y])) for y in years}

        def fake_facility(self, airport):
            return _rd(_facility(*facility_by_airport[airport]))

        def fake_growth(self, airport):
            return _rd(_growth(growth_by_airport[airport]))

        return (
            patch("src.analytics.expansion_scorer.BtsT100Client.get_annual_totals", fake_t100),
            patch("src.analytics.expansion_scorer.FaaArcgisClient.get_facility", fake_facility),
            patch("src.analytics.expansion_scorer.MacroDataClient.get_regional_growth", fake_growth),
        )

    def test_higher_growth_and_load_factor_ranks_first(self):
        t100 = {
            "BOS": {2019: (100.0, 200.0), 2024: (180.0, 200.0)},  # strong pax growth, flat seats
            "BDL": {2019: (100.0, 200.0), 2024: (110.0, 200.0)},  # weak pax growth
        }
        facility = {"BOS": (2, 8000), "BDL": (2, 8000)}
        growth = {"BOS": 5.0, "BDL": 1.0}
        p1, p2, p3 = self._patch_all(t100, facility, growth)
        with p1, p2, p3:
            result = score_new_england_expansion(self.AIRPORTS)

        assert result.ranking[0].airport == "BOS"
        scores = [row.overall_score_0_100 for row in result.ranking]
        assert scores == sorted(scores, reverse=True)
        for row in result.ranking:
            assert 0.0 <= row.overall_score_0_100 <= 100.0

    def test_weights_sum_to_one(self):
        t100 = {a: {2019: (100.0, 200.0), 2024: (110.0, 200.0)} for a in self.AIRPORTS}
        facility = {a: (2, 8000) for a in self.AIRPORTS}
        growth = {a: 2.0 for a in self.AIRPORTS}
        p1, p2, p3 = self._patch_all(t100, facility, growth)
        with p1, p2, p3:
            result = score_new_england_expansion(self.AIRPORTS)
        assert round(sum(result.weights.values()), 6) == 1.0

    def test_llm_estimate_is_flagged_on_the_row(self):
        t100 = {a: {2019: (100.0, 200.0), 2024: (110.0, 200.0)} for a in self.AIRPORTS}
        facility = {a: (2, 8000) for a in self.AIRPORTS}
        growth = {a: 2.0 for a in self.AIRPORTS}
        p1, p2, p3 = self._patch_all(t100, facility, growth)

        def fake_facility_with_llm(self, airport):
            trace = [ResolutionStep("primary", "FAA ArcGIS", "failed", "timeout"),
                     ResolutionStep("llm_fallback", "LLM estimate", "success")]
            return ResolvedData(data=_facility(2, 8000), trace=trace)

        with p1, patch(
            "src.analytics.expansion_scorer.FaaArcgisClient.get_facility", fake_facility_with_llm
        ), p3:
            result = score_new_england_expansion(self.AIRPORTS)

        assert all(row.uses_llm_estimate for row in result.ranking)


class TestCongestionComparator:
    def _patch(self, stats_by_airport, facility_by_airport):
        def fake_stats(self, airport):
            s = stats_by_airport[airport]
            data = {
                "airport": airport,
                "source_month": "2026-05",
                "avg_taxi_out_minutes": s[0],
                "departure_delay_ge15_rate": s[1],
                "cancellation_rate": s[2],
                "avg_nas_delay_minutes": s[3],
                "congestion_score": _score(*s),
                "is_llm_estimate": False,
            }
            return _rd(data)

        def fake_facility(self, airport):
            return _rd(_facility(*facility_by_airport[airport]))

        return (
            patch("src.analytics.congestion_comparator.BtsOnTimeClient.get_airport_stats", fake_stats),
            patch("src.analytics.congestion_comparator.FaaArcgisClient.get_facility", fake_facility),
        )

    def test_most_congested_airport_is_identified_correctly(self):
        stats = {"LAX": (25.0, 0.3, 0.02, 15.0), "SNA": (10.0, 0.1, 0.01, 3.0)}
        facility = {"LAX": (4, 12000), "SNA": (2, 6000)}
        p1, p2 = self._patch(stats, facility)
        with p1, p2:
            result = compare_congestion(["LAX", "SNA"])

        assert result.most_congested_airport == "LAX"
        assert result.least_congested_airport == "SNA"
        assert result.congestion_score_gap > 0

    def test_weights_sum_to_one(self):
        stats = {"LAX": (25.0, 0.3, 0.02, 15.0), "SNA": (10.0, 0.1, 0.01, 3.0)}
        facility = {"LAX": (4, 12000), "SNA": (2, 6000)}
        p1, p2 = self._patch(stats, facility)
        with p1, p2:
            result = compare_congestion(["LAX", "SNA"])
        assert round(sum(result.weights.values()), 6) == 1.0

    def test_defaults_to_lax_vs_sna_when_no_list_given(self):
        stats = {"LAX": (25.0, 0.3, 0.02, 15.0), "SNA": (10.0, 0.1, 0.01, 3.0)}
        facility = {"LAX": (4, 12000), "SNA": (2, 6000)}
        p1, p2 = self._patch(stats, facility)
        with p1, p2:
            result = compare_congestion()
        assert {p.airport for p in result.airports} == {"LAX", "SNA"}

    def test_generalizes_to_more_than_two_airports(self):
        stats = {
            "ORD": (28.0, 0.35, 0.03, 18.0),
            "MDW": (12.0, 0.12, 0.01, 4.0),
            "JFK": (20.0, 0.22, 0.02, 10.0),
        }
        facility = {"ORD": (8, 13000), "MDW": (3, 7000), "JFK": (4, 14500)}
        p1, p2 = self._patch(stats, facility)
        with p1, p2:
            result = compare_congestion(["ORD", "MDW", "JFK"])

        assert {p.airport for p in result.airports} == {"ORD", "MDW", "JFK"}
        assert result.most_congested_airport == "ORD"
        assert result.least_congested_airport == "MDW"
        scores = [p.congestion_score_0_100 for p in result.airports]
        assert scores == sorted(scores, reverse=True)


def _score(taxi_out, delay_rate, cancel_rate, nas_delay):
    c = scoring_constants
    taxi_norm = min(taxi_out / c.congestion_taxi_out_reference_minutes, 1.0)
    nas_norm = min(nas_delay / c.congestion_nas_delay_reference_minutes, 1.0)
    return round(
        (
            c.congestion_weight_taxi_out * taxi_norm
            + c.congestion_weight_delay_rate * delay_rate
            + c.congestion_weight_cancellation_rate * cancel_rate
            + c.congestion_weight_nas_delay * nas_norm
        )
        * 100,
        2,
    )


class TestLongHaulCalculator:
    def _patch(self, routes, source_month="2026-05"):
        def fake_routes(self, airport):
            return _rd({"airport": airport, "source_month": source_month, "routes": routes, "is_llm_estimate": False})

        return patch("src.analytics.longhaul_calculator.BtsOnTimeClient.get_airport_routes", fake_routes)

    def test_known_route_classification(self):
        routes = [
            {"dest": "JFK", "distance_miles": 3365.0, "departures_performed": 10},
            {"dest": "SEA", "distance_miles": 1449.0, "departures_performed": 40},
        ]
        with self._patch(routes):
            result = calculate_long_haul_share("ANC")

        by_dest = {r.dest: r for r in result.routes}
        assert by_dest["JFK"].is_long_haul is True
        assert by_dest["SEA"].is_long_haul is False

    def test_flight_share_percentage_is_correct(self):
        routes = [
            {"dest": "JFK", "distance_miles": 3365.0, "departures_performed": 10},
            {"dest": "SEA", "distance_miles": 1449.0, "departures_performed": 40},
        ]
        with self._patch(routes):
            result = calculate_long_haul_share("ANC")
        assert result.long_haul_flight_share_pct == 20.0  # 10 / 50 * 100
        assert result.total_departures == 50.0

    def test_threshold_matches_config(self):
        with self._patch([]):
            result = calculate_long_haul_share("ANC")
        assert result.long_haul_threshold_miles == scoring_constants.long_haul_min_distance_miles

    def test_no_routes_returns_zero_share_not_a_crash(self):
        with self._patch([]):
            result = calculate_long_haul_share("ANC")
        assert result.long_haul_flight_share_pct == 0.0
        assert result.routes == []


class TestUnmetDemandCalculator:
    def _patch(self):
        t100_data = {
            2019: _rd(_t100_row("SFO", 2019, 50_000_000.0, 65_000_000.0)),
            2024: _rd(_t100_row("SFO", 2024, 52_000_000.0, 62_000_000.0)),
        }

        def fake_t100(self, airport, years):
            return {y: t100_data[y] for y in years}

        def fake_fare_leakage(self):
            return _rd(
                {
                    "avg_sfo_fare_usd": 275.0,
                    "avg_oak_sjc_fare_usd": 240.0,
                    "fare_premium_pct": 14.58,
                    "total_leakage_pct": 16.5,
                    "is_llm_estimate": False,
                }
            )

        def fake_ontime(self, airport):
            rates = {"SFO": 0.023, "OAK": 0.012, "SJC": 0.011}
            return _rd({"cancellation_rate": rates[airport], "is_llm_estimate": False})

        return (
            patch("src.analytics.unmet_demand_calculator.BtsT100Client.get_annual_totals", fake_t100),
            patch("src.analytics.unmet_demand_calculator.MacroDataClient.get_sfo_fare_and_leakage", fake_fare_leakage),
            patch("src.analytics.unmet_demand_calculator.BtsOnTimeClient.get_airport_stats", fake_ontime),
        )

    def test_index_within_0_100_and_components_sum_correctly(self):
        p1, p2, p3 = self._patch()
        with p1, p2, p3:
            result = calculate_unmet_demand("SFO")

        assert 0.0 <= result.unmet_demand_index_0_100 <= 100.0
        total = round(sum(c.weighted_contribution for c in result.components), 2)
        assert total == result.unmet_demand_index_0_100

    def test_five_components_present(self):
        p1, p2, p3 = self._patch()
        with p1, p2, p3:
            result = calculate_unmet_demand("SFO")
        names = {c.name for c in result.components}
        assert names == {
            "load_factor_pressure",
            "capacity_growth_gap_pax_minus_seats",
            "market_leakage_to_oak_sjc",
            "fare_premium_vs_oak_sjc",
            "cancellation_gap_vs_oak_sjc",
        }

    def test_dominant_cause_is_a_known_component(self):
        p1, p2, p3 = self._patch()
        with p1, p2, p3:
            result = calculate_unmet_demand("SFO")
        assert result.likely_dominant_cause in {c.name for c in result.components}
        assert len(result.root_cause_notes) >= 1
