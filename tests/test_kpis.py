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

import pytest

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
        "load_factor": round(passengers / seats, 4) if seats else 0.0,
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

    def test_zero_baseline_passengers_does_not_raise_zero_division(self):
        # A new or essentially-inactive airport with zero EARLY_YEAR traffic
        # must not crash the whole ranking - growth is treated as flat (0%).
        t100 = {
            "BOS": {2019: (0.0, 100.0), 2024: (50.0, 100.0)},  # zero baseline passengers
            "BDL": {2019: (100.0, 200.0), 2024: (110.0, 200.0)},
        }
        facility = {a: (2, 8000) for a in self.AIRPORTS}
        growth = {a: 2.0 for a in self.AIRPORTS}
        p1, p2, p3 = self._patch_all(t100, facility, growth)

        with p1, p2, p3:
            result = score_new_england_expansion(self.AIRPORTS)

        bos_row = next(row for row in result.ranking if row.airport == "BOS")
        assert bos_row.passenger_growth_pct == 0.0

    def test_zero_baseline_seats_does_not_raise_zero_division(self):
        t100 = {
            "BOS": {2019: (0.0, 0.0), 2024: (50.0, 100.0)},  # zero baseline seats too
            "BDL": {2019: (100.0, 200.0), 2024: (110.0, 200.0)},
        }
        facility = {a: (2, 8000) for a in self.AIRPORTS}
        growth = {a: 2.0 for a in self.AIRPORTS}
        p1, p2, p3 = self._patch_all(t100, facility, growth)

        with p1, p2, p3:
            result = score_new_england_expansion(self.AIRPORTS)

        bos_row = next(row for row in result.ranking if row.airport == "BOS")
        assert bos_row.seat_growth_pct == 0.0

    def test_missing_runway_data_gets_neutral_score_not_extreme(self):
        # Before the fix, a missing/empty facility record defaulted to
        # num_runways=1, longest_ft=1 -> an artificially extreme constraint
        # value that could make "no data" look like "most constrained
        # airport in the set." It must land at a neutral 0.5 instead.
        t100 = {a: {2019: (100.0, 200.0), 2024: (110.0, 200.0)} for a in self.AIRPORTS}
        growth = {a: 2.0 for a in self.AIRPORTS}
        p1, _unused_facility_patch, p3 = self._patch_all(
            t100, {a: (2, 8000) for a in self.AIRPORTS}, growth
        )

        def fake_facility_missing_for_bos(self, airport):
            if airport == "BOS":
                return _rd({})  # every real source and the LLM fallback failed
            return _rd(_facility(2, 8000))

        with p1, patch(
            "src.analytics.expansion_scorer.FaaArcgisClient.get_facility", fake_facility_missing_for_bos
        ), p3:
            result = score_new_england_expansion(self.AIRPORTS)

        bos_row = next(row for row in result.ranking if row.airport == "BOS")
        assert bos_row.runway_constraint_score == 0.5
        assert any("BOS" in note and "runway" in note.lower() for note in result.assumptions)

    def test_runway_constraint_label_gives_deterministic_direction(self):
        # The LLM has been observed narrating runway_constraint_score
        # backwards (calling a high score "not a limiting factor"). The
        # label is pre-written in code specifically so the agent doesn't
        # have to interpret the raw number's direction itself.
        t100 = {a: {2019: (100.0, 200.0), 2024: (110.0, 200.0)} for a in self.AIRPORTS}
        growth = {a: 2.0 for a in self.AIRPORTS}
        # BOS: many long runways -> low constraint. BDL: few short runways -> high constraint.
        facility = {"BOS": (6, 10000), "BDL": (2, 5000)}
        p1, p2, p3 = self._patch_all(t100, facility, growth)

        with p1, p2, p3:
            result = score_new_england_expansion(self.AIRPORTS)

        by_airport = {row.airport: row for row in result.ranking}
        assert "MOST runway-constrained" in by_airport["BDL"].runway_constraint_label
        assert "LEAST runway-constrained" in by_airport["BOS"].runway_constraint_label

    def test_runway_constraint_label_middle_rank_is_numeric(self):
        t100 = {a: {2019: (100.0, 200.0), 2024: (110.0, 200.0)} for a in ["BOS", "BDL", "PVD"]}
        growth = {a: 2.0 for a in ["BOS", "BDL", "PVD"]}
        facility = {"BOS": (6, 10000), "BDL": (3, 7000), "PVD": (2, 5000)}
        p1, p2, p3 = self._patch_all(t100, facility, growth)

        with p1, p2, p3:
            result = score_new_england_expansion(["BOS", "BDL", "PVD"])

        bdl_row = next(row for row in result.ranking if row.airport == "BDL")
        assert bdl_row.runway_constraint_label == (
            "Ranked 2 of 3 for runway constraint in this comparison (1 = most constrained, 3 = least constrained)."
        )


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

    def test_airport_with_no_data_is_excluded_not_scored_zero(self):
        # Before the fix, a total data failure defaulted to
        # congestion_score=0.0 - indistinguishable from "genuinely zero
        # congestion" - which could make a no-data airport win "least
        # congested" by default. It must be excluded instead.
        def fake_stats(self, airport):
            if airport == "MDW":
                return _rd({})  # every real source and the LLM fallback failed
            s = {"ORD": (28.0, 0.35, 0.03, 18.0)}[airport]
            return _rd(
                {
                    "airport": airport,
                    "source_month": "2026-05",
                    "avg_taxi_out_minutes": s[0],
                    "departure_delay_ge15_rate": s[1],
                    "cancellation_rate": s[2],
                    "avg_nas_delay_minutes": s[3],
                    "congestion_score": _score(*s),
                    "is_llm_estimate": False,
                }
            )

        def fake_facility(self, airport):
            return _rd(_facility(8, 13000))

        with patch(
            "src.analytics.congestion_comparator.BtsOnTimeClient.get_airport_stats", fake_stats
        ), patch("src.analytics.congestion_comparator.FaaArcgisClient.get_facility", fake_facility):
            result = compare_congestion(["ORD", "MDW"])

        assert {p.airport for p in result.airports} == {"ORD"}
        assert result.most_congested_airport == "ORD"
        assert any("MDW" in note for note in result.assumptions)

    def test_raises_when_no_airport_has_any_data(self):
        def fake_stats(self, airport):
            return _rd({})

        def fake_facility(self, airport):
            return _rd({})

        with patch(
            "src.analytics.congestion_comparator.BtsOnTimeClient.get_airport_stats", fake_stats
        ), patch("src.analytics.congestion_comparator.FaaArcgisClient.get_facility", fake_facility):
            with pytest.raises(ValueError):
                compare_congestion(["ORD", "MDW"])


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

    def test_total_resolution_failure_raises_instead_of_fake_zero(self):
        # Before the fix, a total failure (both the live bulk file and the
        # nonexistent LLM fallback) produced the exact same
        # long_haul_flight_share_pct=0.0 as a real, resolved "no routes"
        # result - indistinguishable from genuine data. It must raise.
        def fake_routes_unavailable(self, airport):
            return _rd(None)

        with patch(
            "src.analytics.longhaul_calculator.BtsOnTimeClient.get_airport_routes", fake_routes_unavailable
        ):
            with pytest.raises(ValueError):
                calculate_long_haul_share("ANC")


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

    def test_zero_baseline_passengers_does_not_raise_zero_division(self):
        # Same guard as expansion_scorer: a zero EARLY_YEAR passenger count
        # must not crash the index - growth is treated as flat (0%) instead.
        t100_data = {
            2019: _rd(_t100_row("SFO", 2019, 0.0, 65_000_000.0)),
            2024: _rd(_t100_row("SFO", 2024, 52_000_000.0, 62_000_000.0)),
        }

        def fake_t100(self, airport, years):
            return {y: t100_data[y] for y in years}

        _, fare_leakage_patch, ontime_patch = self._patch()
        with patch(
            "src.analytics.unmet_demand_calculator.BtsT100Client.get_annual_totals", fake_t100
        ), fare_leakage_patch, ontime_patch:
            result = calculate_unmet_demand("SFO")

        assert 0.0 <= result.unmet_demand_index_0_100 <= 100.0

    def test_early_year_llm_estimate_flags_load_factor_and_growth_components(self):
        # Before the fix, only LATE_YEAR's trace was checked, so an
        # EARLY_YEAR value that came from the LLM fallback was silently
        # marked as fully real on load_factor_pressure and
        # capacity_growth_gap_pax_minus_seats.
        t100_data = {
            2019: ResolvedData(
                data=_t100_row("SFO", 2019, 50_000_000.0, 65_000_000.0),
                trace=[
                    ResolutionStep("primary", "BTS T-100", "failed", "timeout"),
                    ResolutionStep("llm_fallback", "LLM estimate", "success"),
                ],
            ),
            2024: _rd(_t100_row("SFO", 2024, 52_000_000.0, 62_000_000.0)),
        }

        def fake_t100(self, airport, years):
            return {y: t100_data[y] for y in years}

        _, fare_leakage_patch, ontime_patch = self._patch()
        with patch(
            "src.analytics.unmet_demand_calculator.BtsT100Client.get_annual_totals", fake_t100
        ), fare_leakage_patch, ontime_patch:
            result = calculate_unmet_demand("SFO")

        by_name = {c.name: c for c in result.components}
        assert by_name["load_factor_pressure"].uses_llm_estimate is True
        assert by_name["capacity_growth_gap_pax_minus_seats"].uses_llm_estimate is True

    def test_comparison_airport_llm_estimate_flags_cancellation_gap_component(self):
        # Before the fix, only the target airport's (SFO's) trace was
        # checked, so an LLM-estimated OAK or SJC cancellation rate was
        # silently marked as fully real on cancellation_gap_vs_oak_sjc.
        p1, p2, _unused_ontime_patch = self._patch()

        def fake_ontime(self, airport):
            rates = {"SFO": 0.023, "OAK": 0.012, "SJC": 0.011}
            trace = [ResolutionStep("primary", "BTS On-Time Performance", "success")]
            if airport == "OAK":
                trace = [
                    ResolutionStep("primary", "BTS On-Time Performance", "failed", "timeout"),
                    ResolutionStep("llm_fallback", "LLM estimate", "success"),
                ]
            return ResolvedData(data={"cancellation_rate": rates[airport], "is_llm_estimate": airport == "OAK"}, trace=trace)

        with p1, p2, patch(
            "src.analytics.unmet_demand_calculator.BtsOnTimeClient.get_airport_stats", fake_ontime
        ):
            result = calculate_unmet_demand("SFO")

        by_name = {c.name: c for c in result.components}
        assert by_name["cancellation_gap_vs_oak_sjc"].uses_llm_estimate is True
