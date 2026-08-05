"""
New England terminal expansion ranking (deterministic, no LLM).

What this file does: scores every requested New England airport (default:
BOS, BDL, PVD, PWM, MHT, BTV) on a 0-100 scale that estimates how strong a
candidate it is for terminal expansion investment.

Why it exists: this answers the business question "which airports in New
England are strong candidates for terminal expansion?" with a repeatable,
auditable formula instead of an LLM's free-form opinion. The LLM's job
(see src/agent/) is only to call this function and explain the result in
plain English - never to invent the ranking itself. Every raw input this
formula uses comes from a real live public API; the LLM is only ever
consulted as a last resort for one specific input if every real source
for it fails (see `resolution_trace` on each result), and even then it
fills in one fact, not the score.

How data moves: `score_new_england_expansion()` pulls traffic data (BTS
T-100), physical constraints (FAA facilities), and regional demand
context (Census population + BEA income) for each airport, combines four
normalized factors with fixed weights from `src/config.py`, and returns a
fully ranked, explainable `ExpansionRankingResult`.

Scoring formula (weights defined in ScoringConstants, sum to 1.0):
    score = 0.35 * capacity_pressure   (passenger growth outpacing seat growth)
          + 0.25 * load_factor         (how full flights already are)
          + 0.20 * runway_constraint   (how physically limited the airfield is)
          + 0.20 * census_growth       (regional population growth)

Each factor is min-max normalized across the airports being compared, so
the result is a *relative* ranking within the requested set, not an
absolute score. Adding or removing airports from the request changes the
normalization range and can shift the resulting numbers - this is a
documented modeling choice, not a bug.
"""

from __future__ import annotations

from src.analytics.kpi_models import AirportExpansionScore, ExpansionRankingResult, ResolutionStepModel
from src.config import scoring_constants
from src.data_sources.bts_t100_client import BtsT100Client
from src.data_sources.faa_arcgis_client import FaaArcgisClient
from src.data_sources.macro_data_client import MacroDataClient

EARLY_YEAR = 2019
LATE_YEAR = 2024


def _min_max_normalize(values: list[float]) -> list[float]:
    """Scale a list of numbers to the 0-1 range. Flat lists (all equal) map to 0.5 for every value."""
    lo, hi = min(values), max(values)
    if hi == lo:
        return [0.5 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def _runway_constraint_raw(facility: dict | None) -> float:
    """Higher output = more physically constrained (fewer/shorter runways)."""
    facility = facility or {}
    num_runways = facility.get("num_runways") or 1
    longest_ft = facility.get("longest_runway_ft") or 1
    capacity_proxy = num_runways * (longest_ft / 10_000)
    return 1.0 / capacity_proxy if capacity_proxy else 0.0


def score_new_england_expansion(airports: list[str] | None = None) -> ExpansionRankingResult:
    """
    Rank New England airports by terminal-expansion investment attractiveness.

    Returns an `ExpansionRankingResult` sorted from strongest to weakest
    candidate, with every input factor - and exactly where it came from -
    exposed so an analyst can see why an airport ranked where it did.
    """
    airports = list(airports or scoring_constants.new_england_airports)

    t100_client = BtsT100Client()
    facility_client = FaaArcgisClient()
    macro_client = MacroDataClient()

    raw_rows: list[dict] = []
    for airport in airports:
        t100_by_year = t100_client.get_annual_totals(airport, [EARLY_YEAR, LATE_YEAR])
        facility_result = facility_client.get_facility(airport)
        growth_result = macro_client.get_regional_growth(airport)

        early_rd, late_rd = t100_by_year.get(EARLY_YEAR), t100_by_year.get(LATE_YEAR)
        early, late = early_rd.data if early_rd else None, late_rd.data if late_rd else None
        if not early or not late:
            continue  # both real API and LLM fallback failed for this airport; can't score it

        trace = list(early_rd.trace) + list(late_rd.trace) + list(facility_result.trace) + list(growth_result.trace)
        uses_llm = any(step.tier == "llm_fallback" and step.status == "success" for step in trace)

        # A zero baseline (a new or essentially-inactive airport in EARLY_YEAR) can't
        # produce a meaningful growth rate - treat it as flat (0%) rather than crash.
        pax_growth_pct = (
            (late["passengers"] - early["passengers"]) / early["passengers"] * 100
            if early["passengers"]
            else 0.0
        )
        seat_growth_pct = (
            (late["available_seats"] - early["available_seats"]) / early["available_seats"] * 100
            if early["available_seats"]
            else 0.0
        )
        facility = facility_result.data or {}
        growth = growth_result.data or {}

        raw_rows.append(
            {
                "airport": airport,
                "capacity_pressure_raw": pax_growth_pct - seat_growth_pct,
                "load_factor_raw": late["load_factor"] or 0.0,
                "runway_constraint_raw": _runway_constraint_raw(facility),
                "census_growth_raw": growth.get("population_growth_pct_5yr", 0.0),
                "passenger_growth_pct": round(pax_growth_pct, 2),
                "seat_growth_pct": round(seat_growth_pct, 2),
                "latest_load_factor": late["load_factor"],
                "uses_llm_estimate": uses_llm,
                "resolution_trace": [ResolutionStepModel.from_step(s) for s in trace],
            }
        )

    if not raw_rows:
        raise ValueError(
            f"Could not resolve T-100 traffic data for any of {airports} from either the real BTS "
            "API or the LLM fallback. Check network access and OPENROUTER_API_KEY."
        )

    capacity_norm = _min_max_normalize([r["capacity_pressure_raw"] for r in raw_rows])
    load_norm = _min_max_normalize([r["load_factor_raw"] for r in raw_rows])
    runway_norm = _min_max_normalize([r["runway_constraint_raw"] for r in raw_rows])
    census_norm = _min_max_normalize([r["census_growth_raw"] for r in raw_rows])

    c = scoring_constants
    scored: list[AirportExpansionScore] = []
    for row, cap, load, rwy, census in zip(raw_rows, capacity_norm, load_norm, runway_norm, census_norm):
        overall = (
            c.expansion_weight_capacity_pressure * cap
            + c.expansion_weight_load_factor * load
            + c.expansion_weight_runway_constraint * rwy
            + c.expansion_weight_census_growth * census
        ) * 100
        scored.append(
            AirportExpansionScore(
                airport=row["airport"],
                capacity_pressure_score=round(cap, 3),
                load_factor_score=round(load, 3),
                runway_constraint_score=round(rwy, 3),
                census_growth_score=round(census, 3),
                latest_load_factor=row["latest_load_factor"],
                passenger_growth_pct=row["passenger_growth_pct"],
                seat_growth_pct=row["seat_growth_pct"],
                overall_score_0_100=round(overall, 2),
                uses_llm_estimate=row["uses_llm_estimate"],
                resolution_trace=row["resolution_trace"],
            )
        )

    scored.sort(key=lambda s: s.overall_score_0_100, reverse=True)

    return ExpansionRankingResult(
        ranking=scored,
        weights={
            "capacity_pressure": c.expansion_weight_capacity_pressure,
            "load_factor": c.expansion_weight_load_factor,
            "runway_constraint": c.expansion_weight_runway_constraint,
            "census_growth": c.expansion_weight_census_growth,
        },
        assumptions=[
            f"Growth is measured {EARLY_YEAR} -> {LATE_YEAR} (5-year window, spans the 2020 pandemic dip).",
            "Each factor is min-max normalized within the requested airport set, so this is a relative "
            "ranking, not an absolute score comparable across different airport groups.",
            "Runway constraint is a proxy (num_runways x longest_runway_ft), not a full airfield capacity "
            "model (it ignores taxiway layout, terminal gate count, and airspace/ATC constraints - the "
            "real FAA facilities dataset does not publish a gate count or hub-size category).",
            "Regional population growth is county-level (US Census ACS 5-year estimate for the airport's "
            "home county), used as a proxy for the whole catchment area.",
            "Any airport whose data_sources show tier='llm_fallback' had at least one real API fail; "
            "that airport's score is less reliable than one built entirely from live data.",
        ],
    )
