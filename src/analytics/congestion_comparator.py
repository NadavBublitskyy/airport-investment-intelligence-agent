"""
Airport congestion comparison across two or more airports (deterministic, no LLM).

What this file does: pulls real per-flight On-Time Performance stats and
real physical runway configuration for a set of airports and compares
their composite Congestion Scores side by side.

Why it exists: this answers the business question "compare LA and Santa
Ana airport congestion levels" with numbers pulled from a real BTS bulk
data file (see `bts_ontime_client.py` for why this isn't a queryable API)
and a fixed formula, not an LLM guess. The LLM's job is only to call this
function and narrate the result. It takes any list of airport codes, not
just a fixed pair - two is the default because that matches the seed
question, but there's nothing in the formula itself that's pairwise; the
result is a ranking, same as the other comparison tools in this project.

How data moves: `compare_congestion()` calls the real BTS On-Time
Performance client (for the Congestion Score) and the real FAA facilities
client (for runway count context) for each requested airport, then
returns a `CongestionComparisonResult` with every profile ranked, the
most/least congested named, and the full resolution trace for every
number involved.
"""

from __future__ import annotations

from src.analytics.kpi_models import AirportCongestionProfile, CongestionComparisonResult, ResolutionStepModel
from src.config import scoring_constants
from src.data_sources.bts_ontime_client import BtsOnTimeClient
from src.data_sources.faa_arcgis_client import FaaArcgisClient

DEFAULT_AIRPORTS = ("LAX", "SNA")


def compare_congestion(airports: list[str] | None = None) -> CongestionComparisonResult:
    """
    Compare real Congestion Scores and physical runway counts across a set of airports.

    Returns a `CongestionComparisonResult` ranked from most to least
    congested, so the agent can answer "which one is worse," "by how
    much," and, for more than two airports, "how do they all compare."
    Defaults to LAX vs. SNA if no list is given.
    """
    airports = list(airports or DEFAULT_AIRPORTS)
    ontime_client = BtsOnTimeClient()
    facility_client = FaaArcgisClient()

    profiles: list[AirportCongestionProfile] = []
    for airport in airports:
        ontime_result = ontime_client.get_airport_stats(airport)
        facility_result = facility_client.get_facility(airport)
        stats = ontime_result.data or {}
        facility = facility_result.data or {}
        trace = list(ontime_result.trace) + list(facility_result.trace)
        uses_llm = any(step.tier == "llm_fallback" and step.status == "success" for step in trace)

        profiles.append(
            AirportCongestionProfile(
                airport=airport,
                source_month=stats.get("source_month", "n/a (LLM estimate)"),
                avg_taxi_out_minutes=stats.get("avg_taxi_out_minutes", 0.0),
                departure_delay_ge15_rate=stats.get("departure_delay_ge15_rate", 0.0),
                cancellation_rate=stats.get("cancellation_rate", 0.0),
                avg_nas_delay_minutes=stats.get("avg_nas_delay_minutes", 0.0),
                congestion_score_0_100=stats.get("congestion_score", 0.0),
                num_runways=facility.get("num_runways"),
                uses_llm_estimate=uses_llm,
                resolution_trace=[ResolutionStepModel.from_step(s) for s in trace],
            )
        )

    profiles.sort(key=lambda p: p.congestion_score_0_100, reverse=True)
    gap = round(profiles[0].congestion_score_0_100 - profiles[-1].congestion_score_0_100, 2)

    c = scoring_constants
    return CongestionComparisonResult(
        airports=profiles,
        most_congested_airport=profiles[0].airport,
        least_congested_airport=profiles[-1].airport,
        congestion_score_gap=gap,
        weights={
            "taxi_out": c.congestion_weight_taxi_out,
            "departure_delay_ge15_rate": c.congestion_weight_delay_rate,
            "cancellation_rate": c.congestion_weight_cancellation_rate,
            "nas_delay": c.congestion_weight_nas_delay,
        },
        assumptions=[
            "Congestion Score = 0.35*taxi-out(normalized) + 0.30*departure-delay->=15min rate "
            "+ 0.20*cancellation rate + 0.15*NAS-delay-minutes(normalized), scaled to 0-100.",
            f"Taxi-out minutes are normalized against a {c.congestion_taxi_out_reference_minutes:.0f}-minute "
            "reference ceiling and NAS delay minutes against a "
            f"{c.congestion_nas_delay_reference_minutes:.0f}-minute reference ceiling, so the two "
            "minutes-based inputs are comparable to the two rate-based inputs.",
            "Based on the most recent full calendar month of real BTS On-Time Performance data "
            "available (see each profile's source_month) - a single-month snapshot, not a full-year "
            "average, chosen so the agent can answer in seconds rather than downloading 12 months "
            "of national flight data per question.",
            "Covers passenger-carrier flights reported to BTS On-Time Performance; cargo-only carrier "
            "operations are not included in this dataset.",
        ],
    )
