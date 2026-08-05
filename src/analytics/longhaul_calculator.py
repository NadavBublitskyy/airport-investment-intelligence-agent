"""
Anchorage (ANC) long-haul flight share (deterministic, no LLM).

What this file does: takes real per-flight BTS On-Time Performance data
for an airport (which carries real destination and distance per flight -
see `bts_ontime_client.py`) and computes what share of its departures are
"long-haul."

Why it exists: this answers the business question "what is the percentage
of long-haul flights out of Anchorage airport?" There is no queryable
public API for route-level T-100 segment data (confirmed during
development - BTS only publishes it as a non-API bulk download); the
real per-flight On-Time Performance file turned out to carry the same
real Origin/Dest/Distance facts needed here, so this module reuses it
rather than needing a second heavy data pipeline.

How data moves: `calculate_long_haul_share()` calls the real BTS
On-Time Performance client's route method, classifies each real route as
long-haul using the fixed `long_haul_min_distance_miles` threshold from
`src/config.py`, and returns a `LongHaulResult` with the percentage, the
full route list, and where the data came from.

Rule (fixed by the assignment, not tunable): long-haul = nonstop distance
>= 3,000 statute miles.
"""

from __future__ import annotations

from src.analytics.kpi_models import LongHaulResult, LongHaulRoute, ResolutionStepModel
from src.config import scoring_constants
from src.data_sources.bts_ontime_client import BtsOnTimeClient


def calculate_long_haul_share(airport: str = "ANC") -> LongHaulResult:
    """
    Compute the real long-haul flight share for one airport, from real per-flight route data.

    Returns a `LongHaulResult`. If no route data is available for the
    airport at all (real API and LLM fallback both failed), all
    percentages come back as 0.0 with an empty route list.
    """
    threshold = scoring_constants.long_haul_min_distance_miles
    ontime_client = BtsOnTimeClient()
    result = ontime_client.get_airport_routes(airport)
    data = result.data or {}
    rows = data.get("routes", [])
    uses_llm = any(step.tier == "llm_fallback" and step.status == "success" for step in result.trace)

    routes: list[LongHaulRoute] = []
    total_departures = 0.0
    long_haul_departures = 0.0

    for row in rows:
        is_long_haul = row["distance_miles"] >= threshold
        routes.append(
            LongHaulRoute(
                dest=row["dest"],
                distance_miles=row["distance_miles"],
                departures_performed=row["departures_performed"],
                is_long_haul=is_long_haul,
            )
        )
        total_departures += row["departures_performed"]
        if is_long_haul:
            long_haul_departures += row["departures_performed"]

    flight_share = round(long_haul_departures / total_departures * 100, 2) if total_departures else 0.0
    routes.sort(key=lambda r: r.distance_miles, reverse=True)

    return LongHaulResult(
        airport=airport,
        source_month=data.get("source_month", "n/a (LLM estimate)"),
        long_haul_threshold_miles=threshold,
        long_haul_flight_share_pct=flight_share,
        total_departures=total_departures,
        routes=routes,
        uses_llm_estimate=uses_llm,
        assumptions=[
            "Long-haul share is by flight count. A passenger-weighted version was in the original "
            "design, but real per-route passenger counts are not published via any free live API "
            "(only route-level distance/departure counts are, via BTS On-Time Performance) - rather "
            "than fabricate a passenger weighting, this metric is flight-count-only and says so.",
            f"Based on the most recent full calendar month of real BTS On-Time Performance data "
            f"available ({data.get('source_month', 'n/a')}), not a full-year average.",
            "Covers passenger-carrier flights reported to BTS On-Time Performance; cargo-only carrier "
            "operations (a meaningful share of ANC's real traffic) are not included in this dataset.",
        ],
        resolution_trace=[ResolutionStepModel.from_step(s) for s in result.trace],
    )
