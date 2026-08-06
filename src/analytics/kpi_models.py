"""
Pydantic schemas for analytics inputs and outputs.

What this file does: defines the typed data shapes that flow out of the
four scoring modules in src/analytics/. Using Pydantic models (not plain
dicts) means every score carries self-documenting field names, gets
validated automatically, and can be turned into JSON for the LLM tool
layer with one call (`.model_dump()`).

Why it exists: keeping these shapes in one file means the data-source
clients, the analytics engines, and the agent tools all agree on exactly
what a "score" or "comparison" looks like. `ResolutionStepModel` is the
Pydantic mirror of `ResolutionStep` in `src/data_sources/base_client.py`
- every analytics result carries a `resolution_trace`, the ordered list
of every data source attempted (real API succeeded, real API failed and
moved on, or LLM last-resort used) so the agent can show its work for
every number in an answer.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.data_sources.base_client import ResolutionStep


class ResolutionStepModel(BaseModel):
    """One attempt to get a piece of data - which source, what tier, and what happened."""

    tier: str = Field(description="'primary', 'secondary', or 'llm_fallback'")
    source_name: str
    status: str = Field(description="'success' or 'failed'")
    detail: str = ""

    @classmethod
    def from_step(cls, step: ResolutionStep) -> "ResolutionStepModel":
        return cls(tier=step.tier, source_name=step.source_name, status=step.status, detail=step.detail)


class AirportExpansionScore(BaseModel):
    """One airport's New England terminal-expansion candidacy score."""

    airport: str
    capacity_pressure_score: float = Field(description="0-1 normalized: passenger growth outpacing seat growth")
    load_factor_score: float = Field(description="0-1 normalized: how full flights are")
    runway_constraint_score: float = Field(description="0-1 normalized: how physically limited the airfield is")
    runway_constraint_label: str = Field(
        description=(
            "Deterministic, pre-written description of this airport's runway-constraint rank within "
            "the comparison set - use this verbatim (or a faithful paraphrase) when narrating; do not "
            "reinterpret runway_constraint_score's direction yourself, since higher/lower is easy to "
            "get backwards."
        )
    )
    census_growth_score: float = Field(description="0-1 normalized: regional population growth")
    latest_load_factor: float
    passenger_growth_pct: float
    seat_growth_pct: float
    overall_score_0_100: float
    uses_llm_estimate: bool = Field(description="True if any input for this airport came from the LLM fallback")
    resolution_trace: list[ResolutionStepModel]


class ExpansionRankingResult(BaseModel):
    """Full ranked list for the New England terminal-expansion question."""

    ranking: list[AirportExpansionScore]
    weights: dict[str, float]
    assumptions: list[str]


class AirportCongestionProfile(BaseModel):
    """One airport's congestion inputs and composite score."""

    airport: str
    source_month: str = Field(description="Which real BTS On-Time Performance month this is based on")
    avg_taxi_out_minutes: float
    departure_delay_ge15_rate: float
    cancellation_rate: float
    avg_nas_delay_minutes: float
    congestion_score_0_100: float
    num_runways: int | None
    uses_llm_estimate: bool
    resolution_trace: list[ResolutionStepModel]


class CongestionComparisonResult(BaseModel):
    """Congestion comparison across two or more airports (defaults to LAX vs. SNA)."""

    airports: list[AirportCongestionProfile]
    most_congested_airport: str
    least_congested_airport: str
    congestion_score_gap: float = Field(description="Congestion score of the most minus the least congested airport")
    weights: dict[str, float]
    assumptions: list[str]


class LongHaulRoute(BaseModel):
    """One real route from the origin airport, from actual BTS flight-level data."""

    dest: str
    distance_miles: float
    departures_performed: float
    is_long_haul: bool


class LongHaulResult(BaseModel):
    """Anchorage (or any airport) long-haul flight share."""

    airport: str
    source_month: str = Field(description="Which real BTS On-Time Performance month this is based on")
    long_haul_threshold_miles: float
    long_haul_flight_share_pct: float = Field(description="% of departures that are long-haul, by flight count")
    total_departures: float
    routes: list[LongHaulRoute]
    uses_llm_estimate: bool
    assumptions: list[str]
    resolution_trace: list[ResolutionStepModel]


class UnmetDemandComponent(BaseModel):
    """One weighted component of the unmet demand index, for transparency."""

    name: str
    raw_value: float
    normalized_0_1: float
    weight: float
    weighted_contribution: float
    uses_llm_estimate: bool


class UnmetDemandResult(BaseModel):
    """SFO (or any airport) estimated unmet demand index and root-cause breakdown."""

    airport: str
    unmet_demand_index_0_100: float
    components: list[UnmetDemandComponent]
    likely_dominant_cause: str
    root_cause_notes: list[str]
    resolution_trace: list[ResolutionStepModel]
