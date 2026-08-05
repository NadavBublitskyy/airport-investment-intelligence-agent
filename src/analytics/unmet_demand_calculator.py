"""
SFO unmet flight demand index (deterministic, no LLM).

What this file does: combines five weighted signals into a single 0-100
"Estimated Unmet Demand Index" for an airport, and picks out which signal
is contributing the most so the agent can explain *why* demand looks
unmet (physical/gate limits, runway bottleneck, weather, or regional
pricing).

Why it exists: this answers the business question "what is the unmet
flight demand in SFO airport and why?" Unmet demand is not directly
observable (nobody counts the passengers who couldn't get a seat), so
this model approximates it from five signals that are each independently
measurable in real public data: real BTS T-100 traffic (load factor,
capacity growth), real BTS airfare survey data (fare premium AND, as a
genuine live proxy, the real passenger split between SFO/OAK/SJC for
shared origin markets - see `macro_data_client.py`), and real BTS
On-Time Performance cancellation rates.

How data moves: `calculate_unmet_demand()` pulls real T-100 growth/load-
factor data, the real SFO-vs-OAK/SJC fare+leakage data, and real
cancellation rates for SFO/OAK/SJC, combines them with fixed weights from
`src/config.py`, and returns an `UnmetDemandResult` with every component's
source traceable.

Formula (weights defined in ScoringConstants, sum to 1.0):
    index = 0.30 * load_factor_pressure
          + 0.25 * (passenger_growth_pct - seat_growth_pct)   [normalized]
          + 0.20 * market_leakage_to_OAK_SJC                  [normalized]
          + 0.15 * fare_premium                                [normalized]
          + 0.10 * cancellation_gap_vs_OAK_SJC                 [normalized]
"""

from __future__ import annotations

from src.analytics.kpi_models import ResolutionStepModel, UnmetDemandComponent, UnmetDemandResult
from src.config import scoring_constants
from src.data_sources.base_client import ResolutionStep
from src.data_sources.bts_ontime_client import BtsOnTimeClient
from src.data_sources.bts_t100_client import BtsT100Client
from src.data_sources.macro_data_client import MacroDataClient

EARLY_YEAR = 2019
LATE_YEAR = 2024
LEAKAGE_COMPARISON_AIRPORTS = ("OAK", "SJC")


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def calculate_unmet_demand(airport: str = "SFO") -> UnmetDemandResult:
    """
    Compute the Estimated Unmet Demand Index (0-100) for one airport.

    Note: the market-leakage and fare-premium components are specific to
    SFO vs. OAK/SJC (the real BTS airfare dataset only supports this kind
    of comparison for multi-airport metro markets - see
    `macro_data_client.py`). Calling this for an airport outside that
    Bay Area comparison isn't meaningful; the function is written for SFO.
    """
    c = scoring_constants
    t100_client = BtsT100Client()
    ontime_client = BtsOnTimeClient()
    macro_client = MacroDataClient()
    all_trace: list[ResolutionStep] = []

    t100_by_year = t100_client.get_annual_totals(airport, [EARLY_YEAR, LATE_YEAR])
    early_rd, late_rd = t100_by_year[EARLY_YEAR], t100_by_year[LATE_YEAR]
    all_trace += early_rd.trace + late_rd.trace
    early, late = early_rd.data, late_rd.data
    if not early or not late:
        raise ValueError(f"Could not resolve T-100 traffic data for {airport} from any source.")

    # A zero baseline (a new or essentially-inactive airport in EARLY_YEAR) can't
    # produce a meaningful growth rate - treat it as flat (0%) rather than crash.
    pax_growth_pct = (
        (late["passengers"] - early["passengers"]) / early["passengers"] * 100 if early["passengers"] else 0.0
    )
    seat_growth_pct = (
        (late["available_seats"] - early["available_seats"]) / early["available_seats"] * 100
        if early["available_seats"]
        else 0.0
    )
    load_factor = late["load_factor"]
    # Both years feed load_factor (via `late`) and capacity_growth_gap (via
    # both `early` and `late`) - checking only late_rd's trace missed the
    # case where EARLY_YEAR came from the LLM fallback but LATE_YEAR was
    # real, which would mark a partly-estimated component as fully real.
    load_factor_llm = any(
        s.tier == "llm_fallback" and s.status == "success" for s in early_rd.trace + late_rd.trace
    )

    fare_leakage_rd = macro_client.get_sfo_fare_and_leakage()
    all_trace += fare_leakage_rd.trace
    fare_leakage = fare_leakage_rd.data or {}
    fare_premium_pct = fare_leakage.get("fare_premium_pct", 0.0)
    leakage_pct = fare_leakage.get("total_leakage_pct", 0.0)
    fare_leakage_llm = any(s.tier == "llm_fallback" and s.status == "success" for s in fare_leakage_rd.trace)

    target_ontime_rd = ontime_client.get_airport_stats(airport)
    all_trace += target_ontime_rd.trace
    target_cancellation_rate = (target_ontime_rd.data or {}).get("cancellation_rate", 0.0)

    comparison_rates = []
    comparison_traces: list[ResolutionStep] = []
    for comp_airport in LEAKAGE_COMPARISON_AIRPORTS:
        comp_rd = ontime_client.get_airport_stats(comp_airport)
        all_trace += comp_rd.trace
        comparison_traces += comp_rd.trace
        comparison_rates.append((comp_rd.data or {}).get("cancellation_rate", 0.0))
    avg_comparison_cancellation_rate = sum(comparison_rates) / len(comparison_rates)
    cancellation_gap_pct = (target_cancellation_rate - avg_comparison_cancellation_rate) * 100
    # cancellation_gap_pct depends on the target airport's rate AND the
    # OAK/SJC comparison rates - checking only the target's trace missed an
    # LLM-estimated OAK or SJC rate, which would mark a partly-estimated
    # component as fully real.
    cancellation_llm = any(
        s.tier == "llm_fallback" and s.status == "success"
        for s in list(target_ontime_rd.trace) + comparison_traces
    )

    # --- Normalize each raw signal to 0-1 ---
    load_factor_pressure_norm = _clip01(
        (load_factor - c.unmet_comfortable_load_factor) / (1.0 - c.unmet_comfortable_load_factor)
    )
    capacity_growth_gap_raw = pax_growth_pct - seat_growth_pct
    capacity_growth_gap_norm = _clip01(capacity_growth_gap_raw / c.unmet_capacity_growth_gap_reference_pct)
    leakage_norm = _clip01(leakage_pct / c.unmet_market_leakage_reference_pct)
    fare_premium_norm = _clip01(fare_premium_pct / c.unmet_fare_premium_reference_pct)
    cancellation_gap_norm = _clip01(cancellation_gap_pct / c.unmet_cancellation_gap_reference_pct)

    components = [
        UnmetDemandComponent(
            name="load_factor_pressure",
            raw_value=round(load_factor, 4),
            normalized_0_1=round(load_factor_pressure_norm, 3),
            weight=c.unmet_weight_load_factor_pressure,
            weighted_contribution=round(c.unmet_weight_load_factor_pressure * load_factor_pressure_norm * 100, 2),
            uses_llm_estimate=load_factor_llm,
        ),
        UnmetDemandComponent(
            name="capacity_growth_gap_pax_minus_seats",
            raw_value=round(capacity_growth_gap_raw, 2),
            normalized_0_1=round(capacity_growth_gap_norm, 3),
            weight=c.unmet_weight_capacity_growth_gap,
            weighted_contribution=round(c.unmet_weight_capacity_growth_gap * capacity_growth_gap_norm * 100, 2),
            uses_llm_estimate=load_factor_llm,
        ),
        UnmetDemandComponent(
            name="market_leakage_to_oak_sjc",
            raw_value=round(leakage_pct, 2),
            normalized_0_1=round(leakage_norm, 3),
            weight=c.unmet_weight_market_leakage,
            weighted_contribution=round(c.unmet_weight_market_leakage * leakage_norm * 100, 2),
            uses_llm_estimate=fare_leakage_llm,
        ),
        UnmetDemandComponent(
            name="fare_premium_vs_oak_sjc",
            raw_value=round(fare_premium_pct, 2),
            normalized_0_1=round(fare_premium_norm, 3),
            weight=c.unmet_weight_fare_premium,
            weighted_contribution=round(c.unmet_weight_fare_premium * fare_premium_norm * 100, 2),
            uses_llm_estimate=fare_leakage_llm,
        ),
        UnmetDemandComponent(
            name="cancellation_gap_vs_oak_sjc",
            raw_value=round(cancellation_gap_pct, 3),
            normalized_0_1=round(cancellation_gap_norm, 3),
            weight=c.unmet_weight_cancellation_gap,
            weighted_contribution=round(c.unmet_weight_cancellation_gap * cancellation_gap_norm * 100, 2),
            uses_llm_estimate=cancellation_llm,
        ),
    ]

    index_0_100 = round(sum(comp.weighted_contribution for comp in components), 2)
    dominant = max(components, key=lambda comp: comp.weighted_contribution)
    root_cause_notes = _build_root_cause_notes(dominant.name, components)

    return UnmetDemandResult(
        airport=airport,
        unmet_demand_index_0_100=index_0_100,
        components=components,
        likely_dominant_cause=dominant.name,
        root_cause_notes=root_cause_notes,
        resolution_trace=[ResolutionStepModel.from_step(s) for s in all_trace],
    )


def _build_root_cause_notes(dominant_component_name: str, components: list[UnmetDemandComponent]) -> list[str]:
    """
    Turn the dominant weighted component into a plain-English root-cause explanation.

    This mapping is a simplifying assumption: each signal is associated
    with ONE most-likely physical explanation, even though in reality
    causes usually overlap. It exists so the LLM has a grounded starting
    point instead of guessing.
    """
    explanations = {
        "load_factor_pressure": (
            "Load factor is running high relative to a comfortable operating threshold, which most often "
            "points to a gate/terminal capacity limit (not enough gates or check-in/security throughput "
            "to add more flights) rather than a runway limit."
        ),
        "capacity_growth_gap_pax_minus_seats": (
            "Passenger demand has grown faster than available seats, suggesting airlines have not been "
            "able (or willing) to add capacity as fast as demand - consistent with slot, gate, or terminal "
            "constraints limiting schedule growth."
        ),
        "market_leakage_to_oak_sjc": (
            "A meaningful share of Bay Area passenger demand is flying from OAK or SJC instead, based on "
            "real ticketed-passenger volumes for the same origin markets - a classic signature of unmet "
            "demand: travelers substituting a less convenient airport because the primary one is full."
        ),
        "fare_premium_vs_oak_sjc": (
            "Average fares at SFO run measurably above OAK/SJC for comparable markets, which in a "
            "supply-constrained market is a pricing signal of scarcity."
        ),
        "cancellation_gap_vs_oak_sjc": (
            "Cancellations run notably higher than at nearby OAK/SJC, which is consistent with a "
            "weather/operational-reliability driver (e.g. marine-layer fog reducing arrival capacity) "
            "rather than a permanent physical constraint."
        ),
    }
    ranked = sorted(components, key=lambda comp: comp.weighted_contribution, reverse=True)
    notes = [explanations[dominant_component_name]]
    notes.append(
        "Component ranking by weighted contribution: "
        + ", ".join(f"{comp.name} ({comp.weighted_contribution})" for comp in ranked)
        + ". This index blends gate/terminal pressure, capacity growth gap, market leakage, fare "
        "signal, and weather-linked cancellations - it does not itself distinguish which physical asset "
        "(gates vs. runways) would need to expand; that requires the facility and weather detail tools."
    )
    return notes
