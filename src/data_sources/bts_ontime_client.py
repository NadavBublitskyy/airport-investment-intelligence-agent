"""
Client for BTS On-Time Performance data (real, live bulk file, not an API).

What this file does: fetches real per-flight operational data - taxi-out
minutes, departure delay, cancellations, NAS delay minutes, and (as a
bonus) real per-flight destination and distance - for an airport, and
derives the Congestion Score and long-haul route breakdown from it.

Why it exists: BTS does not publish this data as a queryable REST API
(verified during development: no Socrata dataset exposes taxi-out/delay/
cancellation fields). It IS published as a real, free, no-key-required
monthly bulk file at a predictable URL:
`https://transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip`
This client downloads that file, extracts it once, and builds a compact
per-airport summary that both the congestion comparator (taxi-out, delay
rate, cancellation rate, NAS delay) and the long-haul calculator (real
per-route distance) read from - one real dataset, two questions answered.

How data moves: the first call for any airport in a given month triggers
one full download + one full scan of the national file (all airports,
all routes, in a single pass), which is cached to disk as a small JSON
summary (`.cache/bts_ontime/{year}_{month}_summary.json`). Every
subsequent call - for any airport, for either question - reads that
small cache instead of re-downloading or re-scanning. If the live file
is unreachable, the client falls back to an LLM estimate (see
`llm_fallback_client.py`), exactly like every other client.
"""

from __future__ import annotations

import csv
import io
import json
import zipfile
from datetime import datetime, timezone
from typing import Any

import requests
from pydantic import BaseModel, Field

from src.config import PROJECT_ROOT, scoring_constants
from src.data_sources.base_client import BaseDataClient, LiveTier, ResolvedData
from src.data_sources.llm_fallback_client import estimate_structured

BULK_FILE_URL = (
    "https://transtats.bts.gov/PREZIP/"
    "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"
)
CACHE_DIR = PROJECT_ROOT / ".cache" / "bts_ontime"
SOURCE_NAME = "BTS On-Time Performance bulk file (transtats.bts.gov)"
MAX_MONTHS_TO_TRY = 6
DOWNLOAD_TIMEOUT_SECONDS = 120


class OnTimeEstimate(BaseModel):
    """Schema for the LLM last-resort estimate of one airport's operating conditions."""

    avg_taxi_out_minutes: float = Field(description="Estimated average taxi-out time in minutes")
    departure_delay_ge15_rate: float = Field(description="Estimated fraction (0-1) of departures delayed >=15 min")
    cancellation_rate: float = Field(description="Estimated fraction (0-1) of flights cancelled")
    avg_nas_delay_minutes: float = Field(description="Estimated average NAS (air traffic system) delay minutes")
    confidence: str = Field(description="'low', 'medium', or 'high' - be honest, this is a rough estimate")
    caveat: str = Field(description="One sentence on why this is uncertain")


class BtsOnTimeClient(BaseDataClient):
    """Fetches real BTS On-Time Performance stats and real per-route distance for an airport."""

    client_name = "BTS On-Time Performance"

    def get_airport_stats(self, airport: str) -> ResolvedData:
        """Get taxi-out/delay/cancellation/NAS-delay stats and the Congestion Score for one airport."""
        return self.resolve(airport=airport, mode="stats")

    def get_airport_routes(self, airport: str) -> ResolvedData:
        """Get real per-destination route counts and distances for one airport (for long-haul share)."""
        return self.resolve(airport=airport, mode="routes")

    # --- BaseDataClient hooks ---

    def _live_tiers(self, **kwargs: Any) -> list[LiveTier]:
        airport = kwargs["airport"]
        mode = kwargs["mode"]
        return [("primary", SOURCE_NAME, lambda: self._fetch(airport, mode))]

    def _llm_fallback(self, **kwargs: Any):
        airport = kwargs["airport"]
        mode = kwargs["mode"]
        if mode != "stats":
            # No sensible LLM estimate for a full per-route breakdown; only
            # the aggregate stats question gets a last-resort estimate.
            return None

        def fetch() -> dict:
            result = estimate_structured(
                schema=OnTimeEstimate,
                question=(
                    f"Estimate {airport}'s typical current operating conditions: average taxi-out "
                    "time in minutes, the fraction of departures delayed 15+ minutes, the fraction "
                    "of flights cancelled, and average NAS (air traffic control system) delay minutes."
                ),
                context=(
                    "The real BTS On-Time Performance bulk file (transtats.bts.gov) was unreachable "
                    f"for the last {MAX_MONTHS_TO_TRY} months. This is airport {airport}, a US "
                    "commercial airport."
                ),
            )
            score = _congestion_score(
                result["avg_taxi_out_minutes"],
                result["departure_delay_ge15_rate"],
                result["cancellation_rate"],
                result["avg_nas_delay_minutes"],
            )
            return {
                "airport": airport,
                "avg_taxi_out_minutes": result["avg_taxi_out_minutes"],
                "departure_delay_ge15_rate": result["departure_delay_ge15_rate"],
                "cancellation_rate": result["cancellation_rate"],
                "avg_nas_delay_minutes": result["avg_nas_delay_minutes"],
                "congestion_score": score,
                "is_llm_estimate": True,
                "llm_confidence": result["confidence"],
                "llm_caveat": result["caveat"],
            }

        return ("LLM estimate (OpenRouter)", fetch)

    # --- Data pipeline: download -> extract -> single-pass aggregate -> cache ---

    def _fetch(self, airport: str, mode: str) -> dict | None:
        summary = self._get_or_build_month_summary()
        if summary is None:
            return None
        by_origin = summary["by_origin"].get(airport)
        if not by_origin:
            return None
        if mode == "stats":
            return _stats_from_summary(airport, by_origin, summary["source_month"])
        return _routes_from_summary(airport, by_origin, summary["source_month"])

    def _get_or_build_month_summary(self) -> dict | None:
        for year, month in _recent_months(MAX_MONTHS_TO_TRY):
            cache_path = CACHE_DIR / f"{year}_{month:02d}_summary.json"
            if cache_path.exists():
                return json.loads(cache_path.read_text())

            csv_bytes = self._download_month(year, month)
            if csv_bytes is None:
                continue  # this month isn't published yet, try an earlier one

            summary = _build_summary(csv_bytes, source_month=f"{year}-{month:02d}")
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(summary))
            return summary
        return None

    @staticmethod
    def _download_month(year: int, month: int) -> bytes | None:
        url = BULK_FILE_URL.format(year=year, month=month)
        response = requests.get(url, timeout=DOWNLOAD_TIMEOUT_SECONDS)
        if response.status_code != 200 or not response.content:
            return None
        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            csv_name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
            return zf.read(csv_name)


def _recent_months(count: int):
    """Yield (year, month) tuples going backward from ~2 months ago (BTS publishes with a lag)."""
    now = datetime.now(timezone.utc)
    year, month = now.year, now.month
    for _ in range(2):  # start 2 months back; the current/prior month is rarely published yet
        month -= 1
        if month == 0:
            month, year = 12, year - 1
    for _ in range(count):
        yield year, month
        month -= 1
        if month == 0:
            month, year = 12, year - 1


def _build_summary(csv_bytes: bytes, source_month: str) -> dict:
    """Decode the national monthly CSV file and aggregate every origin airport's stats and routes."""
    text_stream = io.TextIOWrapper(io.BytesIO(csv_bytes), encoding="utf-8", errors="replace")
    return _aggregate_rows(csv.DictReader(text_stream), source_month)


def _aggregate_rows(rows, source_month: str) -> dict:
    """
    Pure aggregation over an iterable of flight-row dicts (real BTS On-Time
    Performance column names: Origin, Dest, Distance, TaxiOut, DepDel15,
    Cancelled, NASDelay). Separated from CSV/zip decoding so it can be
    unit-tested directly against small fixture rows, without needing a
    real 276MB national file.
    """
    by_origin: dict[str, dict] = {}
    for row in rows:
        origin = row.get("Origin")
        if not origin:
            continue
        agg = by_origin.setdefault(
            origin,
            {
                "departures": 0,
                "cancelled": 0,
                "dep_del15_count": 0,
                "taxi_out_sum": 0.0,
                "taxi_out_n": 0,
                "nas_delay_sum": 0.0,
                "routes": {},
            },
        )
        agg["departures"] += 1
        cancelled = _to_float(row.get("Cancelled")) == 1.0
        if cancelled:
            agg["cancelled"] += 1
        else:
            if _to_float(row.get("DepDel15")) == 1.0:
                agg["dep_del15_count"] += 1
            nas_delay = _to_float(row.get("NASDelay"))
            agg["nas_delay_sum"] += nas_delay or 0.0
        taxi_out = _to_float(row.get("TaxiOut"))
        if taxi_out is not None:
            agg["taxi_out_sum"] += taxi_out
            agg["taxi_out_n"] += 1

        dest = row.get("Dest")
        distance = _to_float(row.get("Distance"))
        if dest and distance is not None:
            route = agg["routes"].setdefault(dest, {"departures": 0, "distance_miles": distance})
            route["departures"] += 1

    return {"source_month": source_month, "by_origin": by_origin}


def _to_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _stats_from_summary(airport: str, agg: dict, source_month: str) -> dict:
    departures = agg["departures"]
    completed = departures - agg["cancelled"]
    avg_taxi_out = agg["taxi_out_sum"] / agg["taxi_out_n"] if agg["taxi_out_n"] else 0.0
    dep_delay_rate = agg["dep_del15_count"] / completed if completed else 0.0
    cancellation_rate = agg["cancelled"] / departures if departures else 0.0
    avg_nas_delay = agg["nas_delay_sum"] / completed if completed else 0.0
    return {
        "airport": airport,
        "source_month": source_month,
        "sample_departures": departures,
        "avg_taxi_out_minutes": round(avg_taxi_out, 2),
        "departure_delay_ge15_rate": round(dep_delay_rate, 4),
        "cancellation_rate": round(cancellation_rate, 4),
        "avg_nas_delay_minutes": round(avg_nas_delay, 2),
        "congestion_score": _congestion_score(avg_taxi_out, dep_delay_rate, cancellation_rate, avg_nas_delay),
        "is_llm_estimate": False,
    }


def _routes_from_summary(airport: str, agg: dict, source_month: str) -> dict:
    routes = [
        {"dest": dest, "distance_miles": r["distance_miles"], "departures_performed": r["departures"]}
        for dest, r in agg["routes"].items()
    ]
    return {
        "airport": airport,
        "source_month": source_month,
        "total_departures": agg["departures"],
        "routes": routes,
        "is_llm_estimate": False,
    }


def _congestion_score(taxi_out: float, delay_rate: float, cancellation_rate: float, nas_delay: float) -> float:
    """Shared formula so both the live path and the LLM-fallback path compute the score identically."""
    c = scoring_constants
    taxi_out_norm = min(taxi_out / c.congestion_taxi_out_reference_minutes, 1.0)
    nas_delay_norm = min(nas_delay / c.congestion_nas_delay_reference_minutes, 1.0)
    score = (
        c.congestion_weight_taxi_out * taxi_out_norm
        + c.congestion_weight_delay_rate * delay_rate
        + c.congestion_weight_cancellation_rate * cancellation_rate
        + c.congestion_weight_nas_delay * nas_delay_norm
    ) * 100
    return round(score, 2)
