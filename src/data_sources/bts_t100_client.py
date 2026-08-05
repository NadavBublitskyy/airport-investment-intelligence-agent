"""
Client for BTS T-100 traffic data (real, live Socrata API).

What this file does: fetches passenger, seat, and departure volume for an
airport - the real "AFF - T100 Segment Summary By Origin Airport" dataset
(Socrata ID `r495-tyji`) published by BTS on `data.bts.gov` - and derives
load factor and flight fulfillment rate from it.

Why it exists: capacity questions ("is this airport out of room to
grow?") start with raw traffic volume, and this is the actual, verified
public dataset for that (verified by direct query during development -
see README for the research notes). BTS does not expose per-route
(origin-destination) T-100 data as a queryable API; that gap is covered
separately by `bts_ontime_client.py`, whose flight-level file also
carries real per-flight distance and destination.

How data moves: `get_annual_totals()` calls `resolve()` from
`base_client.py`, which tries the real SODA query first and, only if
that fails, asks the LLM for a rough estimate (see
`llm_fallback_client.py`). Every call returns a `ResolvedData` carrying
both the data and the full attempt trace, so the agent can show exactly
where a number came from.
"""

from __future__ import annotations

from typing import Any

import requests
from pydantic import BaseModel, Field

from src.data_sources.base_client import BaseDataClient, LiveTier, ResolvedData
from src.data_sources.llm_fallback_client import estimate_structured
from src.config import data_source_settings

SODA_RESOURCE_URL = "https://data.bts.gov/resource/r495-tyji.json"
SOURCE_NAME = "BTS T-100 by-origin-airport summary (data.bts.gov, r495-tyji)"


class T100AnnualEstimate(BaseModel):
    """Schema for the LLM last-resort estimate of one airport-year's T-100 totals."""

    passengers: float = Field(description="Estimated total annual passengers departing this airport")
    available_seats: float = Field(description="Estimated total annual available seats")
    scheduled_departures: float = Field(description="Estimated total annual scheduled departures")
    departures_performed: float = Field(description="Estimated total annual departures actually flown")
    avg_distance_miles: float = Field(description="Estimated average flight distance in miles")
    confidence: str = Field(description="'low', 'medium', or 'high' - be honest, this is a rough estimate")
    caveat: str = Field(description="One sentence on why this is uncertain")


class BtsT100Client(BaseDataClient):
    """Fetches real BTS T-100 by-origin-airport passenger/seat/departure volume."""

    client_name = "BTS T-100"

    def get_annual_totals(self, airport: str, years: list[int]) -> dict[int, ResolvedData]:
        """
        Get one `ResolvedData` per requested year for one airport.

        Each year is resolved independently so a partial failure (e.g.
        2019 data unavailable but 2024 fine) is visible per-year in the
        trace, rather than silently blending into one combined attempt.
        """
        return {year: self.resolve(airport=airport, year=year) for year in years}

    # --- BaseDataClient hooks ---

    def _live_tiers(self, **kwargs: Any) -> list[LiveTier]:
        airport = kwargs["airport"]
        year = kwargs["year"]
        return [("primary", SOURCE_NAME, lambda: self._fetch_soda(airport, year))]

    def _llm_fallback(self, **kwargs: Any):
        airport = kwargs["airport"]
        year = kwargs["year"]

        def fetch() -> dict:
            result = estimate_structured(
                schema=T100AnnualEstimate,
                question=(
                    f"Estimate {airport}'s total commercial air traffic for calendar year {year}: "
                    "total passengers, total available seats, total scheduled departures, total "
                    "departures actually performed, and average flight distance in miles."
                ),
                context=(
                    f"The real BTS T-100 dataset (data.bts.gov, r495-tyji) was unreachable or had no "
                    f"data for {airport} in {year}. This is airport {airport}, a US commercial airport."
                ),
            )
            return {
                "airport": airport,
                "year": year,
                "passengers": result["passengers"],
                "available_seats": result["available_seats"],
                "scheduled_departures": result["scheduled_departures"],
                "departures_performed": result["departures_performed"],
                "avg_distance_miles": result["avg_distance_miles"],
                "is_llm_estimate": True,
                "llm_confidence": result["confidence"],
                "llm_caveat": result["caveat"],
            }

        return ("LLM estimate (OpenRouter)", fetch)

    @staticmethod
    def _fetch_soda(airport: str, year: int) -> dict[str, Any] | None:
        params = {
            "$select": (
                "origin_airport_code, year, sum(total_passengers) as passengers, "
                "sum(total_seats) as seats, sum(total_departures) as departures, "
                "avg(total_distance_flight_sm) as avg_distance"
            ),
            "$where": f"origin_airport_code='{airport}' AND year='{year}'",
            "$group": "origin_airport_code, year",
        }
        response = requests.get(
            SODA_RESOURCE_URL, params=params, timeout=data_source_settings.request_timeout_seconds
        )
        response.raise_for_status()
        rows = response.json()
        if not rows:
            return None
        row = rows[0]
        passengers = float(row.get("passengers", 0))
        seats = float(row.get("seats", 0))
        departures = float(row.get("departures", 0))
        return {
            "airport": airport,
            "year": year,
            "passengers": passengers,
            "available_seats": seats,
            "scheduled_departures": departures,  # SODA source has no separate "scheduled" figure
            "departures_performed": departures,
            "avg_distance_miles": float(row.get("avg_distance", 0)),
            "load_factor": round(passengers / seats, 4) if seats else None,
            "flight_fulfillment_rate": 1.0,  # real dataset only reports departures actually performed
            "is_llm_estimate": False,
        }
