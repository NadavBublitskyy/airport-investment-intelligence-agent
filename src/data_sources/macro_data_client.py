"""
Client for regional macroeconomic, Census, and BTS airfare data (real, live).

What this file does: fetches three real things -
1. County-level population growth (US Census ACS 5-year estimates) and
   per-capita income growth (BEA Regional CAINC1) for an airport's home
   county - used by the New England expansion score.
2. Real average fares and passenger-volume splits between SFO, OAK, and
   SJC, from BTS's real Consumer Airfare Report (`tfrh-tu9e`) - this one
   dataset genuinely covers both the "fare premium" and "market leakage"
   inputs to the SFO unmet-demand index, since it reports real ticketed
   passenger counts by which Bay Area airport people actually flew from.

Why it exists: population/income growth and airfare data are two of the
few inputs in this project that need a signup API key (Census, BEA -
both free, both confirmed working during development). BTS's airfare
data needs no key. There is no free public API for cross-airport "market
leakage" as an abstract concept - but this dataset's raw passenger counts
by airport, for the exact same origin markets, are a genuine live proxy
for it, not an invented number.

How data moves: `get_regional_growth()` and `get_sfo_fare_and_leakage()`
each call `resolve()`, which tries the real Census/BEA/BTS APIs first and
falls back to an LLM estimate only if the live calls fail (e.g. missing
API key, network issue).
"""

from __future__ import annotations

from typing import Any

import requests
from pydantic import BaseModel, Field

from src.config import data_source_settings
from src.data_sources.base_client import BaseDataClient, LiveTier, ResolvedData
from src.data_sources.llm_fallback_client import estimate_structured

CENSUS_URL = "https://api.census.gov/data/{year}/acs/acs5"
BEA_URL = "https://apps.bea.gov/api/data/"
AIRFARE_URL = "https://data.transportation.gov/resource/tfrh-tu9e.json"

EARLY_YEAR = 2019
LATE_YEAR_CENSUS = 2024
LATE_YEAR_BEA = 2023

SOURCE_NAME_CENSUS_BEA = "US Census ACS5 + BEA CAINC1 (api.census.gov, apps.bea.gov)"
SOURCE_NAME_AIRFARE = "BTS Consumer Airfare Report (data.transportation.gov, tfrh-tu9e)"

# Airport -> (state FIPS, county FIPS, human label). Public, stable
# reference facts (like a phone country-code table) used only to query
# the real Census/BEA APIs - not a substitute for their data.
AIRPORT_COUNTY_FIPS = {
    "BOS": ("25", "025", "Suffolk County, MA"),
    "BDL": ("09", "003", "Hartford County, CT"),
    "PVD": ("44", "007", "Providence County, RI"),
    "PWM": ("23", "005", "Cumberland County, ME"),
    "MHT": ("33", "011", "Hillsborough County, NH"),
    "BTV": ("50", "007", "Chittenden County, VT"),
    "LAX": ("06", "037", "Los Angeles County, CA"),
    "SNA": ("06", "059", "Orange County, CA"),
    "ANC": ("02", "020", "Anchorage Municipality, AK"),
    "SFO": ("06", "081", "San Mateo County, CA"),
    "OAK": ("06", "001", "Alameda County, CA"),
    "SJC": ("06", "085", "Santa Clara County, CA"),
}


class GrowthEstimate(BaseModel):
    """Schema for the LLM last-resort estimate of one county's growth."""

    population_growth_pct_5yr: float = Field(description="Estimated 5-year population growth, percent")
    income_growth_pct: float = Field(description="Estimated per-capita income growth over the same period, percent")
    confidence: str = Field(description="'low', 'medium', or 'high' - be honest, this is a rough estimate")
    caveat: str = Field(description="One sentence on why this is uncertain")


class FareLeakageEstimate(BaseModel):
    """Schema for the LLM last-resort estimate of SFO's fare premium and market leakage."""

    fare_premium_pct_vs_oak_sjc: float = Field(description="Estimated % SFO fares run above OAK/SJC fares")
    leakage_pct_to_oak_sjc: float = Field(description="Estimated % of Bay Area demand using OAK/SJC instead")
    confidence: str = Field(description="'low', 'medium', or 'high' - be honest, this is a rough estimate")
    caveat: str = Field(description="One sentence on why this is uncertain")


class MacroDataClient(BaseDataClient):
    """Fetches real regional population/income growth and real Bay Area airfare/leakage data."""

    client_name = "Macro (Census/BEA/BTS Airfare)"

    def get_regional_growth(self, airport: str) -> ResolvedData:
        """Get real 5-year population growth and per-capita income growth for one airport's home county."""
        return self.resolve(airport=airport, mode="growth")

    def get_sfo_fare_and_leakage(self) -> ResolvedData:
        """Get real SFO-vs-OAK/SJC fare premium and passenger-split leakage proxy."""
        return self.resolve(airport="SFO", mode="fare_leakage")

    # --- BaseDataClient hooks ---

    def _live_tiers(self, **kwargs: Any) -> list[LiveTier]:
        airport = kwargs["airport"]
        mode = kwargs["mode"]
        if mode == "growth":
            return [("primary", SOURCE_NAME_CENSUS_BEA, lambda: self._fetch_growth(airport))]
        return [("primary", SOURCE_NAME_AIRFARE, self._fetch_fare_leakage)]

    def _llm_fallback(self, **kwargs: Any):
        airport = kwargs["airport"]
        mode = kwargs["mode"]
        if mode == "growth":
            return self._growth_llm_fallback(airport)
        return self._fare_leakage_llm_fallback()

    def _growth_llm_fallback(self, airport: str):
        def fetch() -> dict:
            label = AIRPORT_COUNTY_FIPS.get(airport, (None, None, airport))[2]
            result = estimate_structured(
                schema=GrowthEstimate,
                question=(
                    f"Estimate the 5-year population growth and per-capita income growth for the "
                    f"area around {airport} airport ({label})."
                ),
                context="The real Census ACS and BEA APIs were unreachable or returned no data (check API keys).",
            )
            return {
                "airport": airport,
                "population_growth_pct_5yr": result["population_growth_pct_5yr"],
                "income_growth_pct_5yr": result["income_growth_pct"],
                "is_llm_estimate": True,
                "llm_confidence": result["confidence"],
                "llm_caveat": result["caveat"],
            }

        return ("LLM estimate (OpenRouter)", fetch)

    def _fare_leakage_llm_fallback(self):
        def fetch() -> dict:
            result = estimate_structured(
                schema=FareLeakageEstimate,
                question="Estimate SFO's average airfare premium over OAK/SJC, and what share of Bay Area air travel demand uses OAK/SJC instead of SFO.",
                context="The real BTS Consumer Airfare Report API was unreachable.",
            )
            return {
                "airport": "SFO",
                "fare_premium_pct": result["fare_premium_pct_vs_oak_sjc"],
                "total_leakage_pct": result["leakage_pct_to_oak_sjc"],
                "is_llm_estimate": True,
                "llm_confidence": result["confidence"],
                "llm_caveat": result["caveat"],
            }

        return ("LLM estimate (OpenRouter)", fetch)

    # --- Live fetch implementations ---

    @staticmethod
    def _fetch_growth(airport: str) -> dict[str, Any] | None:
        location = AIRPORT_COUNTY_FIPS.get(airport)
        if not location or not data_source_settings.census_api_key:
            return None
        state_fips, county_fips, label = location
        timeout = data_source_settings.request_timeout_seconds

        pop_early = _census_population(state_fips, county_fips, EARLY_YEAR, timeout)
        pop_late = _census_population(state_fips, county_fips, LATE_YEAR_CENSUS, timeout)
        if pop_early is None or pop_late is None:
            return None
        population_growth_pct = (pop_late - pop_early) / pop_early * 100

        income_growth_pct = None
        if data_source_settings.bea_api_key:
            income_growth_pct = _bea_income_growth(state_fips, county_fips, timeout)

        return {
            "airport": airport,
            "county_label": label,
            "population_growth_pct_5yr": round(population_growth_pct, 2),
            "income_growth_pct_5yr": round(income_growth_pct, 2) if income_growth_pct is not None else None,
            "is_llm_estimate": False,
        }

    @staticmethod
    def _fetch_fare_leakage() -> dict[str, Any] | None:
        response = requests.get(
            AIRFARE_URL,
            params={
                "$where": "airport_1 in ('SFO','OAK','SJC') OR airport_2 in ('SFO','OAK','SJC')",
                "$select": "airport_1, airport_2, citymarketid_1, fare, passengers",
                "$order": "year DESC, quarter DESC",
                "$limit": 5000,
            },
            timeout=data_source_settings.request_timeout_seconds,
        )
        response.raise_for_status()
        rows = response.json()
        if not rows:
            return None

        bay_area = {"SFO", "OAK", "SJC"}
        sfo_fares, alt_fares = [], []
        sfo_pax, alt_pax = 0.0, 0.0
        for row in rows:
            dest = row["airport_2"] if row["airport_1"] not in bay_area else row["airport_1"]
            if dest not in bay_area:
                continue
            fare = float(row.get("fare") or 0)
            pax = float(row.get("passengers") or 0)
            if dest == "SFO":
                sfo_fares.append(fare)
                sfo_pax += pax
            else:
                alt_fares.append(fare)
                alt_pax += pax

        if not sfo_fares or not alt_fares:
            return None

        avg_sfo_fare = sum(sfo_fares) / len(sfo_fares)
        avg_alt_fare = sum(alt_fares) / len(alt_fares)
        fare_premium_pct = (avg_sfo_fare - avg_alt_fare) / avg_alt_fare * 100
        total_pax = sfo_pax + alt_pax
        leakage_pct = (alt_pax / total_pax * 100) if total_pax else 0.0

        return {
            "airport": "SFO",
            "avg_sfo_fare_usd": round(avg_sfo_fare, 2),
            "avg_oak_sjc_fare_usd": round(avg_alt_fare, 2),
            "fare_premium_pct": round(fare_premium_pct, 2),
            "sample_sfo_passengers": round(sfo_pax, 1),
            "sample_oak_sjc_passengers": round(alt_pax, 1),
            "total_leakage_pct": round(leakage_pct, 2),
            "is_llm_estimate": False,
        }


def _census_population(state_fips: str, county_fips: str, year: int, timeout: float) -> float | None:
    response = requests.get(
        CENSUS_URL.format(year=year),
        params={
            "get": "NAME,B01003_001E",
            "for": f"county:{county_fips}",
            "in": f"state:{state_fips}",
            "key": data_source_settings.census_api_key,
        },
        timeout=timeout,
    )
    if response.status_code != 200:
        return None
    rows = response.json()
    if len(rows) < 2:
        return None
    return float(rows[1][1])


def _bea_income_growth(state_fips: str, county_fips: str, timeout: float) -> float | None:
    response = requests.get(
        BEA_URL,
        params={
            "UserID": data_source_settings.bea_api_key,
            "method": "GetData",
            "datasetname": "Regional",
            "TableName": "CAINC1",
            "LineCode": 3,  # per-capita personal income
            "GeoFips": f"{state_fips}{county_fips}",
            "Year": f"{EARLY_YEAR},{LATE_YEAR_BEA}",
            "ResultFormat": "JSON",
        },
        timeout=timeout,
    )
    if response.status_code != 200:
        return None
    body = response.json()
    data = body.get("BEAAPI", {}).get("Results", {}).get("Data")
    if not data or len(data) < 2:
        return None
    by_year = {row["TimePeriod"]: float(row["DataValue"]) for row in data}
    early, late = by_year.get(str(EARLY_YEAR)), by_year.get(str(LATE_YEAR_BEA))
    if not early or not late:
        return None
    return (late - early) / early * 100
