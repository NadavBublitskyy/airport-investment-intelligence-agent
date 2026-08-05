"""
Client for FAA Aviation Facilities & Runways data (real, live ArcGIS REST API).

What this file does: fetches real physical airport metadata - acreage,
facility use, and every runway's real length, width, and surface type -
from the FAA's own NTAD ArcGIS FeatureServer layers.

Why it exists: physical capacity is a hard constraint no amount of
demand can overcome. This client's field names were verified directly
against the live service during development (see README research notes)
- e.g. BOS really has 6 runways, the longest at 10,083 ft, confirmed by
querying the live `Runways_View` layer. There is no FAA-published "hub
category" or "gate count" field in this dataset, so this client does not
invent one; `airport_category` some other analyses might expect simply
isn't part of the real physical-facility data model.

How data moves: `get_facility()` calls `resolve()`, which queries the
live Facilities layer (for acreage/use) and the live Runways layer (for
runway counts/lengths/surface), joined by the real `ARPT_ID` field. If
either ArcGIS call fails, the LLM fallback estimates a rough runway
count/length instead.
"""

from __future__ import annotations

from typing import Any

import requests
from pydantic import BaseModel, Field

from src.config import data_source_settings
from src.data_sources.base_client import BaseDataClient, LiveTier, ResolvedData
from src.data_sources.llm_fallback_client import estimate_structured

FACILITIES_URL = (
    "https://services.arcgis.com/xOi1kZaI0eWDREZv/arcgis/rest/services/"
    "NTAD_Aviation_Facilities/FeatureServer/0/query"
)
RUNWAYS_URL = (
    "https://services.arcgis.com/xOi1kZaI0eWDREZv/ArcGIS/rest/services/"
    "Runways_View/FeatureServer/0/query"
)
SOURCE_NAME = "FAA NTAD Aviation Facilities & Runways (ArcGIS FeatureServer)"


class FacilityEstimate(BaseModel):
    """Schema for the LLM last-resort estimate of one airport's runway configuration."""

    num_runways: int = Field(description="Estimated number of runways")
    longest_runway_ft: int = Field(description="Estimated length of the longest runway, in feet")
    confidence: str = Field(description="'low', 'medium', or 'high' - be honest, this is a rough estimate")
    caveat: str = Field(description="One sentence on why this is uncertain")


class FaaArcgisClient(BaseDataClient):
    """Fetches real FAA facility and runway metadata for an airport."""

    client_name = "FAA ArcGIS Facilities & Runways"

    def get_facility(self, airport: str) -> ResolvedData:
        """Get real physical metadata (runways, surface, acreage) for one airport."""
        return self.resolve(airport=airport)

    # --- BaseDataClient hooks ---

    def _live_tiers(self, **kwargs: Any) -> list[LiveTier]:
        airport = kwargs["airport"]
        return [("primary", SOURCE_NAME, lambda: self._fetch(airport))]

    def _llm_fallback(self, **kwargs: Any):
        airport = kwargs["airport"]

        def fetch() -> dict:
            result = estimate_structured(
                schema=FacilityEstimate,
                question=f"Estimate {airport}'s number of runways and its longest runway length in feet.",
                context=(
                    "The real FAA NTAD ArcGIS facilities/runways service was unreachable. "
                    f"This is airport {airport}, a US commercial airport."
                ),
            )
            return {
                "airport": airport,
                "num_runways": result["num_runways"],
                "longest_runway_ft": result["longest_runway_ft"],
                "runway_lengths_ft": [result["longest_runway_ft"]],
                "acreage": None,
                "is_llm_estimate": True,
                "llm_confidence": result["confidence"],
                "llm_caveat": result["caveat"],
            }

        return ("LLM estimate (OpenRouter)", fetch)

    @staticmethod
    def _fetch(airport: str) -> dict[str, Any] | None:
        timeout = data_source_settings.request_timeout_seconds

        facility_resp = requests.get(
            FACILITIES_URL,
            params={
                "where": f"ARPT_ID='{airport}'",
                "outFields": "ARPT_ID,ARPT_NAME,STATE_CODE,ACREAGE,FACILITY_USE_CODE",
                "f": "json",
            },
            timeout=timeout,
        )
        facility_resp.raise_for_status()
        facility_features = facility_resp.json().get("features") or []
        if not facility_features:
            return None
        facility_attrs = facility_features[0]["attributes"]

        runway_resp = requests.get(
            RUNWAYS_URL,
            params={
                "where": f"ARPT_ID='{airport}'",
                "outFields": "RWY_ID,RWY_LEN,RWY_WIDTH,SURFACE_TYPE_CODE",
                "f": "json",
            },
            timeout=timeout,
        )
        runway_resp.raise_for_status()
        runway_features = runway_resp.json().get("features") or []
        runways = [f["attributes"] for f in runway_features]
        lengths = sorted((r["RWY_LEN"] for r in runways if r.get("RWY_LEN")), reverse=True)
        surfaces = sorted({r["SURFACE_TYPE_CODE"] for r in runways if r.get("SURFACE_TYPE_CODE")})

        return {
            "airport": airport,
            "airport_name": facility_attrs.get("ARPT_NAME"),
            "state_code": facility_attrs.get("STATE_CODE"),
            "acreage": facility_attrs.get("ACREAGE"),
            "facility_use_code": facility_attrs.get("FACILITY_USE_CODE"),
            "num_runways": len(runways),
            "longest_runway_ft": lengths[0] if lengths else None,
            "runway_lengths_ft": lengths,
            "surface_types": surfaces,
            "is_llm_estimate": False,
        }
