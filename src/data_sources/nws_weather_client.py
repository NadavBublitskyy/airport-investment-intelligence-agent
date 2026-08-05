"""
Client for the National Weather Service (NWS) API (real, live data).

What this file does: fetches real current weather conditions - visibility,
wind speed/gusts, and active alerts - for an airport, using the real NWS
station-observation endpoints (verified during development against
`KBOS`: returned real live visibility, wind, and conditions).

Why it exists: weather is one of the most common root causes of airport
congestion (SFO's marine-layer fog is the textbook example - it reduces
arrival capacity because its close-spaced parallel runways can't both run
simultaneous instrument approaches in low visibility). The SFO unmet
demand analysis needs to be able to say "this dip is weather, not a
permanent physical constraint," and that requires a real current reading.

How data moves: `get_weather()` calls `resolve()`, which tries the ICAO
weather-station code for the airport directly (`K` + IATA code for the
contiguous US; a small override table below handles the few airports in
this project outside that pattern, e.g. Anchorage's real code is `PANC`,
not `KANC`). If that fails, it falls back to the generic NWS lat/lon grid
lookup. Active alerts are fetched by state. If NWS is entirely
unreachable, the LLM fallback gives a rough seasonal estimate.
"""

from __future__ import annotations

from typing import Any

import requests
from pydantic import BaseModel, Field

from src.config import data_source_settings
from src.data_sources.base_client import BaseDataClient, LiveTier, ResolvedData
from src.data_sources.llm_fallback_client import estimate_structured

NWS_BASE_URL = "https://api.weather.gov"
SOURCE_NAME_DIRECT = "NWS station observation (api.weather.gov)"
SOURCE_NAME_GRID = "NWS lat/lon grid lookup (api.weather.gov)"

# ICAO station codes that don't follow the "K" + IATA pattern used across
# the contiguous US. Small, stable reference table (like a phone country
# code list) - not business data, just an identifier mapping.
ICAO_OVERRIDES = {"ANC": "PANC"}

# Needed only to fetch state-wide active alerts; airport -> (lat, lon, state).
# Coordinates and state are public, stable facts about each airport, used
# only to call the real NWS endpoints - not a substitute for their data.
AIRPORT_LOCATION = {
    "BOS": (42.3656, -71.0096, "MA"),
    "BDL": (41.9389, -72.6832, "CT"),
    "PVD": (41.7240, -71.4283, "RI"),
    "PWM": (43.6462, -70.3093, "ME"),
    "MHT": (42.9326, -71.4357, "NH"),
    "BTV": (44.4719, -73.1533, "VT"),
    "LAX": (33.9416, -118.4085, "CA"),
    "SNA": (33.6757, -117.8682, "CA"),
    "ANC": (61.1743, -149.9963, "AK"),
    "SFO": (37.6213, -122.3790, "CA"),
    "OAK": (37.7126, -122.2197, "CA"),
    "SJC": (37.3639, -121.9289, "CA"),
}


class WeatherEstimate(BaseModel):
    """Schema for the LLM last-resort estimate of one airport's typical current weather."""

    visibility_miles: float = Field(description="Estimated visibility in statute miles")
    wind_speed_mph: float = Field(description="Estimated sustained wind speed in mph")
    likely_weather_pattern: str = Field(description="One phrase, e.g. 'clear', 'marine layer fog season'")
    confidence: str = Field(description="'low', 'medium', or 'high' - be honest, this is a rough estimate")
    caveat: str = Field(description="One sentence on why this is uncertain")


class NwsWeatherClient(BaseDataClient):
    """Fetches real current NWS weather conditions and alerts for an airport."""

    client_name = "NWS Weather"

    def get_weather(self, airport: str) -> ResolvedData:
        """Get current real visibility/wind/alerts for one airport."""
        return self.resolve(airport=airport)

    # --- BaseDataClient hooks ---

    def _live_tiers(self, **kwargs: Any) -> list[LiveTier]:
        airport = kwargs["airport"]
        return [
            ("primary", SOURCE_NAME_DIRECT, lambda: self._fetch_direct(airport)),
            ("secondary", SOURCE_NAME_GRID, lambda: self._fetch_via_grid(airport)),
        ]

    def _llm_fallback(self, **kwargs: Any):
        airport = kwargs["airport"]

        def fetch() -> dict:
            result = estimate_structured(
                schema=WeatherEstimate,
                question=f"Estimate {airport}'s typical current weather conditions right now.",
                context=f"The real NWS API was unreachable. This is airport {airport}, a US commercial airport.",
            )
            return {
                "airport": airport,
                "visibility_miles": result["visibility_miles"],
                "wind_speed_mph": result["wind_speed_mph"],
                "conditions": result["likely_weather_pattern"],
                "active_alerts": [],
                "is_llm_estimate": True,
                "llm_confidence": result["confidence"],
                "llm_caveat": result["caveat"],
            }

        return ("LLM estimate (OpenRouter)", fetch)

    # --- Live fetch tiers ---

    def _fetch_direct(self, airport: str) -> dict[str, Any] | None:
        station = ICAO_OVERRIDES.get(airport, f"K{airport}")
        obs = self._get_json(f"{NWS_BASE_URL}/stations/{station}/observations/latest")
        if obs is None:
            return None
        return self._build_result(airport, obs["properties"])

    def _fetch_via_grid(self, airport: str) -> dict[str, Any] | None:
        location = AIRPORT_LOCATION.get(airport)
        if not location:
            return None
        lat, lon, _ = location
        points = self._get_json(f"{NWS_BASE_URL}/points/{lat},{lon}")
        if points is None:
            return None
        stations = self._get_json(points["properties"]["observationStations"])
        features = (stations or {}).get("features") or []
        if not features:
            return None
        station_id = features[0]["properties"]["stationIdentifier"]
        obs = self._get_json(f"{NWS_BASE_URL}/stations/{station_id}/observations/latest")
        if obs is None:
            return None
        return self._build_result(airport, obs["properties"])

    def _build_result(self, airport: str, obs_props: dict) -> dict[str, Any]:
        visibility_m = _unit_value(obs_props.get("visibility"))
        wind_kmh = _unit_value(obs_props.get("windSpeed"))
        gust_kmh = _unit_value(obs_props.get("windGust"))
        return {
            "airport": airport,
            "conditions": obs_props.get("textDescription"),
            "visibility_miles": round(visibility_m / 1609.34, 1) if visibility_m is not None else None,
            "wind_speed_mph": round(wind_kmh / 1.60934, 1) if wind_kmh is not None else None,
            "wind_gust_mph": round(gust_kmh / 1.60934, 1) if gust_kmh is not None else None,
            "active_alerts": self._fetch_alerts(airport),
            "is_llm_estimate": False,
        }

    def _fetch_alerts(self, airport: str) -> list[str]:
        location = AIRPORT_LOCATION.get(airport)
        if not location:
            return []
        _, _, state = location
        alerts = self._get_json(f"{NWS_BASE_URL}/alerts/active", params={"area": state})
        if not alerts:
            return []
        # NWS returns one feature per affected forecast zone, so a single
        # statewide event (e.g. a heat warning) can repeat many times;
        # de-duplicate down to distinct event names, order preserved.
        events = [f["properties"]["event"] for f in alerts.get("features", [])]
        return list(dict.fromkeys(events))

    @staticmethod
    def _get_json(url: str, params: dict | None = None) -> dict | None:
        headers = {"User-Agent": data_source_settings.nws_user_agent}
        response = requests.get(url, params=params, headers=headers, timeout=data_source_settings.request_timeout_seconds)
        if response.status_code != 200:
            return None
        return response.json()


def _unit_value(field: dict | None) -> float | None:
    """NWS returns measurements as {'value': ..., 'unitCode': ...}; None-safe extraction."""
    if not field or field.get("value") is None:
        return None
    return float(field["value"])
