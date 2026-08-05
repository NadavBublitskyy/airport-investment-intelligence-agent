"""
Client for FAA live National Airspace System (NAS) status (real, live XML feed).

What this file does: fetches the FAA's real-time airport status feed -
active ground stops, ground delay programs, closures, and general
arrival/departure delay advisories - for one airport.

Why it exists: the feed genuinely changes in real time (confirmed during
development: it showed live closure notices for SNA and LAX, and a live
arrival/departure delay for LGA, at the moment this was tested). It is
the only client in the project answering "what's happening right now?"
rather than a historical average.

How data moves: `get_status()` calls `resolve()`, which fetches the real
XML feed at `nasstatus.faa.gov` and parses it. The feed's structure is a
list of `<Delay_type>` blocks (e.g. "Airport Closures", "General Arrival/
Departure Delay Info"), each containing a category-specific list whose
exact wrapper tag name varies - the parser here doesn't hard-code those
wrapper names; it walks every element under each `<Delay_type>` looking
for one with an `<ARPT>` child, which is robust to the FAA adding or
renaming a category. If the feed is unreachable, this client has no LLM
fallback: "no live status data available" is itself an honest answer for
a real-time question, so a guessed answer would be actively misleading.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any

import requests

from src.config import data_source_settings
from src.data_sources.base_client import BaseDataClient, LiveTier, ResolvedData

NAS_STATUS_URL = "https://nasstatus.faa.gov/api/airport-status-information"
SOURCE_NAME = "FAA Live NAS Status (nasstatus.faa.gov)"


class FaaNasClient(BaseDataClient):
    """Fetches real, live FAA ground stop / ground delay / closure status for an airport."""

    client_name = "FAA Live NAS Status"

    def get_status(self, airport: str) -> ResolvedData:
        """Get every current live event affecting one airport, or an empty list if none."""
        return self.resolve(airport=airport)

    # --- BaseDataClient hooks ---

    def _live_tiers(self, **kwargs: Any) -> list[LiveTier]:
        airport = kwargs["airport"]
        return [("primary", SOURCE_NAME, lambda: self._fetch(airport))]

    # No _llm_fallback override: a guessed "is there a ground stop right now"
    # would be worse than useless, so this client only ever returns real data
    # or an honest "unavailable."

    @staticmethod
    def _fetch(airport: str) -> dict[str, Any]:
        response = requests.get(NAS_STATUS_URL, timeout=data_source_settings.request_timeout_seconds)
        response.raise_for_status()
        events = _parse_events(response.text)
        matching = [e for e in events if e["airport"] == airport]
        return {
            "airport": airport,
            "events": matching,
            "has_ground_stop": any("Ground Stop" in e["category"] for e in matching),
            "has_ground_delay": any("Ground Delay" in e["category"] for e in matching),
            "has_closure": any(e["category"] == "Airport Closures" for e in matching),
            "has_arrival_departure_delay": any(
                "Arrival/Departure Delay" in e["category"] for e in matching
            ),
            "is_llm_estimate": False,
        }


def _parse_events(xml_text: str) -> list[dict[str, Any]]:
    """
    Walk every `<Delay_type>` block and pull out every event that names an airport.

    Deliberately does not hard-code the per-category list-wrapper tag
    (`Airport_Closure_List`, `Arrival_Departure_Delay_List`, etc.) since
    those vary by category and the FAA could add new ones. Any element
    under a `Delay_type` that has an `<ARPT>` child is treated as one event.
    """
    root = ET.fromstring(xml_text)
    events: list[dict[str, Any]] = []
    for delay_type in root.findall("Delay_type"):
        category = (delay_type.findtext("Name") or "Unknown").strip()
        for elem in delay_type.iter():
            arpt_el = elem.find("ARPT")
            if arpt_el is None or not (arpt_el.text or "").strip():
                continue
            event: dict[str, Any] = {
                "airport": arpt_el.text.strip(),
                "category": category,
                "reason": _text(elem, "Reason"),
                "start": _text(elem, "Start"),
                "reopen": _text(elem, "Reopen"),
            }
            arrival_departure = elem.find("Arrival_Departure")
            if arrival_departure is not None:
                event["delay_direction"] = arrival_departure.get("Type")
                event["min_delay"] = _text(arrival_departure, "Min")
                event["max_delay"] = _text(arrival_departure, "Max")
                event["trend"] = _text(arrival_departure, "Trend")
            events.append(event)
    return events


def _text(elem: ET.Element, tag: str) -> str | None:
    value = elem.findtext(tag)
    return value.strip() if value else None
