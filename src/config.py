"""
App configuration.

What this file does: it is the ONE place that reads environment variables
(from a real .env file or the shell) and turns them into typed Python values
that the rest of the app can import. It also holds the "business constants"
used by the scoring formulas (weights, thresholds, state lists) so that
analysts can tune the model without touching analytics code.

Why it exists: hard-coding API keys, model names, or magic numbers inside
business logic makes the code hard to test and hard to reuse. Centralizing
config here means every other module just does `from src.config import settings`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load variables from a .env file in the project root, if one exists.
# This does nothing in production if the real environment already has
# the variables set (e.g. Docker, CI secrets).
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class LLMSettings:
    """Everything needed to talk to an LLM through OpenRouter."""

    api_key: str = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    model_name: str = field(
        default_factory=lambda: os.getenv("MODEL_NAME", "openai/gpt-oss-20b:free")
    )
    base_url: str = "https://openrouter.ai/api/v1"
    # OpenRouter uses these optional headers for its public leaderboards.
    site_url: str = field(default_factory=lambda: os.getenv("OPENROUTER_SITE_URL", "http://localhost"))
    site_name: str = field(default_factory=lambda: os.getenv("OPENROUTER_SITE_NAME", "Airport Investment Intelligence Agent"))
    max_tool_call_rounds: int = 5
    request_timeout_seconds: int = 60
    # Context-window management (see src/agent/orchestrator.py::_compact_history):
    # how many of the most recent user turns to keep in full before older
    # turns are dropped entirely from the conversation sent to the LLM.
    max_history_turns: int = field(default_factory=lambda: _get_int("MAX_HISTORY_TURNS", 10))


@dataclass(frozen=True)
class DataSourceSettings:
    """
    Network behavior and credentials for the public data-source clients.

    Every client in `src/data_sources/` calls real, live public APIs.
    There is no "use mock data instead" switch here on purpose: mock
    JSON fixtures exist only under `tests/`, loaded directly by test
    code, and are never read by the running application.
    """

    request_timeout_seconds: float = _get_float("DATA_SOURCE_TIMEOUT_SECONDS", 8.0)
    nws_user_agent: str = field(
        default_factory=lambda: os.getenv(
            "NWS_USER_AGENT",
            "airport-investment-intelligence-agent (contact: example@example.com)",
        )
    )
    # Free signup keys: https://api.census.gov/data/key_signup.html and
    # https://apps.bea.gov/api/signup/. Population/income growth in the New
    # England expansion score falls back to an LLM estimate if these are unset.
    census_api_key: str = field(default_factory=lambda: os.getenv("CENSUS_API_KEY", ""))
    bea_api_key: str = field(default_factory=lambda: os.getenv("BEA_API_KEY", ""))


@dataclass(frozen=True)
class ScoringConstants:
    """
    Business/scoring constants used by src/analytics/*.

    These are assumptions made explicit as named, tunable numbers instead of
    being buried inside formulas. An analyst reviewing the model should be
    able to read this class and understand every weight and threshold used.
    """

    # --- Geography ---
    new_england_states: tuple[str, ...] = ("MA", "CT", "RI", "NH", "ME", "VT")
    new_england_airports: tuple[str, ...] = ("BOS", "BDL", "PVD", "PWM", "MHT", "BTV")

    # --- Question 3: Anchorage long-haul definition ---
    long_haul_min_distance_miles: float = 3000.0

    # --- Question 1: New England terminal expansion score weights (sum to 1.0) ---
    expansion_weight_capacity_pressure: float = 0.35
    expansion_weight_load_factor: float = 0.25
    expansion_weight_runway_constraint: float = 0.20
    expansion_weight_census_growth: float = 0.20

    # --- Question 2: LAX vs SNA congestion score weights (sum to 1.0) ---
    congestion_weight_taxi_out: float = 0.35
    congestion_weight_delay_rate: float = 0.30
    congestion_weight_cancellation_rate: float = 0.20
    congestion_weight_nas_delay: float = 0.15
    # Reference "worst case" values used to normalize minutes into a 0-1 range
    # before applying the weights above. Chosen from observed US large-hub
    # extremes so the final composite score is comparable across airports.
    congestion_taxi_out_reference_minutes: float = 25.0
    congestion_nas_delay_reference_minutes: float = 20.0

    # --- Question 4: SFO unmet demand index weights (sum to 1.0) ---
    unmet_weight_load_factor_pressure: float = 0.30
    unmet_weight_capacity_growth_gap: float = 0.25
    unmet_weight_market_leakage: float = 0.20
    unmet_weight_fare_premium: float = 0.15
    unmet_weight_cancellation_gap: float = 0.10
    # A load factor below this is considered "comfortable"; pressure only
    # accrues above it. Modern narrow-body scheduling commonly targets ~80-85%.
    unmet_comfortable_load_factor: float = 0.80
    # A 10-percentage-point gap between passenger growth and seat growth is
    # treated as "maximum" capacity-growth pressure (clipped at 1.0 above this).
    unmet_capacity_growth_gap_reference_pct: float = 10.0
    # 25% of regional demand leaking to secondary airports is treated as "maximum".
    unmet_market_leakage_reference_pct: float = 25.0
    # A 30% fare premium over the national average is treated as "maximum".
    unmet_fare_premium_reference_pct: float = 30.0
    # A 2-percentage-point cancellation-rate gap is treated as "maximum".
    unmet_cancellation_gap_reference_pct: float = 2.0


llm_settings = LLMSettings()
data_source_settings = DataSourceSettings()
scoring_constants = ScoringConstants()
