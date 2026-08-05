"""
Abstract base class for every data-source client, built around a waterfall.

What this file does: defines the shared "resolve a piece of data" pattern
every client follows. For a given metric (e.g. "BOS passenger growth"), a
client tries a small ordered list of real, live sources - a primary API,
then one or two secondary ones if the primary fails - and only if every
live source fails does it fall back to asking the LLM for a structured
estimate. Every attempt, successful or not, is recorded in a trace that
the agent surfaces to the user.

Why it exists: the project's data policy is "live data first, live data
only in production." Local JSON mock files exist purely so `tests/` can
run fast and offline - they are never read by the running app. This base
class is what makes that policy enforceable in one place instead of
being repeated (and possibly gotten wrong) in six different clients.

How data moves: a client subclass declares `_live_tiers()`, an ordered
list of (tier_label, source_name, fetch_function) triples. `resolve()`
calls each in order, catching failures, until one returns real data. If
every live tier fails, `resolve()` calls `_llm_fallback()` (implemented
per-client, using `src/data_sources/llm_fallback_client.py`) as the last
resort. The full list of attempts - what was tried, in what order, and
what happened - travels with the result as a `ResolutionTrace`.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class ResolutionStep:
    """One attempt to get a piece of data, successful or not."""

    tier: str  # "primary", "secondary", or "llm_fallback"
    source_name: str  # human-readable name, e.g. "BTS T-100 (data.bts.gov, r495-tyji)"
    status: str  # "success", "failed", or "skipped"
    detail: str = ""  # why it failed, or a short note on what it returned


@dataclass
class ResolvedData:
    """The result of a waterfall resolution: the data (if any) plus the full attempt trace."""

    data: Any
    trace: list[ResolutionStep] = field(default_factory=list)

    @property
    def final_source(self) -> str:
        """Name of whichever source actually produced the data, or 'none' if all failed."""
        for step in reversed(self.trace):
            if step.status == "success":
                return step.source_name
        return "none"

    @property
    def is_llm_estimate(self) -> bool:
        """True if no live source worked and the data came from the LLM fallback."""
        return self.final_source != "none" and any(
            step.status == "success" and step.tier == "llm_fallback" for step in self.trace
        )


LiveTier = tuple[str, str, Callable[[], Any]]  # (tier_label, source_name, fetch_fn)


class BaseDataClient(ABC):
    """
    Shared waterfall-resolution behavior for all public data-source clients.

    Subclasses implement `_live_tiers(**kwargs)` to declare their ordered
    list of real API attempts, and optionally `_llm_fallback(**kwargs)`
    for the last-resort LLM estimate. `resolve()` handles the looping,
    error handling, and trace-building so subclasses only need to say
    *what* to try, not *how* to try-and-fall-back.
    """

    #: Name used for logging and in the resolution trace, e.g. "BTS T-100".
    client_name: str = "base"

    def resolve(self, **kwargs: Any) -> ResolvedData:
        """
        Get data for this source, trying each live tier in order, then the LLM as a last resort.

        This is the one public entry point every client exposes. It never
        raises - if everything fails, `data` comes back as `None` and the
        trace shows exactly why, so callers can decide how to handle a
        total miss instead of crashing.
        """
        trace: list[ResolutionStep] = []
        for tier_label, source_name, fetch_fn in self._live_tiers(**kwargs):
            try:
                data = fetch_fn()
            except Exception as exc:  # noqa: BLE001 - any failure moves to the next tier
                trace.append(ResolutionStep(tier_label, source_name, "failed", str(exc)))
                logger.info("%s: %s failed (%s), trying next tier", self.client_name, source_name, exc)
                continue
            if data:
                trace.append(ResolutionStep(tier_label, source_name, "success"))
                return ResolvedData(data=data, trace=trace)
            trace.append(ResolutionStep(tier_label, source_name, "failed", "returned no data"))

        llm_fallback = self._llm_fallback(**kwargs)
        if llm_fallback is not None:
            fallback_name, fetch_fn = llm_fallback
            try:
                data = fetch_fn()
            except Exception as exc:  # noqa: BLE001 - last resort also must not raise
                trace.append(ResolutionStep("llm_fallback", fallback_name, "failed", str(exc)))
                return ResolvedData(data=None, trace=trace)
            if data:
                trace.append(ResolutionStep("llm_fallback", fallback_name, "success"))
                return ResolvedData(data=data, trace=trace)
            trace.append(ResolutionStep("llm_fallback", fallback_name, "failed", "returned no data"))

        return ResolvedData(data=None, trace=trace)

    @abstractmethod
    def _live_tiers(self, **kwargs: Any) -> list[LiveTier]:
        """Return the ordered list of real API attempts for this request."""

    def _llm_fallback(self, **kwargs: Any) -> tuple[str, Callable[[], Any]] | None:
        """
        Return (source_name, fetch_fn) for the LLM last-resort, or None if this client has none.

        Overridden per-client. Returning None means "no LLM fallback is
        wired up for this metric" - `resolve()` then simply returns
        data=None with a trace showing every live tier failed.
        """
        return None
