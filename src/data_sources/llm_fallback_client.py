"""
Last-resort data source: ask the LLM for a structured estimate.

What this file does: when every real live API for a piece of data has
failed (see the waterfall in `base_client.py`), this module asks the
configured OpenRouter model to *estimate* that data point, constrained to
a Pydantic schema so the output is still a typed, structured value - not
free-form prose - and can flow into the exact same deterministic formulas
a real API response would.

Why it exists: the user's requirement is "try real data first, try a
couple more real sources if the best one fails, and only if nothing real
works, ask the LLM - and make that visible." This is that last step. It
is deliberately isolated in `src/data_sources/`, not `src/agent/`,
because from the analytics layer's point of view this is just another
data source with its own (lower) reliability - not a reasoning step. The
LLM is never asked to compute a score here, only to fill in one missing
raw fact (e.g. "roughly what is BOS's 5-year population growth?"), and
every value it returns is tagged `is_llm_estimate=True` so nothing
downstream can mistake it for measured data.

How data moves: a client's `_llm_fallback()` builds a Pydantic schema and
a short prompt describing exactly what fact is missing -> this module
calls OpenRouter with `response_format` locked to that schema -> the
response is validated against the schema (raising if the model returns
something malformed, which the waterfall then reports as a failed
attempt) -> the validated object is returned as a plain dict.
"""

from __future__ import annotations

import json
import logging

from openai import OpenAI
from pydantic import BaseModel

from src.config import llm_settings

logger = logging.getLogger(__name__)

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(base_url=llm_settings.base_url, api_key=llm_settings.api_key)
    return _client


def estimate_structured(*, schema: type[BaseModel], question: str, context: str) -> dict:
    """
    Ask the LLM to fill in one missing data point, constrained to a Pydantic schema.

    `schema` must be a Pydantic model whose fields are exactly the values
    needed (plus, by convention, a `confidence` and `caveat` field so the
    estimate is honest about its own uncertainty - see the schemas defined
    in each data-source client). `question` is a short, specific ask (e.g.
    "Estimate BOS's population growth over the last 5 years"). `context`
    gives the model whatever real data IS available, so it can extrapolate
    rather than guess from nothing.

    Raises on any failure (network, malformed response, schema
    validation) - this is intentional. The waterfall in `base_client.py`
    catches the exception and records it as a failed attempt, exactly
    like a failed real API call.
    """
    if not llm_settings.api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set - cannot use the LLM fallback")

    system_prompt = (
        "You are a data-estimation assistant used only as a last resort, when every real public "
        "data API has failed. You do not have live access to the actual figure being asked about. "
        "Give your best-informed estimate based on general knowledge, and be honest about low "
        "confidence and wide uncertainty ranges - do not fabricate false precision. "
        "Respond ONLY with JSON matching the given schema."
    )
    user_prompt = f"{question}\n\nContext / known real data:\n{context}"

    response = _get_client().chat.completions.create(
        model=llm_settings.model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": schema.__name__,
                "strict": True,
                "schema": schema.model_json_schema(),
            },
        },
        timeout=llm_settings.request_timeout_seconds,
        extra_headers={
            "HTTP-Referer": llm_settings.site_url,
            "X-Title": llm_settings.site_name,
        },
    )
    raw = response.choices[0].message.content
    if not raw:
        raise ValueError("LLM fallback returned an empty response")

    parsed = schema.model_validate(json.loads(raw))
    result = parsed.model_dump()
    result["is_llm_estimate"] = True
    return result
