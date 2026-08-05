"""
Shared pytest setup and fixture-loading helpers.

Provides a harmless placeholder OpenRouter API key so importing the
orchestrator never fails just because a real key isn't configured in CI,
and a `load_fixture()` helper for reading the real-shaped JSON/XML files
under `tests/fixtures/`. Fixture files are the ONLY place mock data lives
in this project - the running application never reads them (see
`src/data_sources/base_client.py`); they exist purely so this test suite
can run fast, deterministic, and fully offline.
"""

import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("OPENROUTER_API_KEY", "test-key-not-used")

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def load_json_fixture(filename: str):
    with open(FIXTURES_DIR / filename, encoding="utf-8") as f:
        return json.load(f)


def load_text_fixture(filename: str) -> str:
    with open(FIXTURES_DIR / filename, encoding="utf-8") as f:
        return f.read()


@pytest.fixture
def t100_fixture():
    return load_json_fixture("bts_t100_soda_response.json")


@pytest.fixture
def ontime_fixture():
    return load_json_fixture("bts_ontime_rows.json")


@pytest.fixture
def arcgis_fixture():
    return load_json_fixture("faa_arcgis_responses.json")


@pytest.fixture
def nas_xml_fixture():
    return load_text_fixture("faa_nas_status.xml")


@pytest.fixture
def nws_fixture():
    return load_json_fixture("nws_responses.json")


@pytest.fixture
def census_bea_fixture():
    return load_json_fixture("census_bea_responses.json")


@pytest.fixture
def airfare_fixture():
    return load_json_fixture("bts_airfare_rows.json")
