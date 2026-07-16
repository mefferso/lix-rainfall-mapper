"""Load official NWS boundary layers bundled with the application."""

from __future__ import annotations

import json
from pathlib import Path


DATA_DIRECTORY = Path(__file__).with_name("data")
BOUNDARY_FILES = (
    "nws_counties.geojson",
    "nws_states.geojson",
    "nws_lix_cwa.geojson",
)


def fetch_la_ms_boundaries() -> dict:
    """Return NWS county, state, and WFO LIX CWA boundaries as GeoJSON."""

    features: list[dict] = []
    for filename in BOUNDARY_FILES:
        with (DATA_DIRECTORY / filename).open(encoding="utf-8") as source:
            features.extend(json.load(source)["features"])
    return {
        "type": "FeatureCollection",
        "source": "National Weather Service GIS boundary shapefiles",
        "features": features,
    }
