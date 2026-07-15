"""Fetch lightweight Census cartographic boundaries for map overlays."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import zipfile

import requests
import shapefile


CENSUS_BASE = "https://www2.census.gov/geo/tiger/GENZ2024/shp"


def _read_layer(url: str, layer: str) -> list[dict]:
    response = requests.get(
        url,
        timeout=(15, 90),
        headers={"User-Agent": "LIX-Rainfall-Mapper/1.0"},
    )
    response.raise_for_status()
    archive = zipfile.ZipFile(BytesIO(response.content))
    members = {
        Path(name).suffix.lower(): name
        for name in archive.namelist()
        if Path(name).suffix.lower() in {".shp", ".shx", ".dbf"}
    }
    reader = shapefile.Reader(
        shp=BytesIO(archive.read(members[".shp"])),
        shx=BytesIO(archive.read(members[".shx"])),
        dbf=BytesIO(archive.read(members[".dbf"])),
    )
    fields = [field[0] for field in reader.fields[1:]]
    features: list[dict] = []
    for shape_record in reader.iterShapeRecords():
        properties = dict(zip(fields, shape_record.record))
        state_fips = properties.get("STATEFP")
        if state_fips not in {"22", "28"}:
            continue
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "layer": layer,
                    "statefp": state_fips,
                    "name": properties.get("NAME"),
                },
                "geometry": shape_record.shape.__geo_interface__,
            }
        )
    return features


def fetch_la_ms_boundaries() -> dict:
    """Return Louisiana and Mississippi county/state boundaries as GeoJSON."""

    features = _read_layer(f"{CENSUS_BASE}/cb_2024_us_county_5m.zip", "county")
    features.extend(_read_layer(f"{CENSUS_BASE}/cb_2024_us_state_5m.zip", "state"))
    return {
        "type": "FeatureCollection",
        "source": "U.S. Census Bureau 2024 cartographic boundaries",
        "features": features,
    }
