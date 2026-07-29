"""Ascension Parish-specific rendering behavior.

This module wraps the general map renderer so the August 12-14, 2016
Ascension preset always shows the three complete CoCoRaHS totals and a
clearly emphasized official parish boundary.
"""

from __future__ import annotations

from datetime import date

import matplotlib.patheffects as path_effects
import numpy as np

from . import map as base_map
from .core import TargetGrid


ASCENSION_EVENT_DATES = (date(2016, 8, 12), date(2016, 8, 14))
ASCENSION_GAUGE_TOTALS = (
    ("Prairieville 2.0 S", 30.276934, -90.979147, 15.02, (-68, 24)),
    ("Gonzales 0.8 E", 30.217250, -90.909870, 11.41, (32, 14)),
    ("Gonzales 4.5 S", 30.151899, -90.928910, 19.13, (32, -24)),
)


def _is_ascension_feature(feature: dict) -> bool:
    """Match the official Ascension Parish county feature robustly."""

    properties = feature.get("properties", {})
    if properties.get("layer") != "county":
        return False

    name = str(properties.get("name", "")).strip().lower()
    fips = str(properties.get("fips", "")).strip()
    state = str(properties.get("state", "")).strip().upper()
    return (state == "LA" and name in {"ascension", "ascension parish"}) or fips == "22005"


def _draw_ascension_boundary(ax, boundaries: dict) -> None:
    """Draw a light halo plus dark line around Ascension Parish."""

    for feature in boundaries.get("features", []):
        if not _is_ascension_feature(feature):
            continue
        for ring in base_map._rings(feature["geometry"]):
            coordinates = np.asarray(ring)
            line = ax.plot(
                coordinates[:, 0],
                coordinates[:, 1],
                color="#111111",
                linewidth=2.8,
                alpha=1,
                solid_capstyle="round",
                solid_joinstyle="round",
                zorder=9,
            )[0]
            line.set_path_effects(
                [
                    path_effects.Stroke(linewidth=5.4, foreground="white", alpha=0.95),
                    path_effects.Normal(),
                ]
            )


def _draw_ascension_locations(
    ax,
    data: np.ndarray,
    grid: TargetGrid,
    start: date,
    end: date,
    *,
    show_cities: bool,
    show_city_samples: bool,
) -> bool:
    """Use exactly the three complete event totals for the target event."""

    if (start, end) == ASCENSION_EVENT_DATES:
        for name, latitude, longitude, total, offset in ASCENSION_GAUGE_TOTALS:
            base_map._draw_location_label(
                ax,
                f'{name}\n{total:.2f}"',
                latitude,
                longitude,
                offset=offset,
                leader=True,
            )
        # False suppresses the legacy Donaldsonville-specific annotation block.
        return False

    # For other periods, retain the normal in-parish city/sample behavior.
    for name in sorted(base_map.ASCENSION_CITY_NAMES):
        latitude, longitude, _ = base_map.CITIES[name]
        label = name
        if show_city_samples:
            sample = base_map._sample_grid(data, grid, latitude, longitude)
            if sample is not None:
                label = f'{name}\n{sample:.2f}"'
        base_map._draw_location_label(
            ax,
            label,
            latitude,
            longitude,
            offset=base_map.CITY_LABEL_OFFSETS.get(name, (5, 5)),
        )
    return False


def render_map(*args, **kwargs) -> bytes:
    """Render through the base module with Ascension-specific fixes applied."""

    region_name = kwargs.get("region_name", "")
    start = args[3] if len(args) > 3 else kwargs.get("start")
    end = args[4] if len(args) > 4 else kwargs.get("end")

    if region_name == "Ascension Parish":
        base_map._is_ascension_feature = _is_ascension_feature
        base_map._draw_ascension_boundary = _draw_ascension_boundary
        base_map._draw_ascension_locations = _draw_ascension_locations

        # The historical Ascension map should show its verified point totals
        # automatically, even if the generic city-sample checkbox is off.
        if (start, end) == ASCENSION_EVENT_DATES:
            kwargs["show_city_samples"] = True

    return base_map.render_map(*args, **kwargs)
