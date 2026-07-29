"""Ascension Parish-specific rendering behavior.

This module wraps the general map renderer so the August 2016 comparison
windows show the three complete CoCoRaHS totals alongside NOAA gridded QPE
samples accumulated over the selected map period.
"""

from __future__ import annotations

from datetime import date

import matplotlib.patheffects as path_effects
import numpy as np

from . import map as base_map
from .core import TargetGrid


ASCENSION_GAUGE_DATES = (date(2016, 8, 12), date(2016, 8, 14))
ASCENSION_COMPARISON_WINDOWS = {
    (date(2016, 8, 10), date(2016, 8, 14)),
    (date(2016, 8, 11), date(2016, 8, 14)),
    (date(2016, 8, 12), date(2016, 8, 14)),
}
ASCENSION_GAUGE_TOTALS = (
    # Coordinates verified against the uploaded CoCoRaHS station workbook.
    # Short offsets keep each label close to its actual observing point.
    ("Prairieville 2.0 S", 30.276934, -90.979147, 15.02, (-8, -8)),
    ("Gonzales 0.8 E", 30.217250, -90.909870, 11.41, (8, -8)),
    ("Gonzales 4.5 S", 30.151899, -90.928910, 19.13, (8, -8)),
)
ASCENSION_QPE_SAMPLES = (
    # Existing city locations used by the generic map sampler.
    # Offsets place the city samples opposite the nearby gauge labels.
    ("Prairieville", (8, 8)),
    ("Gonzales", (-8, 8)),
    ("Donaldsonville", (8, 8)),
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


def _draw_simple_location_label(
    ax,
    label: str,
    latitude: float,
    longitude: float,
    *,
    offset: tuple[float, float],
    marker: str,
    filled: bool,
) -> None:
    """Draw a compact point label with plain black type and no text halo."""

    ax.scatter(
        longitude,
        latitude,
        s=19,
        marker=marker,
        facecolor="#111111" if filled else "white",
        edgecolor="#111111",
        linewidth=0.9,
        zorder=10,
    )
    ax.annotate(
        label,
        (longitude, latitude),
        xytext=offset,
        textcoords="offset points",
        fontsize=8.5,
        fontfamily="DejaVu Sans",
        color="#080808",
        weight="semibold",
        ha="left" if offset[0] >= 0 else "right",
        va="bottom" if offset[1] >= 0 else "top",
        zorder=11,
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
    """Draw Ascension references, including the August 2016 comparison mode."""

    if (start, end) in ASCENSION_COMPARISON_WINDOWS:
        for name, latitude, longitude, total, offset in ASCENSION_GAUGE_TOTALS:
            _draw_simple_location_label(
                ax,
                f'{name}\n{total:.2f}"',
                latitude,
                longitude,
                offset=offset,
                marker="o",
                filled=False,
            )

        for name, offset in ASCENSION_QPE_SAMPLES:
            latitude, longitude, _ = base_map.CITIES[name]
            sample = base_map._sample_grid(data, grid, latitude, longitude)
            if sample is None:
                continue
            _draw_simple_location_label(
                ax,
                f'{name}\n{sample:.2f}"',
                latitude,
                longitude,
                offset=offset,
                marker="s",
                filled=True,
            )

        # Tell the base renderer that CoCoRaHS values are present so its source
        # footer includes the point-observation attribution.
        return True

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


def render_map(
    data: np.ndarray,
    grid: TargetGrid,
    boundaries: dict,
    start: date,
    end: date,
    *,
    custom_title: str = "",
    show_counties: bool = True,
    show_cities: bool = True,
    show_city_samples: bool = False,
    region_name: str = "",
) -> bytes:
    """Render through the base module with Ascension-specific fixes applied."""

    if region_name == "Ascension Parish":
        base_map._is_ascension_feature = _is_ascension_feature
        base_map._draw_ascension_boundary = _draw_ascension_boundary
        base_map._draw_ascension_locations = _draw_ascension_locations

        # Always show both datasets for the three requested comparison windows,
        # regardless of the generic city-label and city-sample checkbox states.
        if (start, end) in ASCENSION_COMPARISON_WINDOWS:
            show_cities = True
            show_city_samples = True

    return base_map.render_map(
        data,
        grid,
        boundaries,
        start,
        end,
        custom_title=custom_title,
        show_counties=show_counties,
        show_cities=show_cities,
        show_city_samples=show_city_samples,
        region_name=region_name,
    )
