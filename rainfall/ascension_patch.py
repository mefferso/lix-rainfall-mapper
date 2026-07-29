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
    # Fixed complete CoCoRaHS totals for August 12-14, 2016.
    # Offsets keep the gauge labels separated from nearby QPE city samples.
    ("Prairieville 2.0 S", 30.276934, -90.979147, 15.02, (-74, -24)),
    ("Gonzales 0.8 E", 30.217250, -90.909870, 11.41, (38, -16)),
    ("Gonzales 4.5 S", 30.151899, -90.928910, 19.13, (40, -22)),
)
ASCENSION_QPE_SAMPLES = (
    # Existing city locations used by the generic map sampler.
    ("Prairieville", (24, 28)),
    ("Gonzales", (-24, 28)),
    ("Donaldsonville", (-18, 20)),
)


def _short_date(day: date) -> str:
    return f"{day.month}/{day.day}"


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


def _draw_qpe_location_label(
    ax,
    label: str,
    latitude: float,
    longitude: float,
    *,
    offset: tuple[float, float],
) -> None:
    """Draw a square-marked NOAA QPE sample distinct from round gauge markers."""

    ax.scatter(
        longitude,
        latitude,
        s=22,
        marker="s",
        facecolor="#17212b",
        edgecolor="white",
        linewidth=0.9,
        zorder=10,
    )
    ax.annotate(
        label,
        (longitude, latitude),
        xytext=offset,
        textcoords="offset points",
        fontsize=8.2,
        color="#17212b",
        weight="bold",
        ha="left" if offset[0] >= 0 else "right",
        va="bottom" if offset[1] >= 0 else "top",
        arrowprops={
            "arrowstyle": "-",
            "color": "#25313a",
            "linewidth": 0.9,
            "linestyle": "solid",
            "shrinkA": 2,
            "shrinkB": 3,
        },
        path_effects=[path_effects.withStroke(linewidth=2.4, foreground="white")],
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
        gauge_period = (
            f"{_short_date(ASCENSION_GAUGE_DATES[0])}–"
            f"{_short_date(ASCENSION_GAUGE_DATES[1])}"
        )
        selected_period = f"{_short_date(start)}–{_short_date(end)}"

        for name, latitude, longitude, total, offset in ASCENSION_GAUGE_TOTALS:
            base_map._draw_location_label(
                ax,
                f'{name}\nCoCoRaHS {gauge_period}: {total:.2f}"',
                latitude,
                longitude,
                offset=offset,
                leader=True,
            )

        for name, offset in ASCENSION_QPE_SAMPLES:
            latitude, longitude, _ = base_map.CITIES[name]
            sample = base_map._sample_grid(data, grid, latitude, longitude)
            if sample is None:
                continue
            _draw_qpe_location_label(
                ax,
                f'{name}\nQPE {selected_period}: {sample:.2f}"',
                latitude,
                longitude,
                offset=offset,
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
