"""Ascension Parish-specific rendering behavior.

This module wraps the general map renderer so the August 2016 comparison
windows show the three complete CoCoRaHS totals alongside NOAA gridded QPE
samples accumulated over the selected map period.
"""

from __future__ import annotations

from datetime import date
import threading

import matplotlib.patheffects as path_effects
from matplotlib.axes import Axes
import numpy as np

from . import map as base_map
from .core import TargetGrid


ASCENSION_GAUGE_DATES = (date(2016, 8, 12), date(2016, 8, 14))
ASCENSION_COMPARISON_WINDOWS = {
    (date(2016, 8, 10), date(2016, 8, 14)),
    (date(2016, 8, 11), date(2016, 8, 14)),
    (date(2016, 8, 12), date(2016, 8, 14)),
}

# Each matched station is drawn once with a combined CoCoRaHS/QPE label.
# Coordinates were verified against the uploaded CoCoRaHS station workbook.
ASCENSION_COMPARISON_STATIONS = (
    ("Prairieville 2.0 S", 30.276934, -90.979147, 15.02, (12, 10)),
    ("Gonzales 0.8 E", 30.217250, -90.909870, 11.41, (12, 10)),
    ("Gonzales 4.5 S", 30.151899, -90.928910, 19.13, (12, -10)),
)
DONALDSONVILLE_QPE = ("Donaldsonville", 30.101, -90.993, (8, 8))

_RENDER_LOCK = threading.Lock()


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


def _draw_marker_pair(ax, latitude: float, longitude: float) -> None:
    """Draw one shared comparison symbol at the matched station location."""

    ax.scatter(
        longitude,
        latitude,
        s=36,
        marker="o",
        facecolor="white",
        edgecolor="#111111",
        linewidth=0.9,
        zorder=10,
    )
    ax.scatter(
        longitude,
        latitude,
        s=14,
        marker="s",
        facecolor="#111111",
        edgecolor="#111111",
        linewidth=0.8,
        zorder=11,
    )


def _draw_label_only(
    ax,
    label: str,
    latitude: float,
    longitude: float,
    *,
    offset: tuple[float, float],
) -> None:
    """Draw one clean text label without creating another marker."""

    ax.annotate(
        label,
        (longitude, latitude),
        xytext=offset,
        textcoords="offset points",
        fontsize=10.0,
        fontfamily="DejaVu Sans",
        color="#080808",
        weight="semibold",
        ha="left" if offset[0] >= 0 else "right",
        va="bottom" if offset[1] >= 0 else "top",
        zorder=12,
    )


def _draw_qpe_only_label(
    ax,
    label: str,
    latitude: float,
    longitude: float,
    *,
    offset: tuple[float, float],
) -> None:
    """Draw the standalone Donaldsonville QPE sample."""

    ax.scatter(
        longitude,
        latitude,
        s=22,
        marker="s",
        facecolor="#111111",
        edgecolor="#111111",
        linewidth=0.8,
        zorder=11,
    )
    _draw_label_only(
        ax,
        label,
        latitude,
        longitude,
        offset=offset,
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
        for name, latitude, longitude, gauge_total, offset in ASCENSION_COMPARISON_STATIONS:
            qpe_total = base_map._sample_grid(data, grid, latitude, longitude)
            _draw_marker_pair(ax, latitude, longitude)

            if qpe_total is None:
                label = f'{name}\n○ {gauge_total:.2f}"'
            else:
                label = f'{name}\n○ {gauge_total:.2f}"   ■ {qpe_total:.2f}"'

            _draw_label_only(
                ax,
                label,
                latitude,
                longitude,
                offset=offset,
            )

        name, latitude, longitude, offset = DONALDSONVILLE_QPE
        qpe_total = base_map._sample_grid(data, grid, latitude, longitude)
        if qpe_total is not None:
            _draw_qpe_only_label(
                ax,
                f'{name}\n■ {qpe_total:.2f}"',
                latitude,
                longitude,
                offset=offset,
            )

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

        if (start, end) in ASCENSION_COMPARISON_WINDOWS:
            show_cities = True
            show_city_samples = True

    original_imshow = Axes.imshow
    original_text = Axes.text

    def _ascension_imshow(self, *args, **kwargs):
        kwargs["interpolation"] = "nearest"
        return original_imshow(self, *args, **kwargs)

    def _ascension_text(self, x, y, text, *args, **kwargs):
        if (
            isinstance(text, str)
            and text.startswith("Ascension Parish boundary")
            and (start, end) in ASCENSION_COMPARISON_WINDOWS
        ):
            text = "Ascension Parish boundary\n○ CoCoRaHS gauge   ■ NOAA gridded QPE"
        return original_text(self, x, y, text, *args, **kwargs)

    with _RENDER_LOCK:
        if region_name == "Ascension Parish":
            Axes.imshow = _ascension_imshow
            Axes.text = _ascension_text
        try:
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
        finally:
            Axes.imshow = original_imshow
            Axes.text = original_text
