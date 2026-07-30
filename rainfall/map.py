"""Publication-style map rendering for accumulated precipitation."""

from __future__ import annotations

import math
from datetime import date
from io import BytesIO

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

from .core import TargetGrid


RAIN_LEVELS = [0.01, 0.25, 0.50, 1, 2, 3, 4, 6, 8, 10, 12, 15, 20, 25, 30]
RAIN_COLORS = [
    "#eee4bf",
    "#d8c175",
    "#f4e663",
    "#a7e36d",
    "#48c86c",
    "#36d6bc",
    "#43b7e8",
    "#3d79d8",
    "#6354c7",
    "#9c5ad5",
    "#d555b6",
    "#ee5c7a",
    "#d52d39",
    "#8e1f2e",
]

CITIES = {
    # The third value is the widest map (longitude span in degrees) on which
    # the city is shown. This keeps statewide maps readable while adding
    # more local references to the metro and coastal presets.
    "Alexandria": (31.312, -92.446, 10.0),
    "Amite City": (30.727, -90.509, 2.2),
    "Baton Rouge": (30.451, -91.187, 10.0),
    "Bay St. Louis": (30.309, -89.330, 4.5),
    "Belle Chasse": (29.855, -89.991, 2.2),
    "Biloxi": (30.396, -88.885, 10.0),
    "Bogalusa": (30.791, -89.849, 4.5),
    "Brookhaven": (31.579, -90.441, 4.5),
    "Chalmette": (29.943, -89.963, 2.2),
    "Columbia": (31.252, -89.838, 4.5),
    "Covington": (30.475, -90.100, 4.5),
    "Denham Springs": (30.487, -90.956, 2.2),
    "Donaldsonville": (30.101, -90.993, 2.2),
    "Franklinton": (30.847, -90.154, 4.5),
    "Galliano": (29.442, -90.299, 2.2),
    "Gonzales": (30.239, -90.920, 4.5),
    "Grand Isle": (29.237, -89.987, 4.5),
    "Greenville": (33.411, -91.061, 10.0),
    "Gulfport": (30.367, -89.093, 10.0),
    "Hammond": (30.504, -90.462, 4.5),
    "Hattiesburg": (31.327, -89.290, 10.0),
    "Houma": (29.596, -90.720, 4.5),
    "Jackson": (32.299, -90.185, 10.0),
    "Kenner": (29.994, -90.242, 2.2),
    "Kentwood": (30.938, -90.509, 2.2),
    "LaPlace": (30.067, -90.480, 2.2),
    "Lafayette": (30.224, -92.020, 10.0),
    "Lake Charles": (30.226, -93.217, 10.0),
    "Liberty": (31.158, -90.812, 4.5),
    "Mandeville": (30.358, -90.066, 2.2),
    "McComb": (31.244, -90.453, 10.0),
    "Meridian": (32.365, -88.704, 10.0),
    "Metairie": (29.984, -90.153, 2.2),
    "Monroe": (32.510, -92.120, 10.0),
    "Morgan City": (29.699, -91.207, 4.5),
    "Natchez": (31.560, -91.403, 4.5),
    "New Orleans": (29.951, -90.072, 10.0),
    "New Roads": (30.702, -91.436, 2.2),
    "Ocean Springs": (30.411, -88.828, 2.2),
    "Pascagoula": (30.366, -88.556, 4.5),
    "Picayune": (30.526, -89.679, 4.5),
    "Plaquemine": (30.289, -91.234, 2.2),
    "Poplarville": (30.840, -89.534, 2.2),
    "Prairieville": (30.303, -90.972, 2.2),
    "Raceland": (29.727, -90.598, 2.2),
    "Shreveport": (32.526, -93.750, 10.0),
    "Slidell": (30.275, -89.782, 4.5),
    "St. Francisville": (30.779, -91.377, 2.2),
    "Thibodaux": (29.795, -90.822, 4.5),
    "Tupelo": (34.258, -88.704, 10.0),
    "Tylertown": (31.117, -90.142, 4.5),
    "Wiggins": (30.858, -89.135, 4.5),
    "Woodville": (31.105, -91.300, 4.5),
    "Zachary": (30.649, -91.157, 2.2),
}

CITY_LABEL_OFFSETS = {
    "Baton Rouge": (5, 5),
    "Belle Chasse": (5, -15),
    "Chalmette": (5, 5),
    "Covington": (5, 5),
    "Denham Springs": (5, -15),
    "Gonzales": (5, 5),
    "Kenner": (-34, 5),
    "LaPlace": (-30, 5),
    "Mandeville": (5, -15),
    "Metairie": (5, -15),
    "New Orleans": (5, 5),
    "Prairieville": (5, -15),
}

ASCENSION_STATIONS = (
    ("Prairieville 2.0 S", 30.276934, -90.979147, 15.02),
    ("Gonzales 0.8 E", 30.217250, -90.909870, 11.41),
    ("Gonzales 4.5 S", 30.151899, -90.928910, 19.13),
)
SORRENTO_SAMPLE = {
    "name": "Sorrento",
    "latitude": 30.18,
    "longitude": -90.87,
}
ASCENSION_LABEL_OFFSETS = {
    "Prairieville 2.0 S": (8, 8),
    "Gonzales 0.8 E": (8, 8),
    "Gonzales 4.5 S": (-8, 8),
}

RAINFALL_DISPLAY_MODES = ("Raw", "Smooth")


def _rainfall_interpolation(rainfall_display_mode: str) -> str:
    """Return the display-only interpolation for the rainfall image."""

    if rainfall_display_mode not in RAINFALL_DISPLAY_MODES:
        raise ValueError(
            f"Rainfall display mode must be one of: {', '.join(RAINFALL_DISPLAY_MODES)}"
        )
    return "nearest" if rainfall_display_mode == "Raw" else "bilinear"


def _rings(geometry: dict):
    if geometry["type"] == "Polygon":
        yield from geometry["coordinates"]
    elif geometry["type"] == "MultiPolygon":
        for polygon in geometry["coordinates"]:
            yield from polygon


def _draw_boundaries(ax, boundaries: dict, layer: str, **style) -> None:
    for feature in boundaries["features"]:
        if feature["properties"].get("layer") != layer:
            continue
        for ring in _rings(feature["geometry"]):
            coordinates = np.asarray(ring)
            ax.plot(coordinates[:, 0], coordinates[:, 1], **style)


def _is_ascension_feature(feature: dict) -> bool:
    """Return whether a county-layer feature represents Ascension Parish."""

    properties = feature.get("properties", {})
    if properties.get("layer") != "county":
        return False
    for value in properties.values():
        normalized = str(value).strip().lower()
        if normalized in {"ascension", "ascension parish", "22005"}:
            return True
    return False


def _draw_ascension_boundary(ax, boundaries: dict) -> None:
    """Emphasize the official Ascension Parish outline."""

    for feature in boundaries["features"]:
        if not _is_ascension_feature(feature):
            continue
        for ring in _rings(feature["geometry"]):
            coordinates = np.asarray(ring)
            ax.plot(
                coordinates[:, 0],
                coordinates[:, 1],
                color="#111111",
                linewidth=3.0,
                alpha=1,
                solid_capstyle="round",
                zorder=8,
            )


def _sample_grid(
    data: np.ndarray,
    grid: TargetGrid,
    latitude: float,
    longitude: float,
) -> float | None:
    """Return the nearest valid grid-cell value for a location."""

    if not (grid.west <= longitude <= grid.east and grid.south <= latitude <= grid.north):
        return None
    column = int((longitude - grid.west) / (grid.east - grid.west) * grid.width)
    row = int((grid.north - latitude) / (grid.north - grid.south) * grid.height)
    column = min(max(column, 0), grid.width - 1)
    row = min(max(row, 0), grid.height - 1)
    value = data[row, column]
    return float(value) if np.isfinite(value) else None


def _format_date(day: date) -> str:
    return f"{day.strftime('%B')} {day.day}, {day.year}"


def _add_scale_bar(ax, grid: TargetGrid) -> None:
    longitude_span = grid.east - grid.west
    miles = 10 if longitude_span < 1.0 else 20 if longitude_span < 2.0 else 50
    latitude = grid.south + 0.055 * (grid.north - grid.south)
    start = grid.west + 0.045 * (grid.east - grid.west)
    degrees = miles / (69.172 * math.cos(math.radians(latitude)))
    tick_height = 0.004 * (grid.north - grid.south)
    label_offset = 0.008 * (grid.north - grid.south)
    ax.plot([start, start + degrees], [latitude, latitude], color="#17212b", lw=3.2, zorder=9)
    ax.plot(
        [start, start],
        [latitude - tick_height, latitude + tick_height],
        color="#17212b",
        lw=1.4,
        zorder=9,
    )
    ax.plot(
        [start + degrees, start + degrees],
        [latitude - tick_height, latitude + tick_height],
        color="#17212b",
        lw=1.4,
        zorder=9,
    )
    ax.text(
        start + degrees / 2,
        latitude + label_offset,
        f"{miles} miles",
        ha="center",
        va="bottom",
        fontsize=7.5,
        color="#17212b",
        zorder=9,
    )


def _draw_location_label(
    ax,
    label: str,
    latitude: float,
    longitude: float,
    *,
    offset: tuple[float, float] = (5, 5),
    leader: bool = False,
    marker_size: float = 16,
) -> None:
    ax.scatter(
        longitude,
        latitude,
        s=marker_size,
        facecolor="white",
        edgecolor="#101820",
        linewidth=0.8,
        zorder=9,
    )
    arrowprops = None
    if leader:
        arrowprops = {
            "arrowstyle": "-",
            "color": "#25313a",
            "linewidth": 0.9,
            "linestyle": (0, (1.5, 2.4)),
            "shrinkA": 2,
            "shrinkB": 3,
        }
    ax.annotate(
        label,
        (longitude, latitude),
        xytext=offset,
        textcoords="offset points",
        fontsize=9.0,
        color="#101820",
        weight="bold",
        ha="left" if offset[0] >= 0 else "right",
        va="bottom" if offset[1] >= 0 else "top",
        arrowprops=arrowprops,
        path_effects=[path_effects.withStroke(linewidth=2.4, foreground="white")],
        zorder=10,
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
    """Draw Ascension's fixed observations and dynamic Sorrento raster sample."""

    for name, latitude, longitude, total in ASCENSION_STATIONS:
        _draw_location_label(
            ax,
            f'{name}\n{total:.2f}"',
            latitude,
            longitude,
            offset=ASCENSION_LABEL_OFFSETS[name],
            marker_size=36,
        )

    sorrento_total = _sample_grid(
        data,
        grid,
        SORRENTO_SAMPLE["latitude"],
        SORRENTO_SAMPLE["longitude"],
    )
    if sorrento_total is not None:
        _draw_location_label(
            ax,
            f'{SORRENTO_SAMPLE["name"]}\n{sorrento_total:.2f}"',
            SORRENTO_SAMPLE["latitude"],
            SORRENTO_SAMPLE["longitude"],
            offset=(8, 8),
            marker_size=36,
        )
    return True


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
    rainfall_display_mode: str = "Smooth",
) -> bytes:
    """Render a rainfall accumulation map and return PNG bytes."""

    cmap = ListedColormap(RAIN_COLORS, name="lix_rainfall")
    cmap.set_bad("#f7f4ed")
    cmap.set_under("#f7f4ed")
    cmap.set_over("#5a1421")
    norm = BoundaryNorm(RAIN_LEVELS, cmap.N)

    figure_size = (9.2, 9.2) if region_name == "Ascension Parish" else (11.5, 9.2)
    fig, ax = plt.subplots(figsize=figure_size, dpi=160)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f7f4ed")

    plotted = np.ma.masked_invalid(data)
    image = ax.imshow(
        plotted,
        extent=grid.extent,
        origin="upper",
        cmap=cmap,
        norm=norm,
        interpolation=_rainfall_interpolation(rainfall_display_mode),
        zorder=1,
    )

    if show_counties:
        _draw_boundaries(
            ax,
            boundaries,
            "county",
            color="#a5a5a5",
            linewidth=0.42,
            alpha=0.85,
            zorder=4,
        )
    _draw_boundaries(
        ax,
        boundaries,
        "state",
        color="#17212b",
        linewidth=1.75,
        alpha=1,
        zorder=6,
    )

    if region_name == "WFO LIX":
        _draw_boundaries(
            ax,
            boundaries,
            "cwa",
            color="#111111",
            linewidth=2.8,
            alpha=1,
            zorder=8,
        )
    elif region_name == "Ascension Parish":
        _draw_ascension_boundary(ax, boundaries)

    if region_name == "Ascension Parish":
        _draw_ascension_locations(
            ax,
            data,
            grid,
            start,
            end,
            show_cities=show_cities,
            show_city_samples=show_city_samples,
        )
    elif show_cities or show_city_samples:
        longitude_span = grid.east - grid.west
        for name, (latitude, longitude, maximum_span) in CITIES.items():
            if longitude_span > maximum_span:
                continue
            if grid.west < longitude < grid.east and grid.south < latitude < grid.north:
                label = name
                if show_city_samples:
                    sample = _sample_grid(data, grid, latitude, longitude)
                    if sample is not None:
                        label = f'{name}\n{sample:.2f}"'
                _draw_location_label(
                    ax,
                    label,
                    latitude,
                    longitude,
                    offset=CITY_LABEL_OFFSETS.get(name, (5, 5)),
                )

    title = custom_title.strip() or "Observed Rainfall"
    period = _format_date(start) if start == end else f"{_format_date(start)} – {_format_date(end)}"
    ax.set_title(title, loc="left", fontsize=20, weight="bold", color="#13283a", pad=30)
    ax.text(
        0,
        1.014,
        f"Total multi-sensor precipitation • {period}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.5,
        color="#516170",
    )

    if region_name == "WFO LIX":
        ax.text(
            0.985,
            0.025,
            "WFO LIX forecast area",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7.5,
            weight="semibold",
            color="#111111",
            zorder=10,
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="white",
                edgecolor="#111111",
                alpha=0.9,
            ),
        )
    elif region_name == "Ascension Parish":
        note = "Ascension Parish boundary"
        ax.text(
            0.985,
            0.025,
            note,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7.3,
            weight="semibold",
            color="#111111",
            zorder=10,
            bbox={
                "boxstyle": "round,pad=0.4",
                "facecolor": "white",
                "edgecolor": "#111111",
                "alpha": 0.92,
            },
        )

    ax.set_xlim(grid.west, grid.east)
    ax.set_ylim(grid.south, grid.north)
    ax.set_aspect(1 / math.cos(math.radians((grid.south + grid.north) / 2)))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#4e5963")
        spine.set_linewidth(0.8)

    ax.annotate(
        "N",
        xy=(0.955, 0.905),
        xytext=(0.955, 0.82),
        xycoords="axes fraction",
        textcoords="axes fraction",
        ha="center",
        va="center",
        fontsize=8,
        weight="bold",
        color="#17212b",
        arrowprops=dict(arrowstyle="-|>", lw=1.5, color="#17212b"),
        zorder=9,
    )
    _add_scale_bar(ax, grid)

    colorbar = fig.colorbar(
        image,
        ax=ax,
        orientation="horizontal",
        fraction=0.045,
        pad=0.045,
        aspect=40,
        ticks=RAIN_LEVELS,
        extend="max",
    )
    colorbar.ax.tick_params(labelsize=7.2, length=2.5, pad=2)
    colorbar.set_label("Rainfall (inches)", fontsize=9.5, weight="semibold", labelpad=7)
    colorbar.outline.set_linewidth(0.6)

    source_note = (
        "Source: NOAA/NWS River Forecast Center multi-sensor precipitation estimates"
        " • Daily periods valid 12Z–12Z"
    )
    fig.text(
        0.5,
        0.018,
        source_note,
        ha="center",
        va="bottom",
        fontsize=7.1,
        color="#68747f",
    )
    fig.subplots_adjust(left=0.035, right=0.965, top=0.90, bottom=0.105)

    output = BytesIO()
    save_options = {} if region_name == "Ascension Parish" else {"bbox_inches": "tight"}
    fig.savefig(output, format="png", dpi=180, facecolor="white", **save_options)
    plt.close(fig)
    return output.getvalue()
