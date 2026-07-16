"""Publication-style map rendering for accumulated precipitation."""

from __future__ import annotations

import math
from datetime import date
from io import BytesIO

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
    "Alexandria": (31.312, -92.446),
    "Baton Rouge": (30.451, -91.187),
    "Bay St. Louis": (30.309, -89.330),
    "Biloxi": (30.396, -88.885),
    "Bogalusa": (30.791, -89.849),
    "Covington": (30.475, -90.100),
    "Greenville": (33.411, -91.061),
    "Gulfport": (30.367, -89.093),
    "Hammond": (30.504, -90.462),
    "Hattiesburg": (31.327, -89.290),
    "Houma": (29.596, -90.720),
    "Jackson": (32.299, -90.185),
    "Lafayette": (30.224, -92.020),
    "Lake Charles": (30.226, -93.217),
    "McComb": (31.244, -90.453),
    "Meridian": (32.365, -88.704),
    "Monroe": (32.510, -92.120),
    "Morgan City": (29.699, -91.207),
    "New Orleans": (29.951, -90.072),
    "Picayune": (30.526, -89.679),
    "Shreveport": (32.526, -93.750),
    "Slidell": (30.275, -89.782),
    "Thibodaux": (29.795, -90.822),
    "Tupelo": (34.258, -88.704),
}




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




def _format_date(day: date) -> str:
    return f"{day.strftime('%B')} {day.day}, {day.year}"


def _add_scale_bar(ax, grid: TargetGrid) -> None:
    miles = 20 if (grid.east - grid.west) < 2.0 else 50
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
        start + degrees / 2, latitude + label_offset, f"{miles} miles",
        ha="center", va="bottom", fontsize=7.5, color="#17212b", zorder=9,
    )


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
    region_name: str = "",
) -> bytes:
    """Render a rainfall accumulation map and return PNG bytes."""

    cmap = ListedColormap(RAIN_COLORS, name="lix_rainfall")
    cmap.set_bad("#f7f4ed")
    cmap.set_under("#f7f4ed")
    cmap.set_over("#5a1421")
    norm = BoundaryNorm(RAIN_LEVELS, cmap.N)

    fig, ax = plt.subplots(figsize=(11.5, 9.2), dpi=160)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f7f4ed")

    plotted = np.ma.masked_invalid(data)
    image = ax.imshow(
        plotted,
        extent=grid.extent,
        origin="upper",
        cmap=cmap,
        norm=norm,
        interpolation="bilinear",
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
        color="white",
        linewidth=2.5,
        alpha=0.95,
        zorder=5,
    )
    _draw_boundaries(
        ax,
        boundaries,
        "state",
        color="#17212b",
        linewidth=1.30,
        alpha=1,
        zorder=6,
    )

    if region_name == "WFO LIX":
        _draw_boundaries(
            ax,
            boundaries,
            "cwa",
            color="white",
            linewidth=4.2,
            alpha=0.95,
            zorder=7,
        )
        _draw_boundaries(
            ax,
            boundaries,
            "cwa",
            color="#111111",
            linewidth=2.3,
            alpha=1,
            zorder=8,
        )

    if show_cities:
        for name, (latitude, longitude) in CITIES.items():
            if grid.west < longitude < grid.east and grid.south < latitude < grid.north:
                ax.scatter(
                    longitude,
                    latitude,
                    s=8,
                    facecolor="white",
                    edgecolor="#17212b",
                    linewidth=0.55,
                    zorder=7,
                )
                ax.annotate(
                    name,
                    (longitude, latitude),
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=6.3,
                    color="#101820",
                    weight="semibold",
                    path_effects=[],
                    zorder=8,
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

    fig.text(
        0.5,
        0.018,
        "Source: NOAA/NWS River Forecast Center multi-sensor precipitation estimates • Daily periods valid 12Z–12Z",
        ha="center",
        va="bottom",
        fontsize=7.1,
        color="#68747f",
    )
    fig.subplots_adjust(left=0.035, right=0.965, top=0.90, bottom=0.105)

    output = BytesIO()
    fig.savefig(output, format="png", dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return output.getvalue()
