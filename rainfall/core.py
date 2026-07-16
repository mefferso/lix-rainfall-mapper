"""NOAA precipitation archive selection and raster accumulation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from enum import Enum
from typing import Callable, Sequence

import numpy as np
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, reproject


class DateRangeError(ValueError):
    """Raised when a requested accumulation period is invalid."""


class ArchiveSource(str, Enum):
    STAGE_III = "Stage III"
    STAGE_IV = "Stage IV"


@dataclass(frozen=True)
class DailyProduct:
    valid_date: date
    source: ArchiveSource
    url: str


@dataclass(frozen=True)
class TargetGrid:
    west: float
    south: float
    east: float
    north: float
    width: int
    height: int
    transform: object
    crs: str = "EPSG:4326"

    @property
    def extent(self) -> tuple[float, float, float, float]:
        return self.west, self.east, self.south, self.north


# Bounds are west, south, east, north. Broad state views are followed by
# operational/local presets in the order shown in the Streamlit selector.
REGIONS: dict[str, tuple[float, float, float, float]] = {
    "Louisiana + Mississippi": (-94.35, 28.70, -88.05, 35.10),
    "Louisiana": (-94.25, 28.70, -88.65, 33.15),
    "Mississippi": (-91.80, 29.95, -87.95, 35.10),
    "WFO LIX": (-91.85, 28.70, -88.55, 31.35),
    "New Orleans Metro": (-90.75, 29.55, -89.45, 30.50),
    "Baton Rouge Metro": (-91.65, 30.15, -90.55, 30.85),
    "Southwest Mississippi": (-91.10, 30.70, -89.45, 31.70),
    "Coastal Mississippi": (-89.85, 30.15, -88.30, 30.95),
    "Coastal Louisiana": (-93.95, 28.65, -88.65, 30.75),
}

STAGE_III_LAST_DATE = date(2017, 6, 27)
EARLIEST_SUPPORTED_DATE = date(2005, 1, 1)
MAX_DAYS = 90


def source_for_date(day: date) -> DailyProduct:
    """Return the best directly readable NOAA daily GeoTIFF for a date."""

    stamp = day.strftime("%Y%m%d")
    path = day.strftime("%Y/%m/%d")

    if day <= STAGE_III_LAST_DATE:
        return DailyProduct(
            valid_date=day,
            source=ArchiveSource.STAGE_III,
            url=(
                "https://water.noaa.gov/resources/downloads/precip/"
                f"stageIII/{path}/nws_precip_1day_{stamp}.tif"
            ),
        )

    return DailyProduct(
        valid_date=day,
        source=ArchiveSource.STAGE_IV,
        url=(
            "https://water.noaa.gov/resources/downloads/precip/"
            f"stageIV/{path}/nws_precip_1day_{stamp}_conus.tif"
        ),
    )


def date_sequence(start: date, end: date) -> list[date]:
    """Return every date in an inclusive range."""

    count = (end - start).days + 1
    return [start + timedelta(days=offset) for offset in range(max(count, 0))]


def validate_date_range(start: date, end: date, *, today: date | None = None) -> None:
    """Validate a user-selected inclusive accumulation period."""

    today = today or date.today()
    if start > end:
        raise DateRangeError("The start date must be on or before the end date.")
    if start < EARLIEST_SUPPORTED_DATE:
        raise DateRangeError(
            f"This version supports dates beginning {EARLIEST_SUPPORTED_DATE:%B %d, %Y}."
        )
    if end >= today:
        raise DateRangeError("Choose an end date no later than yesterday.")
    days = (end - start).days + 1
    if days > MAX_DAYS:
        raise DateRangeError(f"Choose a period of {MAX_DAYS} days or fewer.")


def build_target_grid(region_name: str, resolution: float = 0.025) -> TargetGrid:
    """Build a fixed latitude/longitude output grid for a named region."""

    west, south, east, north = REGIONS[region_name]
    width = int(round((east - west) / resolution))
    height = int(round((north - south) / resolution))
    transform = from_bounds(west, south, east, north, width, height)
    return TargetGrid(west, south, east, north, width, height, transform)


def _clean_source_array(dataset) -> np.ndarray:
    raw = dataset.read(1)
    invalid = ~np.isfinite(raw) | (raw < 0)
    if dataset.nodata is not None and np.isfinite(dataset.nodata):
        invalid |= np.isclose(raw, dataset.nodata)
    # Replace extreme legacy nodata sentinels before narrowing float64 to
    # float32; otherwise NumPy correctly warns that the sentinel overflows.
    source = np.where(invalid, 0.0, raw).astype("float32")
    source[invalid] = np.nan
    return source


def accumulate_rasters(
    raster_bytes: Sequence[bytes],
    grid: TargetGrid,
    progress: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """Reproject and sum daily NOAA GeoTIFFs onto a common output grid."""

    if not raster_bytes:
        raise ValueError("At least one daily raster is required.")

    total = np.zeros((grid.height, grid.width), dtype="float32")
    valid_anywhere = np.zeros_like(total, dtype=bool)

    for index, payload in enumerate(raster_bytes, start=1):
        with MemoryFile(payload) as memory_file:
            with memory_file.open() as dataset:
                source = _clean_source_array(dataset)
                daily = np.full_like(total, np.nan)
                # Older Stage III GeoTIFFs use a custom spherical geographic
                # CRS. Their coordinates are already longitude/latitude, but
                # PROJ cannot always derive a WGS84 datum transformation.
                destination_crs = dataset.crs if dataset.crs.is_geographic else grid.crs
                reproject(
                    source=source,
                    destination=daily,
                    src_transform=dataset.transform,
                    src_crs=dataset.crs,
                    src_nodata=np.nan,
                    dst_transform=grid.transform,
                    dst_crs=destination_crs,
                    dst_nodata=np.nan,
                    resampling=Resampling.bilinear,
                    init_dest_nodata=True,
                )

        valid = np.isfinite(daily)
        total[valid] += daily[valid]
        valid_anywhere |= valid
        if progress:
            progress(index, len(raster_bytes))

    total[~valid_anywhere] = np.nan
    total[total < 0] = np.nan
    return total


def grid_cell_centers(grid: TargetGrid) -> tuple[np.ndarray, np.ndarray]:
    """Return one-dimensional longitude and latitude cell-center arrays."""

    longitudes = np.linspace(grid.west, grid.east, grid.width, endpoint=False)
    latitudes = np.linspace(grid.north, grid.south, grid.height, endpoint=False)
    pixel_width = (grid.east - grid.west) / grid.width
    pixel_height = (grid.north - grid.south) / grid.height
    return longitudes + pixel_width / 2, latitudes - pixel_height / 2


def maximum_location(data: np.ndarray, grid: TargetGrid) -> tuple[float, float, float]:
    """Return maximum inches, latitude, and longitude within the displayed grid."""

    row, column = np.unravel_index(np.nanargmax(data), data.shape)
    longitudes, latitudes = grid_cell_centers(grid)
    return float(data[row, column]), float(latitudes[row]), float(longitudes[column])


def make_geotiff(data: np.ndarray, grid: TargetGrid) -> bytes:
    """Serialize an accumulated grid as a compressed GeoTIFF in inches."""

    nodata = -9999.0
    encoded = np.where(np.isfinite(data), data, nodata).astype("float32")
    with MemoryFile() as memory_file:
        with memory_file.open(
            driver="GTiff",
            height=grid.height,
            width=grid.width,
            count=1,
            dtype="float32",
            crs=grid.crs,
            transform=grid.transform,
            nodata=nodata,
            compress="deflate",
            predictor=3,
        ) as dataset:
            dataset.write(encoded, 1)
            dataset.update_tags(
                units="inches",
                description="Accumulated NOAA RFC multi-sensor precipitation estimate",
            )
        return memory_file.read()
