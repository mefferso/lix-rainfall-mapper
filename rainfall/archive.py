"""Resilient NOAA daily Stage IV download: GeoTIFF -> NetCDF -> source GRIB2.

All returned rasters are GeoTIFFs containing precipitation in inches.
Never substitute a different precipitation analysis or an unverified unit.
"""
from __future__ import annotations

import tarfile
import time
from datetime import date
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import rasterio
import requests
from rasterio.io import MemoryFile

from rainfall.core import ArchiveSource, DailyProduct


class ArchiveUnavailable(RuntimeError):
    pass


def candidate_urls(product: DailyProduct) -> list[tuple[str, str]]:
    """Use only NOAA files for the exact same valid day and product."""
    urls = [("GeoTIFF", product.url)]
    if product.source == ArchiveSource.STAGE_IV:
        prefix = product.url.rsplit("/", 1)[0]
        stamp = product.valid_date.strftime("%Y%m%d")
        urls.extend([
            ("NetCDF", product.url[:-4] + ".nc"),
            ("source GRIB2", f"{prefix}/ncep_stage_iv_source_files_{stamp}.tar"),
        ])
    return urls


def _get(session: requests.Session, url: str) -> bytes:
    last_error = None
    for attempt in range(3):
        try:
            response = session.get(
                url,
                headers={"User-Agent": "LIX-Rainfall-Mapper/1.1 (NOAA QPE archive)"},
                timeout=(15, 90),
            )
            if response.status_code == 404:
                raise FileNotFoundError("HTTP 404")
            response.raise_for_status()
            if len(response.content) < 1000:
                raise ValueError("response unexpectedly small")
            return response.content
        except FileNotFoundError:
            raise  # A missing candidate will not become available on an immediate retry.
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(1.25 * (attempt + 1))
    raise ArchiveUnavailable(str(last_error))


def _unit_factor(units: str, *, format_name: str) -> float:
    """Convert known units to inches; reject ambiguity instead of corrupting totals."""
    unit = (units.strip().lower().replace(" ", "").replace("_", "")
            .replace("(", "").replace(")", ""))
    if unit in {"in", "inch", "inches"}:
        return 1.0
    if unit in {"mm", "millimeter", "millimeters", "kgm-2", "kg/m^2", "kg/m2",
                "kgm**-2", "kgm^-2"}:
        return 1.0 / 25.4
    if unit in {"m", "meter", "meters"}:
        return 1000.0 / 25.4
    raise ArchiveUnavailable(
        f"{format_name} precipitation units {units!r} are not recognized; "
        "refusing to assume inches"
    )


def _convert_raster(data: bytes, suffix: str, *, format_name: str) -> bytes:
    """Read GDAL-supported archived grids and normalize into an inch GeoTIFF."""
    with TemporaryDirectory() as temporary:
        path = Path(temporary) / ("daily" + suffix)
        path.write_bytes(data)
        with rasterio.open(path) as container:
            subdatasets = container.subdatasets
        if subdatasets:
            matching = [item for item in subdatasets
                        if item.rsplit(":", 1)[-1].strip('"').lower() == "observation"]
            if len(matching) != 1:
                raise ArchiveUnavailable(
                    f"{format_name}: cannot uniquely identify the observation subdataset: "
                    f"{subdatasets}"
                )
            source_path = matching[0]
        else:
            source_path = str(path)

        with rasterio.open(source_path) as ds:
            if not ds.crs or not ds.transform or ds.count < 1:
                raise ArchiveUnavailable(f"{format_name}: missing georeferencing")
            if format_name == "GRIB2":
                matches = [
                    band for band in range(1, ds.count + 1)
                    if ds.tags(band).get("GRIB_ELEMENT", "").upper() in {"APCP", "TP"}
                ]
                if len(matches) != 1:
                    raise ArchiveUnavailable(
                        f"GRIB2: expected exactly one accumulated-precipitation band, "
                        f"found {len(matches)}"
                    )
                band = matches[0]
            else:
                if ds.count != 1:
                    raise ArchiveUnavailable(
                        f"{format_name}: expected one precipitation band, got {ds.count}"
                    )
                band = 1

            tags = ds.tags(band)
            units = (
                tags.get("units") or tags.get("Units") or
                tags.get("GRIB_UNIT") or
                (ds.units[band - 1] if ds.units else None) or
                ds.tags().get("units") or ds.tags().get("Units")
            )
            if not units:
                raise ArchiveUnavailable(f"{format_name}: precipitation units missing")
            factor = _unit_factor(units, format_name=format_name)
            values = ds.read(band, masked=True).astype("float32")
            scaled = values.filled(np.nan) * factor
            scaled[~np.isfinite(scaled) | (scaled < 0)] = np.nan
            profile = ds.profile.copy()
            profile.update(driver="GTiff", count=1, dtype="float32",
                           nodata=-9999.0, compress="deflate", predictor=3)
            profile.pop("blockxsize", None)
            profile.pop("blockysize", None)
            profile.pop("tiled", None)
            with MemoryFile() as mem:
                with mem.open(**profile) as out:
                    out.write(np.where(np.isfinite(scaled), scaled, -9999.0), 1)
                    out.update_tags(units="inches", source_format=format_name)
                return mem.read()


def _from_tar(data: bytes, expected_date: date) -> bytes:
    stamp = expected_date.strftime("%Y%m%d")
    with tarfile.open(fileobj=BytesIO(data), mode="r:*") as archive:
        # NOAA's source TARs span more than one naming convention.
        # The ST4 timestamp marks the 12Z END of the 24-hour QPE window.
        accepted = {
            f"conus_{stamp}_24.grb2",
            f"conun_{stamp}_24.grb2",
            f"st4.{stamp}12.24h.grb2",
        }
        contents = [member for member in archive.getmembers() if member.isfile()]
        members = [
            member for member in contents
            if Path(member.name).name.lower() in accepted
        ]
        if len(members) != 1:
            seen = ", ".join(Path(member.name).name for member in contents[:15])
            raise ArchiveUnavailable(
                f"Source TAR has {len(members)} matching CONUS 24-hour GRIB2 "
                f"members for {stamp}; archive members: {seen or '(empty)'}"
            )
        member = members[0]
        if member.size > 250_000_000:
            raise ArchiveUnavailable("GRIB2 member exceeds 250 MB safety limit")
        handle = archive.extractfile(member)
        if handle is None:
            raise ArchiveUnavailable("Cannot read CONUS GRIB2 in TAR")
        return _convert_raster(handle.read(), ".grb2", format_name="GRIB2")


def fetch_daily_geotiff(product: DailyProduct, session: requests.Session | None = None) -> bytes:
    """Return original GeoTIFF or a verified equivalent in inches."""
    session = session or requests.Session()
    errors = []
    for format_name, url in candidate_urls(product):
        try:
            payload = _get(session, url)
            if format_name == "GeoTIFF":
                # Preserve the existing known-inches NOAA GeoTIFF path.
                return payload
            if format_name == "NetCDF":
                return _convert_raster(payload, ".nc", format_name="NetCDF")
            return _from_tar(payload, product.valid_date)
        except (ArchiveUnavailable, FileNotFoundError, rasterio.errors.RasterioError,
                tarfile.TarError, OSError, ValueError) as exc:
            errors.append(f"{format_name}: {exc}")
    raise ArchiveUnavailable(
        f"No verified NOAA daily Stage IV grid for {product.valid_date}: "
        + "; ".join(errors)
    )
