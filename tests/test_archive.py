"""Offline tests for NOAA Stage IV archive fallback selection."""

import tarfile
from datetime import date
from io import BytesIO

import numpy as np
import pytest
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

from rainfall.archive import (
    ArchiveUnavailable,
    _convert_raster,
    _from_tar,
    _unit_factor,
    candidate_urls,
    fetch_daily_geotiff,
)
from rainfall.core import source_for_date


def test_stage_iv_candidates_are_same_day_and_ordered():
    product = source_for_date(date(2020, 11, 27))
    choices = candidate_urls(product)
    assert [fmt for fmt, _ in choices] == ["GeoTIFF", "NetCDF", "source GRIB2"]
    assert choices[0][1].endswith("nws_precip_1day_20201127_conus.tif")
    assert choices[1][1].endswith("nws_precip_1day_20201127_conus.nc")
    assert choices[2][1].endswith("ncep_stage_iv_source_files_20201127.tar")


def test_stage_iii_does_not_use_stage_iv_substitutes():
    assert len(candidate_urls(source_for_date(date(2016, 8, 13)))) == 1


@pytest.mark.parametrize("units,expected", [
    ("inches", 1),
    ("mm", 1 / 25.4),
    ("kg m-2", 1 / 25.4),
    ("m", 1000 / 25.4),
])
def test_known_precipitation_units(units, expected):
    assert _unit_factor(units, format_name="test") == pytest.approx(expected)


def test_unknown_units_rejected():
    with pytest.raises(ArchiveUnavailable, match="not recognized"):
        _unit_factor("unverified", format_name="test")


def test_fallback_from_missing_geotiff_to_netcdf(monkeypatch):
    from rainfall import archive

    class Response:
        def __init__(self, status_code, content=b""):
            self.status_code = status_code
            self.content = content

        def raise_for_status(self):
            pass

    class FakeSession:
        def __init__(self):
            self.urls = []

        def get(self, url, **kwargs):
            self.urls.append(url)
            if url.endswith(".tif"):
                return Response(404)
            return Response(200, b"x" * 1024)

    monkeypatch.setattr(
        archive, "_convert_raster",
        lambda data, suffix, *, format_name: b"converted"
    )
    session = FakeSession()
    result = fetch_daily_geotiff(source_for_date(date(2020, 11, 27)), session)
    assert result == b"converted"
    assert [url.rsplit(".", 1)[-1] for url in session.urls] == ["tif", "nc"]


def test_failed_candidates_raise_and_do_not_return_partial_data():
    class Response:
        status_code = 404
        content = b""

        def raise_for_status(self):
            pass

    class FakeSession:
        def get(self, url, **kwargs):
            return Response()

    with pytest.raises(ArchiveUnavailable, match="GeoTIFF.*NetCDF.*source GRIB2"):
        fetch_daily_geotiff(source_for_date(date(2020, 11, 27)), FakeSession())


def test_source_tar_rejects_wrong_date():
    stream = BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as tar:
        blob = b"GRIB" * 500
        info = tarfile.TarInfo("conun_20201126_24.grb2")
        info.size = len(blob)
        tar.addfile(info, BytesIO(blob))
    with pytest.raises(ArchiveUnavailable, match="0 matching CONUS"):
        _from_tar(stream.getvalue(), date(2020, 11, 27))


def test_stage_iv_historical_conus_24h_naming_in_source_tar(monkeypatch):
    from rainfall import archive

    blob = b"GRIB" * 500
    stream = BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as tar:
        for name in ("ak_20201127_24h.grb2", "conus_20201127_24h.grb2"):
            info = tarfile.TarInfo(name)
            info.size = len(blob)
            tar.addfile(info, BytesIO(blob))
    monkeypatch.setattr(
        archive, "_convert_raster",
        lambda data, suffix, *, format_name: b"processed" if data == blob else b"bad",
    )
    assert _from_tar(stream.getvalue(), date(2020, 11, 27)) == b"processed"


def test_unknown_archive_members_reported():
    stream = BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as tar:
        blob = b"GRIB" * 500
        info = tarfile.TarInfo("unexpected_grid_20201127.grb2")
        info.size = len(blob)
        tar.addfile(info, BytesIO(blob))
    with pytest.raises(ArchiveUnavailable, match="unexpected_grid_20201127.grb2"):
        _from_tar(stream.getvalue(), date(2020, 11, 27))


def _one_band_raster(*, units="kg/(m^2)", count=1):
    with MemoryFile() as mem:
        with mem.open(
            driver="GTiff", width=4, height=4, count=count,
            dtype="float32", crs="EPSG:4326",
            transform=from_origin(-91, 31, 0.01, 0.01),
        ) as dst:
            for band in range(1, count + 1):
                dst.write(np.full((4, 4), 25.4, dtype="float32"), band)
                dst.update_tags(band, GRIB_UNIT=f"[{units}]")
        return mem.read()


def test_stage_iv_single_band_missing_grib_element_still_converts_inches():
    payload = _convert_raster(
        _one_band_raster(), ".grb2", format_name="GRIB2"
    )
    with MemoryFile(payload) as mem, mem.open() as ds:
        assert ds.count == 1
        assert np.allclose(ds.read(1), 1.0)
        assert ds.tags().get("units") == "inches"


def test_stage_iv_multiband_missing_grib_element_fails():
    with pytest.raises(ArchiveUnavailable, match="cannot uniquely select"):
        _convert_raster(
            _one_band_raster(count=2), ".grb2", format_name="GRIB2"
        )


def test_grib_ambiguous_units_are_rejected():
    with pytest.raises(ArchiveUnavailable, match="not recognized"):
        _convert_raster(
            _one_band_raster(units="unknown"), ".grb2", format_name="GRIB2"
        )
