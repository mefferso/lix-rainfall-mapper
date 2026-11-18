"""Offline tests for NOAA Stage IV archive fallback selection."""

import tarfile
from datetime import date
from io import BytesIO

import pytest

from rainfall.archive import (
    ArchiveUnavailable,
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
    with pytest.raises(ArchiveUnavailable, match="exactly one CONUS"):
        _from_tar(stream.getvalue(), date(2020, 11, 27))
