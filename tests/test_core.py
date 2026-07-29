from datetime import date

import pytest

from rainfall.core import (
    REGIONS,
    ArchiveSource,
    DateRangeError,
    build_target_grid,
    date_sequence,
    source_for_date,
    validate_date_range,
)


def test_date_sequence_is_inclusive():
    assert date_sequence(date(2016, 8, 1), date(2016, 8, 3)) == [
        date(2016, 8, 1),
        date(2016, 8, 2),
        date(2016, 8, 3),
    ]


def test_2016_uses_stage_iii_geotiff():
    product = source_for_date(date(2016, 8, 13))
    assert product.source is ArchiveSource.STAGE_III
    assert product.url.endswith("stageIII/2016/08/13/nws_precip_1day_20160813.tif")


def test_newer_dates_use_stage_iv_conus_geotiff():
    product = source_for_date(date(2024, 2, 2))
    assert product.source is ArchiveSource.STAGE_IV
    assert product.url.endswith("stageIV/2024/02/02/nws_precip_1day_20240202_conus.tif")


def test_rejects_more_than_90_days():
    with pytest.raises(DateRangeError, match="90 days"):
        validate_date_range(date(2016, 1, 1), date(2016, 4, 1), today=date(2026, 1, 1))


def test_grid_dimensions_and_extent():
    grid = build_target_grid("Louisiana + Mississippi")
    assert grid.width > 200
    assert grid.height > 200
    assert grid.extent == (-94.35, -88.05, 28.70, 35.10)


def test_operational_region_presets_are_available():
    expected = {
        "WFO LIX",
        "New Orleans Metro",
        "Baton Rouge Metro",
        "Ascension Parish",
        "Southwest Mississippi",
        "Coastal Mississippi",
        "Coastal Louisiana",
    }
    assert expected <= REGIONS.keys()


@pytest.mark.parametrize(
    "region_name",
    [
        "WFO LIX",
        "New Orleans Metro",
        "Baton Rouge Metro",
        "Ascension Parish",
        "Southwest Mississippi",
        "Coastal Mississippi",
        "Coastal Louisiana",
    ],
)
def test_operational_region_grids_have_valid_dimensions(region_name):
    grid = build_target_grid(region_name)
    assert grid.width > 0
    assert grid.height > 0
    assert grid.west < grid.east
    assert grid.south < grid.north
