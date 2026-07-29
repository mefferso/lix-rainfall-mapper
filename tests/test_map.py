import numpy as np

from rainfall.core import TargetGrid
from rainfall.map import (
    ASCENSION_EVENT_DATES,
    ASCENSION_GAUGE_TOTALS,
    CITIES,
    _is_ascension_feature,
    _sample_grid,
)


def test_sample_grid_returns_nearest_city_value():
    grid = TargetGrid(
        west=0,
        south=0,
        east=2,
        north=2,
        width=2,
        height=2,
        transform=None,
    )
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert _sample_grid(data, grid, latitude=1.5, longitude=0.5) == 1.0
    assert _sample_grid(data, grid, latitude=0.5, longitude=1.5) == 4.0


def test_sample_grid_rejects_locations_outside_map():
    grid = TargetGrid(0, 0, 2, 2, 2, 2, None)
    data = np.ones((2, 2))
    assert _sample_grid(data, grid, latitude=3, longitude=1) is None


def test_local_city_reference_set_is_expanded():
    assert len(CITIES) >= 50
    for city in ("Woodville", "Franklinton", "Gonzales", "Pascagoula", "Grand Isle"):
        assert city in CITIES


def test_ascension_event_uses_complete_cocorahs_totals():
    assert tuple(str(day) for day in ASCENSION_EVENT_DATES) == (
        "2016-08-12",
        "2016-08-14",
    )
    assert [(row[0], row[3]) for row in ASCENSION_GAUGE_TOTALS] == [
        ("Prairieville 2.0 S", 15.02),
        ("Gonzales 0.8 E", 11.41),
        ("Gonzales 4.5 S", 19.13),
    ]


def test_ascension_boundary_match_accepts_name_or_fips():
    assert _is_ascension_feature(
        {"properties": {"layer": "county", "name": "Ascension"}}
    )
    assert _is_ascension_feature(
        {"properties": {"layer": "county", "fips": "22005"}}
    )
    assert not _is_ascension_feature(
        {"properties": {"layer": "county", "name": "Livingston"}}
    )
