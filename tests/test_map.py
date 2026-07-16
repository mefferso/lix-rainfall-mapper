import numpy as np

from rainfall.core import TargetGrid
from rainfall.map import CITIES, _sample_grid


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
