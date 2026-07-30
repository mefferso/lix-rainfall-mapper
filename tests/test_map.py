from datetime import date
from unittest.mock import Mock, patch

import matplotlib.pyplot as plt
import numpy as np
import pytest

from rainfall.core import TargetGrid
from rainfall.map import (
    ASCENSION_LABEL_OFFSETS,
    ASCENSION_STATIONS,
    CITIES,
    _draw_ascension_locations,
    _is_ascension_feature,
    _rainfall_interpolation,
    _sample_grid,
    render_map,
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


def test_ascension_uses_three_fixed_observations():
    assert [(row[0], row[3]) for row in ASCENSION_STATIONS] == [
        ("Prairieville 2.0 S", 15.02),
        ("Gonzales 0.8 E", 11.41),
        ("Gonzales 4.5 S", 19.13),
    ]
    assert ASCENSION_LABEL_OFFSETS == {
        "Prairieville 2.0 S": (8, 8),
        "Gonzales 0.8 E": (8, 8),
        "Gonzales 4.5 S": (-8, 8),
    }


def test_ascension_observations_ignore_grid_dates_and_city_options():
    ax = Mock()
    grid = TargetGrid(-91.14, 30.03, -90.60, 30.38, 2, 2, None)

    _draw_ascension_locations(
        ax,
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        grid,
        date(2025, 1, 1),
        date(2025, 1, 31),
        show_cities=False,
        show_city_samples=True,
    )

    assert [call.args[:2] for call in ax.scatter.call_args_list] == [
        (-90.979147, 30.276934),
        (-90.909870, 30.217250),
        (-90.928910, 30.151899),
    ]
    assert [call.args[0] for call in ax.annotate.call_args_list] == [
        'Prairieville 2.0 S\n15.02"',
        'Gonzales 0.8 E\n11.41"',
        'Gonzales 4.5 S\n19.13"',
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


def test_rainfall_display_modes_select_only_display_interpolation():
    assert _rainfall_interpolation("Raw") == "nearest"
    assert _rainfall_interpolation("Smooth") == "bilinear"


def test_invalid_rainfall_display_mode_is_rejected():
    with pytest.raises(ValueError, match="Rainfall display mode"):
        _rainfall_interpolation("Blurred")


def test_render_modes_leave_accumulated_grid_unchanged():
    grid = TargetGrid(0, 0, 2, 2, 2, 2, None)
    data = np.array([[0.1, 1.0], [4.0, np.nan]])
    original = data.copy()
    boundaries = {"features": []}

    with patch("matplotlib.axes.Axes.imshow", autospec=True) as imshow:
        # The remaining renderer expects an image-like colorbar input, so stop
        # immediately after recording the actual rainfall imshow call.
        imshow.side_effect = RuntimeError("captured")
        for mode, expected in (("Raw", "nearest"), ("Smooth", "bilinear")):
            with pytest.raises(RuntimeError, match="captured"):
                render_map(
                    data,
                    grid,
                    boundaries,
                    date(2016, 8, 12),
                    date(2016, 8, 14),
                    rainfall_display_mode=mode,
                )
            assert imshow.call_args.kwargs["interpolation"] == expected

    plt.close("all")

    np.testing.assert_equal(data, original)
