from rainfall.boundaries import fetch_la_ms_boundaries


def test_bundled_nws_boundaries_include_expected_layers():
    boundaries = fetch_la_ms_boundaries()
    layers = [feature["properties"]["layer"] for feature in boundaries["features"]]
    assert layers.count("county") == 146
    assert layers.count("state") == 2
    assert layers.count("cwa") == 1


def test_bundled_cwa_is_wfo_lix():
    boundaries = fetch_la_ms_boundaries()
    cwa = next(
        feature for feature in boundaries["features"] if feature["properties"]["layer"] == "cwa"
    )
    assert cwa["properties"]["wfo"] == "LIX"
    assert cwa["properties"]["station"] == "KLIX"
