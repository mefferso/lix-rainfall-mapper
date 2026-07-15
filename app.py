"""Streamlit interface for creating custom NOAA rainfall maps."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
import time

import requests
import streamlit as st

from rainfall.core import (
    REGIONS,
    DateRangeError,
    accumulate_rasters,
    build_target_grid,
    date_sequence,
    make_geotiff,
    maximum_location,
    source_for_date,
    validate_date_range,
)
from rainfall.boundaries import fetch_la_ms_boundaries
from rainfall.map import render_map


st.set_page_config(
    page_title="LIX Rainfall Mapper",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {max-width: 1220px; padding-top: 2.2rem; padding-bottom: 3rem;}
      [data-testid="stSidebar"] {border-right: 1px solid #dbe2e8;}
      .eyebrow {color:#2d6f9f; font-size:.76rem; font-weight:800; letter-spacing:.13em; text-transform:uppercase;}
      .hero-title {font-size:2.55rem; line-height:1.05; font-weight:800; color:#102b3f; margin:.25rem 0 .6rem;}
      .hero-copy {font-size:1.05rem; color:#536675; max-width:760px; margin-bottom:1.5rem;}
      .info-strip {background:#eef6fb; border:1px solid #cfe2ee; border-radius:10px; padding:.8rem 1rem; color:#294b61; font-size:.9rem; margin-bottom:1.3rem;}
      .metric-card {background:white; border:1px solid #dce4ea; border-radius:12px; padding:1rem 1.1rem; min-height:92px;}
      .metric-label {color:#6b7b87; text-transform:uppercase; letter-spacing:.08em; font-size:.7rem; font-weight:700;}
      .metric-value {color:#132f43; font-size:1.35rem; font-weight:800; margin-top:.25rem;}
      .metric-sub {color:#73818b; font-size:.76rem; margin-top:.1rem;}
      div[data-testid="stDownloadButton"] button {width:100%;}
    </style>
    """,
    unsafe_allow_html=True,
)


class ArchiveDownloadError(RuntimeError):
    pass


@st.cache_data(show_spinner=False, ttl=3600, max_entries=100)
def download_daily_raster(url: str) -> bytes:
    """Download one NOAA raster with short retries and Streamlit caching."""

    headers = {"User-Agent": "LIX-Rainfall-Mapper/1.0 (NOAA data visualization)"}
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = requests.get(url, headers=headers, timeout=(15, 90))
            if response.status_code == 404:
                raise ArchiveDownloadError(f"NOAA has no daily file at {url}")
            response.raise_for_status()
            if len(response.content) < 1_000:
                raise ArchiveDownloadError(f"NOAA returned an unexpectedly small file for {url}")
            return response.content
        except (requests.RequestException, ArchiveDownloadError) as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(1.25 * (attempt + 1))
    raise ArchiveDownloadError(str(last_error))


@st.cache_data(show_spinner=False, ttl=604800, max_entries=1)
def get_boundaries() -> dict:
    return fetch_la_ms_boundaries()


def download_period(products, progress_bar, status_box) -> list[bytes]:
    results: dict[date, bytes] = {}
    failures: list[str] = []
    workers = min(8, len(products))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(download_daily_raster, product.url): product for product in products}
        for completed, future in enumerate(as_completed(futures), start=1):
            product = futures[future]
            try:
                results[product.valid_date] = future.result()
            except Exception as exc:
                failures.append(f"{product.valid_date:%Y-%m-%d}: {exc}")
            progress_bar.progress(completed / len(products))
            status_box.caption(f"Downloaded {completed} of {len(products)} daily grids…")

    if failures:
        preview = "\n".join(failures[:4])
        raise ArchiveDownloadError(f"One or more NOAA files could not be retrieved:\n{preview}")
    return [results[product.valid_date] for product in products]


def metric_card(label: str, value: str, subtext: str = "") -> str:
    return (
        '<div class="metric-card">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'<div class="metric-sub">{subtext}</div>'
        "</div>"
    )


st.markdown('<div class="eyebrow">NOAA multi-sensor precipitation</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">LIX Rainfall Mapper</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-copy">Build a clean, downloadable rainfall accumulation map for any supported date range across Louisiana and Mississippi.</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="info-strip"><b>Timing note:</b> Each archived daily grid is valid from 12Z to 12Z. Both selected dates are included in the total.</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Map settings")
    with st.form("map_settings"):
        start_date = st.date_input(
            "Start date",
            value=date(2016, 8, 1),
            min_value=date(2005, 1, 1),
            max_value=date.today(),
        )
        end_date = st.date_input(
            "End date",
            value=date(2016, 8, 20),
            min_value=date(2005, 1, 1),
            max_value=date.today(),
        )
        region_name = st.selectbox("Map area", list(REGIONS), index=0)
        custom_title = st.text_input("Map title", value="Observed Rainfall")
        st.markdown("##### Map layers")
        show_counties = st.checkbox("Parish and county boundaries", value=True)
        show_cities = st.checkbox("City labels", value=True)
        submitted = st.form_submit_button("Generate rainfall map", type="primary", use_container_width=True)

    st.caption("Supported archive: January 1, 2005 through yesterday. Maximum period: 90 days.")

if not submitted and "result" not in st.session_state:
    st.subheader("Ready when you are")
    st.write(
        "The default dates are already set to August 1–20, 2016. Adjust anything in the sidebar, then generate the map."
    )

if submitted:
    try:
        validate_date_range(start_date, end_date)
        days = date_sequence(start_date, end_date)
        products = [source_for_date(day) for day in days]
        progress_bar = st.progress(0)
        status_box = st.empty()
        status_box.caption("Connecting to the NOAA archive…")
        raster_payloads = download_period(products, progress_bar, status_box)

        grid = build_target_grid(region_name)

        def process_progress(done: int, total: int) -> None:
            progress_bar.progress(done / total)
            status_box.caption(f"Processing daily grid {done} of {total}…")

        accumulation = accumulate_rasters(raster_payloads, grid, process_progress)
        status_box.caption("Loading parish and county boundaries…")
        boundaries = get_boundaries()
        png = render_map(
            accumulation,
            grid,
            boundaries,
            start_date,
            end_date,
            custom_title=custom_title,
            show_counties=show_counties,
            show_cities=show_cities,
        )
        geotiff = make_geotiff(accumulation, grid)
        maximum, latitude, longitude = maximum_location(accumulation, grid)
        source_names = sorted({product.source.value for product in products})
        st.session_state.result = {
            "png": png,
            "geotiff": geotiff,
            "maximum": maximum,
            "latitude": latitude,
            "longitude": longitude,
            "days": len(days),
            "sources": " + ".join(source_names),
            "start": start_date,
            "end": end_date,
            "region": region_name,
        }
        progress_bar.empty()
        status_box.empty()
    except (DateRangeError, ArchiveDownloadError, ValueError) as exc:
        st.error(str(exc))
    except Exception as exc:
        st.exception(exc)

if "result" in st.session_state:
    result = st.session_state.result
    columns = st.columns(4)
    cards = [
        ("Period", f"{result['days']} days", "Inclusive date range"),
        ("Maximum", f"{result['maximum']:.2f} in", "Within displayed area"),
        ("Max location", f"{result['latitude']:.2f}°N", f"{abs(result['longitude']):.2f}°W"),
        ("Data", result["sources"], "NOAA/NWS RFC QPE"),
    ]
    for column, card in zip(columns, cards):
        with column:
            st.markdown(metric_card(*card), unsafe_allow_html=True)

    st.image(result["png"], use_container_width=True)
    filename_base = f"rainfall_{result['start']:%Y%m%d}_{result['end']:%Y%m%d}"
    download_columns = st.columns(2)
    with download_columns[0]:
        st.download_button(
            "Download map (PNG)",
            data=result["png"],
            file_name=f"{filename_base}.png",
            mime="image/png",
            type="primary",
            use_container_width=True,
        )
    with download_columns[1]:
        st.download_button(
            "Download rainfall grid (GeoTIFF)",
            data=result["geotiff"],
            file_name=f"{filename_base}.tif",
            mime="image/tiff",
            use_container_width=True,
        )

    st.caption(
        "These are quality-controlled multi-sensor precipitation estimates, not point rain-gauge observations. "
        "Small local differences and radar/gauge-analysis artifacts remain possible."
    )
