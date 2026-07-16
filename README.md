# LIX Rainfall Mapper

A browser-based tool for creating custom rainfall-accumulation maps across Louisiana and Mississippi from NOAA/NWS River Forecast Center multi-sensor precipitation estimates.

## What it does

- Accepts an inclusive start and end date.
- Downloads daily NOAA precipitation grids concurrently.
- Accumulates and reprojects the grids onto a common Louisiana/Mississippi map.
- Adds state, parish/county, and readable city references.
- Optionally samples the rainfall grid at displayed city locations.
- Provides state, WFO LIX, metro, southwest Mississippi, and coastal map presets.
- Exports a publication-ready PNG and a georeferenced GeoTIFF in inches.

The default selection is August 1–20, 2016.

## Data sources and timing

The app automatically selects a directly readable daily NOAA GeoTIFF:

- **January 1, 2005–June 27, 2017:** NCEP Stage III daily accumulation.
- **June 28, 2017–present:** NCEP Stage IV daily CONUS accumulation.

These quality-controlled products combine radar and rain-gauge information from NWS River Forecast Centers. Archived daily periods are valid **12Z–12Z**, and both dates selected in the app are included.

State, parish/county, and WFO LIX overlays are derived from official National Weather Service GIS boundary shapefiles and bundled with the application.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Sign in at [share.streamlit.io](https://share.streamlit.io/) with GitHub.
2. Select **Create app** and choose this repository.
3. Set the main file path to `app.py`.
4. Deploy. No secrets or API keys are required.

## Current scope

- Total precipitation for periods up to 90 days.
- Statewide, combined, WFO LIX, metro, southwest Mississippi, and coastal map areas.
- The official WFO LIX forecast-area outline on the WFO LIX preset.
- Fixed rainfall color intervals for consistent event-to-event comparison.

Planned additions include rolling maximum 24/48/72-hour totals, gauge overlays, parish/county summaries, and Atlas 14 exceedance comparisons.
