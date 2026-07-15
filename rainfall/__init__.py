"""Core package for the LIX Rainfall Mapper."""

from .core import (
    ArchiveSource,
    DateRangeError,
    TargetGrid,
    accumulate_rasters,
    build_target_grid,
    date_sequence,
    make_geotiff,
    source_for_date,
    validate_date_range,
)

__all__ = [
    "ArchiveSource",
    "DateRangeError",
    "TargetGrid",
    "accumulate_rasters",
    "build_target_grid",
    "date_sequence",
    "make_geotiff",
    "source_for_date",
    "validate_date_range",
]
