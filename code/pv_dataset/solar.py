from __future__ import annotations

import numpy as np
import pandas as pd


def solar_elevation_degrees(
    timestamps: pd.Series,
    latitude: float,
    longitude: float,
    utc_offset_hours: float,
) -> np.ndarray:
    """Approximate NOAA solar elevation for local standard timestamps.

    The result is suitable for daylight filtering. Publication analyses should
    record the coordinates and compare a small sample with a validated solar
    position implementation.
    """
    ts = pd.to_datetime(timestamps)
    day = ts.dt.dayofyear.to_numpy(dtype=float)
    hour = (
        ts.dt.hour.to_numpy(dtype=float)
        + ts.dt.minute.to_numpy(dtype=float) / 60.0
        + ts.dt.second.to_numpy(dtype=float) / 3600.0
    )
    gamma = 2.0 * np.pi / 365.0 * (day - 1.0 + (hour - 12.0) / 24.0)
    eqtime = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2 * gamma)
        - 0.040849 * np.sin(2 * gamma)
    )
    decl = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2 * gamma)
        + 0.000907 * np.sin(2 * gamma)
        - 0.002697 * np.cos(3 * gamma)
        + 0.00148 * np.sin(3 * gamma)
    )
    standard_meridian = 15.0 * float(utc_offset_hours)
    true_solar_minutes = hour * 60.0 + eqtime + 4.0 * (float(longitude) - standard_meridian)
    hour_angle = np.deg2rad(true_solar_minutes / 4.0 - 180.0)
    lat = np.deg2rad(float(latitude))
    cos_zenith = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(hour_angle)
    zenith = np.arccos(np.clip(cos_zenith, -1.0, 1.0))
    return 90.0 - np.rad2deg(zenith)
