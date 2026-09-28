from __future__ import annotations

import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .solar import solar_elevation_degrees


TIMESTAMP_PATTERN = re.compile(r"(?<!\d)(20\d{15})(?!\d)")


def parse_image_timestamp(name: str) -> pd.Timestamp | pd.NaT:
    match = TIMESTAMP_PATTERN.search(Path(name).name)
    if not match:
        return pd.NaT
    text = match.group(1)
    base = pd.to_datetime(text[:14], format="%Y%m%d%H%M%S", errors="coerce")
    if pd.isna(base):
        return pd.NaT
    return base + pd.to_timedelta(int(text[14:17]), unit="ms")


def scan_images(path: str | Path) -> pd.DataFrame:
    root = Path(path)
    rows: list[dict] = []
    extensions = {".jpg", ".jpeg", ".png"}
    if root.is_file() and root.suffix.lower() == ".zip":
        archives = [root]
        image_files: list[Path] = []
    elif root.is_dir():
        archives = sorted(root.rglob("*.zip"))
        image_files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in extensions)
    else:
        raise FileNotFoundError(root)

    for image in image_files:
        rows.append({
            "image_storage": "file",
            "image_container": str(image.parent.resolve()),
            "image_member": image.name,
            "image_original_name": image.name,
            "image_size_bytes": image.stat().st_size,
        })
    for archive in archives:
        with zipfile.ZipFile(archive) as handle:
            for info in handle.infolist():
                if info.is_dir() or Path(info.filename).suffix.lower() not in extensions:
                    continue
                rows.append({
                    "image_storage": "zip",
                    "image_container": str(archive.resolve()),
                    "image_member": info.filename,
                    "image_original_name": Path(info.filename).name,
                    "image_size_bytes": int(info.file_size),
                })
    images = pd.DataFrame(rows)
    if images.empty:
        raise FileNotFoundError(f"No images or ZIP image members found in {root}")
    images["image_timestamp_raw"] = images["image_original_name"].map(parse_image_timestamp)
    images["image_timestamp"] = images["image_timestamp_raw"]
    images["image_timestamp_parse_ok"] = images["image_timestamp_raw"].notna()
    return images.sort_values(["image_timestamp_raw", "image_container", "image_member"], na_position="last").reset_index(drop=True)


def apply_daily_image_clock_correction(images: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Correct stable within-day clock segments without discarding raw timestamps."""
    result = images.copy()
    result["image_clock_offset_seconds"] = np.nan
    result["image_clock_residual_mad_seconds"] = np.nan
    result["image_clock_segment"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    result["image_clock_correction_applied"] = False
    pairing = config["pairing"]
    if not pairing.get("correct_image_clock_by_daily_median", False):
        return result
    valid = result["image_timestamp_raw"].notna()
    raw = result.loc[valid, "image_timestamp_raw"]
    seconds = (
        raw.dt.hour * 3600.0
        + raw.dt.minute * 60.0
        + raw.dt.second
        + raw.dt.microsecond / 1_000_000.0
    )
    signed_residual = ((seconds + 150.0) % 300.0) - 150.0
    working = pd.DataFrame({
        "date": raw.dt.date,
        "residual": signed_residual,
    }, index=raw.index)
    residual_delta = ((working["residual"].diff() + 150.0) % 300.0) - 150.0
    new_segment = working["date"].ne(working["date"].shift()) | residual_delta.abs().gt(
        float(pairing.get("clock_segment_jump_seconds", 30.0))
    )
    working["segment"] = new_segment.cumsum().astype(int)
    offsets = pd.Series(np.nan, index=result.index, dtype=float)
    mads = pd.Series(np.nan, index=result.index, dtype=float)
    segments = pd.Series(pd.NA, index=result.index, dtype="Int64")
    minimum_count = int(pairing["clock_min_images_per_day"])
    maximum_offset = float(pairing["clock_max_daily_offset_seconds"])
    maximum_mad = float(pairing["clock_max_daily_mad_seconds"])
    for segment_id, group in working.groupby("segment", sort=False):
        values = group["residual"].to_numpy(dtype=float)
        angle = values * (2.0 * np.pi / 300.0)
        offset = float(np.angle(np.mean(np.exp(1j * angle))) * 300.0 / (2.0 * np.pi))
        deviations = ((values - offset + 150.0) % 300.0) - 150.0
        mad = float(np.median(np.abs(deviations)))
        segments.loc[group.index] = int(segment_id)
        if len(group) >= minimum_count and abs(offset) <= maximum_offset and mad <= maximum_mad:
            offsets.loc[group.index] = offset
            mads.loc[group.index] = mad
    result["image_clock_offset_seconds"] = offsets
    result["image_clock_residual_mad_seconds"] = mads
    result["image_clock_segment"] = segments
    result["image_clock_correction_applied"] = offsets.notna()
    raw_nanoseconds = result["image_timestamp_raw"].astype("datetime64[ns]")
    correction = pd.to_timedelta(offsets.fillna(0.0).to_numpy(dtype=float), unit="s")
    result["image_timestamp"] = raw_nanoseconds - correction
    return result


def _nearest_join(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_time: str,
    right_time: str,
    tolerance_seconds: int,
) -> pd.DataFrame:
    left_sorted = left.sort_values(left_time).copy()
    right_sorted = right.dropna(subset=[right_time]).sort_values(right_time).copy()
    # Pandas 3 can preserve microsecond units from Excel while parsed image
    # timestamps use nanoseconds. merge_asof requires identical units.
    left_sorted[left_time] = pd.to_datetime(left_sorted[left_time]).astype("datetime64[ns]")
    right_sorted[right_time] = pd.to_datetime(right_sorted[right_time]).astype("datetime64[ns]")
    result = pd.merge_asof(
        left_sorted,
        right_sorted,
        left_on=left_time,
        right_on=right_time,
        direction="nearest",
        tolerance=pd.Timedelta(seconds=int(tolerance_seconds)),
    )
    return result.sort_index()


def load_irradiance(path: str | Path) -> pd.DataFrame:
    # The supplied workbook has two title rows and then five positional columns.
    raw = pd.read_excel(path, sheet_name=0, header=None, skiprows=3, usecols="A:E")
    raw.columns = ["irradiance_device_id", "irradiance_node", "irradiance_measurement", "irradiance_value", "irradiance_time"]
    raw["irradiance_time"] = pd.to_datetime(raw["irradiance_time"], errors="coerce")
    raw["irradiance_value"] = pd.to_numeric(raw["irradiance_value"], errors="coerce")
    return raw.dropna(subset=["irradiance_time"]).sort_values("irradiance_time")


def load_original_hourly_weather(path: str | Path) -> pd.DataFrame:
    workbook = pd.ExcelFile(path)
    if len(workbook.sheet_names) < 2:
        raise ValueError("Weather workbook must contain the original hourly-data sheet")
    raw = pd.read_excel(path, sheet_name=1, header=0, usecols="A:E")
    raw.columns = ["weather_time", "temperature_c", "relative_humidity_pct", "precipitation_mm", "wind_direction_deg"]
    raw["weather_time"] = pd.to_datetime(raw["weather_time"], errors="coerce")
    for column in raw.columns[1:]:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    return raw.dropna(subset=["weather_time"]).sort_values("weather_time")


def build_manifest(
    image_path: str | Path,
    power_qc_csv: str | Path,
    config: dict,
    irradiance_workbook: str | Path | None = None,
    weather_workbook: str | Path | None = None,
) -> pd.DataFrame:
    images = apply_daily_image_clock_correction(scan_images(image_path), config)
    valid_images = images.loc[images["image_timestamp_parse_ok"]].copy()
    invalid_images = images.loc[~images["image_timestamp_parse_ok"]].copy()

    power = pd.read_csv(power_qc_csv)
    power["power_timestamp"] = pd.to_datetime(power.pop("timestamp"), errors="coerce")
    keep_power = [
        "power_timestamp", "power_raw_kw", "power_qc_kw", "power_pu_dc",
        "qc_hard_valid", "qc_hard_flags", "qc_soft_flags", "source_file", "source_row",
    ]
    power = power[[column for column in keep_power if column in power.columns]]
    manifest = _nearest_join(
        valid_images,
        power,
        "image_timestamp",
        "power_timestamp",
        int(config["pairing"]["image_power_tolerance_seconds"]),
    )
    manifest["image_power_delta_seconds"] = (
        manifest["image_timestamp"] - manifest["power_timestamp"]
    ).dt.total_seconds()

    if irradiance_workbook:
        irradiance = load_irradiance(irradiance_workbook)
        manifest = _nearest_join(
            manifest,
            irradiance,
            "image_timestamp",
            "irradiance_time",
            int(config["pairing"]["irradiance_power_tolerance_seconds"]),
        )
        manifest["image_irradiance_delta_seconds"] = (
            manifest["image_timestamp"] - manifest["irradiance_time"]
        ).dt.total_seconds()

    if weather_workbook:
        weather = load_original_hourly_weather(weather_workbook)
        manifest = _nearest_join(
            manifest,
            weather,
            "image_timestamp",
            "weather_time",
            int(config["pairing"]["weather_tolerance_seconds"]),
        )
        manifest["image_weather_delta_seconds"] = (
            manifest["image_timestamp"] - manifest["weather_time"]
        ).dt.total_seconds()

    site = config["site"]
    if site.get("latitude") is not None and site.get("longitude") is not None:
        manifest["solar_elevation_deg"] = solar_elevation_degrees(
            manifest["image_timestamp"],
            float(site["latitude"]),
            float(site["longitude"]),
            float(site["utc_offset_hours"]),
        )
    else:
        manifest["solar_elevation_deg"] = np.nan

    manifest["pairing_flags"] = ""
    _mark(manifest, manifest["power_timestamp"].isna(), "power_unmatched")
    if "irradiance_time" in manifest:
        _mark(manifest, manifest["irradiance_time"].isna(), "irradiance_unmatched")
    if "weather_time" in manifest:
        _mark(manifest, manifest["weather_time"].isna(), "weather_unmatched")
    reused = manifest["power_timestamp"].duplicated(keep=False) & manifest["power_timestamp"].notna()
    _mark(manifest, reused, "power_timestamp_reused")
    manifest["timezone"] = site["timezone"]

    if not invalid_images.empty:
        invalid_images["pairing_flags"] = "image_timestamp_parse_error"
        invalid_images["timezone"] = site["timezone"]
        manifest = pd.concat([manifest, invalid_images], ignore_index=True, sort=False)
    manifest.insert(0, "sample_id", [f"IMG_{i:07d}" for i in range(1, len(manifest) + 1)])
    return manifest.sort_values("image_timestamp", na_position="last").reset_index(drop=True)


def _mark(frame: pd.DataFrame, mask: pd.Series, flag: str) -> None:
    empty = frame["pairing_flags"].eq("") & mask
    nonempty = frame["pairing_flags"].ne("") & mask
    frame.loc[empty, "pairing_flags"] = flag
    frame.loc[nonempty, "pairing_flags"] += ";" + flag


def save_manifest(manifest: pd.DataFrame, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(path, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d %H:%M:%S.%f")
    return path
