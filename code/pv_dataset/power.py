from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .solar import solar_elevation_degrees


def find_monthly_power_files(directory: str | Path) -> list[Path]:
    root = Path(directory)
    files = sorted(root.glob("20??_??.xlsx"))
    if not files:
        raise FileNotFoundError(f"No monthly power workbooks found in {root}")
    return files


def load_raw_power(files: Iterable[str | Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for file in files:
        path = Path(file)
        frame = pd.read_excel(path)
        normalized = {str(c).strip().lower(): c for c in frame.columns}
        if "time" not in normalized or "power" not in normalized:
            raise ValueError(f"{path} must contain Time and Power columns")
        frame = frame.rename(
            columns={normalized["time"]: "timestamp", normalized["power"]: "power_raw_kw"}
        )[["timestamp", "power_raw_kw"]]
        frame["source_file"] = path.name
        frame["source_row"] = np.arange(2, len(frame) + 2)
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    result["timestamp"] = pd.to_datetime(result["timestamp"], errors="coerce")
    result["power_raw_kw"] = pd.to_numeric(result["power_raw_kw"], errors="coerce")
    return result.sort_values(["timestamp", "source_file", "source_row"], na_position="last").reset_index(drop=True)


def load_irradiance(path: str | Path) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=0, header=None, skiprows=3, usecols="A:E")
    raw.columns = [
        "irradiance_device_id", "irradiance_node", "irradiance_measurement",
        "irradiance_value", "irradiance_time",
    ]
    raw["irradiance_time"] = pd.to_datetime(raw["irradiance_time"], errors="coerce")
    raw["irradiance_value"] = pd.to_numeric(raw["irradiance_value"], errors="coerce")
    return raw.dropna(subset=["irradiance_time"]).sort_values("irradiance_time").reset_index(drop=True)


def attach_nearest_irradiance(
    power: pd.DataFrame, irradiance: pd.DataFrame, tolerance_seconds: int,
) -> pd.DataFrame:
    result = power.copy()
    result["_original_order"] = np.arange(len(result))
    valid = result["timestamp"].notna()
    left = result.loc[valid].sort_values("timestamp")
    right = irradiance.dropna(subset=["irradiance_time"]).sort_values("irradiance_time").copy()
    left["timestamp"] = pd.to_datetime(left["timestamp"]).astype("datetime64[ns]")
    right["irradiance_time"] = pd.to_datetime(right["irradiance_time"]).astype("datetime64[ns]")
    matched = pd.merge_asof(
        left, right, left_on="timestamp", right_on="irradiance_time",
        direction="nearest", tolerance=pd.Timedelta(seconds=int(tolerance_seconds)),
    )
    invalid = result.loc[~valid].copy()
    for column in right.columns:
        if column not in invalid:
            invalid[column] = np.nan
    combined = pd.concat([matched, invalid], ignore_index=True, sort=False)
    # Concatenating an empty invalid-timestamp branch can coerce datetime columns
    # to object under pandas 3; restore explicit units before subtraction.
    combined["timestamp"] = pd.to_datetime(combined["timestamp"], errors="coerce").astype("datetime64[ns]")
    combined["irradiance_time"] = pd.to_datetime(
        combined["irradiance_time"], errors="coerce"
    ).astype("datetime64[ns]")
    combined["irradiance_delta_seconds"] = (
        combined["timestamp"] - combined["irradiance_time"]
    ).dt.total_seconds()
    return combined.sort_values("_original_order").drop(columns="_original_order").reset_index(drop=True)


def _append_flag(flags: pd.Series, mask: pd.Series | np.ndarray, name: str) -> pd.Series:
    mask_series = pd.Series(mask, index=flags.index).fillna(False).astype(bool)
    flags = flags.copy()
    flags.loc[mask_series & flags.eq("")] = name
    flags.loc[
        mask_series & flags.ne("") & ~flags.str.contains(fr"(?:^|;){name}(?:;|$)")
    ] += ";" + name
    return flags


def _continuous_run_mask(
    condition: pd.Series, timestamps: pd.Series, minimum_points: int, cadence_minutes: int,
) -> pd.Series:
    condition = condition.fillna(False).astype(bool)
    time_continuous = timestamps.diff().eq(pd.Timedelta(minutes=int(cadence_minutes)))
    new_group = condition.ne(condition.shift(fill_value=False)) | ~time_continuous
    group = new_group.cumsum()
    run_length = condition.groupby(group).transform("sum")
    return condition & run_length.ge(int(minimum_points))


def _expand_anchored_runs(
    base_condition: pd.Series,
    anchor: pd.Series,
    timestamps: pd.Series,
    cadence_minutes: int,
) -> pd.Series:
    """Expand high-confidence anchors to their complete continuous base run."""
    base_condition = base_condition.fillna(False).astype(bool)
    anchor = anchor.fillna(False).astype(bool)
    time_continuous = timestamps.diff().eq(pd.Timedelta(minutes=int(cadence_minutes)))
    group = (base_condition.ne(base_condition.shift(fill_value=False)) | ~time_continuous).cumsum()
    group_has_anchor = anchor.groupby(group).transform("any")
    return base_condition & group_has_anchor


def _stuck_mask(
    values: pd.Series,
    timestamps: pd.Series,
    minimum_points: int,
    minimum_power: float,
    cadence_minutes: int,
) -> pd.Series:
    time_continuous = timestamps.diff().eq(pd.Timedelta(minutes=int(cadence_minutes)))
    same_as_previous = values.eq(values.shift()) & values.notna() & time_continuous
    group = (~same_as_previous).cumsum()
    run_length = values.groupby(group).transform("size")
    return values.ge(minimum_power) & run_length.ge(minimum_points)


def quality_control_power(
    raw: pd.DataFrame, config: dict, irradiance: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict]:
    power_cfg = config["power"]
    site_cfg = config["site"]
    cadence = int(power_cfg["expected_interval_minutes"])
    capacity = float(site_cfg["installed_dc_kwp"])
    plausible_max = capacity * float(power_cfg["plausible_max_dc_factor"])
    jump_threshold = capacity * float(power_cfg["isolated_jump_dc_factor"])
    nameplate_soft_factor = float(power_cfg.get("nameplate_soft_factor", 1.0))
    daylight_elevation = float(power_cfg.get("daylight_solar_elevation_degrees", 10.0))
    night_elevation = float(power_cfg.get("night_solar_elevation_degrees", -6.0))
    zero_max = float(power_cfg.get("zero_power_max_kw", 0.05))
    night_positive_min = float(power_cfg.get("night_positive_min_kw", 0.1))
    irradiance_high = float(power_cfg.get("high_irradiance_threshold", 200.0))
    outage_min_points = int(power_cfg.get("probable_outage_min_points", 3))

    df = raw.copy()
    if irradiance is not None:
        tolerance = int(config.get("pairing", {}).get("irradiance_power_tolerance_seconds", 180))
        df = attach_nearest_irradiance(df, irradiance, tolerance)
    else:
        df["irradiance_time"] = pd.NaT
        df["irradiance_value"] = np.nan
        df["irradiance_delta_seconds"] = np.nan

    latitude, longitude = site_cfg.get("latitude"), site_cfg.get("longitude")
    if latitude is not None and longitude is not None:
        df["solar_elevation_deg"] = solar_elevation_degrees(
            df["timestamp"], float(latitude), float(longitude), float(site_cfg["utc_offset_hours"]),
        )
    else:
        df["solar_elevation_deg"] = np.nan

    hard = pd.Series("", index=df.index, dtype="object")
    soft = pd.Series("", index=df.index, dtype="object")
    auxiliary = pd.Series("", index=df.index, dtype="object")

    invalid_time = df["timestamp"].isna()
    missing_power = df["power_raw_kw"].isna()
    duplicate_time = df["timestamp"].duplicated(keep=False) & df["timestamp"].notna()
    negative_power = df["power_raw_kw"].lt(0)
    above_plausible = df["power_raw_kw"].gt(plausible_max)
    for mask, name in [
        (invalid_time, "invalid_timestamp"), (missing_power, "missing_power"),
        (duplicate_time, "duplicate_timestamp"), (negative_power, "negative_power"),
        (above_plausible, "above_plausible_range"),
    ]:
        hard = _append_flag(hard, mask, name)

    time_diff = df["timestamp"].diff().dt.total_seconds().div(60)
    cadence_break = time_diff.notna() & time_diff.ne(cadence)
    soft = _append_flag(soft, cadence_break, "cadence_break_before")
    stuck = _stuck_mask(
        df["power_raw_kw"], df["timestamp"], int(power_cfg["stuck_min_points"]),
        float(power_cfg["stuck_min_power_kw"]), cadence,
    )
    soft = _append_flag(soft, stuck, "possible_stuck_sensor")

    previous, following = df["power_raw_kw"].shift(1), df["power_raw_kw"].shift(-1)
    neighbors_continuous = (
        df["timestamp"].diff().eq(pd.Timedelta(minutes=cadence))
        & df["timestamp"].shift(-1).sub(df["timestamp"]).eq(pd.Timedelta(minutes=cadence))
    )
    isolated_spike = (
        df["power_raw_kw"].sub((previous + following) / 2.0).abs().gt(jump_threshold)
        & previous.sub(following).abs().lt(jump_threshold / 2.0)
        & previous.notna() & following.notna() & neighbors_continuous
    )
    soft = _append_flag(soft, isolated_spike, "possible_isolated_spike")

    above_nameplate = df["power_raw_kw"].gt(capacity * nameplate_soft_factor)
    daylight_zero = df["solar_elevation_deg"].gt(daylight_elevation) & df["power_raw_kw"].le(zero_max)
    night_positive = df["solar_elevation_deg"].lt(night_elevation) & df["power_raw_kw"].gt(night_positive_min)
    high_irradiance_zero = df["irradiance_value"].ge(irradiance_high) & df["power_raw_kw"].le(zero_max)
    outage_anchor = _continuous_run_mask(
        high_irradiance_zero, df["timestamp"], outage_min_points, cadence,
    )
    # If a sustained high-irradiance anchor proves a zero-power run is likely
    # nonoperational, extend the flag across the same uninterrupted daylight-zero
    # run. This avoids splitting one outage whenever irradiance briefly dips.
    probable_nonoperational = _expand_anchored_runs(
        daylight_zero, outage_anchor, df["timestamp"], cadence,
    )
    for mask, name in [
        (above_nameplate, "above_dc_nameplate"),
        (daylight_zero, "daylight_zero_power"),
        (night_positive, "positive_during_solar_night"),
        (high_irradiance_zero, "high_irradiance_zero_power"),
        (probable_nonoperational, "probable_nonoperational_period"),
    ]:
        soft = _append_flag(soft, mask, name)
    auxiliary = _append_flag(auxiliary, df["irradiance_time"].isna(), "irradiance_unmatched")

    df["qc_hard_flags"] = hard
    df["qc_soft_flags"] = soft
    df["auxiliary_flags"] = auxiliary
    df["qc_hard_valid"] = hard.eq("")
    df["power_qc_kw"] = df["power_raw_kw"].where(df["qc_hard_valid"])
    df["power_pu_dc"] = df["power_qc_kw"] / capacity
    df["time_diff_minutes"] = time_diff
    df["probable_nonoperational"] = probable_nonoperational
    df["high_confidence_outage_anchor"] = outage_anchor
    df["power_training_eligible_observed"] = df["qc_hard_valid"]
    df["power_training_eligible_operational"] = df["qc_hard_valid"] & ~probable_nonoperational
    df["power_training_eligible_strict"] = (
        df["power_training_eligible_operational"] & ~stuck
    )
    df["power_operational_qc_kw"] = df["power_qc_kw"].where(df["power_training_eligible_operational"])
    df["power_strict_qc_kw"] = df["power_qc_kw"].where(df["power_training_eligible_strict"])
    df["operational_status"] = "observed_valid"
    df.loc[~df["qc_hard_valid"], "operational_status"] = "hard_invalid"
    df.loc[soft.ne("") & ~probable_nonoperational & df["qc_hard_valid"], "operational_status"] = "review"
    df.loc[probable_nonoperational & df["qc_hard_valid"], "operational_status"] = "probable_nonoperational"

    daylight = df["solar_elevation_deg"].gt(daylight_elevation)
    paired = df["irradiance_time"].notna()
    correlation_frame = df.loc[daylight & paired, ["power_raw_kw", "irradiance_value"]].dropna()
    report = {
        "schema_version": "2.0", "power_unit": "kW", "installed_dc_kwp": capacity,
        "installed_ac_kw": site_cfg.get("installed_ac_kw"),
        "power_measurement_semantics": site_cfg.get("power_measurement"),
        "plausible_max_kw": plausible_max, "row_count": int(len(df)),
        "time_start": None if df["timestamp"].dropna().empty else str(df["timestamp"].min()),
        "time_end": None if df["timestamp"].dropna().empty else str(df["timestamp"].max()),
        "hard_invalid_count": int((~df["qc_hard_valid"]).sum()),
        "soft_flag_row_count": int(df["qc_soft_flags"].ne("").sum()),
        "hard_flag_counts": _flag_counts(df["qc_hard_flags"]),
        "soft_flag_counts": _flag_counts(df["qc_soft_flags"]),
        "auxiliary_flag_counts": _flag_counts(df["auxiliary_flags"]),
        "observed_min_kw": float(df["power_raw_kw"].min()),
        "observed_max_kw": float(df["power_raw_kw"].max()),
        "irradiance_matched_count": int(paired.sum()),
        "irradiance_unmatched_count": int((~paired).sum()),
        "daylight_power_irradiance_pearson_r": (
            None if len(correlation_frame) < 2 else float(correlation_frame.corr().iloc[0, 1])
        ),
        "probable_nonoperational_row_count": int(probable_nonoperational.sum()),
        "high_confidence_outage_anchor_count": int(outage_anchor.sum()),
        "observed_training_eligible_count": int(df["power_training_eligible_observed"].sum()),
        "operational_training_eligible_count": int(df["power_training_eligible_operational"].sum()),
        "strict_training_eligible_count": int(df["power_training_eligible_strict"].sum()),
        "thresholds": {
            "daylight_solar_elevation_degrees": daylight_elevation,
            "night_solar_elevation_degrees": night_elevation,
            "zero_power_max_kw": zero_max,
            "night_positive_min_kw": night_positive_min,
            "high_irradiance_threshold_reported_unit": irradiance_high,
            "probable_outage_min_consecutive_points": outage_min_points,
            "expected_interval_minutes": cadence,
        },
        "notes": [
            "power_qc_kw is never interpolated and only removes hard structural/physical failures.",
            "power_operational_qc_kw additionally masks sustained zero power under high irradiance as a candidate operational subset.",
            "power_strict_qc_kw also masks possible stuck-sensor runs for a conservative sensitivity subset.",
            "Probable nonoperational periods remain soft/auditable because inverter status and the irradiance unit are not yet independently confirmed.",
            "Values above the 15 kWp DC nameplate are soft-flagged, not removed, until AC rating and meter semantics are confirmed.",
        ],
    }
    return df, report


def _flag_counts(series: pd.Series) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in series[series.ne("")]:
        for flag in str(value).split(";"):
            counts[flag] = counts.get(flag, 0) + 1
    return counts


def summarize_flag_events(df: pd.DataFrame, cadence_minutes: int) -> pd.DataFrame:
    rows: list[dict] = []
    flags = [
        "probable_nonoperational_period", "high_irradiance_zero_power",
        "daylight_zero_power", "possible_stuck_sensor", "possible_isolated_spike",
        "positive_during_solar_night", "above_dc_nameplate",
    ]
    for flag in flags:
        mask = df["qc_soft_flags"].fillna("").str.contains(fr"(?:^|;){flag}(?:;|$)")
        continuity = df["timestamp"].diff().eq(pd.Timedelta(minutes=int(cadence_minutes)))
        group_id = (mask.ne(mask.shift(fill_value=False)) | ~continuity).cumsum()
        for _, group in df.loc[mask].groupby(group_id[mask]):
            rows.append({
                "flag": flag, "start_time": group["timestamp"].min(),
                "end_time": group["timestamp"].max(), "point_count": int(len(group)),
                "inclusive_duration_minutes": int(len(group) * cadence_minutes),
                "min_power_kw": float(group["power_raw_kw"].min()),
                "max_power_kw": float(group["power_raw_kw"].max()),
                "max_solar_elevation_deg": float(group["solar_elevation_deg"].max()),
                "max_irradiance_value": None if group["irradiance_value"].dropna().empty else float(group["irradiance_value"].max()),
                "median_irradiance_value": None if group["irradiance_value"].dropna().empty else float(group["irradiance_value"].median()),
            })
    columns = [
        "flag", "start_time", "end_time", "point_count", "inclusive_duration_minutes",
        "min_power_kw", "max_power_kw", "max_solar_elevation_deg",
        "max_irradiance_value", "median_irradiance_value",
    ]
    return pd.DataFrame(rows, columns=columns).sort_values(["start_time", "flag"]).reset_index(drop=True)


def build_monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    working["month"] = working["timestamp"].dt.to_period("M").astype(str)
    has = lambda x, name: int(x.fillna("").str.contains(name).sum())
    return working.groupby("month", as_index=False).agg(
        row_count=("timestamp", "size"),
        hard_invalid_count=("qc_hard_valid", lambda x: int((~x).sum())),
        probable_nonoperational_count=("probable_nonoperational", "sum"),
        operational_training_eligible_count=("power_training_eligible_operational", "sum"),
        strict_training_eligible_count=("power_training_eligible_strict", "sum"),
        irradiance_unmatched_count=("irradiance_time", lambda x: int(x.isna().sum())),
        mean_power_kw=("power_raw_kw", "mean"), max_power_kw=("power_raw_kw", "max"),
        zero_power_count=("power_raw_kw", lambda x: int(x.le(0.05).sum())),
        daylight_zero_count=("qc_soft_flags", lambda x: has(x, "daylight_zero_power")),
        high_irradiance_zero_count=("qc_soft_flags", lambda x: has(x, "high_irradiance_zero_power")),
        possible_stuck_count=("qc_soft_flags", lambda x: has(x, "possible_stuck_sensor")),
        possible_spike_count=("qc_soft_flags", lambda x: has(x, "possible_isolated_spike")),
        above_nameplate_count=("qc_soft_flags", lambda x: has(x, "above_dc_nameplate")),
    )


def write_power_qc_figures(df: pd.DataFrame, monthly: pd.DataFrame, output: Path) -> list[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    daily = df.assign(date=df["timestamp"].dt.floor("D")).groupby("date", as_index=False).agg(
        daily_max_power_kw=("power_raw_kw", "max"),
        probable_nonoperational_points=("probable_nonoperational", "sum"),
    )
    fig, ax1 = plt.subplots(figsize=(14, 5))
    line = ax1.plot(daily["date"], daily["daily_max_power_kw"], color="#2563eb", linewidth=0.8, label="Daily max power")[0]
    ax1.set(xlabel="Date", ylabel="Daily max power (kW)", title="Daily maximum power and probable nonoperational observations")
    ax2 = ax1.twinx()
    fill = ax2.fill_between(daily["date"], daily["probable_nonoperational_points"], color="#dc2626", alpha=0.25, label="Probable nonoperational points")
    ax2.set_ylabel("Flagged 5-minute points")
    ax1.grid(alpha=0.2)
    ax1.legend([line, fill], [line.get_label(), fill.get_label()], loc="lower right")
    fig.tight_layout()
    path = figures / "daily_power_qc.png"; fig.savefig(path, dpi=180); plt.close(fig); paths.append(path)

    paired = df[df["irradiance_value"].notna() & df["solar_elevation_deg"].gt(10)].copy()
    normal, outage = paired[~paired["probable_nonoperational"]], paired[paired["probable_nonoperational"]]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(normal["irradiance_value"], normal["power_raw_kw"], s=3, alpha=0.08, color="#2563eb", rasterized=True, label="Observed")
    ax.scatter(outage["irradiance_value"], outage["power_raw_kw"], s=8, alpha=0.5, color="#dc2626", rasterized=True, label="Probable nonoperational")
    ax.set(xlabel="Irradiance value (reported unit)", ylabel="Power (kW)", title="Daylight power–irradiance relationship")
    ax.legend(); ax.grid(alpha=0.2); fig.tight_layout()
    path = figures / "power_vs_irradiance_qc.png"; fig.savefig(path, dpi=180); plt.close(fig); paths.append(path)

    x = np.arange(len(monthly)); fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - 0.2, monthly["probable_nonoperational_count"], width=0.4, label="Probable nonoperational")
    ax.bar(x + 0.2, monthly["possible_spike_count"], width=0.4, label="Possible isolated spike")
    ax.set_xticks(x, labels=monthly["month"], rotation=60, ha="right", fontsize=8)
    ax.set(xlabel="Month", ylabel="Flagged 5-minute points", title="Monthly power quality flags")
    ax.legend(); ax.grid(axis="y", alpha=0.2); fig.tight_layout()
    path = figures / "monthly_power_qc_flags.png"; fig.savefig(path, dpi=180); plt.close(fig); paths.append(path)
    return paths


def run_power_qc(
    raw_directory: str | Path,
    output_directory: str | Path,
    config: dict,
    irradiance_workbook: str | Path | None = None,
) -> tuple[Path, ...]:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    raw = load_raw_power(find_monthly_power_files(raw_directory))
    irradiance_path = irradiance_workbook or config.get("paths", {}).get("irradiance_workbook")
    irradiance = load_irradiance(irradiance_path) if irradiance_path else None
    qc, report = quality_control_power(raw, config, irradiance)
    events = summarize_flag_events(qc, int(config["power"]["expected_interval_minutes"]))
    monthly = build_monthly_summary(qc)
    figure_paths = write_power_qc_figures(qc, monthly, output)
    csv_path, report_path = output / "power_qc.csv", output / "power_qc_report.json"
    event_path, monthly_path = output / "power_qc_events.csv", output / "monthly_power_qc.csv"
    qc.to_csv(csv_path, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d %H:%M:%S")
    events.to_csv(event_path, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d %H:%M:%S")
    monthly.to_csv(monthly_path, index=False, encoding="utf-8-sig")
    report["event_count_by_flag"] = events["flag"].value_counts().to_dict()
    report["figure_paths"] = [str(path) for path in figure_paths]
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return (csv_path, report_path, event_path, monthly_path, *figure_paths)
