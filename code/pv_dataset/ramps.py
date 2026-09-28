from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .solar import solar_elevation_degrees


def _prepare_power(power_qc: pd.DataFrame, config: dict) -> pd.DataFrame:
    df = power_qc.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    if df["timestamp"].duplicated().any():
        raise ValueError("Duplicate power timestamps remain; resolve them before ramp detection")
    if "qc_hard_valid" not in df:
        df["qc_hard_valid"] = df["power_qc_kw"].notna()
    else:
        df["qc_hard_valid"] = df["qc_hard_valid"].astype(str).str.lower().map(
            {"true": True, "false": False}
        ).fillna(df["qc_hard_valid"].astype(bool))

    requested_power = config.get("ramp", {}).get(
        "primary_power_column", "power_operational_qc_kw"
    )
    power_column = requested_power if requested_power in df else "power_qc_kw"
    if power_column not in df:
        raise ValueError(f"Ramp power column is unavailable: {power_column}")
    if power_column == "power_operational_qc_kw" and "power_training_eligible_operational" in df:
        valid_column = "power_training_eligible_operational"
    else:
        valid_column = "qc_hard_valid"
    df["_ramp_power_kw"] = pd.to_numeric(df[power_column], errors="coerce")
    df["_ramp_valid"] = (
        df[valid_column].astype(str).str.lower().map({"true": True, "false": False})
        .fillna(df[valid_column].astype(bool))
        & df["_ramp_power_kw"].notna()
    )
    df["_ramp_ghi_wm2"] = pd.to_numeric(
        df["irradiance_value"] if "irradiance_value" in df else np.nan,
        errors="coerce",
    )
    df.attrs["ramp_power_column"] = power_column
    df.attrs["ramp_valid_column"] = valid_column

    site = config["site"]
    latitude, longitude = site.get("latitude"), site.get("longitude")
    if latitude is not None and longitude is not None:
        df["solar_elevation_deg"] = solar_elevation_degrees(
            df["timestamp"], float(latitude), float(longitude), float(site["utc_offset_hours"])
        )
    else:
        df["solar_elevation_deg"] = np.nan
    return df


def calculate_changes(power_qc: pd.DataFrame, horizon_minutes: int, config: dict) -> pd.DataFrame:
    cadence = int(config["power"]["expected_interval_minutes"])
    steps = int(horizon_minutes // cadence)
    if steps < 1 or steps * cadence != horizon_minutes:
        raise ValueError("Ramp horizon must be a positive multiple of the power cadence")

    df = _prepare_power(power_qc, config)
    result = pd.DataFrame({
        "start_time": df["timestamp"],
        "end_time": df["timestamp"].shift(-steps),
        "start_power_kw": df["_ramp_power_kw"],
        "end_power_kw": df["_ramp_power_kw"].shift(-steps),
        "solar_elevation_deg": df["solar_elevation_deg"],
        "end_solar_elevation_deg": df["solar_elevation_deg"].shift(-steps),
        "start_ghi_wm2": df["_ramp_ghi_wm2"],
        "end_ghi_wm2": df["_ramp_ghi_wm2"].shift(-steps),
    })
    result["power_column"] = df.attrs["ramp_power_column"]
    result["horizon_minutes"] = int(horizon_minutes)
    result["delta_kw"] = result["end_power_kw"] - result["start_power_kw"]
    result["delta_ghi_wm2"] = result["end_ghi_wm2"] - result["start_ghi_wm2"]
    capacity = float(config["site"].get("installed_ac_kw") or config["site"]["installed_dc_kwp"])
    result["delta_pu_ac"] = result["delta_kw"] / capacity
    # Backward-compatible alias retained for older downstream readers.
    result["delta_pu_dc"] = result["delta_pu_ac"]
    result["rate_kw_per_min"] = result["delta_kw"] / float(horizon_minutes)
    result["rate_pu_per_min"] = result["delta_pu_ac"] / float(horizon_minutes)
    result["direction"] = np.select(
        [result["delta_kw"].gt(0), result["delta_kw"].lt(0)],
        ["ramp_up", "ramp_down"],
        default="stable",
    )

    expected_end = result["start_time"] + pd.to_timedelta(horizon_minutes, unit="m")
    exact_horizon = result["end_time"].eq(expected_end)
    valid_points = df["_ramp_valid"].fillna(False).astype(bool)
    valid_window = pd.concat(
        [valid_points.shift(-offset) for offset in range(steps + 1)], axis=1
    ).fillna(False).all(axis=1)
    min_elevation = float(config["ramp"]["minimum_solar_elevation_degrees"])
    elevation_window = pd.concat(
        [df["solar_elevation_deg"].shift(-offset) for offset in range(steps + 1)], axis=1
    )
    if elevation_window.notna().any().any():
        daylight = elevation_window.notna().all(axis=1) & elevation_window.min(axis=1).ge(min_elevation)
    else:
        daylight = pd.Series(True, index=df.index)
    result["eligible"] = exact_horizon & valid_window & result["delta_kw"].notna() & daylight
    result["exclusion_reason"] = ""
    result.loc[~exact_horizon, "exclusion_reason"] = "non_continuous_time"
    result.loc[exact_horizon & ~valid_window, "exclusion_reason"] = "invalid_power_in_window"
    result.loc[exact_horizon & valid_window & ~daylight, "exclusion_reason"] = "solar_elevation_below_limit"
    return result


def _threshold_definitions(changes: pd.DataFrame, config: dict, training_end: str | None) -> list[dict]:
    capacity = float(config["site"].get("installed_ac_kw") or config["site"]["installed_dc_kwp"])
    definitions = [
        {
            "scheme": "capacity_fraction",
            "label": f"capacity_{fraction:.3f}",
            "threshold_kw": capacity * float(fraction),
            "threshold_pu": float(fraction),
            "training_end": None,
        }
        for fraction in config["ramp"]["capacity_fraction_thresholds"]
    ]
    end_value = training_end or config["ramp"].get("statistical_training_end")
    if end_value:
        cutoff = pd.Timestamp(end_value)
        training = changes.loc[
            changes["eligible"] & changes["start_time"].le(cutoff), "delta_kw"
        ].abs()
        if training.empty:
            raise ValueError("No eligible ramp changes exist before statistical_training_end")
        for quantile in config["ramp"]["statistical_quantiles"]:
            threshold = float(training.quantile(float(quantile)))
            definitions.append({
                "scheme": "training_quantile",
                "label": f"training_q{float(quantile):.3f}",
                "threshold_kw": threshold,
                "threshold_pu": threshold / capacity,
                "training_end": str(cutoff),
            })
    return definitions


def label_candidates(changes: pd.DataFrame, definition: dict) -> pd.DataFrame:
    mask = changes["eligible"] & changes["delta_kw"].abs().ge(float(definition["threshold_kw"]))
    candidates = changes.loc[mask].copy()
    candidates["threshold_scheme"] = definition["scheme"]
    candidates["threshold_label"] = definition["label"]
    candidates["threshold_kw"] = float(definition["threshold_kw"])
    candidates["threshold_pu"] = float(definition["threshold_pu"])
    candidates["threshold_training_end"] = definition["training_end"]
    return candidates


def merge_candidates_into_events(candidates: pd.DataFrame, merge_gap_minutes: int) -> pd.DataFrame:
    columns = [
        "event_id", "threshold_scheme", "threshold_label", "threshold_kw", "threshold_pu",
        "threshold_training_end",
        "horizon_minutes", "direction", "start_time", "end_time", "duration_minutes",
        "start_power_kw", "end_power_kw", "net_change_kw", "max_abs_change_kw",
        "max_abs_rate_kw_per_min", "candidate_count",
    ]
    if candidates.empty:
        return pd.DataFrame(columns=columns)
    candidates = candidates.sort_values("start_time").reset_index(drop=True)
    gap = pd.Timedelta(minutes=int(merge_gap_minutes))
    events: list[dict] = []
    current: list[pd.Series] = []

    def flush(rows: list[pd.Series]) -> None:
        if not rows:
            return
        first, last = rows[0], rows[-1]
        start = first["start_time"]
        end = max(row["end_time"] for row in rows)
        events.append({
            "threshold_scheme": first["threshold_scheme"],
            "threshold_label": first["threshold_label"],
            "threshold_kw": first["threshold_kw"],
            "threshold_pu": first["threshold_pu"],
            "threshold_training_end": first.get("threshold_training_end"),
            "horizon_minutes": int(first["horizon_minutes"]),
            "direction": first["direction"],
            "start_time": start,
            "end_time": end,
            "duration_minutes": (end - start).total_seconds() / 60.0,
            "start_power_kw": first["start_power_kw"],
            "end_power_kw": last["end_power_kw"],
            "net_change_kw": last["end_power_kw"] - first["start_power_kw"],
            "max_abs_change_kw": max(abs(row["delta_kw"]) for row in rows),
            "max_abs_rate_kw_per_min": max(abs(row["rate_kw_per_min"]) for row in rows),
            "candidate_count": len(rows),
        })

    for _, row in candidates.iterrows():
        if not current:
            current = [row]
            continue
        current_end = max(item["end_time"] for item in current)
        if row["direction"] == current[-1]["direction"] and row["start_time"] <= current_end + gap:
            current.append(row)
        else:
            flush(current)
            current = [row]
    flush(current)

    result = pd.DataFrame(events)
    result.insert(0, "event_id", [f"RAMP_{i:06d}" for i in range(1, len(result) + 1)])
    return result[columns]


def run_ramp_detection(
    power_qc_csv: str | Path,
    output_directory: str | Path,
    config: dict,
    training_end: str | None = None,
) -> tuple[Path, Path, Path]:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    power = pd.read_csv(power_qc_csv)
    all_candidates: list[pd.DataFrame] = []
    all_events: list[pd.DataFrame] = []
    summary: list[dict] = []
    for horizon in config["ramp"]["horizons_minutes"]:
        changes = calculate_changes(power, int(horizon), config)
        for definition in _threshold_definitions(changes, config, training_end):
            candidates = label_candidates(changes, definition)
            events = merge_candidates_into_events(candidates, int(config["ramp"]["merge_gap_minutes"]))
            all_candidates.append(candidates)
            all_events.append(events)
            summary.append({
                "horizon_minutes": int(horizon),
                **definition,
                "eligible_change_count": int(changes["eligible"].sum()),
                "candidate_count": int(len(candidates)),
                "event_count": int(len(events)),
                "ramp_up_event_count": int(events["direction"].eq("ramp_up").sum()) if not events.empty else 0,
                "ramp_down_event_count": int(events["direction"].eq("ramp_down").sum()) if not events.empty else 0,
            })
    candidate_path = output / "ramp_candidates.csv"
    event_path = output / "ramp_events.csv"
    summary_path = output / "ramp_summary.json"
    candidate_table = pd.concat(all_candidates, ignore_index=True)
    event_table = pd.concat(all_events, ignore_index=True)
    if not event_table.empty:
        event_table["event_id"] = [f"RAMP_{i:07d}" for i in range(1, len(event_table) + 1)]
    candidate_table.to_csv(candidate_path, index=False, encoding="utf-8-sig")
    event_table.to_csv(event_path, index=False, encoding="utf-8-sig")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return candidate_path, event_path, summary_path
