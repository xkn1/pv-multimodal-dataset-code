from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _as_bool(series: pd.Series, default: bool = False) -> pd.Series:
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.fillna(default).astype(bool)
    values = series.astype("string").str.strip().str.lower()
    mapped = values.map({"true": True, "false": False, "1": True, "0": False})
    return mapped.fillna(default).astype(bool)


def _hash_day_split(value: object, site_id: str, seed: int) -> str:
    if pd.isna(value):
        return "unassigned"
    day = pd.Timestamp(value).strftime("%Y-%m-%d")
    digest = hashlib.sha256(f"{site_id}|{day}|{seed}".encode("utf-8")).digest()
    number = int.from_bytes(digest[:8], "big") / float(2**64)
    if number < 0.70:
        return "train"
    if number < 0.85:
        return "validation"
    return "test"


def _chronological_split(timestamp: pd.Series, config: dict) -> pd.Series:
    pairing = config.get("day3_pairing", {})
    train_end = pd.Timestamp(pairing.get("chronological_train_end", "2025-06-30 23:59:59"))
    validation_end = pd.Timestamp(pairing.get("chronological_validation_end", "2025-10-31 23:59:59"))
    result = pd.Series("unassigned", index=timestamp.index, dtype="string")
    valid = timestamp.notna()
    result.loc[valid & timestamp.le(train_end)] = "train"
    result.loc[valid & timestamp.gt(train_end) & timestamp.le(validation_end)] = "validation"
    result.loc[valid & timestamp.gt(validation_end)] = "test"
    return result


def _mark(frame: pd.DataFrame, mask: pd.Series, flag: str) -> None:
    empty = frame["pairing_flags"].eq("") & mask
    nonempty = frame["pairing_flags"].ne("") & mask
    frame.loc[empty, "pairing_flags"] = flag
    frame.loc[nonempty, "pairing_flags"] += ";" + flag


def build_day3_pairs(
    image_inventory_csv: str | Path,
    power_qc_csv: str | Path,
    config: dict,
) -> pd.DataFrame:
    """Join audited images to Day-2 power without modifying either source."""
    images = pd.read_csv(image_inventory_csv, low_memory=False)
    power = pd.read_csv(power_qc_csv, low_memory=False).rename(
        columns={"timestamp": "power_timestamp"}
    )
    required_images = {
        "sample_id", "image_timestamp", "image_timestamp_raw",
        "image_timestamp_parse_ok", "inspection_ok", "dimension_expected",
    }
    required_power = {
        "power_timestamp", "power_raw_kw", "power_qc_kw",
        "power_training_eligible_observed",
        "power_training_eligible_operational",
        "power_training_eligible_strict",
    }
    missing_images = required_images.difference(images.columns)
    missing_power = required_power.difference(power.columns)
    if missing_images or missing_power:
        raise ValueError(
            f"Missing columns: image={sorted(missing_images)}, power={sorted(missing_power)}"
        )
    if images["sample_id"].duplicated().any():
        raise ValueError("Image inventory sample_id values must be unique")

    for column in ["image_timestamp", "image_timestamp_raw"]:
        images[column] = pd.to_datetime(images[column], errors="coerce").astype(
            "datetime64[ns]"
        )
    power["power_timestamp"] = pd.to_datetime(
        power["power_timestamp"], errors="coerce"
    ).astype("datetime64[ns]")
    if power["power_timestamp"].isna().any() or power["power_timestamp"].duplicated().any():
        raise ValueError("Power timestamps must be valid and unique before pairing")

    # Day-1 inventory contains older power/covariate columns. Drop them so the
    # Day-2 QC table is the single authoritative source for power labels.
    stale_exact = {
        "power_timestamp", "power_raw_kw", "power_qc_kw", "power_pu_dc",
        "qc_hard_valid", "qc_hard_flags", "qc_soft_flags", "source_file",
        "source_row", "image_power_delta_seconds", "pairing_flags",
        "solar_elevation_deg",
    }
    stale_prefixes = ("power_training_", "power_operational_", "power_strict_")
    stale = [
        column for column in images.columns
        if column in stale_exact or column.startswith(stale_prefixes)
    ]
    images = images.drop(columns=stale, errors="ignore")

    tolerance = int(config["pairing"]["image_power_tolerance_seconds"])
    paired = pd.merge_asof(
        images.sort_values("image_timestamp"),
        power.sort_values("power_timestamp"),
        left_on="image_timestamp",
        right_on="power_timestamp",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=tolerance),
    ).sort_values(["image_timestamp", "sample_id"], na_position="last").reset_index(drop=True)
    paired["image_power_delta_seconds"] = (
        paired["image_timestamp"] - paired["power_timestamp"]
    ).dt.total_seconds()

    reused = paired["power_timestamp"].notna() & paired["power_timestamp"].duplicated(keep=False)
    paired["power_timestamp_reused"] = reused
    timestamp_ok = _as_bool(paired["image_timestamp_parse_ok"])
    inspection_ok = _as_bool(paired["inspection_ok"])
    dimensions_ok = _as_bool(paired["dimension_expected"])
    matched = paired["power_timestamp"].notna()
    within_tolerance = paired["image_power_delta_seconds"].abs().le(tolerance).fillna(False)
    paired["pairing_flags"] = ""
    _mark(paired, ~timestamp_ok, "image_timestamp_parse_error")
    _mark(paired, ~inspection_ok, "image_inspection_failed")
    _mark(paired, ~dimensions_ok, "unexpected_image_dimensions")
    _mark(paired, ~matched, "power_unmatched")
    _mark(paired, reused, "power_timestamp_reused")
    if "image_clock_correction_applied" in paired:
        _mark(
            paired,
            ~_as_bool(paired["image_clock_correction_applied"]),
            "image_clock_uncorrected_review",
        )

    paired["image_qc_valid"] = timestamp_ok & inspection_ok & dimensions_ok
    paired["pairing_qc_valid"] = (
        paired["image_qc_valid"] & matched & within_tolerance & ~reused
    )
    minimum_elevation = float(
        config.get("day3_pairing", {}).get("minimum_solar_elevation_degrees", 5.0)
    )
    daylight = pd.to_numeric(paired["solar_elevation_deg"], errors="coerce").ge(
        minimum_elevation
    )
    paired["daylight_modeling_eligible"] = daylight
    for subset in ("observed", "operational", "strict"):
        source = f"power_training_eligible_{subset}"
        paired[f"sample_eligible_{subset}"] = (
            paired["pairing_qc_valid"] & daylight & _as_bool(paired[source])
        )
    paired["sample_eligible_primary"] = paired["sample_eligible_operational"]
    clock_corrected = (
        _as_bool(paired["image_clock_correction_applied"])
        if "image_clock_correction_applied" in paired
        else pd.Series(False, index=paired.index)
    )
    paired["sample_eligible_operational_clock_strict"] = (
        paired["sample_eligible_operational"] & clock_corrected
    )

    paired["sample_date"] = paired["image_timestamp"].dt.strftime("%Y-%m-%d")
    paired["split_chronological"] = _chronological_split(
        paired["image_timestamp"], config
    )
    site_id = str(config.get("site", {}).get("site_id", "site"))
    seed = int(config.get("day3_pairing", {}).get("seasonal_day_hash_seed", 20260822))
    paired["split_seasonal_day_hash"] = paired["image_timestamp"].dt.normalize().map(
        lambda value: _hash_day_split(value, site_id, seed)
    )
    paired["timezone"] = config["site"]["timezone"]
    return paired


def summarize_day3(paired: pd.DataFrame, config: dict) -> dict:
    matched = paired["power_timestamp"].notna()
    delta = paired.loc[matched, "image_power_delta_seconds"].abs()
    primary = _as_bool(paired["sample_eligible_primary"])
    summary = {
        "schema_version": "3.0-day3-pairing",
        "row_count": int(len(paired)),
        "image_time_start": str(paired["image_timestamp"].min()),
        "image_time_end": str(paired["image_timestamp"].max()),
        "power_matched_count": int(matched.sum()),
        "power_unmatched_count": int((~matched).sum()),
        "power_match_fraction": float(matched.mean()),
        "power_timestamp_reused_row_count": int(_as_bool(paired["power_timestamp_reused"]).sum()),
        "image_qc_valid_count": int(_as_bool(paired["image_qc_valid"]).sum()),
        "pairing_qc_valid_count": int(_as_bool(paired["pairing_qc_valid"]).sum()),
        "sample_eligible_observed_count": int(_as_bool(paired["sample_eligible_observed"]).sum()),
        "sample_eligible_operational_count": int(primary.sum()),
        "sample_eligible_strict_count": int(_as_bool(paired["sample_eligible_strict"]).sum()),
        "sample_eligible_operational_clock_strict_count": int(
            _as_bool(paired["sample_eligible_operational_clock_strict"]).sum()
        ),
        "absolute_match_error_seconds": {
            "median": float(delta.median()),
            "p90": float(delta.quantile(0.90)),
            "p95": float(delta.quantile(0.95)),
            "p99": float(delta.quantile(0.99)),
            "maximum": float(delta.max()),
        },
        "primary_split_counts": {
            key: int(value)
            for key, value in paired.loc[primary, "split_chronological"].value_counts().sort_index().items()
        },
        "primary_split_unique_days": {
            key: int(value)
            for key, value in paired.loc[primary].groupby("split_chronological")["sample_date"].nunique().sort_index().items()
        },
        "seasonal_day_hash_primary_split_counts": {
            key: int(value)
            for key, value in paired.loc[primary, "split_seasonal_day_hash"].value_counts().sort_index().items()
        },
        "rules": {
            "image_power_tolerance_seconds": int(config["pairing"]["image_power_tolerance_seconds"]),
            "minimum_solar_elevation_degrees": float(config.get("day3_pairing", {}).get("minimum_solar_elevation_degrees", 5.0)),
            "primary_power_subset": "operational",
            "primary_split": "chronological",
            "split_unit": "whole calendar day",
        },
        "notes": [
            "Original image archives and Day-2 power observations are not modified.",
            "Primary samples require audited images, unique power pairing, daylight, and Day-2 operational eligibility.",
            "The chronological split is the primary leakage-resistant evaluation; the seasonally mixed day-hash split is sensitivity-only.",
            "Uncorrected clock segments are retained only when their direct match error is within tolerance and are explicitly flagged for review.",
        ],
    }
    return summary


def _save_figures(paired: pd.DataFrame, output_dir: Path) -> list[Path]:
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    matched = paired["power_timestamp"].notna()
    delta = paired.loc[matched, "image_power_delta_seconds"].abs()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(delta, bins=np.linspace(0, max(30.0, float(delta.max())), 61), color="#2f6fed")
    ax.axvline(30, color="red", linestyle="--", label="30 s tolerance")
    ax.set(xlabel="Absolute image-power time error (s)", ylabel="Image count", title="Image-power matching error")
    ax.legend(); fig.tight_layout()
    path = figures / "matching_error_histogram.png"; fig.savefig(path, dpi=180); plt.close(fig); outputs.append(path)

    eligible = paired.loc[_as_bool(paired["sample_eligible_primary"])].copy()
    eligible["month"] = pd.to_datetime(eligible["image_timestamp"]).dt.to_period("M").astype(str)
    monthly = eligible.groupby(["month", "split_chronological"]).size().unstack(fill_value=0)
    order = [column for column in ["train", "validation", "test"] if column in monthly]
    fig, ax = plt.subplots(figsize=(14, 5))
    monthly[order].plot(kind="bar", stacked=True, ax=ax, color=["#2f6fed", "#f5a623", "#d64545"][:len(order)])
    ax.set(xlabel="Month", ylabel="Primary eligible image-power pairs", title="Monthly paired samples and chronological split")
    fig.tight_layout()
    path = figures / "monthly_primary_samples.png"; fig.savefig(path, dpi=180); plt.close(fig); outputs.append(path)

    daily = eligible.assign(date=pd.to_datetime(eligible["image_timestamp"]).dt.date).groupby("date").size()
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.plot(pd.to_datetime(daily.index), daily.values, linewidth=0.8, color="#2f6fed")
    ax.set(xlabel="Date", ylabel="Primary eligible samples", title="Daily image-power sample availability")
    fig.tight_layout()
    path = figures / "daily_primary_samples.png"; fig.savefig(path, dpi=180); plt.close(fig); outputs.append(path)
    return outputs


def run_day3_pairing(
    image_inventory_csv: str | Path,
    power_qc_csv: str | Path,
    output_dir: str | Path,
    config: dict,
) -> list[Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paired = build_day3_pairs(image_inventory_csv, power_qc_csv, config)
    manifest_path = output / "image_power_pairs.csv"
    paired.to_csv(manifest_path, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d %H:%M:%S.%f")

    primary = paired.loc[_as_bool(paired["sample_eligible_primary"])].copy()
    index_columns = [
        "sample_id", "image_storage", "image_container", "image_member",
        "image_timestamp_raw", "image_timestamp", "power_timestamp",
        "image_power_delta_seconds", "power_raw_kw", "power_operational_qc_kw",
        "solar_elevation_deg", "split_chronological", "split_seasonal_day_hash",
    ]
    primary_path = output / "primary_operational_index.csv"
    primary[[c for c in index_columns if c in primary]].to_csv(primary_path, index=False, encoding="utf-8-sig")
    for split in ("train", "validation", "test"):
        part = primary.loc[primary["split_chronological"].eq(split)]
        part[[c for c in index_columns if c in part]].to_csv(output / f"primary_{split}_index.csv", index=False, encoding="utf-8-sig")

    monthly = paired.assign(month=paired["image_timestamp"].dt.to_period("M").astype(str)).groupby("month").agg(
        images=("sample_id", "size"),
        power_matched=("power_timestamp", "count"),
        image_qc_valid=("image_qc_valid", "sum"),
        pairing_qc_valid=("pairing_qc_valid", "sum"),
        eligible_observed=("sample_eligible_observed", "sum"),
        eligible_operational=("sample_eligible_operational", "sum"),
        eligible_strict=("sample_eligible_strict", "sum"),
    ).reset_index()
    monthly_path = output / "monthly_pairing_summary.csv"
    monthly.to_csv(monthly_path, index=False, encoding="utf-8-sig")
    summary_path = output / "day3_summary.json"
    summary_path.write_text(json.dumps(summarize_day3(paired, config), ensure_ascii=False, indent=2), encoding="utf-8")
    figures = _save_figures(paired, output)
    return [manifest_path, primary_path, monthly_path, summary_path, *figures]
