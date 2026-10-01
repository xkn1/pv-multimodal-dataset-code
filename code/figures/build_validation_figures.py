from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


VAL: Path
OUT: Path

COLORS = {
    "blue": "#176B87",
    "orange": "#E69F62",
    "green": "#45A99A",
    "red": "#C95C68",
    "purple": "#536B9F",
    "gray": "#71858C",
    "navy": "#123B52",
    "cyan": "#63C5C2",
    "mint": "#BFE3D0",
}
TEAL_CMAP = LinearSegmentedColormap.from_list(
    "paper_teal", ["#F3FAF7", "#CDEBE0", "#82D1C7", "#2D9D9A", "#12566B", "#172B4D"]
)


def finish(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / f"{name}.png", dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def ecdf(ax: plt.Axes, values: pd.Series, color: str, label: str) -> None:
    x = np.sort(pd.to_numeric(values, errors="coerce").dropna().abs().to_numpy())
    if len(x) > 5000:
        idx = np.unique(np.linspace(0, len(x) - 1, 5000).astype(int))
        x = x[idx]
        y = (idx + 1) / len(values.dropna())
    else:
        y = np.arange(1, len(x) + 1) / len(x)
    ax.plot(x, y, color=color, lw=1.8, label=label)
    ax.fill_between(x, 0, y, color=color, alpha=0.11, linewidth=0)


def temporal_alignment() -> None:
    pairs_path = VAL / "day3_pairing_v1/image_power_pairs.csv"
    cols = [
        "image_power_delta_seconds",
        "image_irradiance_delta_seconds",
        "image_weather_delta_seconds",
    ]
    pairs = pd.read_csv(pairs_path, usecols=cols, low_memory=False)
    coverage = pd.read_csv(VAL / "day1_audit_v1/monthly_coverage.csv")
    missing = pd.read_csv(VAL / "day16_missingness_v1/aggregate/missingness_by_group.csv")

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.45), constrained_layout=True)
    ax = axes[0]
    ecdf(ax, pairs["image_power_delta_seconds"], COLORS["blue"], "Image–power")
    ax.axvline(1.957, color=COLORS["blue"], ls="--", lw=1)
    ax.set(xlabel="Absolute time offset (s)", ylabel="Cumulative fraction", xlim=(0, 5), ylim=(0, 1.01))
    ax.set_title("a  Image–power alignment", loc="left", fontweight="bold")
    ax.text(0.98, 0.12, "median 0.574 s\n95th 1.957 s", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7.5)

    ax = axes[1]
    ecdf(ax, pairs["image_irradiance_delta_seconds"], COLORS["orange"], "Image–GHI")
    ax.axvline(120.975, color=COLORS["orange"], ls="--", lw=1)
    ax.set(xlabel="Absolute time offset (s)", ylabel="Cumulative fraction", xlim=(0, 185), ylim=(0, 1.01))
    ax.set_title("b  Image–GHI alignment", loc="left", fontweight="bold")
    ax.text(0.98, 0.12, "median 60.262 s\n95th 120.975 s", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7.5)

    month_order = coverage["month"].astype(str).tolist()
    image_missing = 100 * coverage["zero_image_days"] / coverage["calendar_days"]
    g = missing[(missing["group_type"] == "month") & (missing["variable"] == "irradiance_value_x")]
    g = g.set_index(g["group_value"].astype(str))["missing_pct"].reindex(month_order)
    w = missing[(missing["group_type"] == "month") & (missing["variable"] == "temperature_c")]
    w = w.set_index(w["group_value"].astype(str))["missing_pct"].reindex(month_order)
    x = np.arange(len(month_order))
    ax = axes[2]
    missing_matrix = np.vstack([image_missing.to_numpy(), g.to_numpy(), w.to_numpy()])
    im = ax.imshow(missing_matrix, cmap=TEAL_CMAP, aspect="auto", vmin=0,
                   vmax=max(20, float(np.nanmax(missing_matrix))))
    ax.set_yticks([0, 1, 2], ["Zero-image days", "GHI samples", "Weather samples"])
    ax.set_xticks(x[::3], [month_order[i] for i in x[::3]], rotation=45, ha="right")
    ax.set(xlabel="Month")
    ax.set_title("c  Monthly missingness", loc="left", fontweight="bold")
    cb = fig.colorbar(im, ax=ax, pad=0.015, fraction=0.05)
    cb.set_label("Missing rate (%)")
    ax.tick_params(axis="y", length=0)
    finish(fig, "fig01_temporal_alignment_and_missingness")


def physical_consistency() -> None:
    path = VAL / "day3_pairing_v1/image_power_pairs.csv"
    cols = ["power_qc_kw", "irradiance_value_x", "solar_elevation_deg", "sample_eligible_primary"]
    d = pd.read_csv(path, usecols=cols, low_memory=False)
    d = d[d["sample_eligible_primary"].astype(bool)].dropna(subset=["power_qc_kw", "irradiance_value_x"])
    d = d[(d["irradiance_value_x"].between(0, 1400)) & (d["power_qc_kw"].between(0, 17))]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.75), constrained_layout=True)
    ax = axes[0]
    hb = ax.hexbin(d["irradiance_value_x"], d["power_qc_kw"] / 15.0,
                   gridsize=55, mincnt=1, bins="log", cmap=TEAL_CMAP, linewidths=0)
    bins = pd.cut(d["irradiance_value_x"], np.arange(0, 1450, 50), include_lowest=True)
    med = d.groupby(bins, observed=True).agg(ghi=("irradiance_value_x", "median"),
                                             p=("power_qc_kw", "median")).dropna()
    ax.plot(med["ghi"], med["p"] / 15.0, color=COLORS["navy"], lw=2.2,
            marker="o", ms=2.2, markevery=2, label="50 W m$^{-2}$ bin median")
    ax.set(xlabel="GHI (W m$^{-2}$)", ylabel="Normalized PV power", xlim=(0, 1400), ylim=(0, 1.12))
    ax.set_title("a  Cross-sensor relationship", loc="left", fontweight="bold")
    ax.text(0.03, 0.95, "Pearson r = 0.9093\nSpearman r = 0.9531", transform=ax.transAxes,
            ha="left", va="top", fontsize=7.5)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    cb = fig.colorbar(hb, ax=ax, pad=0.01)
    cb.set_label("log$_{10}$(count)")

    labels = ["Negative power", "Night GHI >20 W m$^{-2}$", "Power >15 kW",
              "High GHI–low power", "Daytime zero power"]
    values = [0, 0, 48, 26, 1967]
    ax = axes[1]
    y = np.arange(len(values))[::-1]
    point_colors = [COLORS["green"], COLORS["green"], COLORS["orange"], COLORS["orange"], COLORS["red"]]
    for yi, value, color in zip(y, values, point_colors):
        ax.hlines(yi, 0, value, color=color, alpha=0.45, lw=1.5)
        ax.scatter(value, yi, s=54, color=color, edgecolor="white", linewidth=0.8, zorder=3)
        ax.annotate(f"{value:,}", (value, yi), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=7.5)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_yticks(y, labels)
    ax.set(xlabel="Flagged records (symlog scale)", xlim=(-0.15, 4200))
    ax.set_title("b  Physical QC flags", loc="left", fontweight="bold")
    ax.grid(axis="x", color="#DDEBE8", lw=0.55, alpha=0.85)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    finish(fig, "fig02_power_ghi_physical_consistency")


def geometry_validation() -> None:
    base = VAL / "day4_preprocess_baseline_v1/rectilinear_v4"
    candidates = pd.read_csv(base / "sun_detection_candidates.csv")
    d = candidates[candidates["orientation_consistent_candidate"].astype(bool)].copy()
    fit = json.loads((base / "sun_camera_fit.json").read_text())["models"][0]
    theta = np.deg2rad(d["solar_zenith_deg"].to_numpy())
    radius = fit["radial_scale_px_128"] * np.tan(theta / 2)
    angle = np.deg2rad(d["solar_azimuth_deg"].to_numpy() + fit["image_angle_minus_solar_azimuth_deg"])
    pred_x = fit["center_x_128"] + radius * np.cos(angle)
    pred_y = fit["center_y_128"] + radius * np.sin(angle)
    error = np.hypot(d["sun_x_128"].to_numpy() - pred_x, d["sun_y_128"].to_numpy() - pred_y)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.45), constrained_layout=True)
    ax = axes[0]
    years = pd.to_datetime(d["image_timestamp"]).dt.year
    for year, color in [(2024, COLORS["blue"]), (2025, COLORS["orange"])]:
        m = years.eq(year)
        ax.scatter(d.loc[m, "sun_x_128"], d.loc[m, "sun_y_128"], s=4, alpha=0.28, color=color, label=str(year))
    ax.scatter(pred_x, pred_y, s=3, alpha=0.28, color=COLORS["red"], label="Fitted positions")
    ax.set(xlabel="Image x (px)", ylabel="Image y (px)", xlim=(0, 128), ylim=(128, 0), aspect="equal")
    ax.set_title("a  Observed and fitted sun track", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=7, loc="upper right")

    ax = axes[1]
    ax.hist(error, bins=np.arange(0, 17, 0.5), color=COLORS["blue"], alpha=0.85)
    ax.axvline(3.183, color=COLORS["orange"], ls="--", lw=1.4, label="Median 3.18 px")
    ax.axvline(8.235, color=COLORS["red"], ls=":", lw=1.4, label="95th 8.24 px")
    ax.set(xlabel="Reprojection error (px)", ylabel="Sun detections")
    ax.set_title("b  Reprojection error", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[2]
    vals = [fit["cross_year"]["2024"]["image_angle_minus_solar_azimuth_deg"],
            fit["cross_year"]["2025"]["image_angle_minus_solar_azimuth_deg"]]
    ax.hlines(0, vals[1], vals[0], color="#B8D8D3", lw=3, zorder=1)
    ax.scatter(vals[0], 0, s=90, color=COLORS["blue"], edgecolor="white", linewidth=1.0, zorder=3)
    ax.scatter(vals[1], 0, s=90, color=COLORS["orange"], edgecolor="white", linewidth=1.0, zorder=3)
    ax.annotate(f"2024  {vals[0]:.3f}°", (vals[0], 0), xytext=(0, 13), textcoords="offset points",
                ha="center", fontsize=7.5)
    ax.annotate(f"2025  {vals[1]:.3f}°", (vals[1], 0), xytext=(0, -18), textcoords="offset points",
                ha="center", fontsize=7.5)
    ax.set_xlim(110.55, 111.22)
    ax.set_ylim(-0.45, 0.45)
    ax.set_yticks([])
    ax.set(xlabel="North-angle estimate (°)")
    ax.set_title("c  Cross-year orientation", loc="left", fontweight="bold")
    ax.text(0.5, 0.15, "cross-year difference = 0.261°\nstatic-mask IoU = 0.919",
            transform=ax.transAxes, ha="center", fontsize=7.5)
    ax.spines["left"].set_visible(False)
    finish(fig, "fig03_all_sky_geometry_validation")


def nwp_integrity() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.45), constrained_layout=True)
    ax = axes[0]
    names = ["ERA5 hours", "GFS hours", "IFS runs"]
    availability = [100.0, 100.0, 99.65]
    y = np.arange(len(names))[::-1]
    colors = [COLORS["gray"], COLORS["blue"], COLORS["orange"]]
    for yi, value, color in zip(y, availability, colors):
        ax.hlines(yi, 98.8, value, color=color, lw=2.2, alpha=0.55)
        ax.scatter(value, yi, s=72, color=color, edgecolor="white", linewidth=0.9, zorder=3)
        ax.annotate(f"{value:.2f}%", (value, yi), xytext=(7, 0), textcoords="offset points",
                    va="center", fontsize=7.5)
    ax.set_yticks(y, names)
    ax.set_xlim(98.8, 100.22)
    ax.set(xlabel="Archive availability (%)")
    ax.set_title("a  Archive completeness", loc="left", fontweight="bold")
    ax.grid(axis="x", color="#DDEBE8", lw=0.55, alpha=0.85)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    ax = axes[1]
    labels = ["IFS hourly\nrecords", "Key-lead\nrecords", "Unavailable-late\nrecords"]
    vals = [70021, 8574, 0]
    y = np.arange(len(labels))[::-1]
    colors = [COLORS["blue"], COLORS["orange"], COLORS["green"]]
    for yi, value, color in zip(y, vals, colors):
        ax.hlines(yi, 0, value, color=color, lw=2.0, alpha=0.50)
        ax.scatter(value, yi, s=72, color=color, edgecolor="white", linewidth=0.9, zorder=3)
        ax.annotate(f"{value:,}", (value, yi), xytext=(7, 0), textcoords="offset points",
                    va="center", fontsize=7.5)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_yticks(y, labels)
    ax.set(xlabel="Record count (symlog scale)", xlim=(-0.15, 180000))
    ax.set_title("b  Availability-time audit", loc="left", fontweight="bold")
    ax.text(0.98, 0.12, "Leakage count = 0\nIFS runs: 1,429 / 1,434",
            transform=ax.transAxes, ha="right", fontsize=7.5)
    ax.grid(axis="x", color="#DDEBE8", lw=0.55, alpha=0.85)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    finish(fig, "fig04_nwp_archive_integrity")


def cloud_validation() -> None:
    p = VAL / "day30_final500_cloud_model_eval_v1/metrics.json"
    data = json.loads(p.read_text())
    summary = next(x for x in data["summaries"] if x["model"] == "round2_multiclass_ensemble" and x["split"] == "independent410")
    classes = ["cloud", "sun", "sky"]
    iou = [summary["classes"][c]["iou"] for c in classes]
    dice = [summary["classes"][c]["dice"] for c in classes]
    weather = ["clear", "partly_cloudy", "mostly_cloudy", "overcast"]
    weather_labels = ["Clear", "Partly cloudy", "Mostly cloudy", "Overcast"]
    cloud_iou = [summary["by_weather"][w]["classes"]["cloud"]["iou"] for w in weather]
    counts = [summary["by_weather"][w]["n_images"] for w in weather]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.55), constrained_layout=True)
    ax = axes[0]
    y = np.arange(3)[::-1]
    for yi, left, right in zip(y, iou, dice):
        ax.hlines(yi, left, right, color="#B8D8D3", lw=2.4, zorder=1)
    ax.scatter(iou, y, s=58, color=COLORS["blue"], marker="o", label="IoU", zorder=3)
    ax.scatter(dice, y, s=58, color=COLORS["orange"], marker="D", label="Dice", zorder=3)
    ax.set_yticks(y, [c.capitalize() for c in classes])
    ax.set(xlabel="Score", xlim=(0.96, 1.002))
    ax.set_title("a  Internal holdout (n = 410)", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    for yi, left, right in zip(y, iou, dice):
        ax.annotate(f"{left:.3f}", (left, yi), xytext=(-5, 7), textcoords="offset points", ha="right", fontsize=7)
        ax.annotate(f"{right:.3f}", (right, yi), xytext=(5, 7), textcoords="offset points", ha="left", fontsize=7)
    ax.grid(axis="x", color="#DDEBE8", lw=0.55, alpha=0.85)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    ax = axes[1]
    y = np.arange(4)[::-1]
    colors = [COLORS["green"], COLORS["blue"], COLORS["orange"], COLORS["purple"]]
    for yi, value, color in zip(y, cloud_iou, colors):
        ax.hlines(yi, 0.95, value, color=color, alpha=0.45, lw=2.2)
    sizes = 38 + np.sqrt(np.asarray(counts)) * 4.0
    ax.scatter(cloud_iou, y, s=sizes, color=colors, edgecolor="white", linewidth=0.9, zorder=3)
    ax.set_yticks(y, weather_labels)
    ax.set(xlabel="Cloud IoU", xlim=(0.95, 0.998))
    ax.set_title("b  Stratified cloud performance", loc="left", fontweight="bold")
    for yi, value, n in zip(y, cloud_iou, counts):
        ax.annotate(f"{value:.3f}  (n={n})", (value, yi), xytext=(7, 0), textcoords="offset points",
                    va="center", fontsize=7)
    ax.grid(axis="x", color="#DDEBE8", lw=0.55, alpha=0.85)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    finish(fig, "fig05_semantic_label_validation")


def ramp_robustness() -> None:
    summary = pd.read_json(VAL / "scientific_data_ramp_sensitivity_v1/ramp_summary.json")
    d = summary[summary["scheme"].eq("capacity_fraction")].copy()
    d["threshold_pct"] = (100*d["threshold_pu"]).round().astype(int)
    grid = d.pivot(index="horizon_minutes", columns="threshold_pct", values="event_count").sort_index()

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55), constrained_layout=True)
    ax = axes[0]
    im = ax.imshow(grid.to_numpy(), cmap=TEAL_CMAP, aspect="auto")
    ax.set_xticks(np.arange(len(grid.columns)), [f"{v}%" for v in grid.columns])
    ax.set_yticks(np.arange(len(grid.index)), [f"{v} min" for v in grid.index])
    ax.set(xlabel="Threshold", ylabel="Window")
    ax.set_title("a  Merged event count", loc="left", fontweight="bold")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(j, i, f"{grid.iloc[i,j]:,}", ha="center", va="center", fontsize=7,
                    color="white" if grid.iloc[i,j] > grid.to_numpy().mean() else "black")

    ax = axes[1]
    horizon_rows = [(10, COLORS["blue"], "o"), (15, COLORS["orange"], "s"), (30, COLORS["green"], "^")]
    y = np.arange(len(horizon_rows))[::-1]
    for yi, (horizon, color, marker) in zip(y, horizon_rows):
        g = d[d["horizon_minutes"].eq(horizon)].sort_values("threshold_pct")
        vals = dict(zip(g["threshold_pct"], g["event_count"]))
        ax.hlines(yi, vals[15], vals[5], color=color, alpha=0.45, lw=3)
        ax.scatter(vals[5], yi, s=58, color=color, marker=marker, edgecolor="white", linewidth=0.8, zorder=3)
        ax.scatter(vals[10], yi, s=62, facecolor="white", edgecolor=color, marker=marker, linewidth=1.8, zorder=4)
        ax.scatter(vals[15], yi, s=58, color=color, marker=marker, edgecolor="white", linewidth=0.8, zorder=3)
        ax.annotate(f"5%  {vals[5]:,}", (vals[5], yi), xytext=(4, 7), textcoords="offset points", fontsize=6.8)
        ax.annotate(f"10%  {vals[10]:,}", (vals[10], yi), xytext=(0, -13), textcoords="offset points", ha="center", fontsize=6.8)
        ax.annotate(f"15%  {vals[15]:,}", (vals[15], yi), xytext=(-4, 7), textcoords="offset points", ha="right", fontsize=6.8)
    ax.set_yticks(y, ["10 min", "15 min", "30 min"])
    ax.set(xlabel="Merged events")
    ax.set_title("b  Threshold retention", loc="left", fontweight="bold")
    ax.grid(axis="x", color="#DDEBE8", lw=0.55, alpha=0.85)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    primary = d[(d["horizon_minutes"].eq(15)) & (d["threshold_pct"].eq(10))].iloc[0]
    ax = axes[2]
    vals = [primary["ramp_up_event_count"], primary["ramp_down_event_count"]]
    ax.set_axis_off()
    ax.set_title("c  Primary definition", loc="left", fontweight="bold")
    top = ax.inset_axes([0.10, 0.53, 0.84, 0.30])
    top.hlines(0, min(vals), max(vals), color="#B8D8D3", lw=3)
    top.scatter(vals[0], 0, s=78, color=COLORS["orange"], edgecolor="white", linewidth=0.9, zorder=3)
    top.scatter(vals[1], 0, s=78, color=COLORS["blue"], edgecolor="white", linewidth=0.9, zorder=3)
    top.annotate(f"Ramp-up\n{int(vals[0]):,}", (vals[0], 0), xytext=(0, 12), textcoords="offset points",
                 ha="center", fontsize=7)
    top.annotate(f"Ramp-down\n{int(vals[1]):,}", (vals[1], 0), xytext=(0, -23), textcoords="offset points",
                 ha="center", fontsize=7)
    top.set_xlim(3150, 3255)
    top.set_ylim(-0.5, 0.5)
    top.set_yticks([])
    top.set_xlabel("Merged events", fontsize=7)
    top.tick_params(axis="x", labelsize=6.5)
    top.spines["left"].set_visible(False)
    top.spines["top"].set_visible(False)
    top.spines["right"].set_visible(False)

    gauge = ax.inset_axes([0.10, 0.12, 0.84, 0.22])
    agreement = 74.87
    gauge.hlines(0, 0, 100, color="#DDEBE8", lw=3)
    gauge.hlines(0, 0, agreement, color=COLORS["green"], lw=3)
    gauge.scatter(agreement, 0, s=72, color=COLORS["green"], edgecolor="white", linewidth=0.9, zorder=3)
    gauge.annotate(f"{agreement:.2f}%", (agreement, 0), xytext=(0, 10), textcoords="offset points",
                   ha="center", fontsize=7.5)
    gauge.set_xlim(0, 100)
    gauge.set_ylim(-0.45, 0.45)
    gauge.set_yticks([])
    gauge.set_xlabel("Power–GHI sign agreement  ·  13,523 valid candidates", fontsize=7)
    gauge.tick_params(axis="x", labelsize=6.5)
    gauge.spines["left"].set_visible(False)
    gauge.spines["top"].set_visible(False)
    gauge.spines["right"].set_visible(False)
    finish(fig, "fig06_ramp_definition_robustness")


def main() -> None:
    global VAL, OUT
    parser = argparse.ArgumentParser(description="Rebuild manuscript validation figures.")
    parser.add_argument("--validation-root", type=Path, required=True,
                        help="Directory containing day1_audit_v1, day3_pairing_v1, etc.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    VAL = args.validation_root.resolve()
    OUT = args.output_dir.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.titleweight": "bold",
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.facecolor": "#FCFEFD",
        "figure.facecolor": "white",
        "axes.edgecolor": "#3D555B",
        "axes.labelcolor": "#23383D",
        "xtick.color": "#344C52",
        "ytick.color": "#344C52",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    temporal_alignment()
    physical_consistency()
    geometry_validation()
    nwp_integrity()
    cloud_validation()
    ramp_robustness()
    print(OUT)


if __name__ == "__main__":
    main()
