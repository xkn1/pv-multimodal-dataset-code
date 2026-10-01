from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from PIL import Image


ROOT: Path
OUT: Path

NAVY = "#123B52"
BLUE = "#176B87"
TEAL = "#45A99A"
CYAN = "#63C5C2"
MINT = "#BFE3D0"
PALE = "#F3FAF7"
ORANGE = "#E69F62"
RED = "#C95C68"
GRAY = "#71858C"
CMAP = LinearSegmentedColormap.from_list(
    "paper_teal", ["#F3FAF7", "#CDEBE0", "#82D1C7", "#2D9D9A", "#12566B", "#172B4D"]
)

mpl.rcParams.update(
    {
        "font.family": "DejaVu Serif",
        "font.size": 9,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.labelcolor": NAVY,
        "axes.edgecolor": GRAY,
        "xtick.color": NAVY,
        "ytick.color": NAVY,
        "text.color": NAVY,
        "axes.facecolor": "#FBFEFD",
        "figure.facecolor": "white",
        "grid.color": "#DCECE8",
        "grid.linewidth": 0.65,
        "pdf.fonttype": 42,
    }
)


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def box(ax, xy, wh, text, color, *, fontsize=8.5, lw=1.1):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        facecolor=color,
        edgecolor=NAVY,
        linewidth=lw,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)


def arrow(ax, start, end, color=TEAL, lw=1.5):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=11, color=color, lw=lw))


def build_workflow() -> None:
    fig, ax = plt.subplots(figsize=(11.3, 6.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.03, 0.95, "Raw multimodal sources", weight="bold", fontsize=11)
    ax.text(0.36, 0.95, "Unified processing", weight="bold", fontsize=11)
    ax.text(0.69, 0.95, "Released data products", weight="bold", fontsize=11)

    source_y = [0.80, 0.64, 0.48, 0.32, 0.16]
    source_text = [
        "All-sky images\n122,213 files",
        "PV power\n226,685 QC records",
        "GHI and ground weather",
        "ERA5 and GFS",
        "IFS archived forecasts\n1,429 runs",
    ]
    source_colors = [MINT, "#D9EEF1", "#DDF1E8", "#E7EDF5", "#DCE5EF"]
    for y, text, color in zip(source_y, source_text, source_colors):
        box(ax, (0.03, y - 0.055), (0.23, 0.105), text, color)

    process_y = [0.79, 0.61, 0.43, 0.25]
    process_text = [
        "Time-zone and timestamp\nstandardization",
        "Quality control and\navailability-time audit",
        "Image-power-weather pairing\nand geometric normalization",
        "Human review, semantic labels\nand ramp-event extraction",
    ]
    for y, text in zip(process_y, process_text):
        box(ax, (0.36, y - 0.062), (0.25, 0.12), text, "#EAF6F2")
    for y0, y1 in zip(process_y[:-1], process_y[1:]):
        arrow(ax, (0.485, y0 - 0.065), (0.485, y1 + 0.065), NAVY, 1.2)

    product_y = [0.76, 0.49, 0.22]
    product_text = [
        "Product 1\nMulti-timescale multimodal tables\n96,481 paired samples",
        "Product 2\nCloud-sun-sky-obstruction labels\n500 human-reviewed masks",
        "Product 3\nPV ramp-event packages\n6,405 primary events",
    ]
    product_colors = ["#CDEBE0", "#B9E0DF", "#DDE8F3"]
    for y, text, color in zip(product_y, product_text, product_colors):
        box(ax, (0.69, y - 0.078), (0.28, 0.15), text, color, fontsize=8.8, lw=1.3)

    for sy in source_y:
        arrow(ax, (0.26, sy), (0.36, 0.61), "#86BFB9", 1.0)
    for py in product_y:
        arrow(ax, (0.61, 0.43), (0.69, py), "#4C9FA1", 1.2)

    ax.text(
        0.50, 0.055,
        "Technical validation: temporal alignment | physical consistency | geometry | archive integrity | semantic accuracy | ramp robustness",
        ha="center", va="center", fontsize=8.7,
        bbox=dict(boxstyle="round,pad=0.45", facecolor=PALE, edgecolor="#9BCFC7"),
    )
    fig.tight_layout()
    save(fig, "fig00_data_resource_workflow")


def build_geometry_example() -> None:
    raw_path = ROOT / "cloud_annotation_xanylabeling" / "CLOUD_0001_raw_preview.jpg"
    norm_path = ROOT / "cloud_annotation_xanylabeling" / "final_500_annotations_v2" / "images" / "CLOUD_0001.jpg"
    mask_path = ROOT / "cloud_annotation_xanylabeling" / "final_500_annotations_v2" / "semantic_masks" / "CLOUD_0001.png"
    static_path = ROOT / "outputs" / "day4_baseline_review" / "static_sky_mask_128.png"

    raw = np.asarray(Image.open(raw_path).convert("RGB"))
    norm = np.asarray(Image.open(norm_path).convert("RGB"))
    sem = np.asarray(Image.open(mask_path))
    static = np.asarray(Image.open(static_path).convert("L"))
    static = np.asarray(Image.fromarray(static).resize((norm.shape[1], norm.shape[0]), Image.Resampling.NEAREST))

    palette = np.array([[0, 0, 0], [201, 92, 104], [255, 215, 70], [83, 126, 159], [130, 139, 141]], dtype=np.uint8)
    sem_rgb = palette[np.clip(sem.astype(int), 0, len(palette) - 1)]
    overlay = (0.62 * norm + 0.38 * sem_rgb).astype(np.uint8)

    # Crop the original around the optical disk to make the comparison readable.
    h, w = raw.shape[:2]
    side = min(h, w)
    raw_crop = raw[(h - side) // 2:(h + side) // 2, (w - side) // 2:(w + side) // 2]

    fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.2))
    panels = [raw_crop, norm, static, overlay]
    titles = [
        "a  Raw fisheye\nimage",
        "b  Fixed-view\nstandardized image",
        "c  Valid-sky and\nstatic mask",
        "d  Four-class\nsemantic overlay",
    ]
    cmaps = [None, None, "gray", None]
    for ax, image, title, cmap in zip(axes, panels, titles, cmaps):
        ax.imshow(image, cmap=cmap, vmin=0 if cmap else None, vmax=255 if cmap else None)
        ax.set_title(title, pad=5, fontsize=8.8)
        ax.axis("off")
    fig.subplots_adjust(wspace=0.04)
    save(fig, "fig00b_image_processing_examples")


def build_dataset_overview() -> None:
    monthly = pd.read_csv(ROOT / "outputs" / "day15_distribution_review" / "monthly_dataset_statistics.csv")
    manifest = pd.read_csv(ROOT / "cloud_annotation_xanylabeling" / "annotation_manifest_500.csv")
    power = pd.read_csv(ROOT / "work" / "validation_output" / "power" / "power_qc.csv", usecols=["power_qc_kw"])
    power_values = power.power_qc_kw.dropna().clip(0, 16.5)
    x = np.arange(len(monthly))

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5))

    ax = axes[0, 0]
    y = monthly.final_samples.to_numpy()
    ax.fill_between(x, y, color=CYAN, alpha=0.30)
    ax.plot(x, y, color=BLUE, lw=2.1, marker="o", ms=3.4)
    ax.set_title("a  Monthly paired image-power samples", loc="left")
    ax.set_ylabel("Paired samples")
    ax.set_xticks(x[::3], monthly.month.iloc[::3], rotation=35, ha="right")
    ax.grid(axis="y")
    ax.annotate(f"Total = {int(y.sum()):,}", (x[-1], y[-1]), xytext=(-8, 14), textcoords="offset points", ha="right")

    ax = axes[0, 1]
    coverage = np.vstack(
        [
            monthly.final_active_days / monthly.study_window_calendar_days * 100,
            monthly.ghi_match_pct_final,
            monthly.weather_match_pct_final,
        ]
    )
    im = ax.imshow(coverage, aspect="auto", cmap=CMAP, vmin=75, vmax=100)
    ax.set_title("b  Monthly source availability", loc="left")
    ax.set_yticks([0, 1, 2], ["Image-active days", "GHI match", "Weather match"])
    ax.set_xticks(x[::3], monthly.month.iloc[::3], rotation=35, ha="right")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.035)
    cbar.set_label("Availability (%)")

    ax = axes[1, 0]
    bins = np.linspace(0, 16.5, 67)
    hist, edges = np.histogram(power_values, bins=bins, density=True)
    centers = (edges[:-1] + edges[1:]) / 2
    ax.fill_between(centers, hist, color=MINT, alpha=0.9)
    ax.plot(centers, hist, color=NAVY, lw=1.7)
    ax.axvline(15, color=ORANGE, lw=1.4, ls="--", label="15 kW AC rating")
    ax.set_title("c  QC power-value distribution", loc="left")
    ax.set_xlabel("AC power (kW)")
    ax.set_ylabel("Density")
    ax.grid(axis="y")
    ax.legend(frameon=False, loc="upper right")

    ax = axes[1, 1]
    season_order = ["spring", "summer", "autumn", "winter"]
    season_labels = ["Spring", "Summer", "Autumn", "Winter"]
    manifest["season"] = pd.Categorical(manifest.season, season_order, ordered=True)
    grouped = manifest.groupby(["season", "solar_bin"], observed=True).agg(
        n=("annotation_id", "size"), elevation=("solar_elevation_deg", "median")
    ).reset_index()
    solar_order = {name: idx for idx, name in enumerate(sorted(grouped.solar_bin.unique()))}
    for idx, season in enumerate(season_order):
        subset = grouped[grouped.season == season]
        ax.scatter(
            np.full(len(subset), idx), subset.elevation,
            s=30 + subset.n * 7, c=subset.elevation, cmap=CMAP,
            vmin=manifest.solar_elevation_deg.min(), vmax=manifest.solar_elevation_deg.max(),
            edgecolor="white", linewidth=0.8, alpha=0.92,
        )
    ax.set_title("d  Human-reference sampling coverage", loc="left")
    ax.set_xticks(range(4), season_labels)
    ax.set_ylabel("Median solar elevation (deg)")
    ax.grid(axis="y")
    ax.text(0.98, 0.04, "Bubble area represents sample count\n(n = 500)", transform=ax.transAxes, ha="right", va="bottom", fontsize=8)

    fig.tight_layout()
    save(fig, "fig00c_dataset_overview")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rebuild manuscript overview figures.")
    parser.add_argument("--source-root", type=Path, required=True,
                        help="Directory containing cloud_annotation_xanylabeling, outputs, and work.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    ROOT = args.source_root.resolve()
    OUT = args.output_dir.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    build_workflow()
    build_geometry_example()
    build_dataset_overview()
    print(OUT)
