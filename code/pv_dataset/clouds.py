from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def _otsu_threshold(values: np.ndarray, low: float = 0.35, high: float = 1.5) -> float:
    clipped = np.clip(values[np.isfinite(values)], low, high)
    if clipped.size < 100:
        return 0.9
    hist, edges = np.histogram(clipped, bins=256, range=(low, high))
    probability = hist.astype(float) / max(hist.sum(), 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    omega = np.cumsum(probability)
    mu = np.cumsum(probability * centers)
    total_mu = mu[-1]
    denominator = omega * (1.0 - omega)
    between = np.zeros_like(denominator)
    valid = denominator > 0
    between[valid] = (total_mu * omega[valid] - mu[valid]) ** 2 / denominator[valid]
    return float(centers[int(np.argmax(between))])


def red_blue_ratio_cloud_baseline(
    input_path: str | Path,
    output_mask_path: str | Path | None = None,
    fixed_threshold: float | None = None,
    valid_sky_mask_path: str | Path | None = None,
) -> dict:
    """Heuristic cloud mask for baseline/QC use, not manual ground truth."""
    with Image.open(input_path) as opened:
        rgb = np.asarray(opened.convert("RGB"), dtype=np.float32) / 255.0
    height, width = rgb.shape[:2]
    yy, xx = np.ogrid[:height, :width]
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    radius = min(width, height) / 2.0
    optical_circle = ((xx - cx) ** 2 + (yy - cy) ** 2) <= radius**2
    if valid_sky_mask_path:
        with Image.open(valid_sky_mask_path) as opened_mask:
            external_mask = np.asarray(
                opened_mask.convert("L").resize((width, height), Image.Resampling.NEAREST)
            ) > 0
        sky = optical_circle & external_mask
    else:
        sky = optical_circle
    brightness = rgb.mean(axis=2)
    saturated = rgb.max(axis=2) >= 0.995
    dark = brightness <= 0.03
    valid_sky = sky & ~dark
    red, blue = rgb[:, :, 0], rgb[:, :, 2]
    ratio = (red + 1e-4) / (blue + 1e-4)
    threshold = float(fixed_threshold) if fixed_threshold is not None else _otsu_threshold(ratio[valid_sky & ~saturated])
    cloud = valid_sky & (ratio >= threshold)
    valid_count = int(valid_sky.sum())
    cloud_fraction = float(cloud.sum() / valid_count) if valid_count else float("nan")
    if output_mask_path:
        output = Path(output_mask_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[cloud] = 255
        Image.fromarray(mask, mode="L").save(output)
    return {
        "method": "red_blue_ratio_heuristic",
        "threshold": threshold,
        "cloud_fraction_of_valid_sky": cloud_fraction,
        "valid_sky_pixel_count": valid_count,
        "saturated_sky_fraction": float((saturated & sky).sum() / max(int(sky.sum()), 1)),
        "external_valid_sky_mask_used": bool(valid_sky_mask_path),
        "warning": "Heuristic baseline only; validate against manually labelled cloud masks before use as ground truth",
    }
